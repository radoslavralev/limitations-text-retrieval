#!/bin/bash
# =============================================================================
# Identity Trade-off Experiment - Training Script
# =============================================================================
# Trains two models:
#   - Model A (Baseline): MS MARCO triplets only (retrieval baseline)
#   - Model B (Structured): MS MARCO + unified negatives (with hard negatives)
# 
# Uses SentenceTransformerTrainer with WandB logging
#
# Usage:
#   ./run_experiments.sh [--output-dir <path>] [--results-dir <path>]
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Multi-Seed Configuration
# =============================================================================
# Run experiments with multiple seeds for error bar computation
SEEDS=(42 43 44)

# =============================================================================
# Argument Parsing
# =============================================================================
OUTPUT_DIR_ARG=""
RESULTS_DIR_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR_ARG="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR_ARG="$2"
            shift 2
            ;;
        --seeds)
            # Override default seeds (space-separated, e.g., --seeds "42 43 44")
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--output-dir <path>] [--results-dir <path>] [--seeds \"42 43 44\"]"
            exit 1
            ;;
    esac
done

# Configuration
# Multiple base models can be specified (space-separated array)
BASE_MODELS=(
    "sentence-transformers/all-MiniLM-L6-v2"
    "sentence-transformers/all-MiniLM-L12-v2"
    "Alibaba-NLP/gte-modernbert-base"
    "thenlper/gte-small"
)
POOLING="cls"
BATCH_SIZE=128
TIME_MINUTES=4  # Training time budget per model (in minutes)

# =============================================================================
# Scaling Law Functions
# =============================================================================
# Estimates training iterations and learning rate based on model size and time budget.
# Throughput estimates are based on L4 GPU (24GB VRAM, ~30 TFLOPS FP16).
#
# The logic:
#   - Larger models have lower throughput (samples/sec)
#   - Iterations = time_budget * throughput
#   - Learning rate scales inversely with sqrt(params) - larger models use smaller LR
# =============================================================================

get_model_params() {
    # Returns approximate parameter count (in millions) for known models
    local model_name="$1"
    case "$model_name" in
        *"all-MiniLM-L6-v2"*)
            echo 22 ;;
        *"all-MiniLM-L12-v2"*)
            echo 33 ;;
        *"gte-small"*)
            echo 33 ;;
        *"gte-modernbert-base"* | *"modernbert-base"*)
            echo 149 ;;
        *"gte-base"* | *"bert-base"*)
            echo 110 ;;
        *"gte-large"* | *"bert-large"*)
            echo 335 ;;
        *)
            # Default: assume medium-sized model
            echo 100 ;;
    esac
}

get_throughput_samples_per_sec() {
    # Estimates training throughput on L4 GPU with batch_size=128, seq_len=128
    # Based on measured data: MiniLM-L6-v2 runs at ~14 it/s = 1792 samples/sec
    # Throughput scales roughly inversely with model size
    local params_millions="$1"
    
    if [ "$params_millions" -lt 30 ]; then
        # Small models (~22M MiniLM-L6): measured ~14 it/s * 128 = 1792 samples/sec
        echo 1800
    elif [ "$params_millions" -lt 50 ]; then
        # Medium-small models (~33M): ~1.5x params -> ~1200 samples/sec
        echo 1200
    elif [ "$params_millions" -lt 120 ]; then
        # Medium models (~100M): ~4.5x params -> ~400 samples/sec
        echo 400
    elif [ "$params_millions" -lt 200 ]; then
        # Medium-large models (~150M): ~7x params -> ~250 samples/sec
        echo 250
    else
        # Large models (>200M): ~120 samples/sec
        echo 120
    fi
}

compute_iterations() {
    # Computes training iterations from time budget and throughput
    # iterations = (time_minutes * 60 * throughput) / batch_size
    local time_minutes="$1"
    local throughput="$2"
    local batch_size="$3"
    
    # Calculate total samples we can process
    local total_samples=$((time_minutes * 60 * throughput))
    # Convert to iterations (steps)
    local iterations=$((total_samples / batch_size))
    
    # Ensure minimum of 500 iterations
    if [ "$iterations" -lt 500 ]; then
        iterations=500
    fi
    
    echo "$iterations"
}

compute_learning_rate() {
    # Scales learning rate based on model size
    # Base LR of 5e-5 for ~100M params, scales with sqrt(100M / params)
    # Smaller models -> higher LR, larger models -> lower LR
    local params_millions="$1"
    
    if [ "$params_millions" -lt 30 ]; then
        # Small models: 5e-5 * sqrt(100/22) ≈ 1e-4
        echo "1e-4"
    elif [ "$params_millions" -lt 50 ]; then
        # Medium-small: 5e-5 * sqrt(100/33) ≈ 8.5e-5
        echo "8e-5"
    elif [ "$params_millions" -lt 120 ]; then
        # Medium: ~5e-5
        echo "5e-5"
    elif [ "$params_millions" -lt 200 ]; then
        # Medium-large: 5e-5 * sqrt(100/150) ≈ 4e-5
        echo "4e-5"
    else
        # Large: 5e-5 * sqrt(100/335) ≈ 2.7e-5
        echo "3e-5"
    fi
}

compute_weight_decay() {
    # Weight decay scaling based on model size and original training configs
    # 
    # References:
    # - MiniLM models: Originally trained with AdamW, weight decay typically 0.01 or lower
    #   (HF card shows AdamW with 2e-5 LR, standard AdamW default is 0.01)
    # - GTE models: Multi-stage contrastive learning, BERT-based, standard 0.01
    # - ModernBERT: Trained with 0.01 weight decay
    #
    # Smaller models benefit from less regularization (fewer params = less overfitting risk)
    local params_millions="$1"
    
    if [ "$params_millions" -lt 30 ]; then
        # Small models (~22M MiniLM-L6): minimal weight decay to avoid underfitting
        echo "0.001"
    elif [ "$params_millions" -lt 50 ]; then
        # Medium-small (~33M MiniLM-L12, gte-small): light weight decay
        echo "0.005"
    elif [ "$params_millions" -lt 120 ]; then
        # Medium (~100M): standard weight decay
        echo "0.01"
    else
        # Large models (>120M modernbert-base, etc.): full regularization
        echo "0.01"
    fi
}
WARMUP_RATIO=0.1
MAX_SEQ_LENGTH=128
LOSS_SCALE=10.0

# LoRA settings (set USE_LORA="--use-lora" to enable, or "" to disable) # currently disabled
USE_LORA="" # "--use-lora"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="Wqkv Wo"  # ModernBERT attention layers

# Dataloader settings
NUM_WORKERS=1
PREFETCH_FACTOR=1

# Evaluation and logging settings
EVAL_STEPS=250
SAVE_STEPS=0
LOGGING_STEPS=250
EVAL_SAMPLES=40000

# Precision (set to --bf16 for bfloat16, or leave empty for fp16)
PRECISION=""  # or "--bf16" # currently disabled

# Directories
DATA_DIR="./data"
OUTPUT_DIR="${OUTPUT_DIR_ARG:-./models_nq}"
RESULTS_DIR="${RESULTS_DIR_ARG:-./results}"

# WandB settings
WANDB_PROJECT="identity-tradeoff"
# Uncomment to disable WandB
# NO_WANDB="--no-wandb"
NO_WANDB=""

# HuggingFace Hub settings - DISABLED for multi-seed runs
# (seed subdirectories create invalid repo names)
PUSH_TO_HUB=""
HUB_ORG=""

echo "============================================================"
echo "Identity Trade-off Experiment (Multi-Seed)"
echo "============================================================"
echo "Base Models: ${BASE_MODELS[*]}"
echo "Number of Models: ${#BASE_MODELS[@]}"
echo "Seeds: ${SEEDS[*]} (${#SEEDS[@]} seeds per configuration)"
echo "Pooling: ${POOLING}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Time Budget: ${TIME_MINUTES} minutes per model per seed"
echo "Precision: ${PRECISION:-fp16}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Results Dir: ${RESULTS_DIR}"
echo "(Iterations and LR computed per-model based on scaling laws)"
if [ -n "${USE_LORA}" ]; then
    echo "LoRA: enabled (r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT})"
    echo "LoRA Modules: ${LORA_TARGET_MODULES}"
else
    echo "LoRA: disabled (full fine-tuning)"
fi
echo "WandB Project: ${WANDB_PROJECT}"
echo "============================================================"

# Loop through all base models
TOTAL_MODELS=${#BASE_MODELS[@]}
MODEL_IDX=0

for BASE_MODEL in "${BASE_MODELS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))
    
    # Extract short model name for run naming (e.g., "all-MiniLM-L6-v2" from "sentence-transformers/all-MiniLM-L6-v2")
    MODEL_SHORT_NAME=$(basename "${BASE_MODEL}")
    
    # Compute scaling law parameters for this model
    MODEL_PARAMS=$(get_model_params "${BASE_MODEL}")
    THROUGHPUT=$(get_throughput_samples_per_sec "${MODEL_PARAMS}")
    ITERATIONS=$(compute_iterations "${TIME_MINUTES}" "${THROUGHPUT}" "${BATCH_SIZE}")
    LEARNING_RATE=$(compute_learning_rate "${MODEL_PARAMS}")
    WEIGHT_DECAY=$(compute_weight_decay "${MODEL_PARAMS}")
    
    echo ""
    echo "############################################################"
    echo "# Processing Base Model ${MODEL_IDX}/${TOTAL_MODELS}: ${BASE_MODEL}"
    echo "############################################################"
    echo "# Scaling Law Parameters:"
    echo "#   Model params: ~${MODEL_PARAMS}M"
    echo "#   Est. throughput: ~${THROUGHPUT} samples/sec on L4"
    echo "#   Computed iterations: ${ITERATIONS} (for ${TIME_MINUTES} min)"
    echo "#   Computed learning rate: ${LEARNING_RATE}"
    echo "#   Computed weight decay: ${WEIGHT_DECAY}"
    echo "############################################################"

    # -----------------------------------------------------------------------------
    # Backbone Evaluation (Pre-training baseline) - Only once, no seed needed
    # -----------------------------------------------------------------------------
    echo ""
    echo "============================================================"
    echo "[${MODEL_IDX}/${TOTAL_MODELS}] [0/3] Evaluating Backbone (Pre-training) - ${MODEL_SHORT_NAME}"
    echo "============================================================"

    uv run python train.py \
        --eval-only \
        --data-path "${DATA_DIR}/nq_triplets_100k.csv" \
        --model-name "${BASE_MODEL}" \
        --run-name "backbone" \
        --results-dir "${RESULTS_DIR}" \
        --no-wandb

    echo ""
    echo "[${MODEL_IDX}/${TOTAL_MODELS}] [0/3] Backbone evaluation complete for ${MODEL_SHORT_NAME}!"
    echo ""

    # -------------------------------------------------------------------------
    # Multi-Seed Training Loop
    # -------------------------------------------------------------------------
    TOTAL_SEEDS=${#SEEDS[@]}
    SEED_IDX=0
    
    for SEED in "${SEEDS[@]}"; do
        SEED_IDX=$((SEED_IDX + 1))
        
        # Create seed-specific directories
        SEED_OUTPUT_DIR="${OUTPUT_DIR}/seed_${SEED}"
        SEED_RESULTS_DIR="${RESULTS_DIR}"
        
        echo ""
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo ">>>> SEED ${SEED_IDX}/${TOTAL_SEEDS}: ${SEED}"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

        # -----------------------------------------------------------------------------
        # Model A: Baseline (MS MARCO triplets only)
        # -----------------------------------------------------------------------------
        echo ""
        echo "============================================================"
        echo "[${MODEL_IDX}/${TOTAL_MODELS}] [1/3] Training Model A (Baseline) - ${MODEL_SHORT_NAME} - Seed ${SEED}"
        echo "============================================================"

        uv run python train.py \
            --data-path "${DATA_DIR}/nq_triplets_100k.csv" \
            --model-name "${BASE_MODEL}" \
            --pooling "${POOLING}" \
            --batch-size ${BATCH_SIZE} \
            --iterations ${ITERATIONS} \
            --learning-rate ${LEARNING_RATE} \
            --weight-decay ${WEIGHT_DECAY} \
            --warmup-ratio ${WARMUP_RATIO} \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --loss-scale ${LOSS_SCALE} \
            --num-workers ${NUM_WORKERS} \
            --prefetch-factor ${PREFETCH_FACTOR} \
            --eval-steps ${EVAL_STEPS} \
            --save-steps ${SAVE_STEPS} \
            --logging-steps ${LOGGING_STEPS} \
            --eval-samples ${EVAL_SAMPLES} \
            --output-dir "${SEED_OUTPUT_DIR}" \
            --results-dir "${SEED_RESULTS_DIR}" \
            --run-name "model-a-baseline/seed_${SEED}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "${MODEL_SHORT_NAME}-model-a-seed${SEED}" \
            --seed ${SEED} \
            ${PRECISION} \
            ${NO_WANDB} \
            ${PUSH_TO_HUB} ${HUB_ORG} \
            ${USE_LORA} \
            --lora-r ${LORA_R} \
            --lora-alpha ${LORA_ALPHA} \
            --lora-dropout ${LORA_DROPOUT} \
            --lora-target-modules ${LORA_TARGET_MODULES}

        echo ""
        echo "[${MODEL_IDX}/${TOTAL_MODELS}] [1/3] Model A training complete for ${MODEL_SHORT_NAME} (Seed ${SEED})!"
        echo ""

        # -----------------------------------------------------------------------------
        # Model B: Structured (MS MARCO + Unified Negatives)
        # -----------------------------------------------------------------------------
        echo ""
        echo "============================================================"
        echo "[${MODEL_IDX}/${TOTAL_MODELS}] [2/3] Training Model B (Structured) - ${MODEL_SHORT_NAME} - Seed ${SEED}"
        echo "============================================================"

        uv run python train.py \
            --data-path "${DATA_DIR}/unified_negatives_with_nq_train.csv" \
            --model-name "${BASE_MODEL}" \
            --pooling "${POOLING}" \
            --batch-size ${BATCH_SIZE} \
            --iterations ${ITERATIONS} \
            --learning-rate ${LEARNING_RATE} \
            --weight-decay ${WEIGHT_DECAY} \
            --warmup-ratio ${WARMUP_RATIO} \
            --max-seq-length ${MAX_SEQ_LENGTH} \
            --loss-scale ${LOSS_SCALE} \
            --num-workers ${NUM_WORKERS} \
            --prefetch-factor ${PREFETCH_FACTOR} \
            --eval-steps ${EVAL_STEPS} \
            --save-steps ${SAVE_STEPS} \
            --logging-steps ${LOGGING_STEPS} \
            --eval-samples ${EVAL_SAMPLES} \
            --output-dir "${SEED_OUTPUT_DIR}" \
            --results-dir "${SEED_RESULTS_DIR}" \
            --run-name "model-b-structured/seed_${SEED}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "${MODEL_SHORT_NAME}-model-b-seed${SEED}" \
            --seed ${SEED} \
            ${PRECISION} \
            ${NO_WANDB} \
            ${PUSH_TO_HUB} ${HUB_ORG} \
            ${USE_LORA} \
            --lora-r ${LORA_R} \
            --lora-alpha ${LORA_ALPHA} \
            --lora-dropout ${LORA_DROPOUT} \
            --lora-target-modules ${LORA_TARGET_MODULES}

        echo ""
        echo "[${MODEL_IDX}/${TOTAL_MODELS}] [2/3] Model B training complete for ${MODEL_SHORT_NAME} (Seed ${SEED})!"
        echo ""
    
    done  # End of SEED loop

done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo "All training complete!"
echo "============================================================"
echo "Processed ${TOTAL_MODELS} base model(s) x ${#SEEDS[@]} seeds:"
for BASE_MODEL in "${BASE_MODELS[@]}"; do
    MODEL_SHORT_NAME=$(basename "${BASE_MODEL}")
    echo "  - ${MODEL_SHORT_NAME}:"
    echo "      - ${RESULTS_DIR}/${MODEL_SHORT_NAME}/backbone (evaluation only)"
    for SEED in "${SEEDS[@]}"; do
        echo "      - ${RESULTS_DIR}/${MODEL_SHORT_NAME}/model-a-baseline/seed_${SEED}"
        echo "      - ${RESULTS_DIR}/${MODEL_SHORT_NAME}/model-b-structured/seed_${SEED}"
    done
done
echo ""
echo "Seeds used: ${SEEDS[*]}"
echo "Check WandB project '${WANDB_PROJECT}' for training logs."
echo "============================================================"
