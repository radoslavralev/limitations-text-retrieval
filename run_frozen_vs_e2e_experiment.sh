#!/bin/bash
# =============================================================================
# Frozen vs End-to-End Verifier Training Experiment
# =============================================================================
#
# This script runs the comprehensive verifier experiment:
# 1. Train all verifiers (F0-F4) with FROZEN encoder on structured data
# 2. Train all verifiers END-TO-END on structured data
# 3. Evaluate on NanoBEIR (NDCG, MRR, ACC@1, ACC@10)
# 4. Evaluate on MRPC negative pairs
#
# Usage:
#   ./run_frozen_vs_e2e_experiment.sh
#   ./run_frozen_vs_e2e_experiment.sh --encoder sentence-transformers/all-MiniLM-L12-v2
#   ./run_frozen_vs_e2e_experiment.sh --skip-frozen
#   ./run_frozen_vs_e2e_experiment.sh --skip-end-to-end
# =============================================================================

set -e

# =============================================================================
# Default Configuration
# =============================================================================

ENCODER="thenlper/gte-small"
DATA_PATH="./data/unified_negatives_with_nq_train.csv"
MRPC_PATH="./data/unified_negatives_test.csv"
OUTPUT_DIR="./verifier_experiment_results"

# Retrieval validation settings
# Valid datasets: climatefever, dbpedia, fever, fiqa2018, hotpotqa, msmarco, nfcorpus, nq, quoraretrieval, scidocs, arguana, scifact, touche2020
RETRIEVAL_VAL_DATASETS="nq msmarco"  # Space-separated list of NanoBEIR datasets
RETRIEVAL_VAL_STEPS=10000
USE_RETRIEVAL_EARLY_STOPPING=""  # Set to empty "" to disable

# Training settings
# Default batch size for lightweight verifiers (F0-F2)
# F3/F4 use smaller batch due to O(B^2) memory in MNRL loss
BATCH_SIZE=128

TIME_MINUTES=4  # Training time budget per verifier (iterations computed dynamically)
MAX_SEQ_LENGTH=128
EARLY_STOPPING_PATIENCE=5000
EVAL_STEPS=10000

# =============================================================================
# Scaling Law Functions
# =============================================================================
# Computes training iterations, batch sizes, and hyperparameters based on:
# - Encoder model size (for E2E learning rate/weight decay)
# - Training mode (frozen vs E2E affects throughput)
# - Verifier complexity (F0-F2 are lightweight, F3-F4 have learnable params)
#
# CALIBRATION INFO:
#   GPU: A10 (44GB VRAM)
#   Date: 2026-01-29
#   Encoder: gte-small (~33M params)
#   
# Key findings:
#   - Lightweight verifiers (F0-F2): batch 128 works, ~400 samples/sec E2E
#   - Heavy verifiers (F3-F4): batch 32 max (OOM at 64+), ~50-160 samples/sec
#   - F3 CNN is slower than F4 Transformer (conv ops vs attention)
#   - MNRL loss has O(B^2) memory scaling due to similarity matrix
#
# To recalibrate, run short benchmarks (~200 steps) and measure it/s:
#   samples/sec = it/s * batch_size
# =============================================================================

get_model_params() {
    # Returns approximate parameter count (in millions) for known encoder models
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

get_verifier_batch_size() {
    # Returns optimal batch size for verifier type
    # F3/F4 have O(B^2) memory due to MNRL loss similarity matrix
    # Calibrated on A10 GPU (44GB VRAM) with gte-small encoder
    local verifier_type="$1"
    local default_batch="$2"
    
    case "$verifier_type" in
        "F3" | "F4")
            # Heavy verifiers: CNN/Transformer + large similarity matrix
            echo 32 ;;
        *)
            # Lightweight verifiers (None, F0, F1, F2): can use full batch
            echo "$default_batch" ;;
    esac
}

get_verifier_throughput_samples_per_sec() {
    # Estimates training throughput based on encoder size, verifier type, and training mode
    # Args: encoder_params_millions, verifier_type, is_frozen (1=frozen, 0=e2e)
    #
    # ==========================================================================
    # CALIBRATION: A10 GPU (44GB VRAM), gte-small encoder (~33M params)
    # Date: 2026-01-29
    # 
    # Measured values:
    #   F0 E2E @ batch 128:    3.1 it/s  -> ~400 samples/sec
    #   F3 Frozen @ batch 32:  1.63 it/s -> ~52 samples/sec
    #   F4 E2E @ batch 32:     4.9 it/s  -> ~157 samples/sec
    #
    # Notes:
    #   - Lightweight verifiers (F0-F2): batch 128, ~400 samples/sec E2E
    #   - Heavy verifiers (F3-F4): batch 32 (OOM at 64+), ~50-160 samples/sec
    #   - F3 CNN is slower than F4 Transformer due to conv operations
    # ==========================================================================
    local encoder_params="$1"
    local verifier_type="$2"
    local is_frozen="$3"  # 1=frozen, 0=e2e
    
    # Throughput depends on verifier type (affects batch size) and training mode
    if [ "$is_frozen" -eq 1 ]; then
        # Frozen encoder: no backward pass through encoder
        case "$verifier_type" in
            "None" | "F0" | "F1" | "F2")
                # Lightweight verifiers @ batch 128
                if [ "$encoder_params" -lt 50 ]; then
                    echo 600   # Small encoder: fast forward only
                elif [ "$encoder_params" -lt 120 ]; then
                    echo 300   # Medium encoder
                else
                    echo 150   # Large encoder
                fi
                ;;
            "F3")
                # CNN verifier @ batch 32 (measured: ~52 samples/sec)
                if [ "$encoder_params" -lt 50 ]; then
                    echo 50
                elif [ "$encoder_params" -lt 120 ]; then
                    echo 30
                else
                    echo 20
                fi
                ;;
            "F4")
                # Transformer verifier @ batch 32
                if [ "$encoder_params" -lt 50 ]; then
                    echo 200
                elif [ "$encoder_params" -lt 120 ]; then
                    echo 120
                else
                    echo 60
                fi
                ;;
        esac
    else
        # E2E training: encoder backward pass dominates
        case "$verifier_type" in
            "None" | "F0" | "F1" | "F2")
                # Lightweight verifiers @ batch 128 (measured: ~400 samples/sec)
                if [ "$encoder_params" -lt 50 ]; then
                    echo 440   # Small encoder E2E
                elif [ "$encoder_params" -lt 120 ]; then
                    echo 165   # Medium encoder E2E
                else
                    echo 90   # Large encoder E2E
                fi
                ;;
            "F3")
                # CNN verifier @ batch 32 (E2E slower than frozen)
                if [ "$encoder_params" -lt 50 ]; then
                    echo 55
                elif [ "$encoder_params" -lt 120 ]; then
                    echo 35
                else
                    echo 25
                fi
                ;;
            "F4")
                # Transformer verifier @ batch 32 (measured: ~157 samples/sec)
                if [ "$encoder_params" -lt 50 ]; then
                    echo 150
                elif [ "$encoder_params" -lt 120 ]; then
                    echo 80
                else
                    echo 40
                fi
                ;;
        esac
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
    
    
    echo "$iterations"
}

compute_encoder_learning_rate() {
    # Scales encoder learning rate based on model size (for E2E training)
    # Larger models use smaller LR to avoid instability
    local params_millions="$1"
    
    if [ "$params_millions" -lt 30 ]; then
        echo "2e-5"  # Small models
    elif [ "$params_millions" -lt 50 ]; then
        echo "1.5e-5"  # Medium-small
    elif [ "$params_millions" -lt 120 ]; then
        echo "1e-5"  # Medium
    else
        echo "5e-6"  # Large models (modernbert-base, etc.)
    fi
}

compute_encoder_weight_decay() {
    # Weight decay for encoder based on model size (for E2E training)
    # Smaller models need less regularization
    local params_millions="$1"
    
    if [ "$params_millions" -lt 30 ]; then
        echo "0.001"  # Small models: minimal
    elif [ "$params_millions" -lt 50 ]; then
        echo "0.005"  # Medium-small
    elif [ "$params_millions" -lt 120 ]; then
        echo "0.01"   # Medium
    else
        echo "0.01"   # Large models
    fi
}

compute_verifier_learning_rate() {
    # Verifier learning rate based on verifier complexity
    # F0-F2: No learnable params (returns placeholder)
    # F3: ~100K-500K params -> higher LR
    # F4: ~1M+ params -> moderate LR
    local verifier_type="$1"
    
    case "$verifier_type" in
        "None" | "F0" | "F1" | "F2")
            echo "1e-4"  # Placeholder (no params to train)
            ;;
        "F3")
            echo "1e-4"  # Small CNN can handle higher LR
            ;;
        "F4")
            echo "1e-4"  # Transformer needs lower LR
            ;;
        *)
            echo "1e-4"
            ;;
    esac
}

compute_verifier_weight_decay() {
    # Verifier weight decay based on verifier complexity
    # F0-F2: No learnable params (returns 0)
    # F3-F4: Standard weight decay
    local verifier_type="$1"
    
    case "$verifier_type" in
        "None" | "F0" | "F1" | "F2")
            echo "0.0"  # No params to regularize
            ;;
        "F3")
            echo "0.01"  # Standard for small CNN
            ;;
        "F4")
            echo "0.01"  # Standard for transformer
            ;;
        *)
            echo "0.01"
            ;;
    esac
}

# Loss function settings
USE_MNRL_LOSS=""  # Empty means use default (MNRL enabled)
NO_MNRL_LOSS=""
MNRL_TEMPERATURE=0.1

# Learning rate scheduler settings
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="linear"

# Conditional penalty settings
USE_CONDITIONAL_PENALTY="" # empty means disabled
PENALTY_BETA=1.0
PENALTY_TAU=0.9

# Which experiments to run
SKIP_FROZEN=""
SKIP_E2E=""

# Verifiers to evaluate (all by default, "None" is embedding-only baseline)
VERIFIERS="None F0 F1 F2 F3 F4"

# Multi-Seed Configuration (for error bar computation)
SEEDS=(42 43 44)

# Device
DEVICE="cuda"

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --encoder)
            ENCODER="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --mrpc-path)
            MRPC_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --time-minutes)
            TIME_MINUTES="$2"
            shift 2
            ;;
        --early-stopping-patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        --eval-steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --verifiers)
            VERIFIERS="$2"
            shift 2
            ;;
        --skip-frozen)
            SKIP_FROZEN="--skip-frozen"
            shift
            ;;
        --skip-end-to-end)
            SKIP_E2E="--skip-end-to-end"
            shift
            ;;
        --use-mnrl-loss)
            USE_MNRL_LOSS="--use-mnrl-loss"
            shift
            ;;
        --no-mnrl-loss)
            NO_MNRL_LOSS="--no-mnrl-loss"
            shift
            ;;
        --mnrl-temperature)
            MNRL_TEMPERATURE="$2"
            shift 2
            ;;
        --warmup-ratio)
            WARMUP_RATIO="$2"
            shift 2
            ;;
        --lr-scheduler-type)
            LR_SCHEDULER_TYPE="$2"
            shift 2
            ;;
        --use-conditional-penalty)
            USE_CONDITIONAL_PENALTY="--use-conditional-penalty"
            shift
            ;;
        --penalty-beta)
            PENALTY_BETA="$2"
            shift 2
            ;;
        --penalty-tau)
            PENALTY_TAU="$2"
            shift 2
            ;;
        --seeds)
            # Override default seeds (space-separated, e.g., --seeds "42 43 44")
            IFS=' ' read -ra SEEDS <<< "$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --retrieval-val-datasets)
            RETRIEVAL_VAL_DATASETS="$2"
            shift 2
            ;;
        --retrieval-val-steps)
            RETRIEVAL_VAL_STEPS="$2"
            shift 2
            ;;
        --use-retrieval-early-stopping)
            USE_RETRIEVAL_EARLY_STOPPING="--use-retrieval-early-stopping"
            shift
            ;;
        --no-retrieval-early-stopping)
            USE_RETRIEVAL_EARLY_STOPPING=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --encoder MODEL        Base encoder (default: thenlper/gte-small)"
            echo "  --data-path PATH       Path to training data (default: ./data/train_structured.csv)"
            echo "  --mrpc-path PATH       Path to MRPC negatives (default: ./data/unified_negatives_test.csv)"
            echo "  --output-dir PATH      Output directory (default: ./verifier_experiment_results)"
            echo "  --batch-size N         Training batch size (default: 16)"
            echo "  --time-minutes N       Training time budget per verifier in minutes (default: 8)"
            echo "                         Iterations and LR/weight decay are computed via scaling laws"
            echo "  --eval-steps N         Evaluation frequency in steps (default: 100)"
            echo "  --verifiers \"V1 V2\"   Space-separated verifiers (default: \"None F0 F1 F2 F3 F4\")"
            echo "  --skip-frozen          Skip frozen encoder experiments"
            echo "  --skip-end-to-end      Skip end-to-end experiments"
            echo "  --use-mnrl-loss        Use MNRL loss with in-batch negatives (default)"
            echo "  --no-mnrl-loss         Use pairwise loss instead of MNRL"
            echo "  --mnrl-temperature T   Temperature for MNRL loss (default: 0.2)"
            echo "  --warmup-ratio R       Warmup ratio for LR scheduler (default: 0.1)"
            echo "  --lr-scheduler-type T  LR scheduler: linear or cosine (default: linear)"
            echo "  --use-conditional-penalty  Enable conditional penalty mode"
            echo "  --penalty-beta B       Structural penalty strength (default: 1.0)"
            echo "  --penalty-tau T        Similarity threshold for penalty (default: 0.9)"
            echo "  --seeds \"N1 N2 N3\"    Random seeds for multi-run (default: \"42 43 44\")"
            echo "  --device DEV           Device (default: cuda)"
            echo "  --retrieval-val-datasets DATASETS  NanoBEIR datasets for retrieval validation"
            echo "  --retrieval-val-steps N   Evaluate retrieval every N steps (default: 100)"
            echo "  --use-retrieval-early-stopping  Use retrieval NDCG@10 for early stopping"
            echo "  --no-retrieval-early-stopping   Use triplet accuracy for early stopping"
            echo "  --help                 Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Compute Encoder-Level Scaling Parameters
# =============================================================================

ENCODER_PARAMS=$(get_model_params "${ENCODER}")
ENCODER_LR=$(compute_encoder_learning_rate "${ENCODER_PARAMS}")
ENCODER_WD=$(compute_encoder_weight_decay "${ENCODER_PARAMS}")

# =============================================================================
# Create Timestamped Output Directory (once for all verifiers)
# =============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ENCODER_SHORT_NAME=$(basename "${ENCODER}")
EXPERIMENT_OUTPUT_DIR="${OUTPUT_DIR}/${ENCODER_SHORT_NAME}_${TIMESTAMP}"
mkdir -p "${EXPERIMENT_OUTPUT_DIR}"

# =============================================================================
# Print Configuration
# =============================================================================

echo "============================================================"
echo "Frozen vs End-to-End Verifier Experiment (Multi-Seed)"
echo "============================================================"
echo "Encoder: ${ENCODER}"
echo "Data: ${DATA_PATH}"
echo "MRPC: ${MRPC_PATH}"
echo "Output: ${EXPERIMENT_OUTPUT_DIR}"
echo ""
echo "Multi-Seed Configuration:"
echo "  Seeds: ${SEEDS[*]} (${#SEEDS[@]} seeds per configuration)"
echo ""
echo "Training Settings:"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Time Budget: ${TIME_MINUTES} minutes per verifier per seed"
echo "  Eval Steps: ${EVAL_STEPS}"
echo "  (Iterations and LR computed per verifier via scaling laws)"
echo ""
echo "Encoder Scaling (for E2E mode):"
echo "  Encoder params: ~${ENCODER_PARAMS}M"
echo "  Encoder LR: ${ENCODER_LR}"
echo "  Encoder weight decay: ${ENCODER_WD}"
echo ""
echo "Loss Settings:"
echo "  Use MNRL Loss: ${USE_MNRL_LOSS:-yes (default)}"
echo "  No MNRL Loss: ${NO_MNRL_LOSS:-no}"
echo "  MNRL Temperature: ${MNRL_TEMPERATURE}"
echo ""
echo "LR Scheduler:"
echo "  Type: ${LR_SCHEDULER_TYPE}"
echo "  Warmup Ratio: ${WARMUP_RATIO}"
echo ""
echo "Conditional Penalty:"
echo "  Enabled: ${USE_CONDITIONAL_PENALTY:-no}"
echo "  Beta: ${PENALTY_BETA}"
echo "  Tau: ${PENALTY_TAU}"
echo ""
echo "Retrieval Validation:"
echo "  Datasets: ${RETRIEVAL_VAL_DATASETS:-none}"
echo "  Eval Steps: ${RETRIEVAL_VAL_STEPS}"
echo "  Early Stopping: ${USE_RETRIEVAL_EARLY_STOPPING:-no}"
echo ""
echo "Verifiers: ${VERIFIERS}"
echo "Skip Frozen: ${SKIP_FROZEN:-no}"
echo "Skip E2E: ${SKIP_E2E:-no}"
echo "============================================================"

# =============================================================================
# Run Experiment
# =============================================================================

# Build retrieval validation arguments
RETRIEVAL_ARGS=""
if [ -n "${RETRIEVAL_VAL_DATASETS}" ]; then
    RETRIEVAL_ARGS="--retrieval-val-datasets ${RETRIEVAL_VAL_DATASETS} --retrieval-val-steps ${RETRIEVAL_VAL_STEPS}"
fi

# Count verifiers
VERIFIER_COUNT=0
for VERIFIER in ${VERIFIERS}; do
    VERIFIER_COUNT=$((VERIFIER_COUNT + 1))
done

# Multi-Seed Loop
TOTAL_SEEDS=${#SEEDS[@]}
SEED_IDX=0

for SEED in "${SEEDS[@]}"; do
    SEED_IDX=$((SEED_IDX + 1))
    
    # Create seed-specific output directory
    SEED_OUTPUT_DIR="${EXPERIMENT_OUTPUT_DIR}/seed_${SEED}"
    mkdir -p "${SEED_OUTPUT_DIR}"
    
    echo ""
    echo "============================================================"
    echo "SEED ${SEED_IDX}/${TOTAL_SEEDS}: ${SEED}"
    echo "Output: ${SEED_OUTPUT_DIR}"
    echo "============================================================"
    
    CURRENT_VERIFIER=0
    for VERIFIER in ${VERIFIERS}; do
        CURRENT_VERIFIER=$((CURRENT_VERIFIER + 1))
        
        echo ""
        echo "############################################################"
        echo "# [Seed ${SEED}] Verifier ${CURRENT_VERIFIER}/${VERIFIER_COUNT}: ${VERIFIER}"
        echo "############################################################"
        
        # Compute verifier-specific hyperparameters
        VERIFIER_LR=$(compute_verifier_learning_rate "${VERIFIER}")
        VERIFIER_WD=$(compute_verifier_weight_decay "${VERIFIER}")
        
        # Get per-verifier batch size (F3/F4 use smaller batch due to memory)
        VERIFIER_BATCH_SIZE=$(get_verifier_batch_size "${VERIFIER}" "${BATCH_SIZE}")
        
        # Compute iterations for frozen mode
        FROZEN_THROUGHPUT=$(get_verifier_throughput_samples_per_sec "${ENCODER_PARAMS}" "${VERIFIER}" 1)
        FROZEN_STEPS=$(compute_iterations "${TIME_MINUTES}" "${FROZEN_THROUGHPUT}" "${VERIFIER_BATCH_SIZE}")
        
        # Compute iterations for E2E mode
        E2E_THROUGHPUT=$(get_verifier_throughput_samples_per_sec "${ENCODER_PARAMS}" "${VERIFIER}" 0)
        E2E_STEPS=$(compute_iterations "${TIME_MINUTES}" "${E2E_THROUGHPUT}" "${VERIFIER_BATCH_SIZE}")
        
        echo "# Scaling Law Parameters for ${VERIFIER}:"
        echo "#   Batch size: ${VERIFIER_BATCH_SIZE}"
        echo "#   Verifier LR: ${VERIFIER_LR}"
        echo "#   Verifier weight decay: ${VERIFIER_WD}"
        echo "#   Frozen mode: ${FROZEN_THROUGHPUT} samples/sec -> ${FROZEN_STEPS} steps"
        echo "#   E2E mode: ${E2E_THROUGHPUT} samples/sec -> ${E2E_STEPS} steps"
        echo "############################################################"
        
        # Run frozen encoder experiment (if not skipped)
        if [ -z "${SKIP_FROZEN}" ]; then
            echo ""
            echo ">>> Running FROZEN encoder experiment for ${VERIFIER} (Seed ${SEED})..."
            
            uv run python train_and_eval_verifiers_experiment.py \
                --encoder "${ENCODER}" \
                --data-path "${DATA_PATH}" \
                --mrpc-path "${MRPC_PATH}" \
                --output-dir "${SEED_OUTPUT_DIR}" \
                --no-timestamp-subdir \
                --batch-size ${VERIFIER_BATCH_SIZE} \
                --max-steps ${FROZEN_STEPS} \
                --learning-rate ${VERIFIER_LR} \
                --encoder-learning-rate ${ENCODER_LR} \
                --weight-decay ${VERIFIER_WD} \
                --encoder-weight-decay ${ENCODER_WD} \
                --max-seq-length ${MAX_SEQ_LENGTH} \
                --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
                --eval-steps ${EVAL_STEPS} \
                --verifiers ${VERIFIER} \
                --seed ${SEED} \
                --device ${DEVICE} \
                --mnrl-temperature ${MNRL_TEMPERATURE} \
                --warmup-ratio ${WARMUP_RATIO} \
                --lr-scheduler-type ${LR_SCHEDULER_TYPE} \
                --penalty-beta ${PENALTY_BETA} \
                --penalty-tau ${PENALTY_TAU} \
                --skip-end-to-end \
                ${USE_MNRL_LOSS} \
                ${NO_MNRL_LOSS} \
                ${USE_CONDITIONAL_PENALTY} \
                ${RETRIEVAL_ARGS} \
                ${USE_RETRIEVAL_EARLY_STOPPING}
            
            echo ">>> FROZEN experiment complete for ${VERIFIER} (Seed ${SEED})"
        fi
        
        # Run end-to-end experiment (if not skipped)
        if [ -z "${SKIP_E2E}" ]; then
            echo ""
            echo ">>> Running E2E experiment for ${VERIFIER} (Seed ${SEED})..."
            
            uv run python train_and_eval_verifiers_experiment.py \
                --encoder "${ENCODER}" \
                --data-path "${DATA_PATH}" \
                --mrpc-path "${MRPC_PATH}" \
                --output-dir "${SEED_OUTPUT_DIR}" \
                --no-timestamp-subdir \
                --batch-size ${VERIFIER_BATCH_SIZE} \
                --max-steps ${E2E_STEPS} \
                --learning-rate ${VERIFIER_LR} \
                --encoder-learning-rate ${ENCODER_LR} \
                --weight-decay ${VERIFIER_WD} \
                --encoder-weight-decay ${ENCODER_WD} \
                --max-seq-length ${MAX_SEQ_LENGTH} \
                --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
                --eval-steps ${EVAL_STEPS} \
                --verifiers ${VERIFIER} \
                --seed ${SEED} \
                --device ${DEVICE} \
                --mnrl-temperature ${MNRL_TEMPERATURE} \
                --warmup-ratio ${WARMUP_RATIO} \
                --lr-scheduler-type ${LR_SCHEDULER_TYPE} \
                --penalty-beta ${PENALTY_BETA} \
                --penalty-tau ${PENALTY_TAU} \
                --skip-frozen \
                ${USE_MNRL_LOSS} \
                ${NO_MNRL_LOSS} \
                ${USE_CONDITIONAL_PENALTY} \
                ${RETRIEVAL_ARGS} \
                ${USE_RETRIEVAL_EARLY_STOPPING}
            
            echo ">>> E2E experiment complete for ${VERIFIER} (Seed ${SEED})"
        fi
    done  # End verifier loop
    
done  # End seed loop

# =============================================================================
# Post-processing
# =============================================================================

echo ""
echo "============================================================"
echo "Experiment Complete!"
echo "============================================================"
echo "Results saved to: ${EXPERIMENT_OUTPUT_DIR}"
echo ""
echo "Seeds used: ${SEEDS[*]} (${#SEEDS[@]} total)"
echo ""
echo "Per-seed results:"
for SEED in "${SEEDS[@]}"; do
    echo "  - ${EXPERIMENT_OUTPUT_DIR}/seed_${SEED}/all_results.json"
done
echo ""
echo "To view results for a single seed:"
echo "  cat ${EXPERIMENT_OUTPUT_DIR}/seed_42/all_results.json | jq '.'"
echo ""
echo "To view aggregated results with error bars, run:"
echo "  python show_verifier_results.py -r ${EXPERIMENT_OUTPUT_DIR}"
echo "============================================================"
