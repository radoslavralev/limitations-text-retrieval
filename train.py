"""
Training Script for Identity Trade-off Experiment.

Trains two SBERT models:
- Model A (Baseline): Trained on standard QQP triplets
- Model B (Structure-Forced): Trained on QQP + structural hard negatives

Key settings:
- Base Model: prajjwal1/bert-small (29M params) or bert-mini (11M params)
- Pooling: CLS token (not mean pooling)
- Loss: MultipleNegativesRankingLoss
- Training: SentenceTransformerTrainer with WandB logging
"""

import json
import csv
import argparse
import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    models,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator

# Optimize CUDA settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_triplets(data_path: Path) -> list:
    """Load triplets from JSON or CSV file, filtering out any with null values.
    
    Supports multiple formats:
    - JSON: List of dicts with 'anchor', 'positive', 'negative' keys
    - CSV (triplet format): Columns 'anchor', 'positive', 'negative', 'category'
    - CSV (pair format): Columns 'sentence1', 'sentence2', 'category' where:
        - sentence1 is used as both anchor and positive (identity)
        - sentence2 is the hard negative
    """
    data_path = Path(data_path)
    
    if data_path.suffix.lower() == '.csv':
        # Load from CSV format
        triplets = []
        with open(data_path, "r", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            
            # Detect format based on column names
            is_triplet_format = 'anchor' in fieldnames and 'positive' in fieldnames and 'negative' in fieldnames
            
            for row in reader:
                if is_triplet_format:
                    # Triplet format: anchor, positive, negative, category
                    anchor = row.get("anchor", "").strip()
                    positive = row.get("positive", "").strip()
                    negative = row.get("negative", "").strip()
                    category = row.get("category", "unknown").strip()
                    
                    if anchor and positive and negative:
                        triplets.append({
                            "anchor": anchor,
                            "positive": positive,
                            "negative": negative,
                            "negative_type": category,
                        })
                else:
                    # Pair format: sentence1, sentence2, category (identity pairs)
                    sentence1 = row.get("sentence1", "").strip()
                    sentence2 = row.get("sentence2", "").strip()
                    category = row.get("category", "unknown").strip()
                    
                    if sentence1 and sentence2:
                        triplets.append({
                            "anchor": sentence1,
                            "positive": sentence1,  # Identity: anchor == positive
                            "negative": sentence2,
                            "negative_type": category,
                        })
        
        format_type = "triplet" if is_triplet_format else "identity pairs"
        print(f"Loaded {len(triplets)} triplets from CSV ({format_type} format)")
        return triplets
    else:
        # Load from JSON format (original behavior)
        with open(data_path, "r") as f:
            triplets = json.load(f)
        
        # Filter out triplets with null anchor, positive, or negative
        valid_triplets = [
            t for t in triplets
            if t.get("anchor") is not None 
            and t.get("positive") is not None 
            and t.get("negative") is not None
        ]
        
        if len(valid_triplets) < len(triplets):
            print(f"Warning: Filtered out {len(triplets) - len(valid_triplets)} triplets with null values")
        
        return valid_triplets


def triplets_to_dataset(triplets: list) -> Dataset:
    """Convert triplets list to HuggingFace Dataset format."""
    return Dataset.from_dict({
        "anchor": [t["anchor"] for t in triplets],
        "positive": [t["positive"] for t in triplets],
        "negative": [t["negative"] for t in triplets],
    })


def create_model(
    model_name: str = "prajjwal1/bert-small",
    pooling_mode: str = "cls",
    max_seq_length: int = 128,
    device: str = "cuda",
) -> SentenceTransformer:
    """
    Create a SentenceTransformer model with specified pooling.
    
    Model is loaded in fp32 - the trainer handles mixed precision via bf16/fp16 flags.
    
    Args:
        model_name: HuggingFace model identifier
        pooling_mode: 'cls' for CLS token, 'mean' for mean pooling
        max_seq_length: Maximum sequence length
        device: Device to use
    
    Returns:
        SentenceTransformer model
    """
    # Load transformer model in fp32 - trainer handles mixed precision
    # word_embedding_model = models.Transformer(
    #     model_name,
    #     max_seq_length=max_seq_length,
    #     model_args={"add_pooling_layer": False},
    # )
    
    # # Configure pooling
    # pooling_model = models.Pooling(
    #     word_embedding_model.get_word_embedding_dimension(),
    #     pooling_mode_cls_token=(pooling_mode == "cls"),
    #     pooling_mode_mean_tokens=(pooling_mode == "mean"),
    #     pooling_mode_max_tokens=False,
    # )
    
    # Combine into SentenceTransformer
    model = SentenceTransformer(
        model_name,
        device=device,
    )
    model.max_seq_length = max_seq_length
    
    return model


def apply_lora(
    model: SentenceTransformer,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list = None,
) -> SentenceTransformer:
    """
    Apply LoRA (Low-Rank Adaptation) to a SentenceTransformer model.
    
    Args:
        model: SentenceTransformer model to apply LoRA to
        lora_r: LoRA rank (dimension of low-rank matrices)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
    
    Returns:
        SentenceTransformer model with LoRA applied
    """
    if target_modules is None:
        target_modules = ["Wqkv", "Wo"]
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    # Apply LoRA to the underlying transformer model (first module in the model)
    # SentenceTransformer stores modules as a ModuleList, with transformer at index 0
    transformer_module = model[0]
    if hasattr(transformer_module, 'auto_model'):
        # Apply PEFT to the underlying HuggingFace model
        # get_peft_model automatically freezes base model weights
        transformer_module.auto_model = get_peft_model(
            transformer_module.auto_model, 
            lora_config
        )
    else:
        raise ValueError("Could not find auto_model in transformer module")
    
    # Explicitly freeze all non-LoRA parameters to ensure only adapters train
    for name, param in model.named_parameters():
        if 'lora_' not in name.lower():
            param.requires_grad = False
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SBERT models for identity trade-off experiment"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to training data file (JSON or CSV, 10% used for validation). Not required for --eval-only mode."
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name", type=str, default="prajjwal1/bert-small",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--pooling", type=str, default="cls", choices=["cls", "mean"],
        help="Pooling strategy: cls or mean"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=128,
        help="Maximum sequence length"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use-lora", action="store_true",
        help="Enable LoRA (Low-Rank Adaptation) for training"
    )
    parser.add_argument(
        "--lora-r", type=int, default=8,
        help="LoRA rank (dimension of low-rank matrices)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=16,
        help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.1,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--lora-target-modules", type=str, nargs="+", 
        default=["Wqkv", "Wo"],
        help="Target modules for LoRA (default: Wqkv and Wo for ModernBERT)"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--iterations", type=int, default=10000,
        help="Total number of training iterations (steps)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--loss-scale", type=float, default=20.0,
        help="Scale for MultipleNegativesRankingLoss"
    )
    
    # Precision arguments
    parser.add_argument(
        "--bf16", action="store_true",
        help="Use bfloat16 precision (else fp16)"
    )
    
    # Dataloader arguments
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Prefetch factor for dataloader"
    )
    
    # Evaluation and saving arguments
    parser.add_argument(
        "--eval-steps", type=int, default=500,
        help="Steps between evaluations"
    )
    parser.add_argument(
        "--save-steps", type=int, default=1000,
        help="Steps between checkpoint saves"
    )
    parser.add_argument(
        "--logging-steps", type=int, default=100,
        help="Steps between logging"
    )
    parser.add_argument(
        "--save-total-limit", type=int, default=3,
        help="Maximum number of checkpoints to keep"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=5000,
        help="Maximum evaluation samples (for speed)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="./models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this training run (default: auto-generated)"
    )
    
    # WandB arguments
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="WandB run name (default: same as --run-name)"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="identity-tradeoff",
        help="WandB project name"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging"
    )
    
    # Hub arguments
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push model to HuggingFace Hub after training"
    )
    parser.add_argument(
        "--hub-organization", type=str, default=None,
        help="HuggingFace organization to push to"
    )
    parser.add_argument(
        "--hub-repo-name", type=str, default=None,
        help="Repository name on HuggingFace Hub"
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda, cpu, mps)"
    )
    
    # Eval-only mode
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation on the model (no training). Requires --model-name to be a valid model."
    )
    
    return parser.parse_args()


def run_eval_only(args):
    """Run evaluation only on a pre-trained model."""
    from pathlib import Path
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Generate run name if not provided
    if args.run_name is None:
        # Use model name as run name for eval-only
        args.run_name = Path(args.model_name).name
    
    print("=" * 60)
    print("Identity Trade-off Experiment - Evaluation Only")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Run name: {args.run_name}")
    print("=" * 60)
    
    # Create model
    print("\nLoading model...")
    model = create_model(
        model_name=args.model_name,
        pooling_mode=args.pooling,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total_params:,} parameters")
    
    # Load BEIR datasets for final evaluation
    beir_corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus", trust_remote_code=True)
    beir_queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries", trust_remote_code=True)
    beir_relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test", trust_remote_code=True)
    
    # Run final NanoBEIR evaluation with all datasets
    print("\nRunning NanoBEIR evaluation (all datasets)...")
    final_nano_beir_evaluator = NanoBEIREvaluator(
        dataset_id="lightonai/NanoBEIR-en",
        batch_size=32,
    )
    nano_beir_scores = final_nano_beir_evaluator(model=model)
    print(f"NanoBEIR scores: {nano_beir_scores}")
    
    # Run BEIR Touche-2020 evaluation
    print("\nRunning BEIR Touche-2020 evaluation...")
    
    # Concatenate title and text for the corpus
    beir_corpus = beir_corpus.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])
    
    # Shrink the corpus size to relevant documents + 30,000 random documents
    required_corpus_ids = set(map(str, beir_relevant_docs_data["corpus-id"]))
    required_corpus_ids |= set(random.sample(beir_corpus["_id"], k=30_000))
    beir_corpus = beir_corpus.filter(lambda x: x["_id"] in required_corpus_ids)
    
    # Convert the datasets to dictionaries
    beir_corpus_dict = dict(zip(beir_corpus["_id"], beir_corpus["text"]))
    beir_queries_dict = dict(zip(beir_queries["_id"], beir_queries["text"]))
    beir_relevant_docs = {}
    for qid, corpus_ids in zip(beir_relevant_docs_data["query-id"], beir_relevant_docs_data["corpus-id"]):
        qid = str(qid)
        corpus_ids = str(corpus_ids)
        if qid not in beir_relevant_docs:
            beir_relevant_docs[qid] = set()
        beir_relevant_docs[qid].add(corpus_ids)
    
    # Create BEIR evaluator
    beir_evaluator = InformationRetrievalEvaluator(
        queries=beir_queries_dict,
        corpus=beir_corpus_dict,
        relevant_docs=beir_relevant_docs,
        name="BeIR-touche2020-subset-test",
    )
    beir_scores = beir_evaluator(model=model)
    print(f"BEIR Touche-2020 scores: {beir_scores}")
    print(f"Primary metric ({beir_evaluator.primary_metric}): {beir_scores[beir_evaluator.primary_metric]}")
    
    # Save final metrics
    final_metrics = {
        "nano_beir": nano_beir_scores,
        "beir_touche2020": beir_scores,
    }
    
    # Save to <results_dir>/<backbone_name>/<run_name>/
    backbone_name = Path(args.model_name).name
    results_path = Path(args.results_dir) / backbone_name / args.run_name
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    print("\nEvaluation complete!")


def main():
    args = parse_args()
    
    # Handle eval-only mode
    if args.eval_only:
        run_eval_only(args)
        return
    
    # Validate data-path is provided for training
    if args.data_path is None:
        raise ValueError("--data-path is required for training. Use --eval-only for evaluation only mode.")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_name = Path(args.data_path).stem
        args.run_name = f"{data_name}_{args.pooling}_{timestamp}"
    
    # Set wandb run name
    if args.wandb_run_name is None:
        args.wandb_run_name = args.run_name
    
    # Create output directory
    output_path = Path(args.output_dir) / args.run_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Hub model ID
    hub_model_id = None
    if args.push_to_hub:
        hub_repo = args.hub_repo_name or args.run_name
        hub_model_id = f"{args.hub_organization}/{hub_repo}" if args.hub_organization else hub_repo
    
    print("=" * 60)
    print("Identity Trade-off Experiment - Training")
    print("=" * 60)
    print(f"Base model: {args.model_name}")
    print(f"Pooling: {args.pooling}")
    print(f"Data: {args.data_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Precision: {'bf16' if args.bf16 else 'fp16'}")
    if args.use_lora:
        print(f"LoRA: enabled (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})")
        print(f"LoRA target modules: {args.lora_target_modules}")
    else:
        print("LoRA: disabled (full fine-tuning)")
    if hub_model_id:
        print(f"Hub model ID: {hub_model_id}")
    print("=" * 60)
    
    # Load training data
    print("\nLoading training data...")
    all_triplets = load_triplets(Path(args.data_path))
    print(f"Loaded {len(all_triplets)} total examples")
    
    # Split into train (90%) and eval (10%)
    random.shuffle(all_triplets)
    split_idx = int(len(all_triplets) * 0.9)
    train_triplets = all_triplets[:split_idx]
    eval_triplets = all_triplets[split_idx:]
    
    # Subsample eval if too large
    if len(eval_triplets) > args.eval_samples:
        eval_triplets = eval_triplets[:args.eval_samples]
    
    train_dataset = triplets_to_dataset(train_triplets)
    eval_dataset = triplets_to_dataset(eval_triplets)
    print(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
    
    # Build NanoBEIR evaluator for validation
    print("\nBuilding NanoBEIR evaluator for validation...")
    nano_beir_evaluator = NanoBEIREvaluator(
        ["msmarco", "nq"],
        dataset_id="lightonai/NanoBEIR-en",
        batch_size=args.batch_size,
    )
    
    # Create model card data
    model_card_data = SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name=f"Identity Trade-off Experiment: {args.run_name}",
        model_id=hub_model_id,
        task_name="semantic similarity",
        tags=[
            "sentence-transformers",
            "semantic-similarity",
            "identity-tradeoff",
        ],
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=args.model_name,
        pooling_mode=args.pooling,
        max_seq_length=args.max_seq_length,
        device=args.device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Apply LoRA if enabled
    if args.use_lora:
        print("\nApplying LoRA...")
        model = apply_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA applied: {trainable_params:,} trainable parameters ({100 * trainable_params / total_params:.2f}%)")
    beir_corpus = load_dataset("BeIR/webis-touche2020", "corpus", split="corpus", trust_remote_code=True)
    beir_queries = load_dataset("BeIR/webis-touche2020", "queries", split="queries", trust_remote_code=True)
    beir_relevant_docs_data = load_dataset("BeIR/webis-touche2020-qrels", split="test", trust_remote_code=True)

    # Create loss function
    loss = MultipleNegativesRankingLoss(model, scale=args.loss_scale)
    
    print(f"\nTraining for {args.iterations} iterations")
    
    # Initialize training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        max_steps=args.iterations,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Precision
        bf16=args.bf16,
        fp16=not args.bf16,
        # Dataloader    
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_dir=str(output_path / "logs"),
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_cosine_ndcg@10",
        greater_is_better=True,
        # WandB
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.wandb_run_name,
        # Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=hub_model_id,
        # Misc
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        evaluator=nano_beir_evaluator,
        loss=loss,
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nTraining complete! Model saved to: {output_path}")
    
    # Push to Hub if requested
    if args.push_to_hub and hub_model_id:
        print(f"\nPushing model to HuggingFace Hub: {hub_model_id}")
        # Convert to fp32 for hub upload
        model.to(dtype=torch.float32)
        model.push_to_hub(hub_model_id, exist_ok=True)
        print(f"Model pushed successfully: https://huggingface.co/{hub_model_id}")
    
    # Run final NanoBEIR evaluation with all datasets
    print("\nRunning final NanoBEIR evaluation (all datasets)...")
    final_nano_beir_evaluator = NanoBEIREvaluator(
        dataset_id="lightonai/NanoBEIR-en",
        batch_size=32,
    )
    nano_beir_scores = final_nano_beir_evaluator(model=model, output_path=str(output_path))
    print(f"NanoBEIR scores: {nano_beir_scores}")
    
    # Run BEIR Touche-2020 evaluation
    print("\nRunning BEIR Touche-2020 evaluation...")
    
    # Load the Touche-2020 IR dataset
    
    # Concatenate title and text for the corpus
    beir_corpus = beir_corpus.map(lambda x: {'text': x['title'] + " " + x['text']}, remove_columns=['title'])
    
    # Shrink the corpus size to relevant documents + 30,000 random documents
    required_corpus_ids = set(map(str, beir_relevant_docs_data["corpus-id"]))
    required_corpus_ids |= set(random.sample(beir_corpus["_id"], k=30_000))
    beir_corpus = beir_corpus.filter(lambda x: x["_id"] in required_corpus_ids)
    
    # Convert the datasets to dictionaries
    beir_corpus_dict = dict(zip(beir_corpus["_id"], beir_corpus["text"]))
    beir_queries_dict = dict(zip(beir_queries["_id"], beir_queries["text"]))
    beir_relevant_docs = {}
    for qid, corpus_ids in zip(beir_relevant_docs_data["query-id"], beir_relevant_docs_data["corpus-id"]):
        qid = str(qid)
        corpus_ids = str(corpus_ids)
        if qid not in beir_relevant_docs:
            beir_relevant_docs[qid] = set()
        beir_relevant_docs[qid].add(corpus_ids)
    
    # Create BEIR evaluator
    beir_evaluator = InformationRetrievalEvaluator(
        queries=beir_queries_dict,
        corpus=beir_corpus_dict,
        relevant_docs=beir_relevant_docs,
        name="BeIR-touche2020-subset-test",
    )
    beir_scores = beir_evaluator(model=model, output_path=str(output_path))
    print(f"BEIR Touche-2020 scores: {beir_scores}")
    print(f"Primary metric ({beir_evaluator.primary_metric}): {beir_scores[beir_evaluator.primary_metric]}")
    
    # Save final metrics
    final_metrics = {
        "nano_beir": nano_beir_scores,
        "beir_touche2020": beir_scores,
    }
    with open(output_path / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Also save to <results_dir>/<backbone_name>/<run_name>/
    backbone_name = Path(args.model_name).name  # e.g., "all-MiniLM-L6-v2" from "sentence-transformers/all-MiniLM-L6-v2"
    results_path = Path(args.results_dir) / backbone_name / args.run_name
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
