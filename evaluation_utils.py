"""
Shared Evaluation Utilities for Verifier Evaluation.

This module contains common classes and functions used by both:
- eval_verifiers.py (standalone NanoBEIR evaluation)
- train_and_eval_verifiers_experiment.py (comprehensive experiment with evaluation)

Components:
- NANOBEIR_DATASETS: List of NanoBEIR dataset names
- NanoBEIRDataset: Wrapper class for loading and processing NanoBEIR data
- compute_retrieval_metrics(): Compute NDCG, MRR, Accuracy, Recall at various K values
"""

from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
from datasets import load_dataset


# Mapping from lowercase dataset name to human-readable name (matches NanoBEIREvaluator)
DATASET_NAME_TO_HUMAN_READABLE = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}

# All NanoBEIR dataset names (lowercase, compatible with NanoBEIREvaluator)
NANOBEIR_DATASETS = list(DATASET_NAME_TO_HUMAN_READABLE.keys())

# Alias for backwards compatibility
NANOBEIR_DATASETS_LOWERCASE = NANOBEIR_DATASETS


class NanoBEIRDataset:
    """
    Wrapper for NanoBEIR dataset loading and processing.
    
    Loads corpus, queries, and qrels from the HuggingFace dataset and provides
    easy access to the data in a format suitable for retrieval evaluation.
    
    Uses the same loading format as sentence_transformers.evaluation.NanoBEIREvaluator:
    - Dataset ID: "sentence-transformers/NanoBEIR-en" or "lightonai/NanoBEIR-en"
    - Subsets: "corpus", "queries", "qrels"
    - Splits: "NanoMSMARCO", "NanoNQ", etc. (human-readable name with "Nano" prefix)
    
    Attributes:
        name: Dataset name (e.g., "msmarco", "nq")
        corpus: Dict mapping doc_id to document text
        queries: Dict mapping query_id to query text
        qrels: Dict mapping query_id to {doc_id: relevance}
        corpus_ids: List of corpus document IDs
        query_ids: List of query IDs
    """
    
    def __init__(self, dataset_name: str, dataset_id: str = "lightonai/NanoBEIR-en"):
        """
        Initialize NanoBEIRDataset.
        
        Args:
            dataset_name: Lowercase dataset name (e.g., "msmarco", "nq")
            dataset_id: HuggingFace dataset ID (default: "lightonai/NanoBEIR-en")
        """
        self.name = dataset_name
        
        # Convert lowercase name to human-readable split name
        # e.g., "msmarco" -> "MSMARCO" -> "NanoMSMARCO"
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower not in DATASET_NAME_TO_HUMAN_READABLE:
            raise ValueError(
                f"Dataset '{dataset_name}' is not a valid NanoBEIR dataset. "
                f"Valid names: {list(DATASET_NAME_TO_HUMAN_READABLE.keys())}"
            )
        
        human_readable = DATASET_NAME_TO_HUMAN_READABLE[dataset_name_lower]
        split_name = f"Nano{human_readable}"
        
        # Load dataset splits using same format as NanoBEIREvaluator
        # load_dataset(dataset_id, subset, split=split_name)
        corpus = load_dataset(dataset_id, "corpus", split=split_name)
        queries = load_dataset(dataset_id, "queries", split=split_name)
        qrels = load_dataset(dataset_id, "qrels", split=split_name)
        
        # Build corpus dict: id -> text
        self.corpus = {}
        for item in corpus:
            doc_id = str(item["_id"])
            text = item.get("text", "")
            title = item.get("title", "")
            if title:
                text = f"{title} {text}"
            self.corpus[doc_id] = text
        
        # Build queries dict: id -> text
        self.queries = {}
        for item in queries:
            query_id = str(item["_id"])
            self.queries[query_id] = item["text"]
        
        # Build qrels: query_id -> {doc_id: relevance}
        # Note: corpus-id may be a list in some datasets
        self.qrels = defaultdict(dict)
        for item in qrels:
            query_id = str(item["query-id"])
            corpus_ids = item.get("corpus-id")
            score = item.get("score", 1)
            
            if isinstance(corpus_ids, list):
                for doc_id in corpus_ids:
                    self.qrels[query_id][str(doc_id)] = score
            else:
                self.qrels[query_id][str(corpus_ids)] = score
        
        self.corpus_ids = list(self.corpus.keys())
        self.query_ids = list(self.queries.keys())
    
    def get_corpus_texts(self) -> List[str]:
        """Get all corpus documents as a list of texts."""
        return [self.corpus[doc_id] for doc_id in self.corpus_ids]
    
    def get_query_texts(self) -> List[str]:
        """Get all queries as a list of texts."""
        return [self.queries[query_id] for query_id in self.query_ids]


def compute_retrieval_metrics(
    rankings: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute retrieval metrics (NDCG, MRR, Accuracy, Recall) at various K values.
    
    Args:
        rankings: Dict mapping query_id to ranked list of doc_ids
        qrels: Dict mapping query_id to {doc_id: relevance}
        k_values: K values to compute metrics for (default: [1, 3, 5, 10])
    
    Returns:
        Dict of metric_name -> value (e.g., "ndcg@10" -> 0.85)
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    metrics = {}
    
    for k in k_values:
        ndcg_scores = []
        mrr_scores = []
        accuracy_scores = []
        recall_scores = []
        
        for query_id, ranked_docs in rankings.items():
            if query_id not in qrels:
                continue
            
            relevant_docs = qrels[query_id]
            ranked_docs_k = ranked_docs[:k]
            
            # NDCG@k
            dcg = 0.0
            for i, doc_id in enumerate(ranked_docs_k):
                if doc_id in relevant_docs:
                    rel = relevant_docs[doc_id]
                    dcg += rel / np.log2(i + 2)  # i+2 because positions are 1-indexed
            
            # Ideal DCG
            ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
            
            # MRR@k
            mrr = 0.0
            for i, doc_id in enumerate(ranked_docs_k):
                if doc_id in relevant_docs:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores.append(mrr)
            
            # Accuracy@k (hit rate)
            hit = any(doc_id in relevant_docs for doc_id in ranked_docs_k)
            accuracy_scores.append(float(hit))
            
            # Recall@k
            num_relevant = len(relevant_docs)
            num_retrieved_relevant = sum(1 for doc_id in ranked_docs_k if doc_id in relevant_docs)
            recall = num_retrieved_relevant / num_relevant if num_relevant > 0 else 0.0
            recall_scores.append(recall)
        
        metrics[f"ndcg@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        metrics[f"mrr@{k}"] = np.mean(mrr_scores) if mrr_scores else 0.0
        metrics[f"accuracy@{k}"] = np.mean(accuracy_scores) if accuracy_scores else 0.0
        metrics[f"recall@{k}"] = np.mean(recall_scores) if recall_scores else 0.0
    
    return metrics
