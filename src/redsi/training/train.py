#!/usr/bin/env python3
"""
ReDSI seq2seq training with Hugging Face Seq2SeqTrainer.

Expected JSONL schema:
  {"kind":"document"|"query","docid":"...","input_ids":[...],"labels":[...]}
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import socket
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, fields
from inspect import signature

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Sampler
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    get_constant_schedule_with_warmup,
    set_seed,
)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
VALIDATION_INDEX_SPLIT = "validation_index"
DOCID_VOCAB_SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, VALIDATION_INDEX_SPLIT)
GENERATION_LENGTH_SPLITS = (VALIDATION_SPLIT, VALIDATION_INDEX_SPLIT)
DATASET_FORMAT_JSON = "json"

DOCID_COL = "docid"
KIND_COL = "kind"
LABELS_COL = "labels"
TOKENS_KEY = "tokens"
TOKEN_KEY = "token"
QUERY_KIND = "query"
DOCUMENT_KIND = "document"

LOGGER_NAME = "dsi_trainer"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"

DEFAULT_DATA_DIR = "datasets/NQ/data/nq/nq10k"
DEFAULT_TRAIN_FILE = "train.jsonl"
DEFAULT_VALIDATION_FILE = "validation.jsonl"
DEFAULT_VALIDATION_INDEX_FILE = "validation_index.jsonl"
DEFAULT_TOKENS_FILE = "tokens.jsonl"
DEFAULT_TOKENS_INIT_FILE = "tokens_init.jsonl"
DEFAULT_METADATA_FILE = "metadata.json"
DEFAULT_MODEL_NAME = "google-t5/t5-base"
DEFAULT_OUTPUT_ROOT = "outputs/dsi_trainer"
DEFAULT_QUERY_INPUT_MAX_LENGTH = 64

METADATA_CONFIG_KEY = "config"
METADATA_ARGUMENTS_KEY = "arguments"
METADATA_MODEL_NAME_KEY = "model_name"

LOGGING_STRATEGY_STEPS = "steps"
EVAL_STRATEGY_EPOCH = "epoch"
EVAL_STRATEGY_STEPS = "steps"
SAVE_STRATEGY_EPOCH = "epoch"
SAVE_STRATEGY_STEPS = "steps"
REPORT_TO_TENSORBOARD = "tensorboard"
REPORT_TO_NONE = "none"
PADDING_LONGEST = "longest"
LABEL_PAD_TOKEN_ID = -100
LOSS_HF = "hf"
LOSS_T5X = "t5x"
QUERY_SAMPLING_NONE = "none"
QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH = "one_query_per_doc_per_epoch"
DOCUMENT_REPEATS_EQUAL_QUERIES = "query"
DOCUMENT_REPEATS_EQUAL_QUERIES_ALIASES = {"=", "query", "queries"}
DOCUMENT_REPEAT_SAMPLER_CHUNK_SIZE = 8192
AUTO_MAX_NEW_TOKENS = 0
ADAFACTOR_CLIP_THRESHOLD = 1.0
ADAFACTOR_RELATIVE_STEP = False
ADAFACTOR_SCALE_PARAMETER = True
ADAFACTOR_WARMUP_INIT = False

METRIC_HITS_PREFIX = "hits"
METRIC_MRR_PREFIX = "mrr"
METRIC_INVALID_RATE = "invalid_rate"
QUERY_EVAL_NAME = "query"
INDEX_EVAL_NAME = "index"
ATOMIC_TARGET_EVAL_SPLITS = (
    *GENERATION_LENGTH_SPLITS,
    QUERY_EVAL_NAME,
    INDEX_EVAL_NAME,
)
BEST_MODEL_METRIC = "eval_query_mrr@10"
EVAL_METRIC_PREFIX = "eval"
INITIAL_EVAL_METRIC_PREFIX = "initial_eval"
FINAL_EVAL_METRIC_PREFIX = "final_eval"
METRICS_HISTORY_FILE = "metrics_history.jsonl"
JOB_METRICS_DIR = "jobs"
PRECISION_AUTO = "auto"
PRECISION_FP16 = "fp16"
PRECISION_BF16 = "bf16"
PRECISION_FP32 = "fp32"
ACCELERATOR_CUDA = "cuda"
ACCELERATOR_MPS = "mps"
ACCELERATOR_CPU = "cpu"

LOG_TARGET_VOCAB_SIZE = "Target vocabulary size: {size:,}"
LOG_TRAIN_DATASET_SIZE = "Training examples: {size:,}"
LOG_TOKEN_MANIFEST = "Loaded token manifest: {size:,} tokens from {path}"
LOG_TOKEN_INIT = "Initialized {size:,} token embedding(s) from {path}"
LOG_TOKEN_INIT_SKIPPED = "Skipping token initialization from {path}: no new embedding rows were added"
LOG_METADATA = "Loaded metadata from {path}"
LOG_RUN_CONFIG = "Run config: model_name_or_path={model_name_or_path}, output_dir={output_dir}"
LOG_EVALUATION_SCHEDULE = (
    "Evaluation schedule: eval_strategy={eval_strategy}, save_strategy={save_strategy}, "
    "eval_steps={eval_steps}, save_steps={save_steps}, steps_per_epoch={steps_per_epoch}"
)
LOG_GENERATION_CONFIG = (
    "Generation config: max_new_tokens={max_new_tokens}, "
    "num_beams={num_beams}, num_return_sequences={num_return_sequences}"
)
LOG_ATOMIC_EVAL_CONFIG = (
    "Atomic target evaluation: using first decoder-step logits over {size:,} added tokens; "
    "generation and beam search are disabled."
)
LOG_OPTIMIZER_CONFIG = (
    "Optimizer config: Adafactor lr={learning_rate}, weight_decay={weight_decay}, "
    "warmup_steps={warmup_steps}, scale_parameter={scale_parameter}"
)
LOG_DEVICE_CONFIG = (
    "Device config: accelerator={accelerator}, requested_precision={requested_precision}, "
    "precision={precision}, fp16={fp16}, bf16={bf16}, tf32={tf32}, cuda={cuda}"
)
LOG_DATALOADER_WORKERS = "Dataloader workers: {workers}"
LOG_DATALOADER_PIN_MEMORY = "Dataloader pin_memory: {pin_memory}"
LOG_LOSS = "Loss: {value}"
LOG_Z_LOSS = "Z-loss: {value}"
LOG_QUERY_SAMPLING = "Query sampling: {value}"
LOG_QUERY_INPUT_MAX_LENGTH = "Query input max length: {value}"
LOG_DOCUMENT_REPEATS = "Document repeats: {value}"
LOG_DOCUMENT_REPEATS_SUMMARY = (
    "Document repeat summary: input_rows={input_rows:,}, "
    "virtual_rows={output_rows:,}, base_document_rows={base_document_rows:,}, "
    "virtual_document_rows={document_rows:,}, query_rows={query_rows:,}"
)
LOG_ONE_QUERY_PER_DOC_SAMPLER = (
    "One-query-per-doc sampler: document_rows={document_rows:,}, "
    "query_rows={query_rows:,}, query_docs={query_docs:,}, "
    "samples_per_epoch={samples_per_epoch:,}"
)
LOG_SEEN_UNSEEN_REFERENCE = (
    "Seen/unseen reference: {source}; query_docs={query_docs:,}, query_token_ids={query_token_ids:,}"
)


def setup_logger():
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger


@dataclass
class Config:
    data_dir: str = DEFAULT_DATA_DIR
    train_file: str = DEFAULT_TRAIN_FILE
    val_file: str = DEFAULT_VALIDATION_FILE
    validation_index_file: str = DEFAULT_VALIDATION_INDEX_FILE
    tokens_file: str = DEFAULT_TOKENS_FILE
    tokens_init_file: str = DEFAULT_TOKENS_INIT_FILE
    metadata_file: str = DEFAULT_METADATA_FILE
    no_validation_index: bool = False
    dataset_keep_in_memory: bool = False
    seen_unseen_reference_train_file: str = ""

    model_name_or_path: str = None
    seed: int = 42

    output_dir: str = None
    resume_from_checkpoint: str = None
    num_train_epochs: float = 500.0
    max_steps: int = -1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    per_device_train_batch_size: int = 512
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    logging_steps: int = 100
    logging_nan_inf_filter: bool = False
    fail_on_nonfinite_metrics: bool = True
    debug_nonfinite_forward: bool = False
    t5_fp16_clamp: bool = False
    eval_steps: int = 0
    save_steps: int = 0
    save_total_limit: int = 3
    save_safetensors: bool = True
    dataloader_num_workers: int = 8
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 2
    group_by_length: bool = False
    train_queries_only: bool = False
    document_repeats: str = "1"
    loss: str = LOSS_HF
    z_loss: float = 0.0
    query_sampling: str = QUERY_SAMPLING_NONE
    query_input_max_length: int = DEFAULT_QUERY_INPUT_MAX_LENGTH
    max_eval_examples: int = 0
    gradient_checkpointing: bool = False
    precision: str = PRECISION_AUTO
    fp16: bool = False
    fp16_full_eval: bool = False
    fp16_opt_level: str = "O1"
    fp16_backend: str = ""
    bf16: bool = False
    tf32: bool = False
    report_to: str = REPORT_TO_TENSORBOARD
    metric_for_best_model: str = BEST_MODEL_METRIC
    greater_is_better: bool = True

    max_new_tokens: int = AUTO_MAX_NEW_TOKENS
    eval_beam_size: int = 40
    eval_return_sequences: int = 40
    hits_ks: tuple[int, ...] = (1, 5, 10)
    mrr_ks: tuple[int, ...] = (10,)


def parse_args():
    parser = HfArgumentParser(Config)
    (config,) = parser.parse_args_into_dataclasses(args=normalise_cli_args(sys.argv[1:]))
    return config


def normalise_cli_args(args):
    field_names = {field.name for field in fields(Config)}
    normalised = []

    for arg in args:
        if not arg.startswith("--") or arg == "--":
            normalised.append(arg)
            continue

        key, has_value, value = arg.partition("=")
        candidate = key[2:].replace("-", "_")
        if candidate in field_names:
            key = f"--{candidate}"

        if has_value:
            normalised.append(f"{key}={value}")
        else:
            normalised.append(key)

    return normalised


def metric_depth(config):
    return max(config.eval_return_sequences, *config.hits_ks, *config.mrr_ks)


def new_metric_bucket(hit_ks, reciprocal_rank_ks):
    return {
        "hits": {hit_k: 0 for hit_k in hit_ks},
        "reciprocal_rank_sums": {mrr_k: 0.0 for mrr_k in reciprocal_rank_ks},
        "total": 0,
        "total_predictions": 0,
        "invalid_predictions": 0,
    }


def update_metric_bucket(
    bucket,
    gold_target,
    filtered_predictions,
    prediction_count,
    invalid_count,
    hit_ks,
    reciprocal_rank_ks,
):
    bucket["total"] += 1
    bucket["total_predictions"] += int(prediction_count)
    bucket["invalid_predictions"] += int(invalid_count)

    for hit_k in hit_ks:
        if gold_target in filtered_predictions[:hit_k]:
            bucket["hits"][hit_k] += 1

    for mrr_k in reciprocal_rank_ks:
        for rank, predicted_target in enumerate(filtered_predictions[:mrr_k], start=1):
            if gold_target == predicted_target:
                bucket["reciprocal_rank_sums"][mrr_k] += 1.0 / rank
                break


def add_metric_bucket(metrics, bucket, hit_ks, reciprocal_rank_ks, prefix=""):
    key_prefix = f"{prefix}_" if prefix else ""
    if bucket["total"] <= 0:
        metrics[f"{key_prefix}count"] = 0
        return

    for hit_k in hit_ks:
        metrics[f"{key_prefix}{METRIC_HITS_PREFIX}@{hit_k}"] = bucket["hits"][hit_k] / bucket["total"]
    for mrr_k in reciprocal_rank_ks:
        metrics[f"{key_prefix}{METRIC_MRR_PREFIX}@{mrr_k}"] = bucket["reciprocal_rank_sums"][mrr_k] / bucket["total"]
    metrics[f"{key_prefix}{METRIC_INVALID_RATE}"] = bucket["invalid_predictions"] / max(bucket["total_predictions"], 1)
    if prefix:
        metrics[f"{key_prefix}count"] = bucket["total"]


def make_compute_metrics(tokenizer, valid_targets, hits_ks, mrr_ks, seen_targets=None):
    hit_ks = sorted(set(int(hit_k) for hit_k in hits_ks))
    reciprocal_rank_ks = sorted(set(int(mrr_k) for mrr_k in mrr_ks))
    max_metric_k = max(hit_ks + reciprocal_rank_ks)
    seen_targets = None if seen_targets is None else {str(target).strip() for target in seen_targets}

    def compute_metrics(eval_preds):
        predictions = eval_preds.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = eval_preds.label_ids
        prediction_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        prediction_texts = [prediction.strip() for prediction in prediction_texts]

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        labels = np.where(labels == LABEL_PAD_TOKEN_ID, pad_token_id, labels)
        gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gold_texts = [gold.strip() for gold in gold_texts]

        batch_size = len(gold_texts)
        return_sequences = (
            len(prediction_texts) // batch_size if batch_size > 0 and len(prediction_texts) % batch_size == 0 else 1
        )

        full_bucket = new_metric_bucket(hit_ks, reciprocal_rank_ks)
        subgroup_buckets = None
        if seen_targets is not None:
            subgroup_buckets = {
                "seen": new_metric_bucket(hit_ks, reciprocal_rank_ks),
                "unseen": new_metric_bucket(hit_ks, reciprocal_rank_ks),
            }

        for index in range(batch_size):
            gold_target = gold_texts[index]
            example_predictions = prediction_texts[index * return_sequences : (index + 1) * return_sequences]
            invalid_count = sum(1 for predicted_target in example_predictions if predicted_target not in valid_targets)

            filtered = []
            seen = set()
            for predicted_target in example_predictions:
                if predicted_target not in valid_targets:
                    continue
                if predicted_target in seen:
                    continue
                seen.add(predicted_target)
                filtered.append(predicted_target)
                if len(filtered) >= max_metric_k:
                    break

            update_metric_bucket(
                full_bucket,
                gold_target,
                filtered,
                len(example_predictions),
                invalid_count,
                hit_ks,
                reciprocal_rank_ks,
            )
            if subgroup_buckets is not None:
                subgroup = "seen" if gold_target in seen_targets else "unseen"
                update_metric_bucket(
                    subgroup_buckets[subgroup],
                    gold_target,
                    filtered,
                    len(example_predictions),
                    invalid_count,
                    hit_ks,
                    reciprocal_rank_ks,
                )

        metrics = {}
        add_metric_bucket(metrics, full_bucket, hit_ks, reciprocal_rank_ks)
        if subgroup_buckets is not None:
            for subgroup, bucket in subgroup_buckets.items():
                add_metric_bucket(metrics, bucket, hit_ks, reciprocal_rank_ks, subgroup)
        return metrics

    return compute_metrics


def first_label_token_id(label_ids):
    for token_id in label_ids:
        token_id = int(token_id)
        if token_id != LABEL_PAD_TOKEN_ID:
            return token_id
    return None


def make_atomic_compute_metrics(
    valid_target_token_ids,
    hits_ks,
    mrr_ks,
    seen_target_token_ids=None,
):
    valid_target_token_ids = tuple(int(token_id) for token_id in valid_target_token_ids)
    valid_target_token_id_set = set(valid_target_token_ids)
    valid_target_token_id_array = np.asarray(valid_target_token_ids)
    hit_ks = sorted(set(int(hit_k) for hit_k in hits_ks))
    reciprocal_rank_ks = sorted(set(int(mrr_k) for mrr_k in mrr_ks))
    max_metric_k = max(hit_ks + reciprocal_rank_ks)
    seen_target_token_ids = (
        None if seen_target_token_ids is None else {int(token_id) for token_id in seen_target_token_ids}
    )

    def compute_metrics(eval_preds):
        predictions = eval_preds.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.asarray(predictions)
        labels = np.asarray(eval_preds.label_ids)

        if predictions.ndim == 3:
            first_step_scores = predictions[:, 0, valid_target_token_id_array]
            top_k = min(max_metric_k, first_step_scores.shape[-1])
            topk_indices = np.argpartition(
                -first_step_scores,
                kth=top_k - 1,
                axis=-1,
            )[:, :top_k]
            topk_scores = np.take_along_axis(first_step_scores, topk_indices, axis=-1)
            order = np.argsort(-topk_scores, axis=-1)
            topk_indices = np.take_along_axis(topk_indices, order, axis=-1)
            predictions = valid_target_token_id_array[topk_indices]
        elif predictions.ndim == 1:
            predictions = predictions[:, None]

        full_bucket = new_metric_bucket(hit_ks, reciprocal_rank_ks)
        subgroup_buckets = None
        if seen_target_token_ids is not None:
            subgroup_buckets = {
                "seen": new_metric_bucket(hit_ks, reciprocal_rank_ks),
                "unseen": new_metric_bucket(hit_ks, reciprocal_rank_ks),
            }

        for index, label_ids in enumerate(labels):
            gold_target = first_label_token_id(label_ids)
            if gold_target is None:
                continue

            example_predictions = predictions[index]
            invalid_count = sum(
                1 for predicted_target in example_predictions if int(predicted_target) not in valid_target_token_id_set
            )

            filtered = []
            seen = set()
            for predicted_target in example_predictions:
                predicted_target = int(predicted_target)
                if predicted_target not in valid_target_token_id_set:
                    continue
                if predicted_target in seen:
                    continue
                seen.add(predicted_target)
                filtered.append(predicted_target)
                if len(filtered) >= max_metric_k:
                    break

            update_metric_bucket(
                full_bucket,
                gold_target,
                filtered,
                len(example_predictions),
                invalid_count,
                hit_ks,
                reciprocal_rank_ks,
            )
            if subgroup_buckets is not None:
                subgroup = "seen" if gold_target in seen_target_token_ids else "unseen"
                update_metric_bucket(
                    subgroup_buckets[subgroup],
                    gold_target,
                    filtered,
                    len(example_predictions),
                    invalid_count,
                    hit_ks,
                    reciprocal_rank_ks,
                )

        metrics = {}
        add_metric_bucket(metrics, full_bucket, hit_ks, reciprocal_rank_ks)
        if subgroup_buckets is not None:
            for subgroup, bucket in subgroup_buckets.items():
                add_metric_bucket(metrics, bucket, hit_ks, reciprocal_rank_ks, subgroup)
        return metrics

    return compute_metrics


def make_atomic_preprocess_logits_for_metrics(target_token_ids, max_metric_k):
    target_token_ids = tuple(int(token_id) for token_id in target_token_ids)
    target_ids_by_device = {}

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]

        first_step_logits = logits[:, 0, :]
        device = first_step_logits.device
        if device not in target_ids_by_device:
            target_ids_by_device[device] = torch.as_tensor(
                target_token_ids,
                dtype=torch.long,
                device=device,
            )
        target_ids = target_ids_by_device[device]
        target_scores = first_step_logits.index_select(dim=-1, index=target_ids)
        top_k = min(int(max_metric_k), int(target_ids.numel()))
        top_positions = torch.topk(target_scores, k=top_k, dim=-1).indices
        return target_ids[top_positions]

    return preprocess_logits_for_metrics


def existing_file(path, *, required):
    if os.path.exists(path):
        return path
    if required:
        raise FileNotFoundError(f"Missing required file: {path}")
    return None


def load_datasets(config):
    data_files = {
        TRAIN_SPLIT: existing_file(os.path.join(config.data_dir, config.train_file), required=True),
        VALIDATION_SPLIT: existing_file(os.path.join(config.data_dir, config.val_file), required=True),
    }

    if not config.no_validation_index:
        validation_index = existing_file(
            os.path.join(config.data_dir, config.validation_index_file),
            required=False,
        )
        if validation_index is not None:
            data_files[VALIDATION_INDEX_SPLIT] = validation_index

    raw_datasets = load_dataset(
        DATASET_FORMAT_JSON,
        data_files=data_files,
        keep_in_memory=config.dataset_keep_in_memory,
    )

    if int(config.max_eval_examples) > 0:
        limit = int(config.max_eval_examples)
        for split in (VALIDATION_SPLIT, VALIDATION_INDEX_SPLIT):
            if split in raw_datasets:
                raw_datasets[split] = raw_datasets[split].select(range(min(limit, len(raw_datasets[split]))))

    return raw_datasets


def load_metadata(config, logger):
    metadata_path = os.path.join(config.data_dir, config.metadata_file)
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    logger.info(LOG_METADATA.format(path=metadata_path))
    return metadata


def metadata_arguments(metadata):
    return metadata.get(METADATA_CONFIG_KEY, {}).get(METADATA_ARGUMENTS_KEY, {})


def metadata_model_name(metadata):
    return metadata_arguments(metadata).get(METADATA_MODEL_NAME_KEY)


def safe_model_name(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name)).strip("_") or "model"


def default_output_dir(config):
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return os.path.join(config.data_dir, safe_model_name(config.model_name_or_path), job_id)
    return os.path.join(DEFAULT_OUTPUT_ROOT, safe_model_name(config.model_name_or_path))


def apply_metadata_defaults(config, metadata, logger):
    if config.model_name_or_path is None:
        config.model_name_or_path = metadata_model_name(metadata) or DEFAULT_MODEL_NAME

    if config.output_dir is None:
        config.output_dir = default_output_dir(config)

    logger.info(
        LOG_RUN_CONFIG.format(
            model_name_or_path=config.model_name_or_path,
            output_dir=config.output_dir,
        )
    )


def load_added_tokens(config):
    tokens_path = os.path.join(config.data_dir, config.tokens_file)
    if not os.path.exists(tokens_path):
        return []

    tokens = []
    with open(tokens_path, encoding="utf-8") as tokens_file:
        for line in tokens_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if TOKENS_KEY in row:
                tokens.extend(row[TOKENS_KEY])
            elif TOKEN_KEY in row:
                tokens.append(row[TOKEN_KEY])

    return tokens


def load_token_init_payloads(path):
    if not path or not os.path.exists(path):
        return {}

    payloads = {}
    with open(path, encoding="utf-8") as input_file:
        for line_idx, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            token = row.get(TOKEN_KEY)
            payload = row.get("payload")

            if token is None or payload is None:
                raise ValueError(f"Invalid token init row in {path}:{line_idx}: expected 'token' and 'payload'")
            if not isinstance(payload, list) or not payload:
                raise ValueError(
                    f"Invalid token init payload for {token!r} in {path}:{line_idx}: expected a non-empty list"
                )

            token = str(token)
            if token in payloads:
                raise ValueError(f"Duplicate token init payload for {token!r} in {path}:{line_idx}")

            try:
                payloads[token] = [float(value) for value in payload]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric token init payload for {token!r} in {path}:{line_idx}") from exc

    return payloads


def load_token_init_payloads_for_config(config):
    if not config.tokens_init_file:
        return {}, ""

    path = os.path.join(config.data_dir, config.tokens_init_file)
    return load_token_init_payloads(path), path


def apply_token_init_payloads(model, tokenizer, token_init_payloads, logger, path):
    if not token_init_payloads:
        return 0

    input_embeddings = model.get_input_embeddings()
    input_weight = input_embeddings.weight
    embedding_dim = int(input_weight.shape[1])

    token_ids = []
    vectors = []
    missing_tokens = []
    bad_dimensions = []

    for token, payload in token_init_payloads.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            missing_tokens.append(token)
            continue

        if len(payload) != embedding_dim:
            bad_dimensions.append((token, len(payload)))
            continue

        token_ids.append(int(token_id))
        vectors.append(payload)

    if missing_tokens:
        preview = ", ".join(repr(token) for token in missing_tokens[:5])
        raise ValueError(f"Token init payload references token(s) absent from tokenizer: {preview}")

    if bad_dimensions:
        preview = ", ".join(f"{token!r}:{dim}" for token, dim in bad_dimensions[:5])
        raise ValueError(f"Token init payload dimension mismatch; model dim is {embedding_dim}, got {preview}")

    if not token_ids:
        return 0

    with torch.no_grad():
        indices = torch.tensor(token_ids, dtype=torch.long, device=input_weight.device)
        values = torch.tensor(vectors, dtype=input_weight.dtype, device=input_weight.device)
        input_weight.index_copy_(0, indices, values)

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            output_weight = output_embeddings.weight
            if output_weight.shape == input_weight.shape and output_weight.data_ptr() != input_weight.data_ptr():
                output_weight.index_copy_(0, indices.to(output_weight.device), values.to(output_weight.device))

    logger.info(LOG_TOKEN_INIT.format(size=len(token_ids), path=path))
    return len(token_ids)


def collect_valid_targets(raw_datasets, added_tokens):
    if added_tokens:
        return {str(token).strip() for token in added_tokens}

    valid_targets = set()
    for split in DOCID_VOCAB_SPLITS:
        if split in raw_datasets and DOCID_COL in raw_datasets[split].column_names:
            valid_targets.update(str(docid).strip() for docid in raw_datasets[split][DOCID_COL])
    return valid_targets


def collect_train_query_targets(raw_datasets):
    if TRAIN_SPLIT not in raw_datasets:
        return set()

    dataset = raw_datasets[TRAIN_SPLIT]
    if DOCID_COL not in dataset.column_names:
        return set()

    docids = dataset[DOCID_COL]
    kinds = dataset[KIND_COL] if KIND_COL in dataset.column_names else [QUERY_KIND] * len(dataset)
    return {str(docid).strip() for docid, kind in zip(docids, kinds) if kind == QUERY_KIND}


def collect_train_query_target_token_ids(raw_datasets):
    if TRAIN_SPLIT not in raw_datasets:
        return set()

    dataset = raw_datasets[TRAIN_SPLIT]
    if LABELS_COL not in dataset.column_names:
        return set()

    labels_list = dataset[LABELS_COL]
    kinds = dataset[KIND_COL] if KIND_COL in dataset.column_names else [QUERY_KIND] * len(dataset)

    token_ids = set()
    for labels, kind in zip(labels_list, kinds):
        if kind != QUERY_KIND:
            continue
        token_id = first_label_token_id(labels)
        if token_id is not None:
            token_ids.add(int(token_id))
    return token_ids


def collect_query_reference_from_rows(rows, limit=None):
    seen_targets = set()
    seen_target_token_ids = set()

    for row_idx, row in enumerate(rows):
        if limit is not None and row_idx >= limit:
            break
        if row.get(KIND_COL) != QUERY_KIND:
            continue

        if DOCID_COL in row:
            seen_targets.add(str(row[DOCID_COL]).strip())

        labels = row.get(LABELS_COL)
        if labels is not None:
            token_id = first_label_token_id(labels)
            if token_id is not None:
                seen_target_token_ids.add(int(token_id))

    return seen_targets, seen_target_token_ids


def collect_query_reference_from_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as input_file:
        for row_idx, line in enumerate(input_file):
            if limit is not None and row_idx >= limit:
                break
            rows.append(json.loads(line))
    return collect_query_reference_from_rows(rows)


def metadata_original_train_rows(metadata):
    control = metadata.get("synthetic_query_control") or {}
    value = control.get("original_train_rows")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_seen_unseen_reference_targets(config, raw_datasets, metadata, logger):
    reference_file = str(config.seen_unseen_reference_train_file or "").strip()
    if reference_file:
        seen_targets, seen_target_token_ids = collect_query_reference_from_jsonl(reference_file)
        logger.info(
            LOG_SEEN_UNSEEN_REFERENCE.format(
                source=f"file={reference_file}",
                query_docs=len(seen_targets),
                query_token_ids=len(seen_target_token_ids),
            )
        )
        return seen_targets, seen_target_token_ids

    original_train_rows = metadata_original_train_rows(metadata)
    if original_train_rows is not None and TRAIN_SPLIT in raw_datasets:
        seen_targets, seen_target_token_ids = collect_query_reference_from_rows(
            raw_datasets[TRAIN_SPLIT],
            limit=original_train_rows,
        )
        logger.info(
            LOG_SEEN_UNSEEN_REFERENCE.format(
                source=f"original train prefix rows={original_train_rows:,}",
                query_docs=len(seen_targets),
                query_token_ids=len(seen_target_token_ids),
            )
        )
        return seen_targets, seen_target_token_ids

    seen_targets = collect_train_query_targets(raw_datasets)
    seen_target_token_ids = collect_train_query_target_token_ids(raw_datasets)
    logger.info(
        LOG_SEEN_UNSEEN_REFERENCE.format(
            source="current train split",
            query_docs=len(seen_targets),
            query_token_ids=len(seen_target_token_ids),
        )
    )
    return seen_targets, seen_target_token_ids


def added_token_ids(tokenizer, added_tokens):
    if not added_tokens:
        return []

    token_ids = tokenizer.convert_tokens_to_ids(added_tokens)
    missing_tokens = [
        token
        for token, token_id in zip(added_tokens, token_ids)
        if token_id is None or token_id == tokenizer.unk_token_id
    ]
    if missing_tokens:
        preview = ", ".join(missing_tokens[:5])
        raise ValueError(f"Could not resolve added token ids for: {preview}")

    return [int(token_id) for token_id in token_ids]


def is_atomic_target_label_sequence(labels, target_token_id_set, eos_token_id):
    label_ids = [int(token_id) for token_id in labels if int(token_id) != LABEL_PAD_TOKEN_ID]
    if not label_ids:
        return False
    if label_ids[0] not in target_token_id_set:
        return False
    if eos_token_id is None:
        return len(label_ids) == 1
    return all(token_id == eos_token_id for token_id in label_ids[1:])


def is_atomic_target_eval_dataset(raw_datasets, target_token_ids, tokenizer):
    if not target_token_ids:
        return False

    target_token_id_set = {int(token_id) for token_id in target_token_ids}
    checked = 0
    for split in ATOMIC_TARGET_EVAL_SPLITS:
        if split not in raw_datasets:
            continue
        if LABELS_COL not in raw_datasets[split].column_names:
            continue
        for labels in raw_datasets[split][LABELS_COL]:
            checked += 1
            if not is_atomic_target_label_sequence(
                labels,
                target_token_id_set,
                tokenizer.eos_token_id,
            ):
                return False

    return checked > 0


def infer_max_new_tokens(raw_datasets):
    max_label_length = 0
    for split in GENERATION_LENGTH_SPLITS:
        if split not in raw_datasets:
            continue
        for labels in raw_datasets[split][LABELS_COL]:
            max_label_length = max(max_label_length, len(labels))
    return max(1, max_label_length)


def load_model_tokenizer(config, added_tokens, logger):
    model_path = config.resume_from_checkpoint or config.model_name_or_path
    tokenizer_path = config.model_name_or_path

    if config.resume_from_checkpoint:
        logger.info(
            "Resuming model weights from %s; loading tokenizer from base model %s",
            config.resume_from_checkpoint,
            tokenizer_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if added_tokens:
        tokenizer.add_tokens(added_tokens)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    added_embedding_rows = False
    if added_tokens:
        current_embedding_size = model.get_input_embeddings().num_embeddings
        if len(tokenizer) > current_embedding_size:
            model.resize_token_embeddings(len(tokenizer))
            added_embedding_rows = True
        elif len(tokenizer) < current_embedding_size:
            logger.warning(
                "model embeddings have %s rows, but tokenizer has %s tokens; not resizing down",
                current_embedding_size,
                len(tokenizer),
            )

        logger.info(
            LOG_TOKEN_MANIFEST.format(
                size=len(added_tokens),
                path=os.path.join(config.data_dir, config.tokens_file),
            )
        )

    token_init_payloads, token_init_path = load_token_init_payloads_for_config(config)
    if token_init_payloads:
        if added_embedding_rows:
            apply_token_init_payloads(
                model=model,
                tokenizer=tokenizer,
                token_init_payloads=token_init_payloads,
                logger=logger,
                path=token_init_path,
            )
        else:
            logger.info(LOG_TOKEN_INIT_SKIPPED.format(path=token_init_path))

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def setup_eval_datasets(raw_datasets):
    eval_datasets = {QUERY_EVAL_NAME: raw_datasets[VALIDATION_SPLIT]}
    if VALIDATION_INDEX_SPLIT in raw_datasets:
        eval_datasets[INDEX_EVAL_NAME] = raw_datasets[VALIDATION_INDEX_SPLIT]
    return eval_datasets


def is_query_example(example):
    return example[KIND_COL] == QUERY_KIND


def validate_loss(config):
    value = str(config.loss).strip().lower()
    aliases = {
        LOSS_HF: LOSS_HF,
        "transformers": LOSS_HF,
        "default": LOSS_HF,
        LOSS_T5X: LOSS_T5X,
        "t5x_sum": LOSS_T5X,
        "sum": LOSS_T5X,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported loss={config.loss!r}; expected one of " + ", ".join(sorted(set(aliases))))
    config.loss = aliases[value]


def validate_z_loss(config):
    value = float(config.z_loss)
    if value < 0.0:
        raise ValueError("--z-loss must be >= 0")
    config.z_loss = value
    if value > 0.0 and str(config.loss).strip().lower() != LOSS_T5X:
        raise ValueError("--z-loss is only supported with --loss t5x")


def validate_query_sampling(config):
    value = str(config.query_sampling).strip().lower()
    aliases = {
        QUERY_SAMPLING_NONE: QUERY_SAMPLING_NONE,
        "false": QUERY_SAMPLING_NONE,
        "off": QUERY_SAMPLING_NONE,
        "disabled": QUERY_SAMPLING_NONE,
        QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH: QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH,
        "one_query_per_document_per_epoch": QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH,
        "per_document_per_epoch": QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH,
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported query_sampling={config.query_sampling!r}; expected one of " + ", ".join(sorted(set(aliases)))
        )
    config.query_sampling = aliases[value]


def validate_query_input_max_length(config):
    value = int(config.query_input_max_length)
    if value < 0:
        raise ValueError("--query-input-max-length must be >= 0")
    config.query_input_max_length = value


def validate_document_repeats(config):
    value = str(config.document_repeats).strip().lower()
    if value in DOCUMENT_REPEATS_EQUAL_QUERIES_ALIASES:
        config.document_repeats = DOCUMENT_REPEATS_EQUAL_QUERIES
        return

    try:
        repeat_count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--document-repeats must be a non-negative integer or 'query'") from exc

    if repeat_count < 0:
        raise ValueError("--document-repeats must be a non-negative integer or 'query'")
    config.document_repeats = str(repeat_count)


def uses_t5x_loss(config):
    return str(config.loss).strip().lower() == LOSS_T5X


def uses_z_loss(config):
    return float(config.z_loss) > 0.0


def uses_custom_loss(config):
    return uses_t5x_loss(config)


def uses_query_sampling(config):
    return str(config.query_sampling).strip().lower() != QUERY_SAMPLING_NONE


def uses_document_repeat_sampler(config):
    return not bool(config.train_queries_only) and str(config.document_repeats).strip().lower() != "1"


def uses_query_input_truncation(config):
    return int(config.query_input_max_length) > 0


def is_document_repeat_equal_queries(config):
    return str(config.document_repeats).strip() == DOCUMENT_REPEATS_EQUAL_QUERIES


def document_repeat_count(config):
    if is_document_repeat_equal_queries(config):
        return None
    return int(str(config.document_repeats).strip())


def validate_sampling_combination(config):
    if uses_query_sampling(config) and uses_document_repeat_sampler(config):
        raise ValueError("--query-sampling is not currently compatible with --document-repeats values other than 1")


def collect_document_query_indices(dataset):
    if KIND_COL not in dataset.column_names:
        return [], list(range(len(dataset))), {}

    kinds = dataset[KIND_COL]
    docids = dataset[DOCID_COL] if DOCID_COL in dataset.column_names else [None] * len(dataset)
    document_indices = []
    query_indices = []
    query_counts_by_docid = defaultdict(int)

    for index, (kind, docid) in enumerate(zip(kinds, docids)):
        if kind == DOCUMENT_KIND:
            document_indices.append(index)
        elif kind == QUERY_KIND:
            query_indices.append(index)
            query_counts_by_docid[str(docid)] += 1
        else:
            query_indices.append(index)

    return document_indices, query_indices, query_counts_by_docid


def virtual_document_repeat_count(train_dataset, config):
    document_indices, query_indices, query_counts_by_docid = collect_document_query_indices(train_dataset)
    repeat_count = document_repeat_count(config)

    if repeat_count == 1:
        return {
            "input_rows": len(train_dataset),
            "output_rows": len(train_dataset),
            "base_document_rows": len(document_indices),
            "document_rows": len(document_indices),
            "query_rows": len(query_indices),
        }

    if is_document_repeat_equal_queries(config):
        docids = train_dataset[DOCID_COL] if DOCID_COL in train_dataset.column_names else [None] * len(train_dataset)
        virtual_document_rows = sum(
            max(1, query_counts_by_docid.get(str(docids[index]), 0)) for index in document_indices
        )
    else:
        virtual_document_rows = len(document_indices) * int(repeat_count)

    return {
        "input_rows": len(train_dataset),
        "output_rows": virtual_document_rows + len(query_indices),
        "base_document_rows": len(document_indices),
        "document_rows": virtual_document_rows,
        "query_rows": len(query_indices),
    }


def query_sampling_summary(train_dataset):
    kinds = train_dataset[KIND_COL] if KIND_COL in train_dataset.column_names else [QUERY_KIND] * len(train_dataset)
    docids = train_dataset[DOCID_COL] if DOCID_COL in train_dataset.column_names else [None] * len(train_dataset)

    document_rows = 0
    query_rows = 0
    query_docids = set()

    for docid, kind in zip(docids, kinds):
        if kind == DOCUMENT_KIND:
            document_rows += 1
        elif kind == QUERY_KIND:
            query_rows += 1
            query_docids.add(str(docid))

    return {
        "document_rows": document_rows,
        "query_rows": query_rows,
        "query_docs": len(query_docids),
        "samples_per_epoch": document_rows + len(query_docids),
    }


def train_samples_per_epoch(train_dataset, config):
    if not uses_query_sampling(config):
        summary = virtual_document_repeat_count(train_dataset, config)
        return max(1, summary["output_rows"])

    summary = query_sampling_summary(train_dataset)
    if summary["query_rows"] <= 0:
        raise ValueError(f"query_sampling={config.query_sampling} requires at least one query row")
    return max(1, summary["samples_per_epoch"])


def setup_train_dataset(raw_datasets, config, logger):
    train_dataset = raw_datasets[TRAIN_SPLIT]
    input_rows = len(train_dataset)
    if config.train_queries_only:
        train_dataset = train_dataset.filter(is_query_example)

    logger.info(LOG_DOCUMENT_REPEATS_SUMMARY.format(**document_repeats_summary(train_dataset, input_rows, config)))
    return train_dataset


def document_repeats_summary(train_dataset, input_rows, config):
    summary = virtual_document_repeat_count(train_dataset, config)
    summary["input_rows"] = input_rows
    return summary


def truncate_query_input_ids(feature, tokenizer, max_query_tokens):
    if int(max_query_tokens) <= 0:
        return feature

    kind = feature.get(KIND_COL, QUERY_KIND)
    if kind != QUERY_KIND or "input_ids" not in feature:
        return feature

    input_ids = list(feature["input_ids"])
    eos_id = tokenizer.eos_token_id
    had_eos = eos_id is not None and len(input_ids) > 0 and int(input_ids[-1]) == int(eos_id)
    content_ids = input_ids[:-1] if had_eos else input_ids

    if len(content_ids) <= int(max_query_tokens):
        return feature

    truncated = dict(feature)
    new_input_ids = content_ids[: int(max_query_tokens)]
    if had_eos:
        new_input_ids.append(int(eos_id))
    truncated["input_ids"] = new_input_ids
    if "attention_mask" in truncated:
        truncated["attention_mask"] = [1] * len(new_input_ids)
    return truncated


def setup_data_collator(
    model,
    tokenizer,
    strip_extra_columns=False,
    query_input_max_length=0,
):
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=PADDING_LONGEST,
        label_pad_token_id=LABEL_PAD_TOKEN_ID,
        pad_to_multiple_of=8,
    )

    truncate_queries = int(query_input_max_length) > 0
    if not strip_extra_columns and not truncate_queries:
        return base_collator

    model_feature_keys = {
        "input_ids",
        "attention_mask",
        "labels",
        "decoder_input_ids",
    }

    def collate(features):
        clean_features = []
        for feature in features:
            feature = truncate_query_input_ids(
                feature,
                tokenizer,
                query_input_max_length,
            )
            clean_features.append({key: value for key, value in feature.items() if key in model_feature_keys})

        batch = base_collator(clean_features)
        return batch

    return collate


class OneQueryPerDocPerEpochSampler(Sampler):
    def __init__(self, dataset, seed=0):
        if DOCID_COL not in dataset.column_names:
            raise ValueError(f"{QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH} requires a '{DOCID_COL}' column")

        kinds = dataset[KIND_COL] if KIND_COL in dataset.column_names else [QUERY_KIND] * len(dataset)
        docids = dataset[DOCID_COL]
        query_indices_by_docid = defaultdict(list)
        document_indices = []

        for index, (docid, kind) in enumerate(zip(docids, kinds)):
            if kind == DOCUMENT_KIND:
                document_indices.append(index)
            elif kind == QUERY_KIND:
                query_indices_by_docid[str(docid)].append(index)

        if not query_indices_by_docid:
            raise ValueError(f"{QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH} requires at least one query row")

        self.seed = int(seed)
        self.epoch = 0
        self.document_indices = list(document_indices)
        self.query_orders = []

        for doc_offset, docid in enumerate(sorted(query_indices_by_docid)):
            indices = list(query_indices_by_docid[docid])
            if len(indices) > 1:
                generator = torch.Generator()
                generator.manual_seed(self.seed + 7919 * (doc_offset + 1))
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[position] for position in permutation]
            self.query_orders.append(indices)

    def __len__(self):
        return len(self.document_indices) + len(self.query_orders)

    def __iter__(self):
        epoch = self.epoch
        selected_indices = list(self.document_indices)

        for query_order in self.query_orders:
            selected_indices.append(query_order[epoch % len(query_order)])

        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * (epoch + 1))
        permutation = torch.randperm(len(selected_indices), generator=generator).tolist()
        self.epoch += 1

        for position in permutation:
            yield selected_indices[position]


class DocumentRepeatSampler(Sampler):
    def __init__(self, dataset, document_repeats, seed=0):
        self.seed = int(seed)
        self.epoch = 0
        self.document_repeats = str(document_repeats).strip().lower()
        (
            self.document_indices,
            self.query_indices,
            query_counts_by_docid,
        ) = collect_document_query_indices(dataset)

        self.query_count = len(self.query_indices)
        self.document_count = len(self.document_indices)
        self.document_sample_indices = None

        if self.document_repeats in DOCUMENT_REPEATS_EQUAL_QUERIES_ALIASES:
            docids = dataset[DOCID_COL] if DOCID_COL in dataset.column_names else [None] * len(dataset)
            self.document_sample_indices = []
            for index in self.document_indices:
                repeats = max(1, query_counts_by_docid.get(str(docids[index]), 0))
                self.document_sample_indices.extend([index] * repeats)
            self.virtual_document_count = len(self.document_sample_indices)
        else:
            repeat_count = int(self.document_repeats)
            self.virtual_document_count = self.document_count * repeat_count

        self.virtual_length = self.query_count + self.virtual_document_count
        if self.virtual_length <= 0:
            raise ValueError("--document-repeats produced an empty training sampler")

        self.rank, self.world_size = distributed_rank_world()

    def __len__(self):
        if self.rank >= self.virtual_length:
            return 0
        return (self.virtual_length - self.rank + self.world_size - 1) // self.world_size

    def _map_virtual_index(self, virtual_index):
        if virtual_index < self.query_count:
            return self.query_indices[virtual_index]

        document_position = virtual_index - self.query_count
        if self.document_sample_indices is not None:
            return self.document_sample_indices[document_position]

        return self.document_indices[document_position % self.document_count]

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * (self.epoch + 1))
        permutation = torch.randperm(self.virtual_length, generator=generator)
        shard = permutation[self.rank :: self.world_size]
        self.epoch += 1

        for offset in range(0, shard.numel(), DOCUMENT_REPEAT_SAMPLER_CHUNK_SIZE):
            chunk = shard[offset : offset + DOCUMENT_REPEAT_SAMPLER_CHUNK_SIZE].tolist()
            for virtual_index in chunk:
                yield self._map_virtual_index(int(virtual_index))


def distributed_rank_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())

    rank = int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)
    return rank, max(1, world_size)


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
        return

    if isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def tensor_nonfinite_summary(tensor):
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = detached.numel()
    nonfinite_count = total_count - finite_count
    nan_count = int(torch.isnan(detached).sum().item())
    posinf_count = int(torch.isposinf(detached).sum().item())
    neginf_count = int(torch.isneginf(detached).sum().item())

    finite_range = "none"
    if finite_count > 0:
        finite_values = detached[finite_mask]
        finite_range = f"{finite_values.min().item():.6g}..{finite_values.max().item():.6g}"

    return (
        f"dtype={detached.dtype}, shape={tuple(detached.shape)}, "
        f"nonfinite={nonfinite_count}/{total_count}, nan={nan_count}, "
        f"+inf={posinf_count}, -inf={neginf_count}, finite_range={finite_range}"
    )


def is_t5_fp16_clamp_internal_module(module_name):
    return bool(
        re.search(r"\.layer\.[12]\.DenseReluDense(?:\.|$)", module_name)
        or re.search(r"\.layer\.[12]\.dropout$", module_name)
    )


def register_nonfinite_forward_hooks(model, logger, skip_t5_fp16_clamp_internals=False):
    handles = []
    skipped = 0

    def make_hook(module_name):
        def hook(module, _inputs, output):
            for output_index, tensor in enumerate(iter_tensors(output)):
                if not tensor.is_floating_point() or tensor.numel() == 0:
                    continue
                if torch.isfinite(tensor).all().item():
                    continue
                raise FloatingPointError(
                    "Non-finite forward output detected: "
                    f"module={module_name}, class={module.__class__.__name__}, "
                    f"output_index={output_index}, {tensor_nonfinite_summary(tensor)}"
                )

        return hook

    for module_name, module in model.named_modules():
        name = module_name or "<root>"
        if skip_t5_fp16_clamp_internals and is_t5_fp16_clamp_internal_module(name):
            skipped += 1
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    logger.info(
        "Registered non-finite forward hooks on %s modules; skipped %s modules.",
        len(handles),
        skipped,
    )
    return handles


def fp16_clamp_value(margin=1000.0):
    return torch.finfo(torch.float16).max - float(margin)


def clamp_fp16_tensor(tensor, clamp_value):
    if not torch.is_tensor(tensor) or tensor.dtype != torch.float16:
        return tensor
    return torch.clamp(tensor, min=-clamp_value, max=clamp_value)


def is_fp16_stability_dtype(tensor):
    return torch.is_tensor(tensor) and tensor.dtype in (torch.float16, torch.float32)


def clamp_fp16_stability_tensor(tensor, clamp_value):
    if not is_fp16_stability_dtype(tensor):
        return tensor
    return torch.clamp(tensor, min=-clamp_value, max=clamp_value)


def disable_cuda_autocast_if_needed(tensor):
    if torch.is_tensor(tensor) and tensor.is_cuda:
        return torch.amp.autocast("cuda", enabled=False)
    return nullcontext()


def linear_fp32(module, hidden_states):
    bias = getattr(module, "bias", None)
    return F.linear(
        hidden_states.float(),
        module.weight.float(),
        None if bias is None else bias.float(),
    )


def module_dropout_scale(module):
    if not module.training or not hasattr(module, "dropout"):
        return 1.0

    dropout_probability = float(getattr(module.dropout, "p", 0.0))
    if dropout_probability <= 0.0:
        return 1.0
    if dropout_probability >= 1.0:
        return 1.0
    return 1.0 / (1.0 - dropout_probability)


def t5_layer_ff_dropout_scale(module):
    return module_dropout_scale(module)


def is_t5_dense_relu_dense_module(module):
    return all(hasattr(module, name) for name in ("act", "dropout", "wo", "wi_0", "wi_1"))


def t5_dense_relu_dense_fp16_safe_forward(module, hidden_states, clamp_value):
    with disable_cuda_autocast_if_needed(hidden_states):
        hidden_gelu = module.act(linear_fp32(module.wi_0, hidden_states))
        hidden_linear = linear_fp32(module.wi_1, hidden_states)
        forwarded_states = hidden_gelu * hidden_linear

        pre_dropout_clamp = clamp_value / module_dropout_scale(module)
        forwarded_states = torch.clamp(
            forwarded_states,
            min=-pre_dropout_clamp,
            max=pre_dropout_clamp,
        )
        forwarded_states = module.dropout(forwarded_states)
        forwarded_states = torch.clamp(
            forwarded_states,
            min=-clamp_value,
            max=clamp_value,
        )
        forwarded_states = linear_fp32(module.wo, forwarded_states)
        forwarded_states = torch.clamp(
            forwarded_states,
            min=-clamp_value,
            max=clamp_value,
        )
        return forwarded_states.to(dtype=hidden_states.dtype)


def patch_t5_dense_relu_dense_fp16_clamp(model, logger, margin=1000.0):
    clamp_value = fp16_clamp_value(margin)
    patched = 0

    for module in model.modules():
        if not is_t5_dense_relu_dense_module(module):
            continue
        if getattr(module, "_redsi_fp16_forward_patched", False):
            continue

        original_forward = module.forward

        def forward(
            hidden_states,
            *args,
            module=module,
            original_forward=original_forward,
            **kwargs,
        ):
            if args or kwargs or not torch.is_tensor(hidden_states) or not is_fp16_stability_dtype(hidden_states):
                return original_forward(hidden_states, *args, **kwargs)

            return t5_dense_relu_dense_fp16_safe_forward(
                module,
                hidden_states,
                clamp_value,
            )

        module.forward = forward
        module._redsi_fp16_forward_patched = True
        patched += 1

    logger.info(
        "Patched T5 gated DenseReluDense fp16-safe forward on %s modules with clamp_value=%s.",
        patched,
        clamp_value,
    )
    return patched


def patch_t5_layer_ff_fp16_clamp(model, logger, margin=1000.0):
    clamp_value = fp16_clamp_value(margin)
    patched = 0
    gated_safe = 0

    for module in model.modules():
        if module.__class__.__name__ != "T5LayerFF":
            continue
        if not all(hasattr(module, name) for name in ("layer_norm", "DenseReluDense", "dropout")):
            continue
        dense_relu_dense = module.DenseReluDense
        use_gated_safe_forward = is_t5_dense_relu_dense_module(dense_relu_dense)

        def forward(
            hidden_states,
            module=module,
            use_gated_safe_forward=use_gated_safe_forward,
        ):
            forwarded_states = module.layer_norm(hidden_states)
            if (
                use_gated_safe_forward
                and torch.is_tensor(forwarded_states)
                and is_fp16_stability_dtype(forwarded_states)
            ):
                forwarded_states = t5_dense_relu_dense_fp16_safe_forward(
                    module.DenseReluDense,
                    forwarded_states,
                    clamp_value,
                )
            else:
                forwarded_states = module.DenseReluDense(forwarded_states)

            if forwarded_states.dtype == torch.float16 or (
                use_gated_safe_forward and forwarded_states.dtype == torch.float32
            ):
                pre_dropout_clamp = clamp_value / t5_layer_ff_dropout_scale(module)
                forwarded_states = torch.clamp(
                    forwarded_states,
                    min=-pre_dropout_clamp,
                    max=pre_dropout_clamp,
                )

            forwarded_states = module.dropout(forwarded_states)
            hidden_states = hidden_states + forwarded_states
            if use_gated_safe_forward:
                hidden_states = clamp_fp16_stability_tensor(hidden_states, clamp_value)
            else:
                hidden_states = clamp_fp16_tensor(hidden_states, clamp_value)
            return hidden_states

        module.forward = forward
        patched += 1
        if use_gated_safe_forward:
            gated_safe += 1

    logger.info(
        "Patched T5LayerFF fp16 clamp on %s modules with clamp_value=%s (gated_safe=%s).",
        patched,
        clamp_value,
        gated_safe,
    )
    return patched


def patch_t5_fused_rms_norm_fp32(model, logger):
    mixed_dtype_fused_rms_norm_affine = load_apex_mixed_dtype_fused_rms_norm_affine(logger)
    patched = 0

    for _module_name, module in model.named_modules():
        if module.__class__.__name__ != "FusedRMSNorm":
            continue
        if getattr(module, "_redsi_fp32_forward_patched", False):
            continue

        original_forward = module.forward

        def forward(
            hidden_states,
            *args,
            module=module,
            original_forward=original_forward,
            mixed_dtype_fused_rms_norm_affine=mixed_dtype_fused_rms_norm_affine,
            **kwargs,
        ):
            if not torch.is_tensor(hidden_states):
                return original_forward(hidden_states, *args, **kwargs)

            if hidden_states.is_cuda and (cuda_autocast_enabled() or hidden_states.dtype == torch.float16):
                with torch.amp.autocast("cuda", enabled=False):
                    if mixed_dtype_fused_rms_norm_affine is not None:
                        return apex_mixed_dtype_fused_rms_norm_forward(
                            module,
                            hidden_states,
                            mixed_dtype_fused_rms_norm_affine,
                        )
                    return fused_rms_norm_fp32_forward(module, hidden_states)
            if hidden_states.dtype == torch.float16:
                return fused_rms_norm_fp32_forward(module, hidden_states)
            return original_forward(hidden_states, *args, **kwargs)

        module.forward = forward
        module._redsi_fp32_forward_patched = True
        patched += 1

    mode = "mixed-fused" if mixed_dtype_fused_rms_norm_affine is not None else "manual-fp32"
    logger.info("Patched T5 FusedRMSNorm autocast-safe %s forward on %s modules.", mode, patched)
    return patched


def load_apex_mixed_dtype_fused_rms_norm_affine(logger):
    try:
        from apex.normalization.fused_layer_norm import mixed_dtype_fused_rms_norm_affine
    except Exception as exc:
        logger.warning(
            "Apex mixed_dtype_fused_rms_norm_affine is unavailable; "
            "falling back to manual fp32 RMSNorm for native AMP. error=%r",
            exc,
        )
        return None

    return mixed_dtype_fused_rms_norm_affine


def apex_mixed_dtype_fused_rms_norm_forward(module, hidden_states, mixed_dtype_fused_rms_norm_affine):
    return mixed_dtype_fused_rms_norm_affine(
        hidden_states,
        module.weight,
        module.normalized_shape,
        module.eps,
        getattr(module, "memory_efficient", False),
    )


def cuda_autocast_enabled():
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:
        return torch.is_autocast_enabled()


def fused_rms_norm_fp32_forward(module, hidden_states):
    hidden_states = hidden_states.float()
    normalized_shape = tuple(getattr(module, "normalized_shape", hidden_states.shape[-1:]))
    dims = tuple(range(-len(normalized_shape), 0))
    variance = hidden_states.pow(2).mean(dims, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + float(getattr(module, "eps", 1e-6)))

    weight = getattr(module, "weight", None)
    if weight is None:
        return hidden_states
    return weight.float() * hidden_states


def uses_apex_fp16_backend(config):
    return config.fp16 and config.fp16_backend.strip().lower() == "apex"


def setup_generation_config(
    model,
    config,
    raw_datasets,
    logger,
    atomic_target_eval=False,
    atomic_target_token_ids=(),
    atomic_target_eval_log=None,
):
    if atomic_target_eval:
        model.generation_config.max_new_tokens = 1
        model.generation_config.num_beams = 1
        model.generation_config.num_return_sequences = 1
        if atomic_target_eval_log is None:
            atomic_target_eval_log = LOG_ATOMIC_EVAL_CONFIG.format(size=len(atomic_target_token_ids))
        logger.info(atomic_target_eval_log)
        return

    if config.max_new_tokens == AUTO_MAX_NEW_TOKENS:
        max_new_tokens = infer_max_new_tokens(raw_datasets)
    else:
        max_new_tokens = config.max_new_tokens

    max_metric_depth = metric_depth(config)
    model.generation_config.max_new_tokens = int(max_new_tokens)
    model.generation_config.num_beams = int(max(config.eval_beam_size, max_metric_depth))
    model.generation_config.num_return_sequences = int(max_metric_depth)
    model.generation_config.early_stopping = True
    logger.info(
        LOG_GENERATION_CONFIG.format(
            max_new_tokens=model.generation_config.max_new_tokens,
            num_beams=model.generation_config.num_beams,
            num_return_sequences=model.generation_config.num_return_sequences,
        )
    )


def compute_warmup_steps(config, train_dataset):
    explicit_warmup_steps = int(config.warmup_steps)
    if explicit_warmup_steps > 0:
        return explicit_warmup_steps

    warmup_ratio = float(config.warmup_ratio)
    if warmup_ratio <= 0:
        return 0

    if int(config.max_steps) > 0:
        total_steps = int(config.max_steps)
    else:
        total_steps = compute_steps_per_epoch(train_dataset, config) * max(
            0.0,
            float(config.num_train_epochs),
        )

    return int(total_steps * warmup_ratio)


def setup_optimizer_and_scheduler(model, config, train_dataset, logger):
    warmup_steps = compute_warmup_steps(config, train_dataset)
    optimizer = Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        relative_step=ADAFACTOR_RELATIVE_STEP,
        scale_parameter=ADAFACTOR_SCALE_PARAMETER,
        warmup_init=ADAFACTOR_WARMUP_INIT,
        clip_threshold=ADAFACTOR_CLIP_THRESHOLD,
        weight_decay=config.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
    )
    logger.info(
        LOG_OPTIMIZER_CONFIG.format(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=warmup_steps,
            scale_parameter=ADAFACTOR_SCALE_PARAMETER,
        )
    )
    return optimizer, scheduler


def compute_steps_per_epoch(train_dataset, config):
    batch_size = int(config.per_device_train_batch_size)
    gradient_accumulation_steps = int(config.gradient_accumulation_steps)
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)
    effective_batch_size = max(1, batch_size * gradient_accumulation_steps * world_size)
    samples_per_epoch = train_samples_per_epoch(train_dataset, config)
    return max(1, (samples_per_epoch + effective_batch_size - 1) // effective_batch_size)


def setup_evaluation_schedule(config, train_dataset, logger):
    steps_per_epoch = compute_steps_per_epoch(train_dataset, config)
    eval_strategy = EVAL_STRATEGY_EPOCH
    save_strategy = SAVE_STRATEGY_EPOCH
    eval_steps = None
    save_steps = None

    if config.eval_steps > 0:
        eval_steps = int(config.eval_steps)
        eval_strategy = EVAL_STRATEGY_STEPS

    if config.save_steps > 0:
        save_steps = int(config.save_steps)
        save_strategy = SAVE_STRATEGY_STEPS

    if save_strategy == SAVE_STRATEGY_STEPS and eval_strategy != EVAL_STRATEGY_STEPS:
        eval_steps = save_steps
        eval_strategy = EVAL_STRATEGY_STEPS

    if eval_strategy == EVAL_STRATEGY_STEPS and save_strategy != SAVE_STRATEGY_STEPS:
        save_steps = eval_steps
        save_strategy = SAVE_STRATEGY_STEPS

    if config.train_queries_only and steps_per_epoch < int(config.logging_steps):
        eval_steps = max(1, int(config.logging_steps))
        save_steps = eval_steps
        eval_strategy = EVAL_STRATEGY_STEPS
        save_strategy = SAVE_STRATEGY_STEPS

    logger.info(
        LOG_EVALUATION_SCHEDULE.format(
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            steps_per_epoch=steps_per_epoch,
        )
    )
    return eval_strategy, save_strategy, eval_steps, save_steps


def report_to_value(value):
    if value.lower() == REPORT_TO_NONE:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def detect_accelerator():
    import torch

    if torch.cuda.is_available():
        return ACCELERATOR_CUDA
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ACCELERATOR_MPS
    return ACCELERATOR_CPU


def cuda_supports_bf16():
    import torch

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8 and torch.cuda.is_bf16_supported()


def cuda_summary():
    import torch

    if not torch.cuda.is_available():
        return "unavailable"
    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    torch_bf16 = torch.cuda.is_bf16_supported()
    native_bf16 = cuda_supports_bf16()
    return f"name={name}, capability={major}.{minor}, torch_bf16_supported={torch_bf16}, native_bf16={native_bf16}"


def cuda_supports_tf32():
    import torch

    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def dataloader_pin_memory_for_accelerator(accelerator):
    return accelerator == ACCELERATOR_CUDA


def resolve_precision(config, logger):
    precision = str(config.precision).lower()
    valid_precisions = {
        PRECISION_AUTO,
        PRECISION_FP16,
        PRECISION_BF16,
        PRECISION_FP32,
    }
    if precision not in valid_precisions:
        raise ValueError(
            f"Unsupported precision={config.precision!r}; expected one of " + ", ".join(sorted(valid_precisions))
        )

    accelerator = detect_accelerator()
    explicit_fp16 = precision == PRECISION_FP16 or config.fp16
    explicit_bf16 = precision == PRECISION_BF16 or config.bf16

    if accelerator != ACCELERATOR_CUDA and (explicit_fp16 or explicit_bf16):
        requested = PRECISION_FP16 if explicit_fp16 else PRECISION_BF16
        raise ValueError(
            f"precision={requested} was requested, but CUDA is unavailable. "
            "Refusing to continue on CPU because this would make GPU training "
            "silently run much slower."
        )

    if precision == PRECISION_FP32:
        config.fp16 = False
        config.bf16 = False
        config.tf32 = False
    elif precision == PRECISION_FP16:
        config.fp16 = True
        config.bf16 = False
        config.tf32 = False
    elif precision == PRECISION_BF16:
        if accelerator == ACCELERATOR_CUDA and not cuda_supports_bf16():
            raise ValueError(
                "precision=bf16 was requested, but this CUDA device does not appear "
                f"to have native bf16 support ({cuda_summary()})."
            )
        config.fp16 = False
        config.bf16 = True
        config.tf32 = accelerator == ACCELERATOR_CUDA and cuda_supports_tf32()
    elif not (config.fp16 or config.bf16 or config.tf32):
        if accelerator == ACCELERATOR_CUDA:
            if cuda_supports_bf16():
                config.bf16 = True
                config.tf32 = cuda_supports_tf32()
        else:
            config.fp16 = False
            config.bf16 = False
            config.tf32 = False

    if accelerator != ACCELERATOR_CUDA:
        config.fp16 = False
        config.bf16 = False
        config.tf32 = False

    resolved_precision = PRECISION_FP32
    if config.bf16:
        resolved_precision = PRECISION_BF16
    elif config.fp16:
        resolved_precision = PRECISION_FP16

    logger.info(
        LOG_DEVICE_CONFIG.format(
            accelerator=accelerator,
            requested_precision=precision,
            precision=resolved_precision,
            fp16=config.fp16,
            bf16=config.bf16,
            tf32=config.tf32,
            cuda=cuda_summary() if accelerator == ACCELERATOR_CUDA else "n/a",
        )
    )
    logger.info(LOG_DATALOADER_WORKERS.format(workers=config.dataloader_num_workers))
    logger.info(
        LOG_DATALOADER_PIN_MEMORY.format(
            pin_memory=dataloader_pin_memory_for_accelerator(accelerator),
        )
    )
    return accelerator


def setup_trainer_arguments(
    config,
    train_dataset,
    logger,
    predict_with_generate=True,
    accelerator=None,
):
    if int(config.dataloader_num_workers) > 0 and int(config.dataloader_prefetch_factor) < 1:
        raise ValueError("--dataloader-prefetch-factor must be >= 1 when workers are enabled")

    eval_strategy, save_strategy, eval_steps, save_steps = setup_evaluation_schedule(
        config,
        train_dataset,
        logger,
    )

    training_argument_kwargs = {
        "output_dir": config.output_dir,
        "run_name": os.path.basename(config.output_dir),
        "seed": config.seed,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "warmup_steps": config.warmup_steps,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "logging_nan_inf_filter": config.logging_nan_inf_filter,
        "logging_strategy": LOGGING_STRATEGY_STEPS,
        "report_to": report_to_value(config.report_to),
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "predict_with_generate": predict_with_generate,
        "dataloader_pin_memory": dataloader_pin_memory_for_accelerator(accelerator or detect_accelerator()),
        "dataloader_num_workers": config.dataloader_num_workers,
        "group_by_length": config.group_by_length,
        "remove_unused_columns": not (
            uses_query_sampling(config) or uses_document_repeat_sampler(config) or uses_query_input_truncation(config)
        ),
        "fp16": config.fp16,
        "fp16_full_eval": config.fp16_full_eval,
        "fp16_opt_level": config.fp16_opt_level,
        "bf16": config.bf16,
        "tf32": config.tf32,
    }

    training_argument_parameters = signature(Seq2SeqTrainingArguments.__init__).parameters
    if "save_safetensors" in training_argument_parameters:
        training_argument_kwargs["save_safetensors"] = config.save_safetensors
    elif config.save_safetensors:
        logger.warning(
            "Ignoring save_safetensors=True because this transformers version does not "
            "support Seq2SeqTrainingArguments.save_safetensors.",
        )

    if config.fp16_backend:
        if "half_precision_backend" in training_argument_parameters:
            training_argument_kwargs["half_precision_backend"] = config.fp16_backend
        elif "fp16_backend" in training_argument_parameters:
            training_argument_kwargs["fp16_backend"] = config.fp16_backend
        else:
            logger.warning(
                "Ignoring fp16_backend=%r because this transformers version supports "
                "neither Seq2SeqTrainingArguments.fp16_backend nor "
                "Seq2SeqTrainingArguments.half_precision_backend.",
                config.fp16_backend,
            )

    if int(config.dataloader_num_workers) > 0:
        if "dataloader_persistent_workers" in training_argument_parameters:
            training_argument_kwargs["dataloader_persistent_workers"] = config.dataloader_persistent_workers
        if "dataloader_prefetch_factor" in training_argument_parameters:
            training_argument_kwargs["dataloader_prefetch_factor"] = int(config.dataloader_prefetch_factor)

    return Seq2SeqTrainingArguments(**training_argument_kwargs)


def is_nonfinite_number(value):
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return not math.isfinite(float(value))


def json_safe_value(value):
    if value is None or isinstance(value, (bool, str)):
        return value

    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return json_safe_value(detached.item())
        return json_safe_value(detached.tolist())

    if isinstance(value, np.ndarray):
        return json_safe_value(value.tolist())

    if isinstance(value, np.generic):
        return json_safe_value(value.item())

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        if value > 0:
            return "Infinity"
        return "-Infinity"

    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]

    return str(value)


def utc_now_isoformat():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def current_job_id():
    return os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"


def metrics_history_path(output_dir):
    return os.path.join(output_dir, METRICS_HISTORY_FILE)


def metrics_history_context(state=None):
    context = {
        "time": utc_now_isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "world_size": os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS"),
    }

    if state is not None:
        context.update(
            {
                "global_step": json_safe_value(getattr(state, "global_step", None)),
                "epoch": json_safe_value(getattr(state, "epoch", None)),
                "max_steps": json_safe_value(getattr(state, "max_steps", None)),
                "num_train_epochs": json_safe_value(getattr(state, "num_train_epochs", None)),
            }
        )

    return context


def append_metrics_history(output_dir, event, metrics, state=None):
    os.makedirs(output_dir, exist_ok=True)
    record = metrics_history_context(state)
    record["event"] = event
    record["metrics"] = json_safe_value(metrics or {})

    with open(metrics_history_path(output_dir), "a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def job_metrics_dir(output_dir):
    return os.path.join(output_dir, JOB_METRICS_DIR, current_job_id())


def save_job_metrics(output_dir, metric_key_prefix, metrics, state=None):
    out_dir = job_metrics_dir(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    record = metrics_history_context(state)
    record["event"] = f"summary:{metric_key_prefix}"
    record["metrics"] = json_safe_value(metrics or {})

    out_path = os.path.join(out_dir, f"{metric_key_prefix}_results.json")
    with open(out_path, "w", encoding="utf-8") as metrics_file:
        json.dump(record, metrics_file, indent=2, sort_keys=True, allow_nan=False)
        metrics_file.write("\n")


def trainer_is_world_process_zero(trainer):
    is_world_process_zero = getattr(trainer, "is_world_process_zero", None)
    if callable(is_world_process_zero):
        return is_world_process_zero()
    return bool(getattr(trainer.state, "is_world_process_zero", True))


class MetricsHistoryCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        if not getattr(state, "is_world_process_zero", True):
            return control

        append_metrics_history(args.output_dir, "log", logs, state=state)
        return control


class FailOnNonFiniteMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control

        bad_metrics = {key: value for key, value in logs.items() if is_nonfinite_number(value)}
        if bad_metrics:
            raise ValueError(f"Non-finite trainer metrics at step {state.global_step}: {bad_metrics}")

        return control


class T5XSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, z_loss=0.0, **kwargs):
        self.z_loss = float(z_loss)
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        if labels is None:
            kwargs = {"return_outputs": return_outputs}
            compute_loss_parameters = signature(super().compute_loss).parameters
            if num_items_in_batch is not None and "num_items_in_batch" in compute_loss_parameters:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return super().compute_loss(model, inputs, **kwargs)

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        labels = labels.to(logits.device)
        valid_tokens = labels.ne(LABEL_PAD_TOKEN_ID)

        token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=LABEL_PAD_TOKEN_ID,
            reduction="none",
        ).reshape(labels.shape)

        if self.z_loss > 0.0:
            log_z = torch.logsumexp(logits.float(), dim=-1)
            token_loss = token_loss + self.z_loss * log_z.square()

        valid_tokens = valid_tokens.to(token_loss.dtype)
        token_loss = token_loss * valid_tokens
        loss = token_loss.sum()

        return (loss, outputs) if return_outputs else loss


class QuerySamplingTrainerMixin:
    def __init__(
        self,
        *args,
        query_sampling=QUERY_SAMPLING_NONE,
        document_repeats="1",
        **kwargs,
    ):
        self.query_sampling = query_sampling
        self.document_repeats = str(document_repeats).strip().lower()
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset

        if self.query_sampling == QUERY_SAMPLING_ONE_QUERY_PER_DOC_PER_EPOCH:
            return OneQueryPerDocPerEpochSampler(
                train_dataset,
                seed=int(getattr(self.args, "seed", 0)),
            )

        if self.document_repeats != "1":
            return DocumentRepeatSampler(
                train_dataset,
                document_repeats=self.document_repeats,
                seed=int(getattr(self.args, "seed", 0)),
            )

        return super()._get_train_sampler(train_dataset)


class QuerySamplingSeq2SeqTrainer(QuerySamplingTrainerMixin, Seq2SeqTrainer):
    pass


class T5XQuerySamplingSeq2SeqTrainer(
    QuerySamplingTrainerMixin,
    T5XSeq2SeqTrainer,
):
    pass


def setup_trainer(
    config,
    raw_datasets,
    train_dataset,
    model,
    tokenizer,
    trainer_arguments,
    valid_targets,
    optimizer,
    scheduler,
    atomic_target_token_ids=(),
    seen_targets=None,
    seen_target_token_ids=None,
):
    if atomic_target_token_ids:
        compute_metrics = make_atomic_compute_metrics(
            atomic_target_token_ids,
            config.hits_ks,
            config.mrr_ks,
            seen_target_token_ids=seen_target_token_ids,
        )
    else:
        compute_metrics = make_compute_metrics(
            tokenizer,
            valid_targets,
            config.hits_ks,
            config.mrr_ks,
            seen_targets=seen_targets,
        )

    tokenizer_arg_name = (
        "processing_class" if "processing_class" in signature(Seq2SeqTrainer.__init__).parameters else "tokenizer"
    )
    trainer_parameters = signature(Seq2SeqTrainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "args": trainer_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": setup_eval_datasets(raw_datasets),
        tokenizer_arg_name: tokenizer,
        "data_collator": setup_data_collator(
            model,
            tokenizer,
            strip_extra_columns=(uses_query_sampling(config) or uses_document_repeat_sampler(config)),
            query_input_max_length=config.query_input_max_length,
        ),
        "compute_metrics": compute_metrics,
        "optimizers": (optimizer, scheduler),
    }
    if atomic_target_token_ids:
        if "preprocess_logits_for_metrics" not in trainer_parameters:
            raise ValueError(
                "Atomic target evaluation requires a transformers version whose "
                "Seq2SeqTrainer supports preprocess_logits_for_metrics."
            )
        trainer_kwargs["preprocess_logits_for_metrics"] = make_atomic_preprocess_logits_for_metrics(
            atomic_target_token_ids,
            metric_depth(config),
        )

    callbacks = [MetricsHistoryCallback()]
    if config.fail_on_nonfinite_metrics:
        callbacks.append(FailOnNonFiniteMetricsCallback())
    trainer_kwargs["callbacks"] = callbacks

    if uses_query_sampling(config):
        trainer_kwargs["query_sampling"] = config.query_sampling
    if uses_document_repeat_sampler(config):
        trainer_kwargs["document_repeats"] = config.document_repeats

    if uses_custom_loss(config):
        trainer_kwargs["z_loss"] = config.z_loss

    uses_sampling = uses_query_sampling(config) or uses_document_repeat_sampler(config)

    if uses_custom_loss(config) and uses_sampling:
        trainer_cls = T5XQuerySamplingSeq2SeqTrainer
    elif uses_custom_loss(config):
        trainer_cls = T5XSeq2SeqTrainer
    elif uses_sampling:
        trainer_cls = QuerySamplingSeq2SeqTrainer
    else:
        trainer_cls = Seq2SeqTrainer
    return trainer_cls(**trainer_kwargs)


def evaluate_and_save(trainer, metric_key_prefix):
    metrics = trainer.evaluate(metric_key_prefix=metric_key_prefix)
    trainer.log_metrics(metric_key_prefix, metrics)
    trainer.save_metrics(metric_key_prefix, metrics)
    if trainer_is_world_process_zero(trainer):
        append_metrics_history(
            trainer.args.output_dir,
            f"summary:{metric_key_prefix}",
            metrics,
            state=trainer.state,
        )
        save_job_metrics(
            trainer.args.output_dir,
            metric_key_prefix,
            metrics,
            state=trainer.state,
        )
    return metrics


def train_and_save(trainer, config, train_dataset):
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    metrics = train_result.metrics
    samples_per_epoch = train_samples_per_epoch(train_dataset, config)
    metrics["train_samples"] = samples_per_epoch
    metrics["train_physical_samples"] = len(train_dataset)
    metrics["train_samples_per_epoch"] = samples_per_epoch
    trainer.log_metrics(TRAIN_SPLIT, metrics)
    trainer.save_metrics(TRAIN_SPLIT, metrics)
    trainer.save_state()
    if trainer_is_world_process_zero(trainer):
        append_metrics_history(
            trainer.args.output_dir,
            f"summary:{TRAIN_SPLIT}",
            metrics,
            state=trainer.state,
        )
        save_job_metrics(
            trainer.args.output_dir,
            TRAIN_SPLIT,
            metrics,
            state=trainer.state,
        )
    return metrics


def main():
    config = parse_args()
    set_seed(config.seed)
    logger = setup_logger()
    validate_loss(config)
    validate_z_loss(config)
    validate_query_sampling(config)
    validate_query_input_max_length(config)
    validate_document_repeats(config)
    validate_sampling_combination(config)
    logger.info(LOG_LOSS.format(value=config.loss))
    logger.info(LOG_Z_LOSS.format(value=config.z_loss))
    logger.info(LOG_QUERY_SAMPLING.format(value=config.query_sampling))
    logger.info(LOG_DOCUMENT_REPEATS.format(value=config.document_repeats))
    query_input_max_length_log = (
        "disabled"
        if config.query_input_max_length <= 0
        else f"{config.query_input_max_length} content tokens (+ EOS when present)"
    )
    logger.info(LOG_QUERY_INPUT_MAX_LENGTH.format(value=query_input_max_length_log))

    metadata = load_metadata(config, logger)
    apply_metadata_defaults(config, metadata, logger)
    accelerator = resolve_precision(config, logger)

    raw_datasets = load_datasets(config)
    added_tokens = load_added_tokens(config)
    valid_targets = collect_valid_targets(raw_datasets, added_tokens)
    logger.info(LOG_TARGET_VOCAB_SIZE.format(size=len(valid_targets)))
    seen_targets, seen_target_token_ids = collect_seen_unseen_reference_targets(
        config,
        raw_datasets,
        metadata,
        logger,
    )

    train_dataset = setup_train_dataset(raw_datasets, config, logger)
    logger.info(LOG_TRAIN_DATASET_SIZE.format(size=len(train_dataset)))
    if uses_query_sampling(config):
        sampler_summary = query_sampling_summary(train_dataset)
        logger.info(LOG_ONE_QUERY_PER_DOC_SAMPLER.format(**sampler_summary))

    model, tokenizer = load_model_tokenizer(config, added_tokens, logger)
    if config.t5_fp16_clamp:
        if uses_apex_fp16_backend(config):
            logger.info("Skipping T5 FusedRMSNorm native-AMP patch because fp16_backend=apex.")
        else:
            patch_t5_fused_rms_norm_fp32(model, logger)
        patch_t5_dense_relu_dense_fp16_clamp(model, logger)
        patch_t5_layer_ff_fp16_clamp(model, logger)
    if config.debug_nonfinite_forward:
        register_nonfinite_forward_hooks(
            model,
            logger,
            skip_t5_fp16_clamp_internals=config.t5_fp16_clamp,
        )
    target_token_ids = added_token_ids(tokenizer, added_tokens)
    atomic_target_eval = is_atomic_target_eval_dataset(
        raw_datasets,
        target_token_ids,
        tokenizer,
    )
    setup_generation_config(
        model,
        config,
        raw_datasets,
        logger,
        atomic_target_eval=atomic_target_eval,
        atomic_target_token_ids=target_token_ids,
    )
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config, train_dataset, logger)

    trainer_arguments = setup_trainer_arguments(
        config,
        train_dataset,
        logger,
        predict_with_generate=not atomic_target_eval,
        accelerator=accelerator,
    )
    trainer = setup_trainer(
        config,
        raw_datasets,
        train_dataset,
        model,
        tokenizer,
        trainer_arguments,
        valid_targets,
        optimizer,
        scheduler,
        atomic_target_token_ids=target_token_ids if atomic_target_eval else (),
        seen_targets=seen_targets,
        seen_target_token_ids=seen_target_token_ids,
    )

    evaluate_and_save(trainer, INITIAL_EVAL_METRIC_PREFIX)
    train_and_save(trainer, config, train_dataset)
    evaluate_and_save(trainer, FINAL_EVAL_METRIC_PREFIX)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
