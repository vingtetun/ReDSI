#!/usr/bin/env python3
"""
Evaluate a ReDSI seq2seq checkpoint.

Regular identifiers use beam-search generation. Atomic identifiers use first
decoder-step logits over the added atomic tokens, matching trainer evaluation.

Expected JSONL schema:
  {"kind":"document"|"query","docid":"...","input_ids":[...],"labels":[...]}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from inspect import signature

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from redsi.training import train

DEFAULT_BEAM_SIZE = 40
DEFAULT_NUM_RETURN_SEQUENCES = 40
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 16
DEFAULT_OUTPUT_DIR = "outputs/dsi_eval"
HITS_KS = (1, 5, 10)
MRR_KS = (10,)
RAW_HITS_PREFIX = "raw_hits"
RAW_MRR_PREFIX = "raw_mrr"
SPLIT_ALL = "all"
SPLIT_VALIDATION = "validation"
SPLIT_VALIDATION_INDEX = "validation_index"
EVAL_SPLIT_NAMES = {
    SPLIT_VALIDATION: train.QUERY_EVAL_NAME,
    SPLIT_VALIDATION_INDEX: train.INDEX_EVAL_NAME,
}
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_ATOMIC_EVAL_CONFIG = (
    "Atomic target evaluation: using first decoder-step logits over %s added tokens; "
    "generation and beam search are disabled."
)


@dataclass
class DatasetFiles:
    train_file: str
    validation_file: str
    validation_index_file: str
    tokens_file: str


def setup_logger():
    logger = logging.getLogger("dsi_evaluator")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a ReDSI seq2seq checkpoint.")
    parser.add_argument(
        "data_dir_arg",
        nargs="?",
        help="Exported dataset directory, or checkpoint directory when passed alone.",
    )
    parser.add_argument("checkpoint_arg", nargs="?", help="Model checkpoint directory.")
    parser.add_argument("--data-dir", default=None, help="Exported dataset directory.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint directory.")
    parser.add_argument(
        "--tokenizer-name-or-path",
        default=None,
        help="Tokenizer path/name. Defaults to --checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=(SPLIT_ALL, SPLIT_VALIDATION, SPLIT_VALIDATION_INDEX),
        default=SPLIT_ALL,
        help="Evaluation split. Default: all available eval splits.",
    )
    parser.add_argument("--train-file", default=train.DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", default=train.DEFAULT_VALIDATION_FILE)
    parser.add_argument(
        "--validation-index-file",
        default=train.DEFAULT_VALIDATION_INDEX_FILE,
    )
    parser.add_argument("--tokens-file", default=train.DEFAULT_TOKENS_FILE)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help=f"Beam size for non-atomic generation. Default: {DEFAULT_BEAM_SIZE}.",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=DEFAULT_NUM_RETURN_SEQUENCES,
        help=(
            f"Number of generated sequences returned per non-atomic example. Default: {DEFAULT_NUM_RETURN_SEQUENCES}."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=train.AUTO_MAX_NEW_TOKENS,
        help="Maximum generated target tokens. 0 infers from labels. Default: 0.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
    )
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path where the metrics JSON should be written.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--metric-key-prefix",
        default=train.EVAL_METRIC_PREFIX,
        help=f"Metric key prefix. Default: {train.EVAL_METRIC_PREFIX}.",
    )
    args = parser.parse_args()
    resolve_data_dir_and_checkpoint(args, parser)
    validate_args(args, parser)
    return args


def resolve_data_dir_and_checkpoint(args, parser):
    positional = [value for value in (args.data_dir_arg, args.checkpoint_arg) if value]

    if len(positional) == 2:
        position_data_dir, position_checkpoint = positional
        if args.data_dir and args.data_dir != position_data_dir:
            parser.error("data_dir was provided both positionally and via --data-dir")
        if args.checkpoint and args.checkpoint != position_checkpoint:
            parser.error("checkpoint was provided both positionally and via --checkpoint")
        args.data_dir = args.data_dir or position_data_dir
        args.checkpoint = args.checkpoint or position_checkpoint
    elif len(positional) == 1:
        position_value = positional[0]
        if args.data_dir and args.checkpoint:
            parser.error(f"unexpected positional argument: {position_value}")
        if args.data_dir:
            args.checkpoint = args.checkpoint or position_value
        elif args.checkpoint:
            args.data_dir = position_value
        else:
            args.checkpoint = position_value

    if not args.checkpoint:
        parser.error("checkpoint is required")
    if not args.data_dir:
        args.data_dir = infer_data_dir_from_checkpoint(args, parser)


def has_dataset_files(path, args):
    return os.path.isfile(os.path.join(path, args.train_file)) and os.path.isfile(os.path.join(path, args.val_file))


def infer_data_dir_from_checkpoint(args, parser):
    path = os.path.abspath(args.checkpoint)
    if os.path.isfile(path):
        path = os.path.dirname(path)

    while True:
        if has_dataset_files(path, args):
            return path

        parent = os.path.dirname(path)
        if parent == path:
            parser.error("could not infer data_dir from checkpoint path; pass --data-dir explicitly")
        path = parent


def validate_args(args, parser):
    if args.beam_size < 1:
        parser.error("--beam-size must be >= 1")
    if args.num_return_sequences < 1:
        parser.error("--num-return-sequences must be >= 1")
    if args.num_return_sequences > args.beam_size:
        parser.error("--num-return-sequences must be <= --beam-size for beam search")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be >= 0")
    if args.per_device_eval_batch_size < 1:
        parser.error("--per-device-eval-batch-size must be >= 1")
    if args.max_eval_examples < 0:
        parser.error("--max-eval-examples must be >= 0")


def dataset_files(args):
    return DatasetFiles(
        train_file=os.path.join(args.data_dir, args.train_file),
        validation_file=os.path.join(args.data_dir, args.val_file),
        validation_index_file=os.path.join(args.data_dir, args.validation_index_file),
        tokens_file=os.path.join(args.data_dir, args.tokens_file),
    )


def existing_eval_file(path, *, required):
    if os.path.exists(path):
        return path
    if required:
        raise FileNotFoundError(f"Missing required file: {path}")
    return None


def load_eval_datasets(args, files):
    data_files = {}

    if args.split in (SPLIT_ALL, SPLIT_VALIDATION):
        data_files[SPLIT_VALIDATION] = existing_eval_file(
            files.validation_file,
            required=True,
        )

    if args.split in (SPLIT_ALL, SPLIT_VALIDATION_INDEX):
        validation_index_file = existing_eval_file(
            files.validation_index_file,
            required=args.split == SPLIT_VALIDATION_INDEX,
        )
        if validation_index_file is not None:
            data_files[SPLIT_VALIDATION_INDEX] = validation_index_file

    raw_datasets = load_dataset(
        train.DATASET_FORMAT_JSON,
        data_files=data_files,
        cache_dir=args.cache_dir,
    )

    if args.max_eval_examples > 0:
        for split_name, dataset in raw_datasets.items():
            raw_datasets[split_name] = dataset.select(range(min(args.max_eval_examples, len(dataset))))

    return {EVAL_SPLIT_NAMES[split_name]: dataset for split_name, dataset in raw_datasets.items()}


def load_added_tokens(tokens_file):
    if not os.path.exists(tokens_file):
        return []

    tokens = []
    with open(tokens_file, encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if train.TOKENS_KEY in row:
                tokens.extend(row[train.TOKENS_KEY])
            elif train.TOKEN_KEY in row:
                tokens.append(row[train.TOKEN_KEY])
    return tokens


def iter_jsonl_rows(path):
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)


def collect_docid_targets(paths: Iterable[str]):
    targets = set()
    for path in paths:
        for row in iter_jsonl_rows(path):
            docid = row.get(train.DOCID_COL)
            if docid is not None:
                targets.add(str(docid).strip())
    return targets


def collect_valid_targets(files, added_tokens):
    if added_tokens:
        return {str(token).strip() for token in added_tokens}

    return collect_docid_targets(
        (
            files.train_file,
            files.validation_file,
            files.validation_index_file,
        )
    )


def load_model_and_tokenizer(args, added_tokens, logger):
    tokenizer_path = args.tokenizer_name_or_path or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    if added_tokens:
        tokenizer.add_tokens(added_tokens)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    embedding_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    elif len(tokenizer) < embedding_size:
        logger.warning(
            "model embeddings have %s rows, but tokenizer has %s tokens; not resizing down",
            embedding_size,
            len(tokenizer),
        )

    return model, tokenizer


def infer_max_new_tokens(eval_datasets):
    max_label_length = 0
    for dataset in eval_datasets.values():
        if train.LABELS_COL not in dataset.column_names:
            continue
        for labels in dataset[train.LABELS_COL]:
            max_label_length = max(max_label_length, len(labels))
    return max(1, max_label_length)


def configure_generation(model, args, eval_datasets, logger, atomic_target_token_ids=()):
    if atomic_target_token_ids:
        model.generation_config.max_new_tokens = 1
        model.generation_config.num_beams = 1
        model.generation_config.num_return_sequences = 1
        logger.info(LOG_ATOMIC_EVAL_CONFIG, f"{len(atomic_target_token_ids):,}")
        return

    if args.max_new_tokens == train.AUTO_MAX_NEW_TOKENS:
        max_new_tokens = infer_max_new_tokens(eval_datasets)
    else:
        max_new_tokens = args.max_new_tokens

    model.generation_config.max_new_tokens = int(max_new_tokens)
    model.generation_config.num_beams = int(args.beam_size)
    model.generation_config.num_return_sequences = int(args.num_return_sequences)
    model.generation_config.early_stopping = True

    if args.num_return_sequences < max(HITS_KS):
        logger.warning(
            "num_return_sequences=%s is lower than hits@%s; metrics cannot see more returned candidates",
            args.num_return_sequences,
            max(HITS_KS),
        )

    logger.info(
        "Generation config: max_new_tokens=%s, num_beams=%s, num_return_sequences=%s",
        model.generation_config.max_new_tokens,
        model.generation_config.num_beams,
        model.generation_config.num_return_sequences,
    )


def maybe_adjust_workers_for_mps(args, logger):
    accelerator = train.detect_accelerator()
    if accelerator == train.ACCELERATOR_MPS and args.dataloader_num_workers != 0:
        logger.info("MPS detected; using dataloader_num_workers=0")
        args.dataloader_num_workers = 0
    logger.info(
        "Dataloader pin_memory: %s",
        train.dataloader_pin_memory_for_accelerator(accelerator),
    )
    return accelerator


def trainer_arg_name():
    if "processing_class" in signature(Seq2SeqTrainer.__init__).parameters:
        return "processing_class"
    return "tokenizer"


def reciprocal_rank(predictions, gold_target, k):
    for rank, predicted_target in enumerate(predictions[:k], start=1):
        if predicted_target == gold_target:
            return 1.0 / rank
    return 0.0


def make_compute_metrics(tokenizer, valid_targets, hits_ks, mrr_ks):
    hit_ks = sorted(set(int(hit_k) for hit_k in hits_ks))
    reciprocal_rank_ks = sorted(set(int(mrr_k) for mrr_k in mrr_ks))
    max_metric_k = max(hit_ks + reciprocal_rank_ks)

    def compute_metrics(eval_preds):
        predictions = eval_preds.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        labels = eval_preds.label_ids
        prediction_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        prediction_texts = [prediction.strip() for prediction in prediction_texts]

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        labels = np.where(labels == train.LABEL_PAD_TOKEN_ID, pad_token_id, labels)
        gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        gold_texts = [gold.strip() for gold in gold_texts]

        batch_size = len(gold_texts)
        return_sequences = (
            len(prediction_texts) // batch_size if batch_size > 0 and len(prediction_texts) % batch_size == 0 else 1
        )

        raw_hits = {hit_k: 0 for hit_k in hit_ks}
        filtered_hits = {hit_k: 0 for hit_k in hit_ks}
        raw_reciprocal_rank_sums = {mrr_k: 0.0 for mrr_k in reciprocal_rank_ks}
        filtered_reciprocal_rank_sums = {mrr_k: 0.0 for mrr_k in reciprocal_rank_ks}
        total = 0
        total_predictions = 0
        invalid_predictions = 0

        for index in range(batch_size):
            gold_target = gold_texts[index]
            example_predictions = prediction_texts[index * return_sequences : (index + 1) * return_sequences]

            total_predictions += len(example_predictions)
            invalid_predictions += sum(
                1 for predicted_target in example_predictions if predicted_target not in valid_targets
            )

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

            total += 1
            for hit_k in hit_ks:
                if gold_target in example_predictions[:hit_k]:
                    raw_hits[hit_k] += 1
                if gold_target in filtered[:hit_k]:
                    filtered_hits[hit_k] += 1

            for mrr_k in reciprocal_rank_ks:
                raw_reciprocal_rank_sums[mrr_k] += reciprocal_rank(
                    example_predictions,
                    gold_target,
                    mrr_k,
                )
                filtered_reciprocal_rank_sums[mrr_k] += reciprocal_rank(
                    filtered,
                    gold_target,
                    mrr_k,
                )

        metrics = {f"{RAW_HITS_PREFIX}@{hit_k}": raw_hits[hit_k] / max(total, 1) for hit_k in hit_ks}
        metrics.update(
            {
                f"{RAW_MRR_PREFIX}@{mrr_k}": raw_reciprocal_rank_sums[mrr_k] / max(total, 1)
                for mrr_k in reciprocal_rank_ks
            }
        )
        metrics.update(
            {f"{train.METRIC_HITS_PREFIX}@{hit_k}": filtered_hits[hit_k] / max(total, 1) for hit_k in hit_ks}
        )
        metrics.update(
            {
                f"{train.METRIC_MRR_PREFIX}@{mrr_k}": (filtered_reciprocal_rank_sums[mrr_k] / max(total, 1))
                for mrr_k in reciprocal_rank_ks
            }
        )
        metrics[train.METRIC_INVALID_RATE] = invalid_predictions / max(total_predictions, 1)
        return metrics

    return compute_metrics


def metric_depth():
    return max(*HITS_KS, *MRR_KS)


def build_trainer(
    args,
    model,
    tokenizer,
    eval_datasets,
    valid_targets,
    accelerator,
    atomic_target_token_ids=(),
):
    atomic_target_token_ids = tuple(int(token_id) for token_id in atomic_target_token_ids)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=not bool(atomic_target_token_ids),
        report_to=[],
        dataloader_pin_memory=train.dataloader_pin_memory_for_accelerator(accelerator),
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
    )

    if atomic_target_token_ids:
        trainer_parameters = signature(Seq2SeqTrainer.__init__).parameters
        if "preprocess_logits_for_metrics" not in trainer_parameters:
            raise ValueError(
                "Atomic target evaluation requires a transformers version whose "
                "Seq2SeqTrainer supports preprocess_logits_for_metrics."
            )
        compute_metrics = train.make_atomic_compute_metrics(
            atomic_target_token_ids,
            HITS_KS,
            MRR_KS,
        )
        extra_trainer_kwargs = {
            "preprocess_logits_for_metrics": train.make_atomic_preprocess_logits_for_metrics(
                atomic_target_token_ids,
                metric_depth(),
            ),
        }
    else:
        compute_metrics = make_compute_metrics(
            tokenizer,
            valid_targets,
            HITS_KS,
            MRR_KS,
        )
        extra_trainer_kwargs = {}

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_datasets,
        **{trainer_arg_name(): tokenizer},
        data_collator=train.setup_data_collator(model, tokenizer),
        compute_metrics=compute_metrics,
        **extra_trainer_kwargs,
    )


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_metrics(metrics, output_json):
    text = json.dumps(json_ready(metrics), indent=2, sort_keys=True)
    print(text)
    if output_json:
        with open(output_json, "w", encoding="utf-8") as output_file:
            output_file.write(text)
            output_file.write("\n")


def main():
    args = parse_args()
    logger = setup_logger()
    accelerator = maybe_adjust_workers_for_mps(args, logger)

    files = dataset_files(args)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Resolved data_dir: %s", args.data_dir)
    logger.info("Loading eval data from %s", args.data_dir)
    eval_datasets = load_eval_datasets(args, files)

    added_tokens = load_added_tokens(files.tokens_file)
    valid_targets = collect_valid_targets(files, added_tokens)
    logger.info("Valid target count: %s", f"{len(valid_targets):,}")

    model, tokenizer = load_model_and_tokenizer(args, added_tokens, logger)
    target_token_ids = train.added_token_ids(tokenizer, added_tokens)
    atomic_target_token_ids = (
        target_token_ids if train.is_atomic_target_eval_dataset(eval_datasets, target_token_ids, tokenizer) else ()
    )
    configure_generation(
        model,
        args,
        eval_datasets,
        logger,
        atomic_target_token_ids=atomic_target_token_ids,
    )

    trainer = build_trainer(
        args,
        model,
        tokenizer,
        eval_datasets,
        valid_targets,
        accelerator,
        atomic_target_token_ids=atomic_target_token_ids,
    )
    metrics = trainer.evaluate(metric_key_prefix=args.metric_key_prefix)
    write_metrics(metrics, args.output_json)


if __name__ == "__main__":
    main()
