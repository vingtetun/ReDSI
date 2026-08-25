# ReDSI

ReDSI is a reusable dataset, identifier, and seq2seq training pipeline for
Differentiable Search Index and generative retrieval experiments.

The primary use case is fast and reproducible research iteration. ReDSI handles
the standard dataset, tokenization, export, training, and evaluation plumbing so
a researcher can build a small Python package, register one entry point, and run
experiments against a pre-built Natural Questions dataset.

ReDSI provides:

- pre-built Natural Questions loading through the `nq320k` plugin;
- built-in atomic, naive, semantic, UUID, hash, and identity identifiers;
- tokenization and export for seq2seq DSI training;
- `redsi-train` and `redsi-evaluate` commands;
- plugin discovery through Python package entry points.

## Installation

For a released version:

```bash
python -m pip install redsi
```

For local ReDSI development:

```bash
git clone git@github.com:vingtetun/ReDSI.git
cd ReDSI

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Check that the package and built-in plugins are visible:

```bash
redsi-build --plugins
```

## Quick Start

Run the default pre-built NQ320K pipeline:

```bash
redsi-build --config-name nq320k
```

This loads [`vingtetun/nq320k`](https://huggingface.co/datasets/vingtetun/nq320k),
assigns naive identifiers, tokenizes the dataset, adds a validation index split,
and writes the exported data to:

```text
build/nq320k-htmlnorm-pageid-naive/320K
```

By default, the tokenizer keeps a 32-token document view:

```yaml
transformation.tokenizer.document_max_length: 32
```

Use a 64-token document view when that better matches the experiment:

```bash
redsi-build \
  --config-name nq320k \
  transformation.tokenizer.document_max_length=64 \
  arguments.out_dir=build/nq320k-htmlnorm-pageid-naive-doc64
```

This build-time document length is separate from the training-time
`--query-input-max-length` option shown below.

Train on the exported dataset:

```bash
redsi-train \
  --data-dir build/nq320k-htmlnorm-pageid-naive/320K \
  --model-name-or-path google-t5/t5-base \
  --output-dir outputs/nq320k-htmlnorm-pageid-naive \
  --learning-rate 1e-3 \
  --warmup-steps 0 \
  --num-train-epochs 500 \
  --per-device-train-batch-size 512 \
  --per-device-eval-batch-size 16 \
  --query-input-max-length 64 \
  --eval-beam-size 40 \
  --eval-return-sequences 40
```

These are the common knobs to start with. Use `redsi-train --help` for the full
set of training, precision, sampling, checkpointing, and evaluation options.

Evaluate a checkpoint:

```bash
redsi-evaluate \
  --data-dir build/nq320k-htmlnorm-pageid-naive/320K \
  --checkpoint outputs/nq320k-htmlnorm-pageid-naive
```

## Build An Identifier Package

You do not need to modify ReDSI to try a new identifier. Create a separate
package that depends on ReDSI and registers an identifier plugin.

Minimal `pyproject.toml`:

```toml
[project]
name = "my-redsi-identifiers"
version = "0.1.0"
dependencies = ["redsi>=0.1.0"]

[project.entry-points."redsi.plugins.identifiers"]
my_identifier = "my_redsi_identifiers.plugin:MyIdentifierPlugin"
```

Minimal `my_redsi_identifiers/plugin.py`:

```python
from redsi.formats import formats
from redsi.plugins import Param


class MyIdentifierPlugin:
    description = "Assign simple prefixed integer document identifiers."
    params = [
        Param("prefix", "str", default="doc"),
    ]

    def run(self, documents, params, seed):
        return {
            doc[formats.FEATURE_DOCID]: f"{params.prefix}{index}"
            for index, doc in enumerate(documents)
        }
```

Install your package in the same environment as ReDSI:

```bash
python -m pip install -e .
redsi-build --plugins
```

Run the default NQ320K pipeline with your identifier:

```bash
redsi-build \
  --config-name nq320k \
  '~identifiers' \
  '+identifiers.my_identifier={prefix: exp}'
```

The important part is the last two overrides:

- `~identifiers` removes the default identifier config.
- `+identifiers.my_identifier={...}` adds the identifier discovered from your
  package entry point.

## Natural Questions Data

ReDSI exposes Natural Questions in two different ways.

### Pre-Built NQ320K

Use this path when you want to run experiments from an existing packaged
Natural Questions variant. This is the recommended starting point because it
keeps dataset construction out of the identifier research loop.

The default NQ320K config loads:

```yaml
data_source: vingtetun/nq320k
config_name: htmlnorm-pageid
repo_type: dataset
```

Use another pre-built package variant by overriding `config_name`:

```bash
redsi-build \
  --config-name nq320k \
  datasets.params.config_name=ncinorm-ncititle
```

You can also point `datasets.params.data_source` at a local NQ320K package
directory instead of a Hugging Face repository.

### Building Natural Questions Variants

Use the `natural_questions` plugin when you want to build a dataset variant
from the raw/intermediate Natural Questions source files. This path is for
dataset construction and research iteration, not the shortest path for regular
training.

For example, this builds a full NQ320K package variant from the local
`build/natural_questions` source and writes a reusable package under
`build/nq320k_package/data/ncinorm/ncititle`:

```bash
redsi-build --config-name build/nq320k_package/ncinorm/ncititle
```

The package-build configs are organized by text style and deduplication method:

```text
src/redsi/configs/yaml/build/nq320k_package/
  htmlnorm/
  ncinorm/
  simplenorm/
  simplified/
```

## Plugin Model

ReDSI is designed so a researcher can build a separate package and register new
behavior without modifying ReDSI itself.

Plugins are discovered from Python entry points. The built-in groups are:

- `redsi.plugins.datasets`
- `redsi.plugins.identifiers`
- `redsi.plugins.transformation`
- `redsi.plugins.augmentation`
- `redsi.plugins.export`

Each plugin class must expose:

- `description`: a short string shown by `redsi-build --plugins`;
- `params`: a list of `redsi.plugins.Param(...)` declarations;
- `run(...)`: the callable used by the pipeline.

Dataset plugins may also expose `load(...)`; this is the common pattern for
plugins that load a source dataset and then return preprocessing stages from
`run()`.

This keeps ReDSI as the shared engine while project-specific datasets,
identifier variants, augmentations, or token-initialization methods live in
their own research packages.
