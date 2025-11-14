# just-tri-it

This project aims to detect hallucinations in LLM-genarated code using semantic triangulation.

Mechanized proofs: `proofs/SemanticTriangulation.lean`

## Repository Layout

- `src/`: Core library including our implementation of semantic triangulation and other methods used for comparison, as well as experiment scripts.
- `proofs/`: Proofs for semantic triangulation in Lean.
- `datasets/`: Datasets used for experiment, including LiveCodeBench v6, CodeElo inexact problems and easy tasks for simple testing.
- `results/`: Evaluation results and cache for LiveCodeBench and CodeElo.
- Auxiliary Content:
  - `scripts/`: Small cripts used for dataset processing, plotting, and evaluation of semantic entropy.
  - `tests/`: Unit tests for our semantic triangulation implementation.
  - `doc/`: Documents explaining how we preprocess the datasets to adapt to our framework and how to make a simple try on the evaluated methods.

## Preliminaries

Set your 302.ai API key using the environment variable `AI302_API_KEY`.

The project uses **uv** to manage dependencies.

Run linter and type checker:

    uv run ruff check .
    uvx mypy src/
    
To run tests, execute `uv run pytest`.

## Usage

just-tri-it supports multiple workflows:

### Benchmarking

The goal of the benchmarking tool is to compare various tool configurations on various code generation datasets and compute basic statistics.

To run a benchmark, execute

    uv run benchmark --dataset DATASET [ --task TASK_ID ] --selector TOOL_CONFIG --model MODEL

For example

    uv run benchmark --dataset datasets/test.json --selector Plurality --model gpt-4o
    
For LiveCodeBench v6, first decompress the dataset using

    unzip datasets/lcb_part6.json.zip
    
A specific task can be specified using `--task`:
    
    uv run benchmark --dataset datasets/lcb_part6.json --selector CodeT_IO --model gpt-4o --task atcoder_abc387_b
    
List of configurations:

- `Plurality`
- `MaxTest_Assert`
- `MaxTest_IO`
- `CodeT_Assert`
- `CodeT_IO`
- `Syntactic`
- `OffByOne`
- `Postcondition`
- `FWD_INV`
- `FWD_SINV`
- `ENUM_SINV`

### Experiments

The goal of the experimentation tool is to provide comprehensive measurements and allow for iterative improvements.

To run LiveCodeBench v6, first decompress the dataset using:
    
    unzip datasets/lcb_part6.json.zip
    
Then, to collect experimental results and save (append) into `data_dir`, execute

    uv run experiment --dataset datasets/lcb_part6.json --model gpt-4o --data data_dir

You can specify a specific task to run via the option `--only atcoder_abc387_b`.

Then, to compute measures and generate plots, execute

    uv run analyze --data data_dir --report report_dir

## Reproducibility

just-tri-it provides the following options to manage LLM cache:

- `--cache-root DIR` to set LLM cache (default: `~/.just_tri_it_cache/`)
- `--export-cache DIR` to explore all cached samples used during the run to a different directory
- `--no-cache` to disable cache
- `--replicate` to use only cache; fail in case of cache misses

To replicate our experiments, please unzip the corresponding cache in `/results`, and then:
 - for LCB, add option `--cache-root PATH_TO_LCB_CACHE`
 - for CodeElo, add option `--cache-root PATH_TO_CEl_CACHE`


