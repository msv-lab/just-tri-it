# just-tri-it

This project aims to detect hallucinations in LLM-genarated code using resonance check.

## Preliminaries

Set your 302.ai API key using the environment variable `AI302_API_KEY`.

The project uses uv to manage dependencies.

When runnig the project for the first time, please create an environment for testing generated programs:

    uv venv --no-project --seed --python 3.13 test_venv
    source test_venv/bin/activate
    pip install -r test_requirements.txt
    deactivate
    
Run unit tests with

    uv run pytest

Run linter and type checker:

    uv run ruff check .
    uvx mypy src/

## Usage

just-tri-it supports multiple workflows:

### Benchmarking

The goal of the benchmarking tool is to compare various tool configurations on various code generation datasets and compute basic statistics.

To run a benchmark, execute

    uv run benchmark --test-venv TESTING_ENVIRONMENT --dataset DATASET [ --task TASK_ID ] --selector TOOL_CONFIG --model MODEL

For example

    uv run benchmark --test-venv test_venv/ --dataset datasets/test.json --selector Plurality --model gpt-4o
    
For LiveCodeBench v6, first decompress the dataset using

    unzip dataset/lcb_part6.json.zip
    
A specific task can be specified using `--task`:
    
    uv run benchmark --test-venv test_venv/ --dataset datasets/lcb_part6.json --selector CodeT_IO --model gpt-4o --task atcoder_abc387_b
    
List of configurations:

- `Plurality`
- `MaxTest_Assert`
- `MaxTest_IO`
- `CodeT_Assert`
- `CodeT_IO`
- `Syntactic`
- `OffByOne`
- `Postcondition`
- `FOR_INV`
- `FOR_FIB`

### Experiments

The goal of the experimentation tool is to provide comprehensive measurements and allow for iterative improvements.

To run LiveCodeBench v6, first decompress the dataset using:
    
    unzip datasets/lcb_part6.json.zip
    
Then, to collect experimental results and save (append) into `data_dir`, execute

    uv run experiment --test-venv test_venv/ --dataset datasets/lcb_part6.json --model gpt-4o --data data_dir

Optionally, you can specify the list of tasks to run via the option `--task-list datasets/lcb_top30.txt`.

Then, to compute measures and generate plots, execute

    uv run analyze --data data_dir --report report_dir

## Reproducibility

just-tri-it provides the following options to manage LLM cache:

- `--cache-root DIR` to set LLM cache (default: `~/.just_tri_it_cache/`)
- `--export-cache DIR` to explore all cached samples used during the run to a different directory
- `--no-cache` to disable cache
- `--replicate` to use only cache; fail in case of cache misses

Any experiment should be reproducible if the following pieces of information are provided:

- A commit hash of this repository
- A bash command to be executed from the root of this repository

Additionally, to replicate the experiment, it is sufficient to provide:

- The commit hash of your LLM cache

Cache can be downloaded from

    https://github.com/msv-lab/just-tri-it-cache-USER/archive/COMMIT.zip

where `USER` can be `yihan`, `haotian`, `sijie`, `sergey`.

## Experimental data

Everything that is not trivially derivable from LLM cache, should be stored in https://github.com/msv-lab/just-tri-it-data
