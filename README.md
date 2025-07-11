# VibeRate

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

## Usage

To run the demo, execute

    uv run demo --test-venv TESTING_ENVIRONMENT --input-file PROBLEM_DESCRIPTION

For example

    uv run demo --test-venv test_venv/ --input-file examples/test_prob3.md

Supported options are

- `--cache-root DIR` to set LLM cache (default: `~/.viberate_cache/`)
- `--no-cache` to disable cache

## Reproducibility

Any experiment should be reproducible if the following pieces of information are provided:

- A commit hash of this repository
- A bash command to be executed from the root of this repository

Additionally, to replicate the experiment, it is sufficient to provide:

- The commit hash of your LLM cache

## Managing cache

Cache can be downloaded from

    https://github.com/msv-lab/VibeRate-cache-USER/archive/COMMIT.zip

where `USER` can be `yihan`, `haotian`, `sijie`, `sergey`.

## Experimental data

Everything that is not trivially derivable from LLM cache, should be stored in https://github.com/msv-lab/VibeRate-data
