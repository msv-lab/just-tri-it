# VibeRate

Set your 302.ai API key using the environment variable `AI302_API_KEY`.

To run the project, execute

    uv run main.py OPTIONS
    
Supported options are

- `--cache-root DIR` to set LLM cache (default: `~/.viberate_cache/`)
- `--no-cache` to disable cache

## Reproducibility

Any experiment should be reproducible if the following pieces of information are provided:

- A commit hash of this repository
- A bash command to be executed from the root of this repository

Additionally, to replicate the experiment, it is sufficient to provide:

- The commit hash of your LLM cache
