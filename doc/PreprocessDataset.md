# Replicating Conversion of LiveCodeBench Part 6 into VibeRate Format

First, download LiveCodeBench (fast) data file `test6.jsonl`.

Second, download GPT-4o cache: https://github.com/msv-lab/VibeRate-cache-sergey/archive/livecodebench_part6_conversion.zip

Then, on the commit 02209a156dc48c9b98f9ba7e44e81c0945c373cf, execute

    uv run preprocess_dataset --test-venv test_venv/ --format LiveCodeBench --dataset test6.jsonl --output lcb_part6.json --cache-root livecodebench_part6_conversion
    
The converter generates parsers and printers for all inputs and outputs and checks that their composition is an identity function.

# Inspecting Dataset

The VibeRate format uses LiveCodeBench's test data compression. To decompress test data for a given task, execute

    uv run preprocess_dataset --decompress --dataset lcb_part6.json --task atcoder_abc387_b --output decompressed.json
