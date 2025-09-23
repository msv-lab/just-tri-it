# Replicating Conversion of LiveCodeBench Part 6 into VibeRate Format

First, download LiveCodeBench (fast) data file `test6.jsonl`.

Second, download GPT-4o cache: https://github.com/msv-lab/just-tri-it-cache-sergey/archive/livecodebench_part6_conversion_v2.zip

Then, on the commit f0533f5d6ee48dde0bc97baccd0bb7a1e59d0b30, execute

    uv run preprocess_dataset --test-venv test_venv/ --format LiveCodeBench --dataset test6.jsonl --output lcb_part6.json --cache-root livecodebench_part6_conversion_v2
    
The converter generates parsers and printers for all inputs and outputs and checks that their composition is an identity function.

Old version:

- https://github.com/msv-lab/just-tri-it-cache-sergey/archive/livecodebench_part6_conversion.zip
- 02209a156dc48c9b98f9ba7e44e81c0945c373cf

# Inspecting Dataset

The VibeRate format uses LiveCodeBench's test data compression. To decompress test data for a given task, execute

    uv run preprocess_dataset --decompress --dataset lcb_part6.json --task atcoder_abc387_b --output decompressed.json
    
