# Replicating Conversion of LiveCodeBench Part 6 into VibeRate Format

First, download LiveCodeBench (fast) data file `test6.jsonl`.

Second, download GPT-4o cache: https://github.com/msv-lab/just-tri-it-cache-sergey/archive/livecodebench_part6_conversion_v2.zip

Then, on the commit 857478838718f5b5869f5ddea263cd86c5f54b6d, execute

    uv run preprocess_dataset --format LiveCodeBench --dataset test6.jsonl --output lcb_part6.json --cache-root livecodebench_part6_conversion_v2
    
The converter generates parsers and printers for all inputs and outputs and checks that their composition is an identity function.

Old version:

- https://github.com/msv-lab/just-tri-it-cache-sergey/archive/livecodebench_part6_conversion.zip
- 02209a156dc48c9b98f9ba7e44e81c0945c373cf

# Inspecting Dataset

The VibeRate format uses LiveCodeBench's test data compression. To decompress test data for a given task, execute

    uv run preprocess_dataset --decompress --dataset lcb_part6.json --task atcoder_abc387_b --output decompressed.json
    
# Sanity Check for A Test Suite

If your dataset has the fields `correct_solution` and `incorrect_solution` in the task metadata, i.e.

    [
        {
            "id": ...,
            ...
            "metadata": { 
                "correct_solution": ...
                "incorrect_solution": ...
            }
        }
        ...
    ]
    
To run the sanity check, execute

    uv run construct_judges --sanity-check --dataset datasets/test.json
    
It will check the ability of the test suite to classify the correct solution as correct, and the incorrect as incorrect.
