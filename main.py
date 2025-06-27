from pathlib import Path
import argparse
import os

from llm import Cached, AI302, extract_answer, extract_code


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=str,
        help="Set LLM cache root directory (default: ~/.viberate_cache/)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = "deepseek-v3-huoshan"
    
    m = AI302(model_name, 1.0)
    if not args.no_cache:
        if args.cache_root:
            m = Cached(m, Path(args.cache_root))
        else:
            m = Cached(m, Path.home() / ".viberate_cache")

    prompt = "What is greater, 9.11 or 9.9? Wrap your answer with the tag <answer></answer>."
    responses = m.sample(prompt, k=5)

    for r in responses:
        answer = extract_answer(r)
        print(answer)


if __name__ == "__main__":
    main()
