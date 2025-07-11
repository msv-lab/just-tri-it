import os
import hashlib
from pathlib import Path
from typing import List
import requests
import json
import mistletoe


class Cached:
    def __init__(self, llm, cache_root: Path):
        self.llm = llm
        self.model_name = llm.model_name
        self.temperature = llm.temperature
        self.cache_root = cache_root

    @staticmethod        
    def prompt_id(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _sample_dir(self, prompt: str) -> Path:
        subdir = f"{self.model_name}_{self.temperature}"
        return self.cache_root / subdir / Cached.prompt_id(prompt)

    def _read_cached_samples(self, prompt: str) -> List[str]:
        dir_ = self._sample_dir(prompt)
        if not dir_.exists():
            return []
        cached = []
        i = 0
        while True:
            fname = dir_ / f"{i}.md"
            if fname.exists():
                with open(fname, "r", encoding="utf-8") as f:
                    cached.append(f.read())
                i += 1
            else:
                break
        return cached

    def _write_samples(self, prompt: str, samples: List[str], offset: int = 0):
        d = self._sample_dir(prompt)
        os.makedirs(d, exist_ok=True)
        for idx, sample in enumerate(samples):
            fname = d / f"{offset+idx}.md"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(sample)

    def sample(self, prompt: str, k: int = 1) -> List[str]:
        cached = self._read_cached_samples(prompt)
        n_cached = len(cached)
        missing = max(k - n_cached, 0)

        new_samples = []
        if missing > 0:
            new_samples = self.llm.sample(prompt, missing)
            self._write_samples(prompt, new_samples, offset=n_cached)

        return cached[:k] + new_samples

class AI302:

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature

    def sample(self, prompt, k=1):
        url = "https://api.302.ai/v1/chat/completions"

        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + os.environ['AI302_API_KEY'],
            'Content-Type': 'application/json'
        }

        responses = []
        for _ in range(0, k):
            response = requests.request("POST", url, headers=headers, data=payload)
            responses.append(json.loads(response.text)["choices"][0]["message"]["content"])

        return responses


class DataExtractionFailure(Exception):
    "Raised when failed to parse LLM output"
    pass    
    

def extract_code(content):
    """Extract first markdown code block"""
    parsed = mistletoe.Document(content)
    for child in parsed.children:
        if child.__class__.__name__ == "CodeFence":
            return child.children[0].content
    raise DataExtractionFailure


def extract_answer(s):
    if "<answer>" in s and "</answer>" in s and \
       s.index("<answer>") < s.index("</answer>"):
        return s.split("<answer>", 1)[1].split("</answer>", 1)[0]
    else:
        raise DataExtractionFailure
