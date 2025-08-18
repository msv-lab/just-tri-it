import os
import copy
import hashlib
from pathlib import Path
from typing import Iterable, Optional, Any
from abc import ABC, abstractmethod
import json

import requests
import mistletoe


class LLM(ABC):
    @abstractmethod
    def sample(self, prompt: str) -> Iterable[str]:
        pass


class ReplicationCacheMiss(Exception):
    "Raised when a cache miss occurs during replication"
    pass


class Cached(LLM):
    '''This decorator caches responses from an LLM by storing them in
    a specified directory.  The `replication` argument controls the
    behavior when a cache miss occurs. If `replication` is `False`,
    the system queries the LLM to generate a response. If
    `replication` is `True`, the system raises the exception
    ReplicationCacheMiss instead of querying the LLM.  `alias` is
    useful when different providers use different names for the same
    LLM, or the name contains symbols that cannot be used in
    filesystem paths. If `stateful` is `False`, each call of sample
    returns a new iterator through the same sequence. If `stateful` is
    `True`, each call of sample returns reference to the same
    iterator.

    '''
    def __init__(self, llm, cache_root: Path, replication: bool = False, alias: Optional[str] = None, stateful=False):
        self.llm = llm
        if alias is not None:
            self.alias = alias
        else:
            self.alias = llm.model_name
        self.replication = replication
        self.cache_root = cache_root
        self.cache_export_root = None
        self.base_state: dict[str, Any] = dict()
        self.stateful = stateful
        self.state: dict[str, Any] = dict()

    def _model_dir(self, root: Path):
        return root / f"{self.alias}_{self.llm.temperature}"

    def start_slicing(self, d: Path):
        self.cache_export_root = d

    def stop_slicing(self):
        self.cache_export_root = None

    def get_stateful(self):
        new = copy.copy(self)
        new.base_state = dict() #FIXME: I am not sure about this one
        new.state = dict()
        new.stateful = True
        return new

    def sample(self, prompt: str) -> Iterable[str]:
        if prompt not in self.base_state:
            self.base_state[prompt] = self.llm.sample(prompt)
        if self.stateful:
            if prompt not in self.state:
                self.state[prompt] = Cached._LazyCachedSampler(self, prompt)
            return self.state[prompt]
        else:
            return Cached._LazyCachedSampler(self, prompt)

    @staticmethod        
    def prompt_id(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    class _LazyCachedSampler():

        def __init__(self, base, prompt):
            self.base = base
            self.prompt = prompt
            self.index = 0

        def __iter__(self):
            return self

        def _read_cached_sample(self, root: Path, prompt: str, i: int) -> Optional[str]:
            d = self.base._model_dir(root) / Cached.prompt_id(prompt)
            if not d.exists():
                return None
            fname = d / f"{i}.md"
            if fname.exists():
                with open(fname, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                return None

        def _write_sample(self, root: Path, prompt: str, sample: str, i: int):
            d = self.base._model_dir(root) / Cached.prompt_id(prompt)
            os.makedirs(d, exist_ok=True)
            fname = d / f"{i}.md"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(sample)
        
        def __next__(self):
            sample = self._read_cached_sample(self.base.cache_root, self.prompt, self.index)
            if sample is None:
                if self.base.replication:
                    raise ReplicationCacheMiss(Cached.prompt_id(self.prompt))
                sample = next(self.base.base_state[self.prompt])
                self._write_sample(self.base.cache_root, self.prompt, sample, self.index)
            if self.base.cache_export_root is not None:
                self._write_sample(self.base.cache_export_root,
                                   self.prompt,
                                   sample,
                                   self.index)
            self.index += 1
            return sample


class AI302(LLM):

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature

    def start_slicing(self, d: Path):
        pass

    def stop_slicing(self):
        pass

    def get_stateful(self):
        return self

    def sample(self, prompt: str) -> Iterable[str]:
        url = "https://api.302.ai/v1/chat/completions"
        print("sample sample sample sample")

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

        while True:
            raw_response = requests.request("POST", url, headers=headers, data=payload)
            yield json.loads(raw_response.text)["choices"][0]["message"]["content"]


class XMCP(LLM):

    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature

    def start_slicing(self, d: Path):
        pass

    def stop_slicing(self):
        pass

    def get_stateful(self):
        return self

    def sample(self, prompt: str) -> Iterable[str]:
        url = "https://llm.xmcp.ltd/chat/completions"

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
            'Authorization': 'Bearer ' + os.environ['XMCP_API_KEY_UNLIMITED'],
            'Content-Type': 'application/json'
        }

        while True:
            raw_response = requests.request("POST", url, headers=headers, data=payload)
            yield json.loads(raw_response.text)["choices"][0]["message"]["content"]


class MockModel(LLM):
    def __init__(self, response):
        self.response = response
        self.queries = 0
        self.model_name = "mock-model"
        self.temperature = 1.0

    def start_slicing(self, d: Path):
        pass

    def stop_slicing(self):
        pass

    def get_stateful(self):
        return self

    def sample(self, prompt: str) -> Iterable[str]:
        while True:
            self.queries += 1
            yield self.response


class Human(LLM):
    def __init__(self):
        self.model_name = "human"
        self.temperature = 36.6

    def start_slicing(self, d: Path):
        pass

    def stop_slicing(self):
        pass
    
    def get_stateful(self):
        return self

    def sample(self, prompt: str) -> Iterable[str]:
        while True:
            response = input(f"PROMPT: {prompt}\n")
            yield response


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
