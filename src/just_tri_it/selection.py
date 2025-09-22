import os
import hashlib
import sys
import threading
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, Tuple, List
from abc import ABC, abstractmethod
from collections import defaultdict

from just_tri_it.cached_llm import Model
from just_tri_it.executor import Error
from just_tri_it.utils import extract_code, print_annotated_hr, RawData, ContentAddressable
from just_tri_it.program import Signature, Program, Requirements
from just_tri_it.executor import Executor


# for each program list all its plausibility witnesses
type AgreementOutcome = List[Program, List[ContentAddressable]]


class Agreement(ABC):

    @abstractmethod
    def compute_witnesses(self, model: Model, req: Requirements) -> Tuple[AgreementOutcome, RawData]:
        pass


@dataclass
class Selected:
    """
    A selected program and witnesses of its plausibility (tests, specs, programs)
    """
    program: Program
    witnesses: List[ContentAddressable]


@dataclass
class Abstained:
    pass


type SelectionOutcome = Selected | Abstained


class Selector(ABC):

    @abstractmethod
    def generate_and_select(self, model: Model, req: Requirements) -> Tuple[SelectionOutcome, RawData]:
        pass


class MaxWitness(ABC):

    def __init__(self, agreement):
        self.agreement = agreement

    def generate_and_select(self, model: Model, req: Requirements) -> Tuple[SelectionOutcome, RawData]:
        """Schema:
        {
           "method": "max_witness",
           "max_witness_sets": [ (program, witnesses), ...],
           "argeement": ...
           "agreement_raw_data": ...
        }

        Note: "witness set" refers to a set of independent samples, not their values. Values may repeat.
        """
        agreement, agreement_raw_data = self.agreement.compute_witnesses(model, req)

        raw_data = {
           "method": "max_witness",
           "max_witness_sets": [],
           "agreement": agreement,
           "agreement_raw_data": agreement_raw_data
        }

        if len(agreement) == 0:
            return (Abstained(), raw_data)
        
        max_size = max(len(witnesses) for _, witnesses in agreement)
        
        max_witness_sets = [
            (program, witnesses)
            for program, witnesses in agreement
            if len(witnesses) == max_size
        ]

        raw_data["max_witness_sets"] = max_witness_sets

        return (Selected(*max_witness_sets[0]), raw_data)


class Ransac(ABC):

    def __init__(self, agreement):
        self.agreement = agreement

    def generate_and_select(self, model: Model, req: Requirements) -> Tuple[SelectionOutcome, RawData]:
        """Schema:
        {
           "method": "ransac",
           "ransac_scores": [ (score, program, witnesses), ...],
           "argeement": ...        
           "agreement_raw_data": ...
        }
        """
        agreement, agreement_raw_data = self.agreement.compute_witnesses(model, req)

        raw_data = {
           "method": "ransac",
           "ransac_scores": [],
           "agreement": agreement,
           "agreement_raw_data": agreement_raw_data
        }

        if len(agreement) == 0:
            return (Abstained(), raw_data)

        #NOTE: sorted hashlist is important, because we want identical samples to contribute to the score
        programs_to_witness_hashlists = [(p, tuple(sorted([w.hash_id() for w in ws]))) for (p, ws) in agreement]

        groups = defaultdict(list)

        for program, witness_hashlist in programs_to_witness_hashlists:
            groups[witness_hashlist].append(program)

        ransac_scores = []

        for witness_hashlist, programs in groups.items():
            score = len(witness_hashlist) * len(programs)
            for p in programs:
                for q, ws in agreement:
                    if p.hash_id() == q.hash_id():
                        ransac_scores.append((score, p, ws))

        raw_data["ransac_scores"] = ransac_scores

        _, program, witnesses = max(ransac_scores, key=lambda x: x[0])

        return (Selected(program, witnesses), raw_data)
