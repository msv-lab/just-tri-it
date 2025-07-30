from dataclasses import dataclass
from typing import List

from viberate.program import Test
from viberate.requirements import Requirements


@dataclass
class Task:
    id: str
    requirements: Requirements
    tests: List[Test]
    metadata: dict[str, Any]


type Benchmark = List[Task]
    

