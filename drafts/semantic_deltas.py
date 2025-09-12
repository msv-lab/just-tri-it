from typing import Any, List

class Delta:
    def apply(self, y: Any) -> Any: raise NotImplementedError
    def predicate_tag(self) -> str: raise NotImplementedError
    def __repr__(self): return self.predicate_tag()

# --- if return int/float ---
class IntAdd(Delta):
    def __init__(self, k: int): self.k = k
    def apply(self, y): return y + self.k
    def predicate_tag(self): return f"IntAdd_{self.k}"

class FloatAdd(Delta):
    def __init__(self, k: float): self.k = k
    def apply(self, y): return y + self.k
    def predicate_tag(self): return f"FloatAdd_{self.k}"

# --- return str ---
class StrSuffix(Delta):
    def __init__(self, s: str): self.s = s
    def apply(self, y): return str(y) + self.s
    def predicate_tag(self): return f"StrSuffix_{self.s}"

# --- return list ---
class ListAppendConst(Delta):
    def __init__(self, e: Any): self.e = e
    def apply(self, y): return list(y) + [self.e]
    def predicate_tag(self): return f"ListAppendConst_{repr(self.e)}"

# --- return set ---
class SetUnionConst(Delta):
    def __init__(self, e: Any): self.e = e
    def apply(self, y):
        z = set(y)
        z.add(self.e)
        return z
    def predicate_tag(self): return f"SetUnionConst_{repr(self.e)}"

def delta_candidates(return_type: str) -> List[Delta]:
    t = (return_type or "").lower()
    if t in ("int", "integer"):   return [IntAdd(1)]
    if t in ("float", "double"):  return [FloatAdd(1.0)]
    if t in ("str", "string"):    return [StrSuffix("_x")]
    if t in ("list", "array"):    return [ListAppendConst(0)]
    if t in ("set",):             return [SetUnionConst(0)]
    return [IntAdd(1)] 
