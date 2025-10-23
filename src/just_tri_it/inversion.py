from dataclasses import dataclass

from just_tri_it.program import Requirements, NamedReturnSignature, Signature, Parameter
from just_tri_it.cached_llm import Model, Independent
from just_tri_it.utils import (
    gen_and_extract_answer_with_retry,
    ExperimentFailure,
    extract_answer
)


@dataclass
class ParameterInversion:
    index: int


@dataclass
class ListSuffixInversion:
    index: str
    suffix_length: int


type InversionScheme = ParameterInversion | ListSuffixInversion


def choose_inversion_scheme(model: Model, req: Requirements) -> InversionScheme:
    if len(req.signature.params) == 1:
        return ParameterInversion(0)
    else:
        PROMPT = f"""
The problem below is solved using a function with the signature:

{req.signature.pretty_print()}

Choose a single input parameter of the function to be replaced by
its output, thereby formulating an inverse problem. Determine which
parameter, when inverted, would yield the most natural or well-posed
inverse formulation. Exclude parameters whose values can be readily
deduced from other inputs.

Output only the full name of this parameter (not its type) enclosed
within `<answer>` and `</answer>` tags.

Problem:
{req.description}
        """
        ind_model = Independent(model)
        return_param = None
        tried_samples = []
        for attempt in range(3):
            try:
                sample = next(ind_model.sample(PROMPT, 3))
                tried_samples.append(sample)
                valid_name = extract_answer(sample)
                valid_name = valid_name.strip() if valid_name else None
                if valid_name:
                    return_param = [p.name for p in req.signature.params].index(valid_name)
                    break
                else:
                    continue
            except Exception as e:
                if attempt == 2:
                    raise ExperimentFailure(f"retry failed with {type(e).__name__}: {e}")
        return ParameterInversion(return_param)
