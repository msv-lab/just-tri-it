from dataclasses import dataclass
import copy

from just_tri_it.program import Requirements, NamedReturnSignature, Signature, Parameter, Program
from just_tri_it.cached_llm import Model, Independent
from just_tri_it.logic import Demonic
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


def list_split_signature(index: int, s: Signature):
    new_sig = copy.deepcopy(s)
    new_sig.name = new_sig.name + "_split_list_adapter"
    new_sig.params = new_sig.params[0:index] + \
        [ Parameter(new_sig.params[index].name + "_prefix", new_sig.params[index].type),
          Parameter(new_sig.params[index].name + "_suffix", new_sig.params[index].type) ] + \
        new_sig.params[index+1:]
    return new_sig



def fwd_program_adapter(inversion_scheme):
    match inversion_scheme:
        case ParameterInversion():
            return lambda p: p
        case ListSuffixInversion(index, suffix_length):
            def adapter(p):
                new_sig = list_split_signature(index, p.signature)
                ADAPTER_CODE=f"""
def {new_sig.name}(*args):
    args = list(args)
    new_args = args[:{index}] + [ args[{index}] + args[{index+1}] ] + args[{index+2}:]
    return {p.signature.name}(*new_args)
                """
                return Program(new_sig, p.code + "\n" + ADAPTER_CODE)
            return adapter


def fwd_input_adapter(inversion_scheme):
    match inversion_scheme:
        case ParameterInversion():
            return lambda a: a
        case ListSuffixInversion(index, suffix_length):
            def adapter(args):
                args = copy.deepcopy(args)
                lst = args[index]
                if not isinstance(lst, list):
                    return [Demonic()]
                split_at = max(0, len(lst) - suffix_length)
                prefix = lst[:split_at]
                suffix = lst[split_at:]
                args[index:index+1] = [prefix, suffix]
                return args                
            return adapter


def choose_inversion_scheme(model: Model, req: Requirements) -> InversionScheme:
    if len(req.signature.params) == 1:
        if req.signature.params[0].type.lower().startswith('list'):
            return ListSuffixInversion(0, 1)
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
