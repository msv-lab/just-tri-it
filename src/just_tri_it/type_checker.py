from typing import List, Any
from just_tri_it.program import Signature


def check_type(data, type_str):
    type_str = type_str.lower()
    match type_str:
        case 'bool':
            return isinstance(data, bool)
        case 'int':
            return isinstance(data, int)
        case 'float':
            return isinstance(data, float)
        case 'str':
            return isinstance(data, str)
        case 'list[int]':
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, int):
                    return False
            return True
        case 'list[str]':
            if not isinstance(data, list):
                return False
            return all(isinstance(item, str) for item in data)
        case 'list[tuple[int, int]]':
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 2:
                    return False
                if not all(isinstance(x, int) for x in item):
                    return False
            return True
        case 'list[tuple[int, int, int]]':
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 3:
                    return False
                if not all(isinstance(x, int) for x in item):
                    return False
            return True
        case 'list[list[int]]':
            if not isinstance(data, list):
                return False
            for inner in data:
                if not isinstance(inner, list):
                    return False
                for item in inner:
                    if not isinstance(item, int):
                        return False
            return True
        case 'list[list[str]]':
            if not isinstance(data, list):
                return False
            for inner in data:
                if not isinstance(inner, list):
                    return False
                for item in inner:
                    if not isinstance(item, str):
                        return False
            return True
        case 'list[tuple[str, int]]':
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 2:
                    return False
                if not isinstance(item[0], str) or not isinstance(item[1], int):
                    return False
            return True
        case 'list[tuple[int, str, str]]':
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 3:
                    return False
                if not isinstance(item[0], int):
                    return False
                if not isinstance(item[1], str):
                    return False
                if not isinstance(item[2], str):
                    return False
            return True
        case 'list[tuple[int, list[int]]]':
            if not isinstance(data, list):
                return False
            for element in data:
                if not isinstance(element, tuple) or len(element) != 2:
                    return False
                first, second = element
                if not isinstance(first, int):
                    return False
                if not isinstance(second, list):
                    return False
                if not all(isinstance(x, int) for x in second):
                    return False
            return True
        case 'list[tuple[int, int, list[tuple[int, int, int]]]]':
            if not isinstance(data, list):
                return False
            for outer_tuple in data:
                if not (isinstance(outer_tuple, tuple) and len(outer_tuple) == 3):
                    return False
                if not (isinstance(outer_tuple[0], int) and isinstance(outer_tuple[1], int)):
                    return False
                inner_list = outer_tuple[2]
                if not isinstance(inner_list, list):
                    return False
                for inner_tuple in inner_list:
                    if not (isinstance(inner_tuple, tuple) and len(inner_tuple) == 3):
                        return False
                    if not all(isinstance(x, int) for x in inner_tuple):
                        return False
            return True
        case 'list[tuple[int, int, int, int, int, int]]':
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 6:
                    return False
                if not all(isinstance(x, int) for x in item):
                    return False
            return True
        case 'list[tuple[int, int, list[str]]]':
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple) or len(item) != 3:
                    return False
                if not isinstance(item[0], int) or not isinstance(item[1], int):
                    return False
                if not isinstance(item[2], list):
                    return False
                for s in item[2]:
                    if not isinstance(s, str):
                        return False
            return True
        case 'union[list[int], int]' | 'union[int, list[int]]':
            if isinstance(data, int):
                return True
            if isinstance(data, list):
                if len(data) == 0:
                    return True
                return all(isinstance(item, int) for item in data)
            return False
        case "tuple[int, list[int]]":
            if not isinstance(data, tuple):
                return False
            if len(data) != 2:
                return False
            if not isinstance(data[0], int):
                return False
            if not isinstance(data[1], list):
                return False
            if len(data[1]) > 0:
                return all(isinstance(item, int) for item in data[1])
            return True
        case "tuple[int, int]":
            if not isinstance(data, tuple):
                return False
            if len(data) != 2:
                return False
            return all(isinstance(item, int) for item in data)
        case "list[tuple[int, list[int], list[int], list[int]]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 4:
                    return False
                if not isinstance(item[0], int):
                    return False
                for i in range(1, 4):
                    if not isinstance(item[i], list):
                        return False
                    if not all(isinstance(x, int) for x in item[i]):
                        return False
            return True
        case "list[tuple[int, list[tuple[int, int]]]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 2:
                    return False
                if not isinstance(item[0], int):
                    return False
                if not isinstance(item[1], list):
                    return False
                for sub_item in item[1]:
                    if not isinstance(sub_item, tuple):
                        return False
                    if len(sub_item) != 2:
                        return False
                    if not all(isinstance(x, int) for x in sub_item):
                        return False
            return True
        case "list[tuple[int, int, list[tuple[int, int]]]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 3:
                    return False
                if not all(isinstance(x, int) for x in item[:2]):
                    return False
                if not isinstance(item[2], list):
                    return False
                for sub_item in item[2]:
                    if not isinstance(sub_item, tuple):
                        return False
                    if len(sub_item) != 2:
                        return False
                    if not all(isinstance(x, int) for x in sub_item):
                        return False
            return True
        case "list[tuple[int, list[int]]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 2:
                    return False
                if not isinstance(item[0], int):
                    return False
                if not isinstance(item[1], list):
                    return False
                if not all(isinstance(x, int) for x in item[1]):
                    return False
            return True
        case "list[tuple[int, int, str]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 3:
                    return False
                if not isinstance(item[0], int):
                    return False
                if not isinstance(item[1], int):
                    return False
                if not isinstance(item[2], str):
                    return False
            return True
        case "list[tuple[int, int, int, int, str]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 5:
                    return False
                if not all(isinstance(x, int) for x in item[:4]):
                    return False
                if not isinstance(item[4], str):
                    return False
            return True
        case "list[tuple[int, str]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 2:
                    return False
                if not isinstance(item[0], int):
                    return False
                if not isinstance(item[1], str):
                    return False
            return True
        case "list[tuple[str, str]]":
            if not isinstance(data, list):
                return False
            for item in data:
                if not isinstance(item, tuple):
                    return False
                if len(item) != 2:
                    return False
                if not isinstance(item[0], str):
                    return False
                if not isinstance(item[1], str):
                    return False
            return True
        case 'tuple[int, list[tuple[int, int]]]':
            if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], int) and isinstance(data[1], list):
                for item in data[1]:
                    if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], int)):
                        return False
                return True
            else:
                return False
        case _:
            raise ValueError(f"unsupported type: {type_str}")


def args_match_signature(args: List[Any], sig: Signature):
    for index in range(len(sig.params)):
        if not check_type(args[index], sig.params[index].type):
            return False
    return True
