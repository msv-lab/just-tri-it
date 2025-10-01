
def type_checker(data, type_str):
    type_str = type_str.lower()
    match type_str:
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
        case _:
            raise ValueError(f"undefined: {type_str}")
