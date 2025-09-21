from just_tri_it.program import Signature
import ast


def test_from_function_ast():
    code = """
def foo(a: int, b: str) -> bool:
    return str(a) == b
    """
    tree = ast.parse(code)
    fn_node = tree.body[0]
    signature = Signature.from_function_ast(fn_node)
    assert signature.pretty_print() == "def foo(a: int, b: str) -> bool"
