from viberate.llm import MockModel, Cached


def test_lazy_sampling(tmp_path):
    m = MockModel("this is a mock response")
    c = Cached(m, tmp_path)
    prompt = "this is a mock prompt"
    s1 = c.sample(prompt)
    s2 = c.sample(prompt)
    assert next(s1) == m.response
    assert next(s2) == m.response
    assert next(s1) == m.response
    assert next(s2) == m.response
    assert next(s1) == m.response
    assert m.queries == 3
