from neurallambda.util import transform_runs

def test_transform_runs():
    assert transform_runs([1, 1, 2, 3, 3, 3], lambda x, y: x == y, lambda run: sum(run) if len(run) > 1 else run[0]) == [2, 2, 9]
    assert transform_runs([1, 2, 3, 4], lambda x, y: x == y, lambda run: sum(run) if len(run) > 1 else run[0]) == [1, 2, 3, 4]
    assert transform_runs([5], lambda x, y: x == y, lambda run: sum(run) if len(run) > 1 else run[0]) == [5]
    assert transform_runs([], lambda x, y: x == y, lambda run: sum(run) if len(run) > 1 else run[0]) == []
    assert transform_runs([7, 7, 7, 7], lambda x, y: x == y, lambda run: sum(run) if len(run) > 1 else run[0]) == [28]
    assert transform_runs(["a", "A", "a", "b"], lambda x, y: x.lower() == y.lower(), lambda run: "".join(run)) == ["aAa", "b"]
