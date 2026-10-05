"""Exercise optimizer dispatch without importing Linux-only mjbatch."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run-so101-sysid.py"
TREE = ast.parse(SCRIPT.read_text(encoding="utf-8"))
FIT = next(node for node in TREE.body
           if isinstance(node, ast.FunctionDef) and node.name == "fit_parameters")


class Params:
    def __init__(self):
        self.vector = np.array([1.0])

    def as_vector(self):
        return self.vector.copy()

    def update_from_vector(self, vector):
        self.vector = np.asarray(vector, dtype=float).copy()


def fit_with(regression, cem, params, *, two_stage):
    calls = []

    def optimize(current, residual_fn, *, optimizer, max_iters):
        calls.append((optimizer, max_iters, current.as_vector().copy()))
        return current.as_vector().copy(), "result"

    namespace = {"ParameterDict": Params, "np": np,
                 "run_regression": regression, "cem_optimize": cem,
                 "optimize": optimize}
    exec(compile(ast.Module(body=[FIT], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    args = SimpleNamespace(two_stage=two_stage, reg_samples=2, reg_iters=3,
                           cem_samples=4, cem_elite=2, cem_iters=5,
                           optimizer="scipy", max_iters=6)
    result = namespace["fit_parameters"](params, None, args)
    return result, calls


class SysidFitTests(unittest.TestCase):
    def test_default_runs_optimizer(self):
        params = Params()
        result, calls = fit_with(lambda *_: self.fail("regression ran"),
                                 lambda *_: self.fail("CEM ran"), params,
                                 two_stage=False)
        self.assertEqual(result[1], "result")
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][2], [1.0])

    def test_cem_starts_from_best_regression_result(self):
        params = Params()

        def regression(current, *_):
            current.update_from_vector([99.0])
            return [(1.0, np.array([2.0])), (2.0, np.array([3.0]))]

        def cem(current, *_):
            np.testing.assert_array_equal(current.as_vector(), [2.0])
            return np.array([4.0]), 0.5

        _, calls = fit_with(regression, cem, params, two_stage=True)
        np.testing.assert_array_equal(calls[0][2], [4.0])

    def test_failed_regression_restores_initial_seed(self):
        params = Params()

        def regression(current, *_):
            current.update_from_vector([99.0])
            return []

        _, calls = fit_with(regression, lambda *_: self.fail("CEM ran"),
                            params, two_stage=True)
        np.testing.assert_array_equal(calls[0][2], [1.0])


if __name__ == "__main__":
    unittest.main()
