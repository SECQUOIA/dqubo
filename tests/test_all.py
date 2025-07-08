# Copyright 2025 Dell Inc. or its subsidiaries. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from copy import deepcopy
from pyqubo import Array, LogEncInteger, Binary
from dqubo.encoder_decoder import My_RuntimeEncoder, My_RuntimeDecoder
import json
from time import time
import numpy as np
from dqubo.utils import assert_qubo_equal
from dqubo.apis.neal import Neal
from dqubo.apis.base import AnnealerSolver
from dqubo.apis.openjij import OpenJij
from dqubo import DQUBO

# To run pytest test.py
from random import random, randrange, randint
from itertools import combinations
import dqubo.dqubo
from numpy import float64, int64
from typing import Dict, List, Optional, Union


def test_non_str_vars() -> None:
    # declare the following problem
    model_dqubo = DQUBO()
    with pytest.raises(AssertionError):
        model_dqubo.add_variables(list_var=[float(i) + 0.01 for i in range(3)])

    assert len(model_dqubo.list_var) == 0, "List var size must be 0"

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    assert len(model_dqubo.list_var) == 3, "List var size must be 3"


@pytest.mark.parametrize("sparse", [False, True])
def test_cons_with_mismatches(sparse: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add linear inequality constraints 0 <= x_1+3*x_2 <= 15 ; 2 <= 4*x_0+5*x_1 <= 25
    list_cons = [["x[1]", "x[2]"], ["x[0]", "x[1]"]]
    list_coefs = [[1], [4, 5]]
    lhs = [0, 2]
    rhs = [15, 25]

    with pytest.raises(AssertionError):
        model_dqubo.add_linear_ineq_cons(
            list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
        )

    # add objective (quadratic constraint) 5*x_1*x_5" 7*x_0*x_1
    list_q_pairs = [["x[1]", "x[1]"], ["x[0]", "x[1]"]]
    list_q_coefs = [5]
    with pytest.raises(AssertionError):
        model_dqubo.add_quadratic_cons(
            list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=51.0
        )

    # add objective (quadratic constraint) 5*x_1*x_1*x_1" 7*x_0*x_1
    list_q_pairs = [["x[1]", "x[1]", "x[1]"], ["x[0]", "x[1]"]]
    list_q_coefs = [5, 6]
    with pytest.raises(AssertionError):
        model_dqubo.add_quadratic_cons(
            list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=51.0
        )


@pytest.mark.parametrize("sparse", [False, True])
def test_cons_with_wrong_types(sparse: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add linear inequality constraints 0 <= x_1+3*x_2 <= 15 ; 2 <= 4*x_0+5*0.01 <= 25
    list_cons = [[1, "x[2]"], ["x[0]", 0.01]]
    list_coefs = [[1, 2], [4, 5]]
    lhs = [0, 2]
    rhs = [15, 25]

    with pytest.raises(TypeError):
        model_dqubo.add_linear_ineq_cons(
            list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
        )

    # add linear inequality constraints 0 <= x_1+2*x_2 <= 15 ; 2 <= 4*x_0+5*x_1 <= 25
    list_cons = [["x[1]", "x[2]"], ["x[0]", "x[1]"]]
    list_coefs = [["1", 2], ["4", 5]]
    lhs = [0, 2]
    rhs = [15, 25]

    with pytest.raises(TypeError):
        model_dqubo.add_linear_ineq_cons(
            list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
        )

    list_cons = [["x[1]", "x[2]"], ["x[0]", "x[1]"]]
    list_coefs = [[1, 2], [4, 5]]
    lhs = ["0", 2]
    rhs = [15, "25"]

    with pytest.raises(TypeError):
        model_dqubo.add_linear_ineq_cons(
            list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
        )


@pytest.mark.parametrize("sparse", [False, True])
def test_cons_with_wrong_vars(sparse: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add linear inequality constraints 0 <= x_1+3*x_2 <= 15 ; 2 <= 4*x_0+5*x_1 <= 25
    list_cons = [["x[1]", "x[2]"], ["x[0]", "x[1]"]]
    list_coefs = [[1, 3], [4, 5]]
    lhs = [0, 2]
    rhs = [15, 25]
    model_dqubo.add_linear_ineq_cons(
        list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
    )

    # add linear inequality constraint 2 <= 4*x_0+5*x_5 <= 2
    list_cons = [["x[0]", "x[5]"]]
    list_coefs = [[4, 5]]
    lhs = [2]
    rhs = [2]

    with pytest.raises(KeyError):
        model_dqubo.add_linear_ineq_cons(
            list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
        )

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x[0]", "x[1]", "x[5]"]]
    list_coefs = [[1, 4, 5]]
    rhs = [10]
    with pytest.raises(KeyError):
        model_dqubo.add_linear_eq_cons(
            list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=19.0
        )

    # add objective (quadratic constraint) 5*x_1*x_5" 7*x_0*x_1
    list_q_pairs = [["x[1]", "x[5]"], ["x[0]", "x[1]"]]
    list_q_coefs = [5, 7]
    with pytest.raises(KeyError):
        model_dqubo.add_quadratic_cons(
            list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=51.0
        )


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_empty_vars(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[])

    assert len(model_dqubo.list_var) == 0, "List var must be 0"

    model_dqubo.compile()
    with pytest.raises(AssertionError):
        qubo_dqubo, offset_dqubo = model_dqubo.get_qubo()


def test_add_qubo_obj_lagrangian() -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO()

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo.add_qubo_obj(
        list_pairs=list_q_pairs, list_coefs=list_q_coefs, lagragian=1.0
    )

    # add objective  2*(7*x_0*x_0" +6*x_0*x_2 + 5*x_2*x_1)
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [7, 6, 5]
    model_dqubo.add_qubo_obj(
        list_pairs=list_q_pairs, list_coefs=list_q_coefs, lagragian=2.0
    )

    assert (
        model_dqubo.Ho[(0, 0)] == 16.0
        and model_dqubo.Ho[(0, 2)] == 15.0
        and model_dqubo.Ho[(2, 1)] == 18.0
    ), "Objective variables are not attached with the right coefficients (after lagragian)"


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_empty_cons(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_qubo_obj(list_pairs=[], list_coefs=[])

    with pytest.raises(AssertionError):
        model_dqubo.add_linear_ineq_cons(
            list_cons=[], list_coefs=[], penalty=1.0, lhs=[], rhs=[]
        )
    with pytest.raises(AssertionError):
        model_dqubo.add_linear_eq_cons(list_cons=[], list_coefs=[], penalty=1.0, rhs=[])

    with pytest.raises(AssertionError):
        model_dqubo.add_quadratic_cons(list_pairs=[], list_coefs=[], penalty=1.0)

    model_dqubo.compile()

    with pytest.raises(AssertionError):
        qubo_dqubo, offset_dqubo = model_dqubo.get_qubo()


@pytest.mark.parametrize("sparse", [False, True])
def test_log_enc(sparse: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse)
    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add linear inequality constraints 0 <= x_1+3*x_2 <= 15 ; 2 <= 4*x_0+5*x_1 <= 25
    list_cons = [["x[1]", "x[2]"], ["x[0]", "x[1]"]]
    list_coefs = [[1, 3], [4, 5]]
    lhs = [0, 2]
    rhs = [15, 25]
    model_dqubo.add_linear_ineq_cons(
        list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
    )

    slack_0 = LogEncInteger("slack_0", (lhs[0], rhs[0]))

    slack_0_n = sum([1 for key in model_dqubo.list_var if "slack_0" in key])

    assert slack_0_n == len(slack_0.array.bit_list), "Slack_0 sizes do not match"

    slack_1 = LogEncInteger("slack_1", (lhs[1], rhs[1]))

    slack_1_n = sum([1 for key in model_dqubo.list_var if "slack_1" in key])

    assert slack_1_n == len(slack_1.array.bit_list), "Slack_1 sizes do not match"


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_inequality_cons(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add linear inequality constraints 0 <= x_1+3*x_2 <= 15 ; 2 <= 4*x_0+5*x_1 <= 25
    list_cons = [["x[1]", "x[2]"], ["x[0]", "x[1]"]]
    list_coefs = [[1, 3], [4, 5]]
    lhs = [0, 2]
    rhs = [15, 25]
    model_dqubo.add_linear_ineq_cons(
        list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, lhs=lhs, rhs=rhs
    )

    # add linear inequality constraints x_0+3*x_2 <= 8
    list_cons = [["x[0]", "x[1]"]]
    list_coefs = [[1, 3]]

    rhs = [8]
    model_dqubo.add_linear_ineq_cons(
        list_cons=list_cons, list_coefs=list_coefs, penalty=10.0, rhs=rhs
    )

    # add linear inequality constraints 2*x_1+3*x_2
    list_cons = [["x[1]", "x[2]"]]
    list_coefs = [[2, 3]]
    with pytest.raises(AssertionError):
        model_dqubo.add_linear_ineq_cons(
            list_cons=list_cons, list_coefs=list_coefs, penalty=10.0
        )

    model_dqubo_start = time()
    model_dqubo.compile()
    model_dqubo_end = time()
    print("dqubo compile: ", model_dqubo_end - model_dqubo_start)

    qubo_dqubo, offset_dqubo = model_dqubo.get_qubo()


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_objective(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    model_dqubo_start = time()
    model_dqubo.compile()
    model_dqubo_end = time()
    print("dqubo compile: ", model_dqubo_end - model_dqubo_start)

    # pyqubo
    x = Array.create("x", shape=(3,), vartype="BINARY")

    H = 2 * x[0] ** 2 + 3 * x[0] * x[2] + 8 * x[2] * x[1]

    model_pyqubo_start = time()
    model_pyqubo = H.compile()
    model_pyqubo_end = time()
    print("pyqubo compile: ", model_pyqubo_end - model_pyqubo_start)

    qubo_pyqubo, offset_pyqubo = model_pyqubo.to_qubo()

    qubo_dqubo, offset_dqubo = model_dqubo.get_qubo("dict_labelled")

    assert_qubo_equal(qubo_pyqubo, qubo_dqubo), "different dictionaries"
    assert offset_pyqubo == offset_dqubo, "different offsets"


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_changing_penalty(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    # model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)
    model_dqubo = DQUBO()

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x[0]", "x[1]", "x[2]"], ["x[0]", "x[2]"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 20]
    model_dqubo.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=19.0
    )
    # add objective (quadratic constraint) 5*x_1*x_1" 7*x_0*x_1
    list_q_pairs = [["x[1]", "x[1]"], ["x[0]", "x[1]"]]
    list_q_coefs = [5, 7]
    model_dqubo.add_quadratic_cons(
        list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=51.0
    )

    model_dqubo.compile()

    model_dqubo.compile({0: 82.0, 1: 27.0})
    _, _, penalty = model_dqubo.cons[0]
    _, penalty2 = model_dqubo.cons[1]
    assert penalty == 82.0 and penalty2 == 27.0, "Penalties did not change"


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_equality_cons(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x[0]", "x[1]", "x[2]"], ["x[0]", "x[2]"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 20]
    model_dqubo.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0
    )
    model_dqubo_start = time()
    model_dqubo.compile()
    model_dqubo_end = time()
    print("dqubo compile: ", model_dqubo_end - model_dqubo_start)

    # pyqubo
    x = Array.create("x", shape=(3,), vartype="BINARY")

    H = 10.0 * (
        (x[0] + 4 * x[1] + 5 * x[2] - 10) ** 2 + (6 * x[0] + 9 * x[2] - 20) ** 2
    )

    model_pyqubo_start = time()
    model_pyqubo = H.compile()
    model_pyqubo_end = time()
    print("pyqubo compile: ", model_pyqubo_end - model_pyqubo_start)

    qubo_pyqubo, offset_pyqubo = model_pyqubo.to_qubo()

    qubo_dqubo, offset_dqubo = model_dqubo.get_qubo("dict_labelled")

    assert_qubo_equal(qubo_pyqubo, qubo_dqubo), "different dictionaries"
    assert offset_pyqubo == offset_dqubo, "different offsets"


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_check_energy(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x[0]", "x[1]", "x[2]"], ["x[0]", "x[2]"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 20]
    model_dqubo.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0, name="C1"
    )

    model_dqubo.compile()

    result_total = {"x[0]": 1, "x[1]": 1, "x[2]": 1}
    result_dqubo = {"x[0]": 1, "x[2]": 1}

    fixed_vars = {"x[1]": 1}

    model_dqubo.fix_vars(fixed_vars)

    # pyqubo
    x = Array.create("x", shape=(3,), vartype="BINARY")

    Ho = 2 * x[0] ** 2 + 3 * x[0] * x[2] + 8 * x[2] * x[1]
    Hc = 10.0 * (
        (x[0] + 4 * x[1] + 5 * x[2] - 10) ** 2 + (6 * x[0] + 9 * x[2] - 20) ** 2
    )
    H = Ho + Hc
    model_pyqubo = H.compile()

    energy_pyqubo = model_pyqubo.energy(result_total, vartype="BINARY")

    energy_dqubo = model_dqubo.energy(result_dqubo)

    assert energy_pyqubo == energy_dqubo, "energies does not match"
    return


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_check_objective(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x[0]", "x[1]", "x[2]"], ["x[0]", "x[2]"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 20]
    model_dqubo.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0, name="C1"
    )

    model_dqubo.compile()

    result_total = {"x[0]": 1, "x[1]": 1, "x[2]": 1}
    result_dqubo = {"x[0]": 1, "x[2]": 1}

    fixed_vars = {"x[1]": 1}

    model_dqubo.fix_vars(fixed_vars)

    # pyqubo
    x = Array.create("x", shape=(3,), vartype="BINARY")

    Ho = 2 * x[0] ** 2 + 3 * x[0] * x[2] + 8 * x[2] * x[1]

    H = Ho
    model_pyqubo = H.compile()

    energy_pyqubo = model_pyqubo.energy(result_total, vartype="BINARY")

    objective_dqubo = model_dqubo.objective(result_dqubo)

    assert energy_pyqubo == objective_dqubo, "energies does not match"
    return


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_check_cons(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 15
    list_cons = [["x[0]", "x[1]", "x[2]"], ["x[0]", "x[2]"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 15]
    model_dqubo.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0, name="C1"
    )

    list_q_pairs = [["x[1]", "x[1]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo.add_quadratic_cons(
        list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=9.0
    )

    model_dqubo.compile()

    result = {"x[0]": 1, "x[1]": 1, "x[2]": 1}
    checked_cons = model_dqubo.check_constraints(result, print_results=True)

    assert checked_cons["C1_0"][0] and checked_cons["C1_1"][0]

    result = {"x[0]": 1, "x[1]": 1, "x[2]": 0}
    checked_cons = model_dqubo.check_constraints(result, print_results=True)

    assert checked_cons["C1_0"][0] == False and checked_cons["C1_1"][0] == False


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_encode_decode(sparse: bool, enable_async: bool) -> None:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    model_dqubo.add_variables(list_var=[f"x[{i}]" for i in range(3)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x[0]", "x[1]", "x[2]"], ["x[0]", "x[2]"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 20]
    model_dqubo.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0, name="C1"
    )

    json_object = json.dumps(model_dqubo, ensure_ascii=False, cls=My_RuntimeEncoder)
    model_dqubo_loaded = json.loads(json_object, cls=My_RuntimeDecoder)

    model_dqubo.compile()
    model_dqubo_loaded.compile()

    qubo_dqubo_loaded, offset_dqubo_loaded = model_dqubo_loaded.get_qubo(
        "dict_labelled"
    )
    qubo_dqubo, offset_dqubo = model_dqubo.get_qubo("dict_labelled")

    assert_qubo_equal(qubo_dqubo_loaded, qubo_dqubo), "different dictionaries"
    assert offset_dqubo_loaded == offset_dqubo, "different offsets"
    return


def test_to_dense_to_sparse() -> None:
    model_dqubo_1 = DQUBO(sparse=True)
    model_dqubo_1.add_variables(list_var=[f"x[{i}]" for i in range(10)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[0]", "x[0]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo_1.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x[0]", "x[1]", "x[2]"], ["x[0]", "x[2]"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 20]
    model_dqubo_1.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0, name="C1"
    )

    # add objective (quadratic constraint) 2*x_1*x_1" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x[1]", "x[1]"], ["x[0]", "x[2]"], ["x[2]", "x[1]"]]
    list_q_coefs = [2, 3, 8]
    model_dqubo_1.add_quadratic_cons(
        list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=1.0
    )

    model_dqubo_2 = deepcopy(model_dqubo_1)

    model_dqubo_1.to_dense()

    model_dqubo_1.to_sparse()

    model_dqubo_1.compile()

    model_dqubo_2.compile()

    qubo_dqubo1, offset_1 = model_dqubo_1.get_qubo("dict_labelled")

    qubo_dqubo2, offset_2 = model_dqubo_2.get_qubo("dict_labelled")

    assert_qubo_equal(qubo_dqubo1, qubo_dqubo2)

    assert offset_1 == offset_2, "Offsets do not match"


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_solvers(sparse: bool, enable_async: bool) -> None:
    model = create_qubo(sparse=sparse, enable_async=enable_async)

    fixed_vars = {"x[1]": 1, "x[3]": 0}

    model.fix_vars(fixed_vars)

    model.get_qubo("dict_labelled")

    neal_solution = model.solve(
        Neal(), num_reads=1, num_sweeps=10, beta_range=[10, 200]
    )

    check_solution(neal_solution)

    oj_solution = model.solve(OpenJij(), num_reads=1, num_sweeps=10)

    check_solution(oj_solution)

    return

def test_matrix_qubo():
    sparse = True
    enable_async = True
    model = DQUBO(sparse=sparse, enable_async=enable_async)
    model = populate_model(model)

    qubo_sparse, offset_sparse = model.get_qubo(input_format="matrix")

    sparse = False
    enable_async = True
    model = DQUBO(sparse=sparse, enable_async=enable_async)
    model = populate_model(model)

    qubo_dense, offset_dense = model.get_qubo(input_format="matrix")

    sparse = True
    enable_async = False
    model = DQUBO(sparse=sparse, enable_async=enable_async)
    model = populate_model(model)

    qubo_sparse_async, offset_sparse_async = model.get_qubo(input_format="matrix")

    sparse = False
    enable_async = False
    model = DQUBO(sparse=sparse, enable_async=enable_async)
    model = populate_model(model)

    qubo_dense_async, offset_dense_async = model.get_qubo(input_format="matrix")

    assert (
        offset_dense == offset_dense_async
    ), f"offsets do not match ( {offset_sparse},{offset_dense} )"

    assert (
        offset_sparse == offset_sparse_async
    ), f"offsets do not match ( {offset_sparse_async},{offset_sparse} )"

    assert (
        offset_dense == offset_sparse
    ), f"offsets do not match ( {offset_dense_async},{offset_sparse} )"

    assert np.array_equal(
        qubo_dense, qubo_dense_async
    ), "QUBO in numpy does not match - dense"
    assert np.array_equal(
        qubo_sparse.todense(), qubo_sparse_async.todense()
    ), "QUBO in numpy does not match - sparse"

    assert np.array_equal(
        qubo_dense, qubo_sparse.todense()
    ), "QUBO in numpy does not match - dense and sparse"

    return


def test_sparse_dense_dqubo() -> None:
    model_sparse = DQUBO()
    model_sparse = populate_model(model_sparse)

    qubo_sparse, offset_sparse = model_sparse.get_qubo("dict_labelled")

    model_dense = DQUBO(sparse=False)
    model_dense = populate_model(model_dense)

    qubo_dense, offset_dense = model_dense.get_qubo("dict_labelled")

    model_sparse_asyncFalse = DQUBO(enable_async=False)
    model_sparse_asyncFalse = populate_model(model_sparse_asyncFalse)

    (
        qubo_sparse_asyncFalse,
        offset_sparse_asyncFalse,
    ) = model_sparse_asyncFalse.get_qubo("dict_labelled")

    model_dense_asyncFalse = DQUBO(sparse=False, enable_async=False)
    model_dense_asyncFalse = populate_model(model_dense_asyncFalse)

    qubo_dense_asyncFalse, offset_dense_asyncFalse = model_dense_asyncFalse.get_qubo(
        "dict_labelled"
    )

    assert (
        offset_dense == offset_dense_asyncFalse
    ), f"offsets do not match ( {offset_sparse},{offset_dense_asyncFalse} )"

    assert (
        offset_sparse == offset_sparse_asyncFalse
    ), f"offsets do not match ( {offset_sparse_asyncFalse},{offset_sparse} )"

    assert (
        offset_dense == offset_sparse
    ), f"offsets do not match ( {offset_dense},{offset_sparse} )"

    assert_qubo_equal(qubo_dense_asyncFalse, qubo_dense)
    assert_qubo_equal(qubo_sparse_asyncFalse, qubo_sparse)
    assert_qubo_equal(qubo_sparse, qubo_dense)


@pytest.mark.parametrize(
    "sparse,enable_async", [(True, True), (True, False), (False, True), (False, False)]
)
def test_var_cons_compile_iterative(sparse: bool, enable_async: bool) -> None:
    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    for it in range(5):
        model_dqubo.add_variables(list_var=[f"x[{it}][{i}]" for i in range(3)])

        # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
        list_q_pairs = [
            [f"x[{it}][0]", f"x[{it}][0]"],
            [f"x[{it}][0]", f"x[{it}][2]"],
            [f"x[{it}][2]", f"x[{it}][1]"],
        ]
        list_q_coefs = [2, 3, 8]
        model_dqubo.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

        # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 15
        list_cons = [
            [f"x[{it}][0]", f"x[{it}][1]", f"x[{it}][2]"],
            [f"x[{it}][0]", f"x[{it}][2]"],
        ]
        list_coefs = [[1, 4, 5], [6, 9]]
        rhs = [it - 10, it + 15]
        model_dqubo.add_linear_eq_cons(
            list_cons=list_cons,
            list_coefs=list_coefs,
            rhs=rhs,
            penalty=10.0,
            name=f"C{it}",
        )

        list_q_pairs = [
            [f"x[{it}][1]", f"x[{it}][1]"],
            [f"x[{it}][0]", f"x[{it}][2]"],
            [f"x[{it}][2]", f"x[{it}][1]"],
        ]
        list_q_coefs = [it + 2, it + 3, it + 8]
        model_dqubo.add_quadratic_cons(
            list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=1.0, name="QC{it}"
        )

        model_dqubo.compile()

    qubo_dqubo, offset_dqubo = model_dqubo.get_qubo("dict_labelled")

    model_dqubo = DQUBO(sparse=sparse, enable_async=enable_async)

    for it in range(5):
        model_dqubo.add_variables(list_var=[f"x[{it}][{i}]" for i in range(3)])

        # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
        list_q_pairs = [
            [f"x[{it}][0]", f"x[{it}][0]"],
            [f"x[{it}][0]", f"x[{it}][2]"],
            [f"x[{it}][2]", f"x[{it}][1]"],
        ]
        list_q_coefs = [2, 3, 8]
        model_dqubo.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

        # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 15
        list_cons = [
            [f"x[{it}][0]", f"x[{it}][1]", f"x[{it}][2]"],
            [f"x[{it}][0]", f"x[{it}][2]"],
        ]
        list_coefs = [[1, 4, 5], [6, 9]]
        rhs = [it - 10, it + 15]
        model_dqubo.add_linear_eq_cons(
            list_cons=list_cons,
            list_coefs=list_coefs,
            rhs=rhs,
            penalty=10.0,
            name=f"C{it}",
        )
        list_q_pairs = [
            [f"x[{it}][1]", f"x[{it}][1]"],
            [f"x[{it}][0]", f"x[{it}][2]"],
            [f"x[{it}][2]", f"x[{it}][1]"],
        ]
        list_q_coefs = [it + 2, it + 3, it + 8]
        model_dqubo.add_quadratic_cons(
            list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=1.0, name="QC{it}"
        )

    model_dqubo.compile()

    qubo_dqubo_2, offset_dqubo_2 = model_dqubo.get_qubo("dict_labelled")

    assert_qubo_equal(qubo_dqubo, qubo_dqubo_2)
    assert offset_dqubo == offset_dqubo_2, "offsets do not match"


@pytest.mark.parametrize("sparse", [True, False])
def test_fix_vars(sparse: bool) -> None:
    model = DQUBO(sparse=sparse)
    model.add_variables(list_var=[f"x[{i}]" for i in range(4)])

    H = np.array([[2, 0, 0, 0], [5, 8, 0, 0], [7, 8, 3, 0], [9, 4, 2, 3]], np.float32)
    if sparse:
        from scipy.sparse import csr_matrix

        model.H = csr_matrix(H)
    else:
        model.H = H

    model.compiled = True

    fixed_vars = {"x[1]": 1, "x[3]": 0}
    model.fix_vars(fixed_vars)

    qubo, offset = model.get_qubo()

    true_qubo = {(0, 0): 2 + 5, (2, 2): 3 + 8, (2, 0): 7}
    true_offset = 8

    assert_qubo_equal(qubo, true_qubo)
    assert offset == true_offset, "offsets do not match"
    return


@pytest.mark.parametrize("sparse", [True, False])
def test_lpread(sparse):
    model = DQUBO(sparse=sparse)

    model.readlp("tests/example.lp")

    Ho = np.array([[-1, -4, -2], [0, -1, 0], [0, 0, 0]])

    Ho = np.pad(Ho, ((0, 3), (0, 3)), mode="constant", constant_values=0)

    Hom = model.Ho.toarray() if sparse else model.Ho

    assert np.allclose(Ho, Hom), "objective is not equal"

    Ac1 = np.array([1, 1, 0, 0, 0, 0])

    Ac1m = model.cons[("c1",)][0].toarray() if sparse else model.cons[("c1",)][0]

    assert np.allclose(Ac1, Ac1m), "c1 is not equal"

    Ac23 = np.array([[-1, 1, 0, 0, 0, 0], [1, 0, -1, -1, -2, -1]])

    Ac23m = (
        model.cons[("c2", "c3")][0].toarray() if sparse else model.cons[("c2", "c3")][0]
    )

    assert np.allclose(Ac23, Ac23m), "c2, c3 are not equal"

    return


def check_solution(
    solution: Dict[str, List[Dict[str, Optional[Union[float64, int64]]]]]
) -> None:
    assert isinstance(solution, dict), "solution is not a dictionary"
    assert "samples" in solution.keys(), "no samples on dictionary"
    assert "sol" in solution["samples"][0], "no sol in dictionary"
    assert "energy" in solution["samples"][0], "no energy in dictionary"
    assert "repetitions" in solution["samples"][0], "no repetition in dictionary"
    return


def create_qubo(
    QUBITS: int = 50,
    DENSITY: float = 0.5,
    sparse: bool = True,
    enable_async: bool = True,
) -> dqubo.dqubo.DQUBO:
    d_model = DQUBO(sparse, enable_async)

    d_model.add_variables([f"x[{i}]" for i in range(QUBITS)])

    # populate objective function
    Ho = 0
    list_pairs = list()
    list_coefs = list()
    for i, j in combinations(range(QUBITS), 2):
        # print(f" {a}, {b}")
        if random() < DENSITY:
            coef = randrange(20)
            list_pairs.append((f"x[{i}]", f"x[{j}]"))
            list_coefs.append(coef)

    d_model.add_qubo_obj(list_pairs=list_pairs, list_coefs=list_coefs)

    Nc = randint(1, 2 * QUBITS)

    list_cons = list()
    list_coefs = list()
    rhs = list()
    penalty = 1
    for c in range(Nc):
        con = list()
        coefs = list()
        for i in range(QUBITS):
            # print(f" {a}, {b}")
            if random() < DENSITY / 2:
                coef = randrange(10)

                con.append(f"x[{i}]")
                coefs.append(coef)
        rhsc = randrange(20)
        penalty = randrange(100)

        list_cons.append(con)
        list_coefs.append(coefs)
        rhs.append(rhsc)

    d_model.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=penalty
    )

    list_cons = list()
    list_coefs = list()
    rhs = list()
    lhs = list()
    for c in range(Nc):
        con = list()
        coefs = list()
        for i in range(QUBITS):
            # print(f" {a}, {b}")
            if random() < DENSITY / 2:
                coef = randrange(10)

                con.append(f"x[{i}]")
                coefs.append(coef)
        lhsc = randrange(20)
        rhsc = lhsc + randrange(20)
        penalty = randrange(100)

        list_cons.append(con)
        list_coefs.append(coefs)
        lhs.append(lhsc)
        rhs.append(rhsc)

    d_model.add_linear_ineq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, lhs=lhs, penalty=penalty
    )

    d_model.compile()

    return d_model


def populate_model(model: dqubo.dqubo.DQUBO) -> dqubo.dqubo.DQUBO:
    # declare the following problem
    # min x^T*Q*x  s.t. Ax = b , Cx < d

    model.add_variables(list_var=[f"x_{i}" for i in range(3)])

    # add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x_0", "x_0"], ["x_0", "x_2"], ["x_2", "x_1"]]
    list_q_coefs = [2, 3, 8]
    model.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    # add linear equality constraints x_0+4*x_1+5*x_2 = 10 ; 6*x_0+9*x_2 = 20
    list_cons = [["x_0", "x_1", "x_2"], ["x_0", "x_2"]]
    list_coefs = [[1, 4, 5], [6, 9]]
    rhs = [10, 20]
    model.add_linear_eq_cons(
        list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0
    )

    # add linear inequality constraints 0 <= x_1+3*x_2 <= 15 ; 2 <= 4*x_0+5*x_1 =< 25
    list_cons = [["x_1", "x_2"], ["x_0", "x_1"]]
    list_coefs = [[1, 3], [4, 5]]
    lhs = [0, 2]
    rhs = [15, 25]
    model.add_linear_ineq_cons(
        list_cons=list_cons, list_coefs=list_coefs, penalty=1.0, lhs=lhs, rhs=rhs
    )

    # add linear inequality constraints with only lhs 2 <= 2x_0+x_2  ; 5 <= 3*x_2+7*x_1
    list_cons = [["x_0", "x_2"], ["x_2", "x_1"]]
    list_coefs = [[2, 1], [3, 7]]
    lhs = [2, 5]
    model.add_linear_ineq_cons(
        list_cons=list_cons, list_coefs=list_coefs, penalty=1.0, lhs=lhs
    )

    # add objective (quadratic constraint) 2*x_1*x_1" +3*x_0*x_2 + 8*x_2*x_1
    list_q_pairs = [["x_1", "x_1"], ["x_0", "x_2"], ["x_2", "x_1"]]
    list_q_coefs = [2, 3, 8]
    model.add_quadratic_cons(
        list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=1.0
    )
    model.compile()

    return model
