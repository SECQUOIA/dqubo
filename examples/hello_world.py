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

from dqubo import DQUBO
import numpy as np

model = DQUBO()

print("Running example")

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

# add linear inequality constraints 0=<x_1+3*x_2 =< 15 ; 2<= 4*x_0+5*x_1 =< 25
list_cons = [["x_1", "x_2"], ["x_0", "x_1"]]
list_coefs = [[1, 3], [4, 5]]
lhs = [0, 2]
rhs = [15, 25]
model.add_linear_ineq_cons(
    list_cons=list_cons, list_coefs=list_coefs, penalty=1.0, lhs=lhs, rhs=rhs
)

# add linear inequality constraints with only lhs 5 =< 2x_0+x_2  ; 2<= 3*x_2+7*x_1
list_cons = [["x_0", "x_2"], ["x_2", "x_1"]]
list_coefs = [[2, 1], [3, 7]]
lhs = [2, 5]
model.add_linear_ineq_cons(
    list_cons=list_cons, list_coefs=list_coefs, penalty=1.0, lhs=lhs
)

# add objective 2*x_0*x_0" +3*x_0*x_2 + 8*x_2*x_1
list_q_pairs = [["x_1", "x_1"], ["x_1", "x_2"]]
list_q_coefs = [5, 10]
model.add_quadratic_cons(list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=1.0)
model.compile()
qubo, offset = model.get_dict()
print(qubo)
qubo2, offset = model.get_dict_with_labels()
print(qubo2)
