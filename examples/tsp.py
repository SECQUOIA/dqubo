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

from dqubo.apis.openjij import OpenJij
from dqubo.apis.neal import Neal


def create_qubo(point_num, dlength, d_min, max_len):
    model = DQUBO()
    # Hamiltonian of the distance between two points

    # Spin definition

    model.add_variables(
        list_var=[f"x[{i}][{j}]" for i in range(point_num) for j in range(point_num)]
    )

    # Building "Hd"
    list_q_pairs = []
    list_q_coefs = []
    for j in range(point_num):
        for k in range(point_num):
            if k != j:
                for i in range(point_num):
                    list_q_pairs.append([f"x[{i}][{j}]", f"x[{(i+1)%point_num}][{k}]"])
                    list_q_coefs.append(float(dlength[j][k] - d_min[j]))

    model.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

    penalty = max_len * 0.15
    # add linear equality constraints
    # Building "Hb"
    list_cons_hb = [
        [f"x[{i}][{j}]" for j in range(point_num)] for i in range(point_num)
    ]
    list_coefs_hb = [[1.0] * point_num] * point_num
    rhs_hb = [1.0] * point_num

    model.add_linear_eq_cons(
        list_cons=list_cons_hb, list_coefs=list_coefs_hb, rhs=rhs_hb, penalty=penalty
    )

    # Building "Ha"
    list_cons_ha = [
        [f"x[{i}][{j}]" for i in range(point_num)] for j in range(point_num)
    ]
    list_coefs_ha = [[1.0] * point_num] * point_num
    rhs_ha = [1.0] * point_num

    model.add_linear_eq_cons(
        list_cons=list_cons_ha, list_coefs=list_coefs_ha, rhs=rhs_ha, penalty=penalty
    )
    model.compile()
    qubo, offset = model.get_dict_with_labels()

    return qubo, offset, model


if __name__ == "__main__":
    point_num = 5
    dlength = np.random.random_sample((point_num, point_num))
    max_len = float(dlength.max())
    d_min = np.amin(dlength, axis=0, initial=max_len, where=dlength > 0)

    qubo, offset, model = create_qubo(point_num, dlength, d_min, max_len)

    model.get_qubo(keys_str=True)

    oj_solution = model.solve(OpenJij(), num_reads=1, num_sweeps=10)
