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

from dqubo.apis.base import AnnealerSolver
import dimod as di
from neal import SimulatedAnnealingSampler
from dimod.sampleset import SampleSet
from typing import Dict, List, Tuple, Union


class Neal(AnnealerSolver):
    def __init__(self) -> None:
        """Constructor of neal solver interface class."""
        self.sampler = SimulatedAnnealingSampler()
        return

    def build_model(self, qubo: Dict[Tuple[str, str], float], offset: float) -> None:
        """Builds neal solver model using QUBO dictionary.

        Args:
            qubo (Dict[Tuple[str, str], float]): QUBO dictionary with the pairs and coefficients.
            offset (float): constant part of the objective function.
        """
        self.qubo_as_bqm = di.BQM(qubo, "BINARY")

        return

    def actual_solve(
        self, **kwargs
    ) -> Dict[str, List[Dict[str, Union[Dict[str, int], float, int]]]]:
        """Executes the solution part of the neal solver.

        Returns:
            Dict[str, List[Dict[str, Union[Dict[str, int], float, int]]]]: dictionary of dqubo solution.
        """
        num_reads = kwargs["num_reads"]
        num_sweeps = kwargs["num_sweeps"]
        beta_range = kwargs.get("beta_range", None)

        sampleset = self.sampler.sample(
            self.qubo_as_bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            beta_range=beta_range,
        )

        return self.convert_solution(sampleset)

    def convert_solution(
        self, o_sol: SampleSet
    ) -> Dict[str, List[Dict[str, Union[Dict[str, int], float, int]]]]:
        """Converts a solution from neal's format to DQUBO's format.

        Args:
            o_sol (SampleSet): object that contains the solution on neal's format.

        Returns:
            n_sol (Dict[str, List[Dict[str, Union[Dict[str, int], float, int]]]]): dictionary on dqubo's format with the solution.
        """
        n_sol = {}

        n_sol["samples"] = []

        for s in o_sol.data():
            sample = {}
            sample["sol"] = s[0]
            sample["energy"] = s[1]
            sample["repetitions"] = s[2]
            n_sol["samples"].append(sample)

        return n_sol
