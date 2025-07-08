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

"""DQUBO class implementation."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, dia_matrix, lil_matrix, tril
import asyncio
import nest_asyncio
import pyparsing as pp

try:
    from dqubo.apis.neal import Neal
except:
    # log warning
    Neal = None

try:
    from dqubo.apis.openjij import OpenJij
except ImportError:
    #  log warning
    OpenJij = None

from dqubo.apis.base import AnnealerSolver
from numpy import bool_, float32, float64, int64, ndarray
from typing import Any, Dict, List, Optional, Tuple, Union


class DQUBO:
    """
    DQUBO class implementation.

    This can be instantiated in several ways:
        DQUBO()
            with sparse matrix representation and parallization enabled for compilation.

        DQUBO(sparse=False)
            with dense matrix representation and parallization enabled for compilation.

        DQUBO(sparse=False, enable_async=False)
            with dense matrix representation and parallization disabled for compilation.

    Attributes
    ----------
    n_var : int
        Number of binary variables in the model
    H : float
        Hamiltonian expression for the whole problem
    Ho : float
        Hamiltonian expression for the problem's objective function(s)
    var_dic : dict
        Dictionary to store all variables
    list_var : list
        List of all binary variables
    fixed_vars : dict
        Dictionary to store all variables and their fixed values
    offset : float
        Offset value
    n_ineq : int
        Number of inequalities of the model
    cons : dict
        Dictionary to store all constraints of the model
    sparse : bool
        Flag to define if sparse mode must be considered
    compiled : bool
        Flag to define if model's compilation was completed
    qubo : dict
        Dicionary to store the resulting QUBO
    self.constraints: int
        Number of constraints in added in the QUBO
    self.constraints_coefs: int
        Number of coefficients added in all constraints
    """

    def __init__(self, sparse: bool = False, enable_async: bool = True) -> None:
        """Represents the Constructor definition for DQUBO. Initializes all variables to their defaults values.

        Args:
            sparse (bool, optional): All matrix operations will use sparse mode (if true), otherwise, all operarations will be conducted on dense representation using Numpy. Defaults to True.
            enable_async (bool, optional): Enable asynchornous calls to accelerate compilation. Defaults to True.

        """
        self.n_var = 0
        self.H = float(0.0)
        self.Ho = float(0.0)
        self.var_dic = dict()
        self.list_var = list()
        self.fixed_vars = dict()
        self.offset = float(0.0)
        self.n_ineq = 0
        self.cons = dict()
        self.sparse = sparse
        self.compiled = False
        self.enable_async = enable_async
        self.qubo = dict()
        if self.enable_async:
            nest_asyncio.apply()
        self.constraints = 0
        self.constraints_coefs = 0

    @property
    def size(self):
        """Get the size of the Hamiltonian after compilation.

        Raises:
            RuntimeError: Raises whenever this function is called before compilation.

        Returns:
           int: Elements in Hamiltonian
        """
        if self.compiled:
            if self.sparse:
                return self.H.size
            else:
                return np.count_nonzero(self.H)
        else:
            raise RuntimeError("size is not available before compilation")

    def add_variables(self, list_var: list) -> None:
        """Adds new variables (list_var) to the variables dict (var_dict).

        Args:
            list_var (list): Input variables to be added to the list
        """
        for var in list_var:
            assert type(var) == str, "All variables must be str type"
        self.list_var.extend(list_var)
        temp_dic = {var: self.n_var + i for i, var in enumerate(list_var)}

        self.var_dic.update(temp_dic)

        if self.n_var > 0:
            self.n_var += len(temp_dic)
            self.resize_matrizes()
        else:
            self.n_var += len(temp_dic)

        return

    def fix_vars(self, fixed_vars: Dict[str, int | float]) -> None:
        """Sets input variables to be fixed in the model.

        Args:
            fixed_vars (Dict[str, int | float]): A dictionary of variables to be fixed by the model
        """
        self.fixed_vars = fixed_vars

        return

    def add_qubo_obj(
        self, list_pairs: list, list_coefs: list, lagragian: float = 1.0
    ) -> None:
        """Adds an objective function to the model.

        Args:
            list_pairs (list): lists of pairs, where each list represents the variables of the pair.
            list_coefs (list): lists of coefficients, where each list represents the coefficients of the respective pair.
            lagragian (float, optional): A multiplier factor to the all given input pairs. Defaults to 1.0.
        """
        if self.sparse:
            Ho = self._create_qubo_matrix_sparse(list_pairs, list_coefs)
        else:
            Ho = self._create_qubo_matrix_dense(list_pairs, list_coefs)

        self.Ho += Ho * lagragian
        return

    def _create_qubo_matrix_dense(self, list_pairs: list, list_coefs: list) -> ndarray:
        """Adds an objective function to the model in dense representation.

        Args:
            list_pairs (list): lists of pairs, where each list represents the variables of the pair.
            list_coefs (list): lists of coefficients, where each list represents the coefficients of the respective pair.
        """
        row = [self.var_dic[x_i] for x_i, _ in list_pairs]
        col = [self.var_dic[x_j] for _, x_j in list_pairs]

        Ho = np.zeros((self.n_var, self.n_var), dtype=np.float32)

        Ho[row, col] = list_coefs

        return Ho

    def _create_qubo_matrix_sparse(
        self, list_pairs: list, list_coefs: list
    ) -> coo_matrix:
        """Adds an objective function to the model in sparse mode.

        Args:
            list_pairs (list): lists of pairs, where each list represents the variables of the pair.
            list_coefs (list): lists of coefficients, where each list represents the coefficients of the respective pair.
        """
        row = np.array([self.var_dic[x_i] for x_i, x_j in list_pairs])
        col = np.array([self.var_dic[x_j] for x_i, x_j in list_pairs])
        Ho = coo_matrix(
            (list_coefs, (row, col)), shape=(self.n_var, self.n_var), dtype=np.float32
        )
        Ho.sum_duplicates()

        return Ho

    def _compile_obj(self) -> None:
        """Compiles the objective to generate H Hamiltonian matrix."""
        if not isinstance(self.Ho, float):
            if self.sparse:
                self.H += tril(self.Ho.T) + tril(self.Ho, k=-1)
            else:
                self.H = np.tril(self.Ho.T) + np.tril(self.Ho, k=-1)

        return

    def add_linear_eq_cons(
        self,
        list_cons: list,
        list_coefs: list,
        rhs: list,
        penalty: float,
        name: Union[str, Tuple[str]] = None,
    ) -> None:
        """Adds a list of linear equalities as constraints to the model.

        Args:
            list_cons (list): lists of constraints, where each list represents the variables of the constraint.
            list_coefs (list): lists of coefficients, where each list represents the coefficients of the respective constraint.
            rhs (list): each element represents the right-hand side of the respective constraint.
            penalty (float): A penalty factor for all constraints.
            name (Union[str, Tuple[str]], optional): Constraint name. Defaults to None, then the constraint name will be a numerical index.
        """
        self._assert_list_cons_sizes(list_cons, list_coefs, rhs)
        if self.sparse:
            self._add_linear_eq_cons_sparse(list_cons, list_coefs, rhs, penalty, name)
        else:
            self._add_linear_eq_cons_dense(list_cons, list_coefs, rhs, penalty, name)

        return

    def _add_linear_eq_cons_dense(
        self,
        list_cons: list,
        list_coefs: list,
        rhs: list,
        penalty: float,
        name: Union[str, Tuple[str]] = None,
    ) -> None:
        """Adds a list of linear equalities as constraints to the model in dense representation.

        Args:
            list_cons (list): lists of constraints, where each list represents the variables of the constraint.
            list_coefs (list): lists of coefficients, where each list represents the coefficients of the respective constraint.
            rhs (list): each element represents the right-hand side of the respective constraint.
            penalty (float): A penalty factor for all constraints.
            name (Union[str, Tuple[str]], optional): Constraint name. Defaults to None, then the constraint name will be a numerical index.
        """
        n_cons = len(list_cons)

        row = [i for i in range(n_cons) for _ in range(len(list_coefs[i]))]
        col = [self.var_dic[var] for list_vars in list_cons for var in list_vars]
        data = [coef for con_coefs in list_coefs for coef in con_coefs]

        A = np.zeros((n_cons, self.n_var), dtype=np.float32)

        A[row, col] = data

        b = np.array(rhs, dtype=np.float32)

        if name is None:
            name = len(self.cons)
        self.cons[name] = (A, b, penalty)

        return

    def _add_linear_eq_cons_sparse(
        self,
        list_cons: list,
        list_coefs: list,
        rhs: list,
        penalty: float,
        name: Union[str, Tuple[str]] = None,
    ) -> None:
        """Adds a list of linear equalities as constraints to the model in sparse mode.

        Args:
            list_cons (list): lists of constraints, where each list represents the variables of the constraint.
            list_coefs (list): lists of coefficients, where each list represents the coefficients of the respective constraint.
            rhs (list): each element represents the right-hand side of the respective constraint.
            penalty (float): A penalty factor for all constraints.
            name (Union[str, Tuple[str]], optional): Constraint name. Defaults to None, then the constraint name will be a numerical index.
        """
        n_cons = len(list_cons)

        indptr = np.cumsum([0] + [len(list_coefs[i]) for i in range(n_cons)])
        indices = np.array(
            [self.var_dic[var] for list_vars in list_cons for var in list_vars]
        )
        data = np.array(
            [coef for con_coefs in list_coefs for coef in con_coefs], dtype=np.float32
        )
        A = csr_matrix(
            (data, indices, indptr), shape=(n_cons, self.n_var), dtype=np.float32
        )
        b = np.array(rhs, dtype=np.float32)

        if name is None:
            name = len(self.cons)
        self.cons[name] = (A, b, penalty)

        return

    def _compile_cons(self, with_penalty: Dict[int, float]) -> None:
        """Compiles constraints to generate H Hamiltonian matrix.

        Args:
            with_penalty (Dict[int  |  str, float], optional): Dictionary of penalties of constraints that overwrites the previous penalties. Defaults to dict().
        """
        for key in with_penalty.keys():
            if key in self.cons:
                cons = self.cons[key]
                if len(cons) == 3:
                    a, b, penalty = self.cons[key]
                    self.cons[key] = (a, b, with_penalty[key])
                elif len(cons) == 2:
                    a, penalty = self.cons[key]
                    self.cons[key] = (a, with_penalty[key])
        eligible_cons = []
        for name, cons in self.cons.items():
            penalty = None
            if len(cons) == 3:
                _, _, penalty = cons
            elif len(cons) == 2:
                _, penalty = cons
            if penalty != 0.0 and penalty != None:
                eligible_cons.append((name, cons))

        results = []
        if self.enable_async:
            # # Create a new event loop
            loop = asyncio.get_event_loop()

            # Create all async calls
            looper = asyncio.gather(
                *[self._compile_cons_async(name, cons) for name, cons in eligible_cons]
            )

            # Block all calls at this line below
            results = loop.run_until_complete(looper)
        else:
            if self.sparse:
                for name, cons in eligible_cons:
                    results.append(self._compile_cons_sparse(name, cons))
            else:
                for name, cons in eligible_cons:
                    results.append(self._compile_cons_dense(name, cons))

        for Hc, penalty, bb in results:
            # save as a quadratic constraint compiled
            self.H += penalty * Hc
            if bb != None:
                self.offset += penalty * bb
        return

    # Async function #1
    def background(f):
        """Implements the async background function (decorator used in _compile_cons_async()) that is used in the parallezation part of compilation.

        Args:
            f (Function): Function to be run on background for async calls
        """

        def wrapped(*args, **kwargs):
            return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

        return wrapped

    # Async function #2
    @background
    def _compile_cons_async(self, name, cons):
        """Async function that calls to compile the constraints into the Hamiltonian. Sparsity flag is applied.

        Args:
            name (Union[str, Tuple[str]], optional): Constraint name.
            cons (Union[Tuple[coo_matrix, float],List[Union[csr_matrix, ndarray, float]],Tuple[csr_matrix, ndarray, float]): Constraint elements (matrix, array and penalty).

        Returns:
           Hc (csr_matrix) : Hamiltonian for the constraints.
           penalty (float) : penalty for all constraints of Hc.
           bb (float) : scalar of the objective function for this constraint.
        """
        if self.sparse:
            return self._compile_cons_sparse(name, cons)
        else:
            return self._compile_cons_dense(name, cons)

    def _compile_cons_sparse(
        self,
        name: Union[int, str],
        cons: Union[
            Tuple[coo_matrix, float],
            List[Union[csr_matrix, ndarray, float]],
            Tuple[csr_matrix, ndarray, float],
        ],
    ) -> Union[
        Tuple[csr_matrix, float, None],
        Tuple[csr_matrix, float, float32],
        Tuple[csr_matrix, int, float32],
    ]:
        """Compiles the constraints into the Hamiltonian using sparse mode.

        Args:
            name (Union[str, Tuple[str]], optional): Constraint name.
            cons (Union[Tuple[coo_matrix, float],List[Union[csr_matrix, ndarray, float]],Tuple[csr_matrix, ndarray, float]): Constraint elements (matrix, array and penalty)

        Returns:
           Hc (csr_matrix) : Hamiltonian for the constraints.
           penalty (float) : penalty for all constraints of Hc.
           bb (float) : scalar of the objective function for this constraint.
        """
        # linear
        Hc = None
        penalty = None
        bb = None
        if len(cons) == 3:
            A, b, penalty = cons

            Ab = A.T.dot(b)
            bb = b.T.dot(b)
            A2 = A.T.dot(A)

            # work only with the diagonal
            data = A2.diagonal() - 2 * Ab
            offsets = np.zeros_like(data)[-2]
            diag = dia_matrix(
                (data, offsets), shape=(data.size, data.size), dtype=np.float32
            )
            # the diagonal plus 2 times the inferior part since is simetric
            Hc = diag + 2 * tril(A2, k=-1)

        # quadratic
        elif len(cons) == 2:
            Hc, penalty = cons
            Hc = tril(Hc.T) + tril(Hc, k=-1)

        return Hc, penalty, bb

    def _compile_cons_dense(
        self,
        name: Union[str, List[str]],
        cons: Union[
            List[Union[ndarray, float]],
            Tuple[ndarray, float],
            Tuple[ndarray, ndarray, float],
        ],
    ) -> Union[Tuple[ndarray, float, None], Tuple[ndarray, float, float32]]:
        """Compiles the constraints into the Hamiltonian using dense representation.

        Args:
            name (Union[str, Tuple[str]], optional): Constraint name.
            cons (Union[Tuple[coo_matrix, float],List[Union[csr_matrix, ndarray, float]],Tuple[csr_matrix, ndarray, float]): Constraint elements (matrix, array and penalty)

        Returns:
           Hc (csr_matrix) : Hamiltonian for the constraints
           penalty (float) : penalty for all constraints of Hc
           bb (float) : scalar of the objective function for this constraint
        """
        Hc = None
        penalty = None
        bb = None
        # linear
        if len(cons) == 3:
            A, b, penalty = cons

            Ab = A.T.dot(b)
            bb = b.T.dot(b)
            A2 = A.T.dot(A)

            # work only with the diagonal
            data = A2.diagonal() - 2 * Ab

            diag = np.diag(data)
            # the diagonal plus 2 times the inferior part since is simetric
            Hc = diag + 2 * np.tril(A2, k=-1)

        # quadratic
        elif len(cons) == 2:
            Hc, penalty = cons
            Hc = np.tril(Hc.T) + np.tril(Hc, k=-1)

        return Hc, penalty, bb

    def _assert_list_cons_sizes(
        self,
        list_cons: List[List[str]],
        list_coefs: List[Union[List[float], List[int], float, int]],
        rhs: Optional[List[int | float]],
        lhs: Optional[List[int | float]] = None,
        quad: bool = False,
    ) -> None:
        """Asserts every given constraint (variables, coefficients, rhs and lhs) to an expected format.

        Args:
            list_cons (list): lists of constraints, where each list represents the variables of the constraint.
            list_coefs (list): lists of coefficients, where each list represents the coefficients of the respective constraint.
            rhs (Optional[List[int  |  float]]): list of right-hand side values for each constraint.
            lhs (Optional[List[int  |  float]], optional): list of left-hand side values for each constraint.. Defaults to None.
            quad (bool, optional): Flag to define if the given set of constraints is quadratic or linear. Defaults to False.

        Raises:
            TypeError: Indicates that at least one format violation was found in the given constraints.
        """
        assert (
            len(list_cons) > 0 and len(list_coefs) > 0
        ), "List cons and coefs must be greater than zero in size"

        assert len(list_cons) == len(
            list_coefs
        ), "List cons and list_coefs must be equal in size"

        if quad:
            for i, cons in enumerate(list_cons):
                assert (
                    len(cons) == 2
                ), "Quadratic constraints must have exactly 2 variables"

            for i, cons in enumerate(list_cons):
                for var in cons:
                    if type(var) != str:
                        raise TypeError("Constraint variables must be in string format")
            for coef in list_coefs:
                if not (type(coef) == int or type(coef) == float):
                    raise TypeError("Variable coefficients must be integer or float")

        else:
            for i, cons in enumerate(list_cons):
                assert len(cons) == len(
                    list_coefs[i]
                ), "(Linear constraints) coefficients and variables do not match"

            for i, cons in enumerate(list_cons):
                for var in cons:
                    if type(var) != str:
                        raise TypeError("Constraint variables must be in string format")
                for coef in list_coefs[i]:
                    if not (type(coef) == int or type(coef) == float):
                        raise TypeError(
                            "Variable coefficients must be integer or float"
                        )

        if rhs != None:
            assert len(list_cons) == len(rhs), "List cons and rhs must be equal in size"
            for coef in rhs:
                if not (type(coef) == int or type(coef) == float):
                    raise TypeError("Rhs values must be integer or float")

        if lhs != None:
            assert len(list_cons) == len(lhs), "List cons and lhs must be equal in size"
            for coef in lhs:
                if not (type(coef) == int or type(coef) == float):
                    raise TypeError("Rhs values must be integer or float")

    def add_linear_ineq_cons(
        self,
        list_cons: List[List[str]],
        list_coefs: List[List[int | float]],
        penalty: float,
        lhs: Optional[List[int | float]] = None,
        rhs: Optional[List[int | float]] = None,
        name: str = None,
    ) -> None:
        """Adds a list of linear inequalities as constraints to the model.

        Args:
            list_cons (List[List[str]]): lists of constraints, where each list represents the variables of the constraint.
            list_coefs (List[List[int  |  float]]): lists of coefficients, where each list represents the coefficients of the respective constraint.
            penalty (float): A penalty factor for all constraints.
            lhs (Optional[List[int  |  float]], optional): each element represents the left-hand side of the respective constraint.. Defaults to None.
            rhs (Optional[List[int  |  float]], optional): each element represents the right-hand side of the respective constraint.. Defaults to None.
            name (str, optional): Constraint name. Defaults to None, then the constraint name will be a numerical index.
        """
        self._assert_list_cons_sizes(list_cons, list_coefs, rhs, lhs)

        assert (
            lhs != None or rhs != None
        ), "lhs and rhs must not be None at the same time"

        n_cons = len(list_cons)
        if lhs == None:
            lhs = [sum(coef for coef in coefs if coef < 0) for coefs in list_coefs]

        if rhs == None:
            rhs = [sum(coef for coef in coefs if coef > 0) for coefs in list_coefs]

        assert all(lhs[i] <= rhs[i] for i in range(n_cons)), "lhs > rhs"

        span = [rhs[i] - lhs[i] for i in range(n_cons)]

        span_list_var, span_list_coefs = self.add_slacks_log(span)

        list_cons2 = [list_cons[i] + span_list_var[i] for i in range(n_cons)]
        list_coefs2 = [list_coefs[i] + span_list_coefs[i] for i in range(n_cons)]

        self.n_ineq += n_cons
        self.add_linear_eq_cons(list_cons2, list_coefs2, lhs, penalty, name)

        return

    def add_quadratic_cons(
        self,
        list_pairs: List[Tuple[str, str]],
        list_coefs: List[int | float],
        penalty: float,
        name: str = None,
    ) -> None:
        """Adds quadratic constraits to the set of constraints.

        Args:
            list_pairs (List[Tuple[str, str]]): list of pairs that represente the multiplication of two variables.
            list_coefs (List[int  |  float]): list of coefficients that correspond to the list_pairs.
            penalty (float): constant that multiplies to the quadratic constraints.
            name (str, optional): Name of the quadratic constraint.
        """
        self._assert_list_cons_sizes(
            list_pairs, list_coefs, rhs=None, lhs=None, quad=True
        )
        if self.sparse:
            self._add_quadratic_cons_sparse(list_pairs, list_coefs, penalty, name)
        else:
            self._add_quadratic_cons_dense(list_pairs, list_coefs, penalty, name)
        return

    def _add_quadratic_cons_sparse(
        self,
        list_pairs: List[Tuple[str, str]],
        list_coefs: List[int | float],
        penalty: float,
        name: str = None,
    ) -> None:
        """Adds quadratic constraits to the set of constraints when is in sparse mode.

        Args:
            list_pairs (List[Tuple[str, str]]): list of pairs that represente the multiplication of two variables.
            list_coefs (List[int  |  float]): list of coefficients that correspond to the list_pairs.
            penalty (float): constant that multiplies to the quadratic constraints.
            name (str, optional): Name of the quadratic constraint.
        """
        row = np.array([self.var_dic[x_i] for x_i, _ in list_pairs], dtype=np.float32)
        col = np.array([self.var_dic[x_j] for _, x_j in list_pairs], dtype=np.float32)
        Hc = coo_matrix(
            (list_coefs, (row, col)), shape=(self.n_var, self.n_var), dtype=np.float32
        )

        if name is None:
            name = len(self.cons)
        self.cons[name] = (Hc, penalty)

        return

    def _add_quadratic_cons_dense(
        self,
        list_pairs: List[Tuple[str, str]],
        list_coefs: List[int | float],
        penalty: float,
        name: str = None,
    ) -> None:
        """Adds quadratic constraits to the set of constraints when is in dense mode.

        Args:
            list_pairs (List[Tuple[str, str]]): list of pairs that represente the multiplication of two variables.
            list_coefs (List[int  |  float]): list of coefficients that correspond to the list_pairs.
            penalty (float): constant that multiplies to the quadratic constraints.
            name (str, optional): Name of the quadratic constraint.
        """
        row = [self.var_dic[x_i] for x_i, _ in list_pairs]
        col = [self.var_dic[x_j] for _, x_j in list_pairs]
        Hc = np.zeros((self.n_var, self.n_var), dtype=np.float32)

        Hc[row, col] = list_coefs

        if name is None:
            name = len(self.cons)
        self.cons[name] = (Hc, penalty)

        return

    def add_slacks_log(
        self, rhs: List[int | float]
    ) -> Tuple[List[List[str]], List[List[int | float]]]:
        """Creates slack variables to convert inequalities into equalities.

        Args:
            rhs (List[int  |  float]): right hand side of the inequality constraint.

        Returns:
            list_var (List[List[str]]): list of new slack variables.
            list_coefs (List[List[int | float]]): list of coefficients of list_var.
        """
        list_var = list()
        list_names = list()
        list_coefs = list()
        for i, b in enumerate(rhs):
            if b == 0:
                list_var.append([])
                list_coefs.append([])
                continue
            else:
                label = "slack_" + str(self.n_ineq + i)
                names, coefs = log_enc(label, (0, b))
                coefs = [-coef for coef in coefs]
                list_names.extend(names)
                list_var.append(names)
                list_coefs.append(coefs)

        self.add_variables(list_names)

        return list_var, list_coefs

    def resize_matrizes(self) -> None:
        """Resizes all matrizes when new variables are added."""
        if self.sparse:
            self._resize_matrizes_sparse()
        else:
            self._resize_matrizes_dense()

        return

    def _resize_matrizes_sparse(self) -> None:
        """Resizes all matrizes when new variables are added in the sparse mode."""
        if not isinstance(self.H, float):
            self.H.resize((self.n_var, self.n_var))

        if not isinstance(self.Ho, float):
            self.Ho.resize((self.n_var, self.n_var))

        for name, cons in self.cons.items():
            # linear
            if len(cons) == 3:
                A, b, penalty = cons
                A.resize(A.shape[0], self.n_var)
                self.cons[name] = (A, b, penalty)
            # quadratic
            elif len(cons) == 2:
                Hc, penalty = cons
                Hc.resize((self.n_var, self.n_var))
                self.cons[name] = (Hc, penalty)

        return

    def _resize_matrizes_dense(self) -> None:
        """Resizes all matrizes when new variables are added in the dense mode."""
        if not isinstance(self.H, float):
            n_old = self.H.shape[0]
            n_add = self.n_var - n_old
            self.H = np.pad(
                self.H, ((0, n_add), (0, n_add)), mode="constant", constant_values=0
            )

        if not isinstance(self.Ho, float):
            n_old = self.Ho.shape[0]
            n_add = self.n_var - n_old
            self.Ho = np.pad(
                self.Ho, ((0, n_add), (0, n_add)), mode="constant", constant_values=0
            )

        for name, cons in self.cons.items():
            # linear
            if len(cons) == 3:
                A, b, penalty = cons
                n_old = A.shape[1]
                n_add = self.n_var - n_old
                A = np.pad(A, ((0, 0), (0, n_add)), mode="constant", constant_values=0)
                self.cons[name] = (A, b, penalty)
            # quadratic
            elif len(cons) == 2:
                Hc, penalty = cons
                n_old = Hc.shape[0]
                n_add = self.n_var - n_old
                Hc = np.pad(
                    Hc, ((0, n_add), (0, n_add)), mode="constant", constant_values=0
                )
                self.cons[name] = (Hc, penalty)

        return

    def compile(self, with_penalty: Dict[int | str, float] = dict()) -> None:
        """Compiles the objective and constraints to generate H Hamiltonian matrix.

        Args:
            with_penalty (Dict[int  |  str, float], optional): Dictionary of penalties of constraints that overwrites the previous penalties. Defaults to dict().
        """
        self.H = float(0.0)
        self.offset = float(0.0)
        self._compile_obj()

        self._compile_cons(with_penalty)

        self.compiled = True
        return

    def objective(self, x_dict: Dict[str, int]) -> float:
        """Calculates the objective function for a given solution x_dict.

        Args:
            x_dict (Dict[str, int]): Dictionary that contains a solution with names on the key and 0 or 1 on values.

        Returns:
            obj (float): value of the objective function.
        """
        assert self.compiled, "model not compiled"
        x = np.array(
            [
                self.fixed_vars[vname] if vname in self.fixed_vars else x_dict[vname]
                for vname in self.list_var
            ],
            dtype=np.float32,
        )

        obj = self.Ho.dot(x).dot(x)  # +self.offset
        return obj

    def energy(self, x_dict: Dict[str, int]) -> float:
        """Calculates the energy for a given solution x_dict.

        Args:
            x_dict (Dict[str, int]): Dictionary that contains a solution with names on the key and 0 or 1 on values.

        Returns:
            e (float): value of the energy.
        """
        assert self.compiled, "model not compiled"
        x = np.array(
            [
                self.fixed_vars[vname] if vname in self.fixed_vars else x_dict[vname]
                for vname in self.list_var
            ],
            dtype=np.float32,
        )

        e = self.H.dot(x).dot(x) + self.offset
        return e

    def get_qubo(
        self,
        input_format: str = "dict_int",
    ) -> Union[
        Tuple[Dict[Tuple[int, int], float], float],
        Tuple[Dict[Tuple[str, str], float], float],
    ]:
        """Gets the final QUBO dictionary on different formats.

        Args:
            format (str, optional): QUBO output format. Defaults to "dict". Acception options: ["dict_int", "dict_str", "dict_labelled", "matrix"]

        Raises:
            NotImplementedError: raises if any format except "dict" is given.

        Returns:
            Union[ Tuple[Dict[Tuple[int, int], float], float], Tuple[Dict[Tuple[str, str], float], float], ]: _description_
        """
        assert self.compiled, "model not compiled"
        assert len(self.list_var) > 0, "model does not have variables"
        accepted_input_formats = [
            "dict_int",
            "dict_str",
            "dict_labelled",
            "matrix",
        ]

        assert (
            input_format in accepted_input_formats
        ), "Only dict or numpy array are accepted options"

        input_format_split = input_format.split("_")

        qubo = None
        offset = float(0.0)

        if input_format_split[0] == "dict":
            keys_str = input_format_split[1] == "str"

            if self.sparse:
                qubo, offset = self._get_qubo_dict_sparse(keys_str)
            else:
                qubo, offset = self._get_qubo_dict_dense(keys_str)

            if input_format_split[1] == "labelled":
                qubo = {
                    (self.list_var[i], self.list_var[j]): val
                    for (i, j), val in qubo.items()
                }
        elif input_format_split[0] == "matrix":
            if self.sparse:
                qubo, offset = self._get_qubo_matrix_sparse()
            else:
                qubo, offset = self._get_qubo_matrix_dense()

        self.qubo = qubo
        self.offset = offset
        return qubo, offset

    def _get_qubo_dict_sparse(
        self, keys_str: bool = False
    ) -> Union[
        Tuple[Dict[Tuple[int, int], float], float],
        Tuple[Dict[Tuple[str, str], float], float],
    ]:
        """Gets a QUBO dictionary representation on two formats with keys as strings or integers. Such hose keys are tuples of the index of the variables for sparse mode.

        Args:
            keys_str (bool, optional): True if Keys are strings. Defaults to "False".

        Returns:
            qubo ( Union[ Dict[Tuple[int, int], float], Dict[Tuple[str, str], float]] ): QUBO dictionary with the pairs and coefficients.
            offset (float): constant part of the objective function.
        """
        if self.fixed_vars:
            Hf, const = self._fix_vars_H_sparse()
        else:
            self.H = self.H.tocoo()
            Hf = self.H
            const = 0

        if keys_str:
            # openjij case
            qubo = dict(
                zip(
                    zip(map(str, Hf.row), map(str, Hf.col)),
                    map(float, Hf.data),
                )
            )
        else:
            qubo = dict(
                zip(
                    zip(map(int, Hf.row), map(int, Hf.col)),
                    map(float, Hf.data),
                )
            )
        return qubo, float(self.offset + const)

    def _get_qubo_dict_dense(
        self, keys_str: bool = False
    ) -> Union[
        Tuple[Dict[Tuple[int, int], float], float],
        Tuple[Dict[Tuple[str, str], float], float],
    ]:
        """Gets a QUBO dictionary representations on two formats with keys as strings or integers. Such keys are tuples of the index of the variables for the dense mode.

        Args:
            keys_str (bool, optional): True if Keys are strings. Defaults to "False".

        Returns:
            qubo ( Union[ Dict[Tuple[int, int], float], Dict[Tuple[str, str], float]] ): QUBO dictionary with the pairs and coefficients.
            offset (float): constant part of the objective function.
        """
        if self.fixed_vars:
            Hf, const = self._fix_vars_H_dense()
        else:
            Hf = self.H
            const = 0

        indexes = np.nonzero(Hf)
        if keys_str:
            # openjij case
            qubo = dict(
                zip(
                    zip(map(str, indexes[0]), map(str, indexes[1])),
                    map(float, Hf[indexes]),
                )
            )
        else:
            qubo = dict(
                zip(
                    zip(map(int, indexes[0]), map(int, indexes[1])),
                    map(float, Hf[indexes]),
                )
            )
        return qubo, float(self.offset + const)

    def _fix_vars_H_sparse(self) -> Tuple[coo_matrix, float]:
        fixed_indexs = [self.var_dic[name] for name in self.fixed_vars.keys()]
        var = list(set(self.list_var).difference(set(self.fixed_vars.keys())))
        var_indexs = [self.var_dic[vname] for vname in var]

        H11 = self.H.copy().tolil()
        H11[fixed_indexs, :] = 0
        H11[:, fixed_indexs] = 0

        H22 = self.H.copy().tolil()
        H22[var_indexs, :] = 0
        H22[:, var_indexs] = 0

        H11_mask = lil_matrix(self.H.shape, dtype=bool)
        H11_mask[fixed_indexs, :] = True
        H11_mask[:, fixed_indexs] = True
        H22_mask = lil_matrix(self.H.shape, dtype=bool)
        H22_mask[var_indexs, :] = True
        H22_mask[:, var_indexs] = True

        H12_mask = H11_mask.multiply(H22_mask)

        H12 = lil_matrix(self.H.shape, dtype=np.float32)
        H12[H12_mask] = self.H[H12_mask]

        H12 = H12.T + H12

        x_f = np.array(
            [
                self.fixed_vars[vname] if vname in self.fixed_vars else 0
                for vname in self.list_var
            ],
            dtype=np.float32,
        )

        data = H12.dot(x_f)
        offsets = np.zeros_like(data)[-2]
        D = dia_matrix((data, offsets), shape=(data.size, data.size), dtype=np.float32)

        H11 += D

        const = H22.dot(x_f).dot(x_f)

        return H11.tocoo(), const

    def _fix_vars_H_dense(self) -> Tuple[ndarray, float]:
        """Fixes the variables during the process of QUBO dictionary generation for dense mode.

        Returns:
            H11 (coo_matrix): croped matrix version of H when variables are fixed.
            cons (float): new constant part of the objective function when variables are fixed.
        """
        fixed_indexs = [self.var_dic[name] for name in self.fixed_vars.keys()]
        var = list(set(self.list_var).difference(set(self.fixed_vars.keys())))
        var_indexs = [self.var_dic[vname] for vname in var]

        H11 = self.H.copy()
        H11[fixed_indexs, :] = 0
        H11[:, fixed_indexs] = 0

        H22 = self.H.copy()
        H22[var_indexs, :] = 0
        H22[:, var_indexs] = 0

        H11_mask = np.zeros_like(self.H, dtype=bool)
        H11_mask[fixed_indexs, :] = True
        H11_mask[:, fixed_indexs] = True
        H22_mask = np.zeros_like(self.H, dtype=bool)
        H22_mask[var_indexs, :] = True
        H22_mask[:, var_indexs] = True

        H12_mask = np.multiply(H11_mask, H22_mask)

        H12 = np.zeros_like(self.H, dtype=np.float32)
        H12[H12_mask] = self.H[H12_mask]

        H12 = H12.T + H12

        x_f = np.array(
            [
                self.fixed_vars[vname] if vname in self.fixed_vars else 0
                for vname in self.list_var
            ],
            dtype=np.float32,
        )

        D = np.diag(H12.dot(x_f))

        H11 += D

        const = H22.dot(x_f).dot(x_f)

        return H11, const

    def _get_qubo_matrix_dense(self):
        Hf = None
        const = float(0.0)
        if self.fixed_vars:
            Hf, const = self._fix_vars_H_dense()
        else:
            Hf = self.H
            const = 0
        const += self.offset
        return Hf, const

    def _get_qubo_matrix_sparse(self):
        Hf = None
        const = float(0.0)

        if self.fixed_vars:
            Hf, const = self._fix_vars_H_sparse()
        else:
            self.H = self.H.tocoo()
            Hf = self.H
            const = 0
        const += self.offset
        return Hf, const

    def _check_constraint(
        self, name: Union[str, List[str]], x_dict: dict, tol: float = 1e-6
    ) -> Union[Dict[int, Tuple[bool_, float]], Dict[str, Tuple[bool_, float]]]:
        """Checks an individual constraint if is feasible with the given x_dic solution under a precision tolerance tol.

        Args:
            name (Union[str, List[str]]): constraint name.
            x_dict (dict): dictionary solution to be tested.
            tol (float, optional): tolerance for determine if the constraint is attained by the solution. Defaults to 1e-6.

        Returns:
            checked_cons (Union[Dict[int, Tuple[bool_, float]], Dict[str, Tuple[bool_, float]]]): dictionary with the names of constraints as keys and tuples of an indicator if the constraint is atained and a mistmatch r.
        """
        cons = self.cons[name]
        x = np.array([x_dict[vname] for vname in self.list_var], dtype=np.float32)
        checked_cons = {}
        if len(cons) == 3:
            A, b, _ = cons
            r = A.dot(x) - b
            n_cons = len(r)
            logical_values = abs(r) <= tol
            if isinstance(name, str):
                names = [name + "_" + str(i) for i in range(n_cons)]
            else:
                names = name
            checked_cons = {names[i]: (logical_values[i], r[i]) for i in range(n_cons)}
        # quadratic
        elif len(cons) == 2:
            Hc, _ = cons
            r = Hc.dot(x).dot(x)
            logical_values = abs(r) <= tol
            checked_cons[name] = (logical_values, r)

        return checked_cons

    def check_constraints(
        self, x_dict: Dict[str, int], print_results: bool = False
    ) -> Dict[Union[str, int], Tuple[bool_, float]]:
        """Checks all constraints if is feasible with the given x_dic solution.

        Args:
            x_dict (dict): dictionary solution to be tested.
            print_results (bool, optional): indicator if print the results or not. Defaults to False.

        Returns:
            checked_cons (Union[Dict[int, Tuple[bool_, float]], Dict[str, Tuple[bool_, float]]]): dictionary with the names of constraints as keys and tuples of an indicator if the constraint is atained and a mistmatch r.
        """
        assert self.compiled, "model not compiled"
        checked_cons = {}
        for name in self.cons.keys():
            cc = self._check_constraint(name, x_dict)
            checked_cons.update(cc)

        if print_results:
            print(checked_cons)

        return checked_cons

    def to_dense(self) -> None:
        """Switchs the matrices formats from sparse to dense mode."""
        if self.sparse:
            if self.compiled:
                self.H = self.H.toarray()
            if not isinstance(self.Ho, float):
                self.Ho = self.Ho.toarray()

            for name, cons in self.cons.items():
                # linear
                if len(cons) == 3:
                    A, b, penalty = cons
                    A = A.toarray()
                    self.cons[name] = (A, b, penalty)
                # quadratic
                elif len(cons) == 2:
                    Hc, penalty = cons
                    Hc = Hc.toarray()
                    self.cons[name] = (Hc, penalty)

            self.sparse = False
        return

    def to_sparse(self) -> None:
        """Switchs the matrices formats from dense to sparse mode."""
        if not self.sparse:
            if self.compiled:
                self.H = coo_matrix(self.H)
            if not isinstance(self.Ho, float):
                self.Ho = coo_matrix(self.Ho)

            for name, cons in self.cons.items():
                # linear
                if len(cons) == 3:
                    A, b, penalty = cons
                    A = csr_matrix(A)
                    self.cons[name] = (A, b, penalty)
                # quadratic
                elif len(cons) == 2:
                    Hc, penalty = cons
                    Hc = coo_matrix(Hc)
                    self.cons[name] = (Hc, penalty)

            self.sparse = True

        return

    def solve(self, solver: AnnealerSolver, **kwargs) -> Dict[str, List[Dict]]:
        """Solves the model using an external solver.

        Args:
            solver (AnnealerSolver): Receives the encapsulated solver to solve the model.

        Returns:
            solution (Dict[str, List[Dict]]): dictionary with the solution of the model.
        """
        assert self.compiled, "model not compiled"
        solver.build_model(self.qubo, self.offset)

        solution = solver.actual_solve(**kwargs)

        # put fixed bars
        if self.fixed_vars:
            for sample in solution["samples"]:
                sample["sol"].update(self.fixed_vars)

        if len(self.list_var) == len(solution["samples"][0]["sol"]):
            for sample in solution["samples"]:
                for var in self.list_var:
                    if var not in sample["sol"]:
                        sample["sol"][var] = 0
        return solution

    def readlp(self, filename: str) -> None:
        """Reads lp file format and populate the dqubo model with variables, objective and constraints. Adapted from https://github.com/aphi/Lp-Parser.

        Args:
            filename (str): filename for reading the file.

        Raises:
            IOError: Raise if the given filename cannot open the file.
        """
        # read input lp file
        try:
            fp = open(filename)
            fullDataString = fp.read()
            fp.close()
        except IOError:
            print(f"Could not find input lp file {filename}")
            raise IOError

        # name char ranges for objective, constraint or variable
        allNameChars = pp.alphanums + "!\"#$%&()/,.;?@_'`\{\}|~"
        firstChar = multiRemove(
            allNameChars, pp.nums + "eE."
        )  # <- can probably use CharsNotIn instead

        name = pp.Word(firstChar, allNameChars, max=255)
        keywords = [
            "inf",
            "infinity",
            "max",
            "maximum",
            "maximize",
            "min",
            "minimum",
            "minimize",
            "s.t.",
            "st",
            "bound",
            "bounds",
            "bin",
            "binaries",
            "binary",
            "gen",
            "general",
            "end",
        ]
        pyKeyword = pp.MatchFirst(map(pp.CaselessKeyword, keywords))
        validName = ~pyKeyword + name
        validName = validName.setResultsName("name")

        colon = pp.Suppress(pp.oneOf(": ::"))
        plusMinus = pp.oneOf("+ -")
        inf = pp.oneOf("inf infinity", caseless=True)
        number = pp.Word(pp.nums + ".")
        sense = pp.oneOf("< <= =< = > >= =>").setResultsName("sense")

        # section tags
        objTagMax = pp.oneOf("max maximum maximize", caseless=True)
        objTagMin = pp.oneOf("min minimum minimize", caseless=True)
        objTag = (objTagMax | objTagMin).setResultsName("objSense")

        constraintsTag = pp.oneOf(
            ["subj to", "subject to", "s.t.", "st"], caseless=True
        )

        boundsTag = pp.oneOf("bound bounds", caseless=True)
        binTag = pp.oneOf("bin binaries binary", caseless=True)
        genTag = pp.oneOf("gen general", caseless=True)

        endTag = pp.CaselessLiteral("end")

        # linear expression
        # coefficient on a variable (includes sign)
        firstVarCoef = pp.Optional(plusMinus, "+") + pp.Optional(number, "1")
        firstVarCoef.setParseAction(
            lambda tokens: eval("".join(tokens))
        )  # TODO: can't this just be eval(tokens[0] + tokens[1])?

        coef = plusMinus + pp.Optional(number, "1")
        coef.setParseAction(
            lambda tokens: eval("".join(tokens))
        )  # TODO: can't this just be eval(tokens[0] + tokens[1])?

        # variable (coefficient and name)
        firstVar = pp.Group(
            firstVarCoef.setResultsName("coef")
            + validName
            + pp.ZeroOrMore("*" + validName)
        )
        var = pp.Group(
            coef.setResultsName("coef") + validName + pp.ZeroOrMore("*" + validName)
        )

        linvarExpr = firstVar + pp.ZeroOrMore(var)
        linvarExpr = linvarExpr.setResultsName("linExpr")

        # objective
        objective = objTag + pp.Optional(validName + colon) + pp.Optional(linvarExpr)
        objective = objective.setResultsName("objective")

        # constraint rhs
        rhs = pp.Optional(plusMinus, "+") + number
        rhs = rhs.setResultsName("rhs")
        rhs.setParseAction(lambda tokens: eval("".join(tokens)))

        # constraints
        constraint = pp.Group(pp.Optional(validName + colon) + linvarExpr + sense + rhs)
        constraints = pp.ZeroOrMore(constraint)
        constraints = constraints.setResultsName("constraints")

        # bounds
        signedInf = (plusMinus + inf).setParseAction(
            lambda tokens: (tokens[0] == "+") * 1e30
        )
        signedNumber = (pp.Optional(plusMinus, "+") + number).setParseAction(
            lambda tokens: eval("".join(tokens))
        )  # this is different to previous, because "number" is mandatory not optional
        numberOrInf = (signedNumber | signedInf).setResultsName("numberOrInf")
        ineq = numberOrInf & sense
        sensestmt = pp.Group(
            pp.Optional(ineq).setResultsName("leftbound")
            + validName
            + pp.Optional(ineq).setResultsName("rightbound")
        )
        freeVar = pp.Group(validName + pp.Literal("free"))

        boundstmt = freeVar | sensestmt
        bounds = boundsTag + pp.ZeroOrMore(boundstmt).setResultsName("bounds")

        # generals
        generals = genTag + pp.ZeroOrMore(validName).setResultsName("generals")

        # binaries
        binaries = binTag + pp.ZeroOrMore(validName).setResultsName("binaries")

        varInfo = pp.ZeroOrMore(bounds | generals | binaries)

        grammar = objective + constraintsTag + constraints + varInfo + endTag

        # commenting
        commentStyle = pp.Literal("\\") + pp.restOfLine
        grammar.ignore(commentStyle)

        # parse input string
        parseOutput = grammar.parseString(fullDataString)

        if parseOutput.objSense in ["max", "maximum", "maximize"]:
            sense = -1

        self.add_variables(list(parseOutput.binaries))

        list_pairs = []
        list_coefs = []
        for term in parseOutput.objective[2:]:
            if len(term) == 2:
                c = term[0]
                x = term[1]
                list_pairs.append((x, x))
                list_coefs.append(c)
            elif len(term) == 4:
                c = term[0]
                x = term[1]
                y = term[3]
                list_pairs.append((x, y))
                list_coefs.append(c)

        self.add_qubo_obj(list_pairs=list_pairs, list_coefs=list_coefs, lagragian=sense)

        list_eqs = []
        list_eqs_coef = []
        rhs_eqs = []
        name_eqs = []
        list_ineqs = []
        list_ineqs_coef = []
        rhs_ineqs = []
        name_ineqs = []
        for con in parseOutput.constraints:
            if con["sense"] == "=":
                list_eqs.append([])
                list_eqs_coef.append([])
                rhs_eqs.append(con["rhs"])
                name_eqs.append(con["name"][0])
                for term in con["linExpr"]:
                    list_eqs[-1].append(term["name"][0])
                    list_eqs_coef[-1].append(term["coef"])

            elif con["sense"] in ["<=", "=<"]:
                list_ineqs.append([])
                list_ineqs_coef.append([])
                rhs_ineqs.append(con["rhs"])
                name_ineqs.append(con["name"][0])
                for term in con["linExpr"]:
                    list_ineqs[-1].append(term["name"][0])
                    list_ineqs_coef[-1].append(term["coef"])

            elif con["sense"] in ["=>", ">="]:
                list_ineqs.append([])
                list_ineqs_coef.append([])
                rhs_ineqs.append(-con["rhs"])
                name_ineqs.append(con["name"][0])
                for term in con["linExpr"]:
                    list_ineqs[-1].append(term["name"][0])
                    list_ineqs_coef[-1].append(-term["coef"])

        if list_eqs:
            self.add_linear_eq_cons(
                list_cons=list_eqs,
                list_coefs=list_eqs_coef,
                rhs=rhs_eqs,
                penalty=1,
                name=tuple(name_eqs),
            )

        if list_ineqs:
            self.add_linear_ineq_cons(
                list_cons=list_ineqs,
                list_coefs=list_ineqs_coef,
                rhs=rhs_ineqs,
                penalty=1,
                name=tuple(name_ineqs),
            )
        return


def multiRemove(baseString: str, removables: str) -> str:
    """Replaces an iterable of strings in removables if removables is a string, each character is removed.

    Args:
        baseString (str): string to be filtered according to removables.
        removables (str): characters to be removed in baseString.

    Returns:
        str: resulting string after removing all removables from baseString
    """
    for r in removables:
        try:
            baseString = baseString.replace(r, "")
        except TypeError:
            raise (TypeError, "Removables must contains only strings elements.")
    return baseString


def log_enc(label: str, value_range: tuple) -> Tuple[List[str], List[int | float]]:
    """Log-encodes an integer variable into a binary value.

    Args:
        label (str): name of the variable.
        value_range (tuple): range of the integers to convert to binary.

    Returns:
        names (List[str]): list of names of the long encoded new binary variables.
        coefs (List[int | float]): respective coefficients of binary variables.
    """
    lower, upper = value_range
    assert upper > lower, "upper value should be larger than lower value"
    assert isinstance(lower, int)
    assert isinstance(upper, int)
    span = upper - lower
    d = int(np.log2(span))
    n_vars = d + 1
    names = [label + "[" + str(lower + i) + "]" for i in range(n_vars)]
    coefs = [2**i for i in range(d)]
    coefs.append((span - (2**d - 1)))

    return names, coefs
