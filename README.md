# DQUBO - Dell QUBO

DQUBO is a simple but efficient open-source Python library for modelling QUBO problems and their subsequent use in any solver.

It implements efficiently operations to define objective functions, linear and quadratic constraints of any QUBO problem. Once the problem is modelled into DQUBO, it is compiled to yield a resulting dictionary of the problem. This dictionary stands for, at each key, a qubit pair with a coefficient as value, where this structure is mandatory as input format for many (Simulated) Quantum Annealing solvers. 


DQUBO also provides an unified abstraction layer for the main QUBO solvers, comprising all pipeline steps to solve a QUBO problem from its modelling to obtaining a final solution.

Otherwise, DQUBO permits that the resulting dict can be returned to users, allowing them to worry only on handcrafted manipulations on the QUBO solvers and/or its usage into non-supported solvers.

# Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [DQUBO Plus](#dqubo-plus)
* [Support](#support)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [Authors and Acknowledgement](#authors-and-acknowledgment)
* [License](#license)

## Installation

Before installing DQUBO, it is necessary to install the Poetry package. Poetry is an advanced Python package manager which supports the instalation and the development of DQUBO. According to its official documentation, poetry *must not* be installed using pip. To install Poetry for your user, you must do the following command:

``` curl -sSL https://install.python-poetry.org | python3 - ```

Ensure you have Python >= 3.8 installed for this step. For more information on Poetry's installation, please visit [the official documentation](https://python-poetry.org/docs/#installing-with-the-official-installer).

We recommend using Python >= 3.12, and individual environment and Poetry to install all DQUBO dependencies.
The following command installs all necessary packages, where *--no-root* assures that dqubo won't be installed in your environment:

```poetry install --only=main  --no-root```


In some settings, the user only wants to install a specific solver (and its dependencies) instead of installing all solvers available on DQUBO.

For only enabling OpenJij:

```poetry install -E "openjij"```

For only enabling Neal (D-Wave): 

```poetry install -E "dwave"```

For enabling both OpenJij and Neal: 

```poetry install --all-extras```

### Full installation

For a full installation, which includes benchmarking DQUBO against PyQUBO and unit tests dependencies (dev dependencies), execute:

```poetry install ```


### Installing with pip

Yet not recommended, the user may use ```pip install -r requirements.txt``` at the root folder to install all necessary packages. 

### compiling package
for compile .so files run:

python3 setup.py build_ext --inplace


## Usage

The objective of the code below is twofold: (1) to validate your installation and (2) to provide a end-to-end tour over all features of DQUBO.


```
from dqubo import DQUBO

# Create DQUBO object.
# You may change the booleang flags of the constructor if the problem
# is sparse and you want to enable a parallel compilation to improve speed.
model = DQUBO(sparse=False, enable_async=True)

print("Running example")

# We will declare the following problem
# min x^T*Q*x  s.t. Ax = b , e <= Cx <= d , x^T*Qc*x = 0

That is a quadratic objective with linear equality, linear inequality and quadratic equality constraints.

# Adding three variables to our model
model.add_variables(list_var=[f"x_{i}" for i in range(3)])

# Add an objective function to minimize: 2*x_0*x_0 + 3*x_0*x_2 + 8*x_2*x_1
# Minimization is defined by default.
# Otherwise, multiplying the whole expression by -1 will turn it into maximization
list_q_pairs = [["x_0", "x_0"], ["x_0", "x_2"], ["x_2", "x_1"]]
list_q_coefs = [2, 3, 8]
model.add_qubo_obj(list_pairs=list_q_pairs, list_coefs=list_q_coefs)

# Add two linear equality constraints at once.
# Both constrains the same penalty factor (i.e., 10), otherwise, add them individually to specify each penalty factor.
# x_0+4*x_1+5*x_2 = 10 and 6*x_0+9*x_2 = 20.
# A parameter "name" can be used to specify the same group constraint name (internally, the group of constraints is a matrix).
list_cons = [["x_0", "x_1", "x_2"], ["x_0", "x_2"]]
list_coefs = [[1, 4, 5], [6, 9]]
rhs = [10, 20]
model.add_linear_eq_cons(
    list_cons=list_cons, list_coefs=list_coefs, rhs=rhs, penalty=10.0
)

# Add two linear inequality constraints 0 <= x_1+3*x_2 <= 15 ; 2 <= 4*x_0+5*x_1 <= 25.
# Linear inequalities will have the same behavior as Linear equalities considering parameters "penalty" and "name" .
list_cons = [["x_1", "x_2"], ["x_0", "x_1"]]
list_coefs = [[1, 3], [4, 5]]
lhs = [0, 2]
rhs = [15, 25]
model.add_linear_ineq_cons(
    list_cons=list_cons, list_coefs=list_coefs, penalty=1.0, lhs=lhs, rhs=rhs
)

# Add two quadratic constraints 2*x_1*x_10" +3*x_0*x_2 + 8*x_2*x_1
list_q_pairs = [["x_1", "x_1"], ["x_1", "x_2"]]
list_q_coefs = [5, 10]
model.add_quadratic_cons(list_pairs=list_q_pairs, list_coefs=list_q_coefs, penalty=1.0)

# The command below will compile all added objectives and constraints into a single expression into a Hamiltonian.
# If desired, the user may change the penalty factors for each input constreaint by indicating the constraint name and its new penalty.
model.compile()

# get_dict and get_dict_with_labels functions will return the dict form of the compiled Hamiltonian.
# The latter function will include variable names in str format and slack variables for inequalities while the former will return all variables as integer
qubo, offset = model.get_dict()
print(f"get_dict():\nQUBO: {qubo}, offset: {offset}")
qubo2, offset2 = model.get_dict_with_labels()
print(f"get_dict_with_labels():\nQUBO: {qubo2}, offset: {offset2}")
```
### Changing penalties
During the compilation is possible to change the previous declared penalties by using a dictionary with the names of the constraints and the new penalties.

```
with_penalty = {0:7.0,1: 3.0,2:4.0}
model.compile( with_penalty = with_penalty)
```

### Checking constraints
With a given solution x_dict, dqubo can check the feasibility of this solution with the constraints.

```
checked_cons = model.check_constraints(x_dict, print_results=True)
```
### Evaluating objective function and energy
DQUBO can evaluate the objective function and energy for a given solution x_dict.

```
obj = model.objective(x_dict)
energy = model.energy(x_dict)
```

### Fixing variables
DQUBO can fix some variables in the process of create the QUBO dictionary, first you need to inform to DQUBO which variables will be fixed.
```
fixed_vars = {"x[1]": 1, "x[3]": 0}

model.fix_vars(fixed_vars)
```
then generate the reduced QUBO dictionary using get_dict() function.

### Supported QUBO solvers

DQUBO will include regularly (Simulated) Quantum Annealing and Simulated Annealing solvers. 

A list of supported QUBO solvers into DQUBO is below:

- OpenJij: https://github.com/OpenJij/OpenJij
    
    ``` 
    from dqubo import OpenJij
    oj_solution = model.solve(OpenJij(), num_reads=1, num_sweeps=10)

    ```

- Neal (D-Wave): https://docs.ocean.dwavesys.com/projects/neal/en/latest/index.html
    
    ``` 
    from dqubo import Neal
    neal_solution = model.solve(Neal(), num_reads=1, num_sweeps=10, beta_range=[10, 200])

    ```

## DQUBO Plus

DQUBO Plus is the enterprise version for DQUBO. It differentiates from DQUBO by adding extra capabilities for addressing QUBO problems and their intelligent orchestration among multiple solvers. 

Features:

- QUBO cutting: subdividing the QUBO into multiple parts, thus, allowing larger problems to be addressed into resource-constrained machines.
 

## Support

Users may create a new issue to get support about DQUBO. Please, access *Issues* tab for starting a new one.

## Roadmap

DQUBO has been constantly enhanced by the implementation of new features and performance improvements in existing ones.

The team has identified the following features to be implemented in a mid-term basis:

- Improved parallelization strategy for compiling QUBOs

## Contributing

DQUBO team is open to contributions from any interested user by means of Pull Requests.

Each new Pull Request (PR) will be carefully evaluated and we expected that unit tests are considered.

Prior to create a PR, we encourage the following steps:

1. Make sure your code is running with the latest version of DQUBO (updates on DQUBO will be done regularly).

2. Have a full installation of the poetry in your environment. It permits you to execute existing unit tests using the following command from the root folder:
``` pytest --cache-clear --cov=dqubo --cov-report term tests/test_all.py --verbosity 1```

3. If all tests passed in the previous step, make sure you create new unit tests for your PR. Reexecute the command of step 2 to make sure your tests passed. New unit tests must account not only aspects of code coverage, but also ensuring that all possible corner cases are addressed. 

## Authors and acknowledgment
Dell Research Team

## License
[Apache 2.0](LICENSE)

