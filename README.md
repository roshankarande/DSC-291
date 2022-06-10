# DSC 291 Project : Numerical Linear Algebra

## Group 3 : Libraries and Tools : Julia and Python

## Members
*   Huning Chen (huc006@ucsd.edu)
*   Roshan Karande (rskarande@ucsd.edu)

## Repository Structure

```powershell
├───autodiff
├───machine_learning
├───optimization
└───other
```

### Repo Description:
-   autodiff - Code related to automatic differentiation
-   machine learning - Code related to machine learning libraries
-   optimization - Code related to optimization libraries


# Running the Code

In order to run the code one would have to install Julia and Python and also the libraries that are listed below - 

# Third Party Modules
- pyomo (Python Optimization)
- cvxpy (Python Convex Optimization)
- JuMP (Julia Optimization)
- Convex.jl (Julia Convex Optimization)
- FluxML(Julia - Machine Learning)
- SimpleChains(Julia - Machine Learning)
- Pytorch(Python - Machine Learning)
- Pytorch Autograd (Python - Automatic Differentiation)
- Jax (Python - Automatic Differentiation)
- ForwardDiff (Julia - Automatic Differentiation)
- ReverseDiff (Julia - Automatic Differentiation)
- Zygote (Julia - Automatic Differentiation)
- Enzyme(Julia - Automatic Differentiation)

# Installation Guide 

## Julia
* [Julia Installation](https://julialang.org/downloads/) - Install the appropriate verision of Julia on your machine
* [Package Installation](https://docs.julialang.org/en/v1/stdlib/Pkg/) - To install a julia package.
  *   Once Julia is installed. Type in command prompt

```julia
julia  # this will open a julia session.
]      # type this key to enter in package mode
add (package name) # to install appropriate package

# For e.g.

add BenchmarkTools
add ForwardDiff
add ReverseDiff
add Zygote
add Enzyme
add Plots
add StaticArrays
add Flux
add MLDatasets
add CUDA
add PyCall
add SimpleChains
add JuMP
add Convex
add ECOS
add Ipopt
add GLPK
add ECOS
add SCS
```

```python
# For installation of pyomo
conda install -c conda-forge pyomo
pip install pyomo
```


Once thse packages are added - 
 ### Automatic Differentiation

  * Run ``` julia ./autodiff/benchmarks.jl ``` . Note that this might take quite some time.
  * This would generate a julia_benchmarks.png file in the present directory.
  * This file would contain the results of automatic differentiation for various julia libraries
  * Run ``` ./autodiff/benchmarks.ipynb ``` for Python Jax and Pytorch benchmarks. We could not generate plots as jax is not well supported on windows. So we had to run jax under WSL on windows and integrated the run it accordingly. If one wants to compare a the results
  * Run ``` autodiff/basics.ipynb |  autodiff/forward_mode.ipynb ``` as a Jupyter notebook with Julia Kernel.


### Machine Learning
  * Run ``` julia ./machine_learning/pytorch_flux_basic1.jl ``` to see benchmarks for pytorch and Flux. They will be printed on console.
  * Run ``` julia ./machine_learning/mnist_flux_basic.jl ``` to see working of a basic flux model
  * Run ``` julia ./machine_learning/mnist_mlp.jl ``` to see working of a mnist mlp model in flux


* Optimization
  * Run ``` ./julia_convex.ipynb and ./julia_opt.ipynb ``` as Jupyter notebooks with Julia kernel to see the results
  * Run ``` ./python_opt.ipynb ``` as Jupyter notebook with Python kernel. Note that some of the solvers would have to be installed manually. [Solvers ](https://www.cvxpy.org/install/index.html) is the link to install appropriate solvers.


# Results
- autodiff/benchmarks.jl
![Auto diff benchmarks](./other/benchmarks.png)

- ml/machine_learning/pytorch_flux_basic1.jl
![Pytorch flux basic](./other/ml.png)


