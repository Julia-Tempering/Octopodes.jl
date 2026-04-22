```@meta
CurrentModule = Octopodes
```

## Overview

Octopodes take as input independent MCMC runs, one for each star/system. 
Each of these MCMC runs approximates the posterior of exoplanet(s) parameters given 
the data of only that system
(for example, in our work, we use 
[Octofitter](https://sefffal.github.io/Octofitter.jl/stable/) to produce these independent MCMC runs).

We assume the results of these MCMC runs have been serialized as a Julia `Dict` in a format described 
below.
From a user's point of view, the main point of this page is to 
explain the expected organization of that `Dict`. 

Internally, we transform that `Dict` into a [`IndepRuns`](@ref) object. 
We then bin the [`IndepRuns`](@ref) into a [`BinnedIndepRuns`](@ref) object. 
We give some information on that lower level process as well in this page. 


## Expected format for the input `Dict`

Here is an example:

```@example dict
using Octopodes
dict = Octopodes.Examples.small_dict()
```

If the `Dict` contains other keys it is not a problem. 

We assume the prior used in the independent MCMC runs are 
specified using [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) objects. 

At the moment we assume that the independent MCMC runs use
uniform priors over log10 periods (`log_P_yr_prior`) 
and log10 mass ratios (`log_q_prior`). 
The support of these priors are used to construct correct
discretization bins. 

The prior on the number of companions can be any discrete 
distribution on `0:max_n_companions`. 

Precise infomation on all these priors is essential to have 
correct joint inference as we need to "cancel them out" in 
a precise way.

The dictionary's `star_data` key should point to a `Vector` where each item in the `Vector` corresponds to an independent MCMC run for one system:

```@example dict
typeof(dict["star_data"])
```

Here are the fields required for each of these items:

```@example dict
traces_for_one_system = first(dict["star_data"])
typeof(traces_for_one_system)
```

Again, if your NamedTuple has other fields this is 
not a problem.

The period (`log_P_yr`) and mass (`log_q`) are encoded as matrices where the number of rows is the maximum number 
of companions allowed, and the number of columns is the number of MCMC samples. If at iteration `i`, `n_planets` is `n`, we will ignore the elements 
in `log_P_yr[:, i]` past index `n` (this setup comes from the use 
of [model saturation in the transdimensional MCMC context](https://www.stat.ubc.ca/~bouchard/courses/stat520-sp2021-22/T8-model-selection-basics.html#(9))). 

We assume they are in `log10` scale. 

```@example dict
traces_for_one_system.log_P_yr
```

## Loading and validating the data

Use:

```@example dict
runs = IndepRuns(dict)
```

This will perform several validation checks. 


## Binning 

To bin the data, proceed as follows. 

First, create a binning by choosing how many grid points for the mass-ratio and period parameters:

```@example dict
b = Binning(runs, n_log_P_yr_intervals = 20, n_log_q_intervals = 20)
```

Then use [`bin`](@ref) to perform the binning. You can select star subsets using 
the `star_selector` option.
The `bin` function also shuffle the samples. See [`bin`](@ref) 
for details. 

```@example dict
using Random

binned = bin(b, runs; star_selector = (x -> startswith(x, "HIP")))
```

## Binarizing

To test our code, it is useful to consider the special 
case where we collapse the number of companions into two cases, 
no companion versus at least one companion, and 
where we also collapse all masses and periods in the same bin. 
This makes each independent MCMC run over a binary space 
(at least one companion versus no companion). 
In that case the true posterior can be computed analytically. 

To re-bin into binary samples, use:

```@example dict
Octopodes.binarize(binned)
```