"""
Discretization of the `(log_P_yr, log_q)` into a hierarchy of regular grids. 

The highest level grid is specified using `StepRangeLen` (see  [`interval_index`](@ref)), which is then 
reused at each sub-level.
"""
struct MultiBinning{G <: StepRangeLen, S <: Tuple}
	multibinning::Vector{Binning{G, S}}
	# derived value
	n_levels::Int
end

function MultiBinning(runs::IndepRuns; n_log_P_yr_intervals::Int, n_log_q_intervals::Int, n_levels::Int) where {G, S}
	bh = Vector{Binning{G,S}}()
	for i in 0:n_levels-1
		push!(bh, Binning(runs, n_log_P_yr_intervals = 2^i*n_log_P_yr_intervals, n_log_q_intervals = 2^i*n_log_q_intervals))
	end
	return MultiBinning(bh, n_levels)
end

function MultiBinning(log_P_yr_grid::StepRangeLen, log_q_grid::StepRangeLen; n_levels::Int) where {G, S}
	logqmin = first(log_q_grid)
	logqmax = last(log_q_grid)
	logqn = length(log_q_grid)-1
	logpmin = first(log_P_yr_grid)
	logpmax =  last(log_P_yr_grid)
	logpn =  length(log_P_yr_grid)-1
	bh = Vector{Binning{G,S}}()
	for i in 0:n_levels-1
		push!(bh, Binning(runs, range(logpmin, logpmax, 2^i*logpn + 1), range(logqmin, logqmax, 2^i*logqn+1)))
	end
	return MultiBinning(bh, n_levels)
end

struct MultiBinnedIndepRuns{B <: Binning, M <: Matrix, V}
	multibinnedruns::Vector{BinnedIndepRuns{B,M,V}}
end

function bin(
        b::MultiBinning, runs::IndepRuns; 
        star_selector = (star_name::String -> true), 
        shuffle_rng::Union{Nothing, AbstractRNG} = Xoshiro(1)
        ) 
   	brh = Vector{BinnedIndepRuns{B,M,V}}()
   	for n in 1:b.n_levels
		push!(brh, bin(b.multibinning[n], runs, star_selector, shuffle_rng))
	end
	return MultiBinnedIndepRuns(brh)
end




