module Examples 
using JLD2

# TODO: use Artifacts.toml to get the large run
small_dict() = JLD2.load(joinpath(@__DIR__, "../test/IndepRuns_demo.jld2"))

end