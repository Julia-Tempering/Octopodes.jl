module Examples 
using JLD2

# TODO: use Artifacts.toml
small_dict() = JLD2.load(joinpath(@__DIR__, "../test/IndepRuns_demo.jld2"))

end