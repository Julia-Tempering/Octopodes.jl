include("lib.jl")

# README.md 
Literate.markdown(
    "$work_dir/README.jl", dirname(work_dir); 
    execute = true,
    postprocess=clean_gensyms,
    flavor = Literate.CommonMarkFlavor())

# Documenter.jl site 
build()