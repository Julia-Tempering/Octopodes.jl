work_dir = @__DIR__ 

using Pkg 
Pkg.activate(work_dir)

using Documenter 
using DocumenterVitepress
using Octopodes 
using Literate

function build(for_preview::Bool = false)

    mkpath("$work_dir/src/generated")

    Literate.markdown(
        "$work_dir/README.jl", "$work_dir/src/generated"; 
        flavor = Literate.DocumenterFlavor())

    makedocs_args_for_preview = for_preview ? 
        (; clean = false) :
        (;)

    repo = "TBD"

    makedocs(;
        modules = [Octopodes], 
        sitename="Octopodes.jl",
        format= for_preview ? 
            DocumenterVitepress.MarkdownVitepress(;
                repo = repo,
                md_output_path = ".",
                build_vitepress = false, ) : 
            DocumenterVitepress.MarkdownVitepress(
                repo = repo
            )
        ,
        pages=[
            "Overview" => "generated/README.md",
            "Input" => "input.md",
            "Reference" => "reference.md",
        ],
        makedocs_args_for_preview...
    )
end

function clean_gensyms(str)
    replace(str, r"var\"##\d+\"\." => "")
end