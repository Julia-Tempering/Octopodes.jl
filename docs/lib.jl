work_dir = @__DIR__ 

parent_path, _ = splitdir(work_dir)
@assert parent_path == abspath(pwd()) "Run doc preview/build with pwd() at the root of the repo"

using Pkg 
Pkg.activate(work_dir)

using Documenter 
using DocumenterVitepress
using Octopodes 
using Literate

function build(for_preview::Bool = false)

    generated = "$work_dir/src/generated"
    if isdir(generated)
        rm(generated; recursive=true)
    end
    mkpath(generated)

    Literate.markdown(
        "$work_dir/README.jl", generated; 
        flavor = Literate.DocumenterFlavor())
    mv("$generated/README.md", "$work_dir/src/index.md", force=true)

    makedocs_args_for_preview = for_preview ? 
        (; clean = false) :
        (;)

    repo = "github.com/Julia-Tempering/Octopodes"

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
            "Overview" => "index.md",
            "Input" => "input.md",
            "Reference" => "reference.md",
        ],
        makedocs_args_for_preview...
    )

    if !for_preview 
        DocumenterVitepress.deploydocs(;
            repo,
            devbranch = "main",
            push_preview = true
        )
    end

    # since it is generated, try to avoid leaving it around, someone could edit it and then lose their changes
    rm("$work_dir/src/index.md")
end

function clean_gensyms(str)
    replace(str, r"var\"##\d+\"\." => "")
end
