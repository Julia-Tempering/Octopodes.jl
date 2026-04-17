include("lib.jl")

build(true)

cd(work_dir) do
    DocumenterVitepress.dev_docs("build", md_output_path = "")
end