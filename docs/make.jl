using VUMPSAutoDiff
using Documenter

DocMeta.setdocmeta!(VUMPSAutoDiff, :DocTestSetup, :(using VUMPSAutoDiff); recursive=true)

makedocs(;
    modules=[VUMPSAutoDiff],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    sitename="VUMPSAutoDiff.jl",
    format=Documenter.HTML(;
        canonical="https://tangwei94.github.io/VUMPSAutoDiff.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/tangwei94/VUMPSAutoDiff.jl",
    devbranch="main",
)
