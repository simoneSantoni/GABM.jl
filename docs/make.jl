using Documenter
using GABM

DocMeta.setdocmeta!(GABM, :DocTestSetup, :(using GABM); recursive=true)

makedocs(
    sitename = "GABM.jl",
    modules = [GABM],
    authors = "Simone Santoni",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://simoneSantoni.github.io/GABM.jl",
        edit_link = "main",
        assets = ["assets/custom.css", "assets/favicon.ico"],
    ),
    repo = "https://github.com/simoneSantoni/GABM.jl/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "User Guide" => [
            "Language Models" => "guide/llm.md",
            "The Memory Stream" => "guide/memory.md",
            "The Cognitive Loop" => "guide/cognition.md",
            "Building Models" => "guide/models.md",
        ],
        "API Reference" => [
            "Personas and Agents" => "api/agents.md",
            "Memory and Retrieval" => "api/memory.md",
            "Cognition and LLMs" => "api/cognition.md",
            "Models and Simulation" => "api/model.md",
        ],
        "References" => "references.md",
    ],
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(
    repo = "github.com/simoneSantoni/GABM.jl.git",
    devbranch = "main",
    versions = [
        "stable" => "dev",
        "dev" => "dev",
    ],
    push_preview = true,
)
