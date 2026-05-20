# Language Models

Every cognitive operation in GABM.jl — rating a memory, synthesising a
reflection, choosing an action — ends in a call to a language model. This page
describes the interface those calls go through and the two backends GABM.jl
ships.

## The `AbstractLLM` interface

GABM.jl never calls a provider's SDK directly. All model access goes through
[`AbstractLLM`](@ref), an abstract type with a deliberately small contract. A
backend is anything that implements one method:

```julia
complete(llm, prompt; system = "", temperature) -> String
```

[`complete`](@ref) takes a prompt and returns a completion. The optional
`system` argument supplies a system prompt; `temperature` controls sampling
randomness. A backend *may* additionally implement

```julia
embed(llm, text) -> Vector{Float64}
```

and signal that it has done so by defining
[`supports_embeddings`](@ref)`(llm) = true`. Embeddings are used only to score
the relevance of memories during [`retrieve`](@ref); a backend without them
still runs a complete model, because retrieval falls back to lexical overlap.

This narrow interface is the reason a generative model is *portable*. The
simulation is written against `AbstractLLM`, so the same model definition runs
against a deterministic mock during development and a live provider in
production, with no change to a single cognitive call.

## The scripted backend

[`ScriptedLLM`](@ref) is a deterministic mock that never touches the network.
It is the backend to develop and test against: it is free, instant, and exactly
reproducible. It has three forms.

A **queue** returns a fixed list of replies, one per call, and raises
[`LLMError`](@ref) when exhausted — so a missing reply fails loudly:

```julia
llm = ScriptedLLM(["ACTION: wave", "ACTION: smile"])
complete(llm, "...")   # "ACTION: wave"
complete(llm, "...")   # "ACTION: smile"
```

A **responder** computes each reply from the prompt. This is the most useful
form for a model with many cognitive calls, because one function can answer
every kind of prompt the simulation issues:

```julia
llm = ScriptedLLM() do prompt
    occursin("Rating:", prompt) ? "7" :
    occursin("ACTION:", prompt) ? "ACTION: cooperate\nREASON: trust pays" :
    "I see."
end
```

A **constant** returns the same string for every call — useful for the
simplest smoke tests:

```julia
llm = ScriptedLLM("yes")
```

Every prompt a `ScriptedLLM` receives is appended to its `log` field, so a test
can assert on exactly what the cognitive layer asked:

```julia
llm.log    # Vector{String} of all prompts seen
```

`GABM.reset!` rewinds a queue-backed `ScriptedLLM` and clears its log, so a
deterministic scenario can be re-run.

## The live backend

[`PromptingToolsLLM`](@ref) is the production backend. It forwards to
[PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl), which is
itself provider-agnostic: a single `PromptingToolsLLM` can drive Anthropic,
OpenAI, Mistral, Groq, or a model served locally by Ollama. The provider is
selected entirely by the `model` string.

```julia
llm = PromptingToolsLLM(model = "claudeh", temperature = 0.7)
```

The keyword arguments are:

| Argument | Default | Meaning |
|----------|---------|---------|
| `model` | `"claudeh"` | Chat model, as a PromptingTools.jl name or alias |
| `embedding_model` | `"text-embedding-3-small"` | Model used by [`embed`](@ref) |
| `temperature` | `0.7` | Default sampling temperature |
| `max_tokens` | `1024` | Maximum tokens per completion |

API keys are read from the environment by PromptingTools.jl, not by GABM.jl —
set `ENV["ANTHROPIC_API_KEY"]`, `ENV["OPENAI_API_KEY"]`, and so on, or register
custom models, following the PromptingTools.jl documentation. When a provider
call fails, the error is wrapped in an [`LLMError`](@ref) that names the model
involved.

## Choosing a backend

A typical project uses both. The model is developed and unit-tested against a
`ScriptedLLM` — fast iteration, no cost, reproducible failures — and then run
for real by substituting a `PromptingToolsLLM`:

```julia
llm = run_for_real ? PromptingToolsLLM(model = "claudeh") :
                     ScriptedLLM(my_responder)

model = generative_abm(; llm = llm, agent_step! = agent_step!)
```

Because the backend is just one object passed to [`generative_abm`](@ref),
nothing else in the model is aware of which one is in use.

## Temperature and reproducibility

A classical Agents.jl model is made reproducible by seeding its random number
generator. A generative model has a second source of randomness: the language
model's own sampling, governed by `temperature`. Two settings bracket the
useful range.

- `temperature = 0.0` makes a provider as close to deterministic as it can be,
  and is the right choice for importance ratings and for runs intended to be
  comparable.
- A higher temperature (`0.7`–`1.0`) produces the behavioural variety that
  makes a population of generative agents heterogeneous even when their
  personas are similar.

For *exact* reproducibility — a regression test, a figure in a paper — use a
`ScriptedLLM`: it is the only backend whose output does not depend on a remote
service at all.

## Implementing a custom backend

Any new backend is a subtype of [`AbstractLLM`](@ref) with a `complete` method.
A backend that wraps an in-house model, adds caching, or logs token usage need
implement nothing more:

```julia
struct CachingLLM <: AbstractLLM
    inner::AbstractLLM
    cache::Dict{String,String}
end

function GABM.complete(llm::CachingLLM, prompt::AbstractString; kwargs...)
    get!(llm.cache, prompt) do
        complete(llm.inner, prompt; kwargs...)
    end
end
```

Such a backend drops straight into [`generative_abm`](@ref); the cognitive
layer cannot tell the difference.

## Where to next

- [The Memory Stream](memory.md) — what the cognitive layer reads and writes.
- [The Cognitive Loop](cognition.md) — the operations that issue these calls.
- [Cognition and LLMs](../api/cognition.md) — the API reference for the
  backend types.
