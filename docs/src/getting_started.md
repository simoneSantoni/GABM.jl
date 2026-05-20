# Getting Started

This page installs GABM.jl and builds a complete generative agent-based model.
The whole walkthrough runs **offline**: it uses the deterministic
[`ScriptedLLM`](@ref) backend, so it needs no API key, costs nothing, and
produces the same result every time. Swapping in a live model is a one-line
change, shown at the end.

## Installation

GABM.jl requires Julia 1.10 or newer. From the Julia REPL:

```julia
using Pkg
Pkg.add(url = "https://github.com/simoneSantoni/GABM.jl")
```

The package pulls in [Agents.jl](https://juliadynamics.github.io/Agents.jl/)
for the simulation engine and
[PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) for live
language-model access. Load it with:

```julia
using GABM
```

## A first mind

The smallest unit of GABM.jl is a [`Mind`](@ref): a [`Persona`](@ref) — the
agent's fixed identity — wrapped around a [`MemoryStream`](@ref) that will fill
up as the simulation runs.

```julia
isabella = Mind(Persona("Isabella Rodriguez";
    age = 34,
    occupation = "café owner",
    traits = ["warm", "organised", "ambitious"],
    background = "Runs Hobbs Cafe and is thinking about hosting an event."))
```

Nothing has happened to Isabella yet, so her memory stream is empty. We give
her an experience with [`observe!`](@ref), passing the current tick:

```julia
observe!(isabella, "The café was busy and three regulars asked for a place " *
                   "to celebrate Valentine's Day."; tick = 1, importance = 7.0)
```

The `importance` argument is the memory's poignancy on the Park et al. 1–10
scale. Omit it and — if a language-model backend is supplied — GABM.jl will
ask the model to rate it; see [`rate_importance`](@ref).

## A scripted backend

Cognition needs a backend. For development we use [`ScriptedLLM`](@ref) in its
*responder* form: a function from prompt to reply. It inspects each prompt and
returns a plausible answer, so the whole model is deterministic.

```julia
function backend(prompt)
    if occursin("ACTION:", prompt)
        return "ACTION: host a Valentine's Day party at the café\n" *
               "REASON: regulars asked for somewhere to celebrate"
    elseif occursin("brief, concrete plan", prompt)
        return "Decorate the café\nInvite the regulars\nPrepare a set menu"
    else
        return "It sounds like a wonderful idea."
    end
end

llm = ScriptedLLM(backend)
```

A real run would instead use `PromptingToolsLLM()`; everything below is
identical either way.

## Asking the agent to decide

[`decide`](@ref) is the act step of the cognitive loop. It retrieves the
memories most relevant to a situation, builds a prompt, and returns a
[`Decision`](@ref).

```julia
d = decide(isabella, llm,
    "It is the first of February. Isabella is considering how to use the café."
    ; tick = 5)

d.action     # "host a Valentine's Day party at the café"
d.reasoning  # "regulars asked for somewhere to celebrate"
```

The decision is grounded in the memory we planted: Isabella acts on the
remembered request from her regulars, not on a rule we wrote down.

## Assembling a model

A single mind is useful for prototyping, but a simulation needs a population, a
clock, and a step function. [`generative_abm`](@ref) builds an Agents.jl
`StandardABM` carrying a shared backend and a tick counter.

```julia
function agent_step!(agent, model)
    tick = abmclock(model)
    d = decide(agent, abmllm(model),
               "Day $tick at the café — what should $(persona(agent).name) do?";
               tick = tick)
    mind(agent).status = d.action
    observe!(agent, "On day $tick I decided to $(d.action)."; tick = tick)
end

model = generative_abm(; llm = llm, agent_step! = agent_step!)
```

We populate it with [`add_generative_agent!`](@ref), which builds a
[`GenerativeAgent`](@ref) — the ready-made non-spatial agent type — around a
persona:

```julia
add_generative_agent!(model, Persona("Isabella Rodriguez";
    occupation = "café owner", traits = ["warm", "ambitious"]))
add_generative_agent!(model, Persona("Tom Moreno";
    occupation = "pharmacist", traits = ["sociable", "busy"]))
```

## Running it

The model is an ordinary Agents.jl model, so it steps and runs with the
Agents.jl API (re-exported by GABM.jl for convenience):

```julia
step!(model, 3)        # advance three ticks
abmclock(model)        # 3
```

To collect data, use Agents.jl's `run!` with an agent-data specification — here
we record each agent's `status` every tick:

```julia
status(agent) = mind(agent).status
adf, _ = run!(model, 3; adata = [status])
adf        # a DataFrame: one row per agent per step
```

`adf` is a normal `DataFrame`; everything Agents.jl offers for analysis,
plotting, and reproducibility applies unchanged.

## Going live

To run the same model against a real language model, change one line:

```julia
llm = PromptingToolsLLM(model = "claudeh")
```

[`PromptingToolsLLM`](@ref) forwards to
[PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl), which reads
provider API keys from your environment (for Anthropic, set
`ENV["ANTHROPIC_API_KEY"]`; see the PromptingTools.jl documentation for other
providers). The model definition, the step function, and the analysis code do
not change at all.

## Where to next

- [Language Models](guide/llm.md) — the backends in depth, and how to choose
  and configure them.
- [The Memory Stream](guide/memory.md) — how memories are stored, scored, and
  retrieved.
- [The Cognitive Loop](guide/cognition.md) — [`observe!`](@ref),
  [`reflect!`](@ref), [`plan!`](@ref), and [`decide`](@ref).
- [Building Models](guide/models.md) — spatial models, schedulers, and larger
  simulations.
