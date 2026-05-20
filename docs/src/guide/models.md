# Building Models

The previous guides covered the parts in isolation — a backend, a memory
stream, the cognitive loop. This page assembles them into a complete,
runnable agent-based model and then shows how to take it spatial.

## A generative model is an Agents.jl model

GABM.jl adds no simulation engine of its own. A generative model *is* an
Agents.jl `StandardABM`; [`generative_abm`](@ref) only guarantees it carries
the two things the cognitive layer needs:

- a shared [`AbstractLLM`](@ref) backend, reachable with [`abmllm`](@ref); and
- an integer clock starting at `0`, read with [`abmclock`](@ref) and advanced
  once per step by [`clock_step!`](@ref).

Everything else — stepping, running, data collection, the random number
generator — is Agents.jl, unchanged and fully available.

## A worked model: deciding to adopt

We build a small model of social adoption. A handful of villagers each decide,
tick by tick, whether to adopt a new irrigation method; they talk to one
another, and what they hear shapes what they later decide. The whole model runs
offline against a [`ScriptedLLM`](@ref).

### The agents

The villagers have no position — they form a single conversational community —
so the ready-made non-spatial [`GenerativeAgent`](@ref) is the right body. Each
gets a [`Persona`](@ref):

```julia
using GABM

personas = [
    Persona("Wbishere"; occupation = "smallholder", traits = ["cautious", "respected"]),
    Persona("Adsila";   occupation = "smallholder", traits = ["curious", "early adopter"]),
    Persona("Catori";   occupation = "smallholder", traits = ["skeptical", "frugal"]),
]
```

### The backend

For development we use a prompt-aware [`ScriptedLLM`](@ref). One responder
function answers every kind of cognitive prompt the model will issue:

```julia
function responder(prompt)
    if occursin("Rating:", prompt)
        return "6"
    elseif occursin("ACTION:", prompt)
        return occursin("early adopter", prompt) ?
            "ACTION: adopt the new method\nREASON: the trial yields look strong" :
            "ACTION: wait and watch the neighbours\nREASON: the method is unproven"
    else
        return "I have been thinking about the new irrigation method."
    end
end

llm = ScriptedLLM(responder)
```

### The step function

The cognitive loop lives in `agent_step!`. Each tick a villager observes the
current state of opinion around it, then [`decide`](@ref)s between two explicit
options — the discrete choice set that keeps the model analysable:

```julia
function agent_step!(agent, model)
    t   = abmclock(model)
    llm = abmllm(model)

    adopters = count(a -> mind(a).status == "adopter", allagents(model))
    observe!(agent, "At time $t, $adopters villagers have adopted the method.";
             tick = t, importance = 4.0)

    d = decide(agent, llm,
               "Villager $(persona(agent).name) must decide about the method.";
               tick = t, options = ["adopt the new method",
                                     "wait and watch the neighbours"])

    if occursin("adopt", d.action)
        mind(agent).status = "adopter"
    end
    return nothing
end
```

The model step is left as the default [`clock_step!`](@ref), which advances the
tick counter.

### Assembling and populating

```julia
model = generative_abm(; llm = llm, agent_step! = agent_step!)

for p in personas
    add_generative_agent!(model, p; status = "undecided")
end
```

[`add_generative_agent!`](@ref) builds a [`Mind`](@ref) around each persona and
adds a [`GenerativeAgent`](@ref) to the model.

### Running and collecting data

The model is an ordinary Agents.jl model, so `run!` drives it and collects
results into a `DataFrame`. We record each agent's `status` every tick:

```julia
status(agent) = mind(agent).status

adf, mdf = run!(model, 10; adata = [status])
adf      # one row per agent per step — the adoption curve
```

`adf` is a standard `DataFrame`; group it, plot it, or compare runs exactly as
with any Agents.jl model.

## Adding conversation

Adoption spreads through talk. To let it, have agents [`converse`](@ref) before
they decide — a conversation writes a shared memory into both participants, and
[`decide`](@ref) will retrieve it:

```julia
function agent_step!(agent, model)
    t, llm = abmclock(model), abmllm(model)
    others = filter(a -> a !== agent, collect(allagents(model)))
    if !isempty(others)
        partner = rand(abmrng(model), others)
        converse(agent, partner, llm; topic = "the new irrigation method",
                 turns = 2, tick = t)
    end
    d = decide(agent, llm, "Decide about the method."; tick = t,
               options = ["adopt the new method", "wait and watch the neighbours"])
    occursin("adopt", d.action) && (mind(agent).status = "adopter")
    return nothing
end
```

Using `abmrng(model)` rather than a bare `rand` keeps the run reproducible:
seed the model's generator and the *sequence of conversation partners* is
fixed, even though the language model's replies may not be.

## Reflection between ticks

The default [`generative_step!`](@ref) calls [`maybe_reflect!`](@ref), so
agents consolidate memories on their own. A custom `agent_step!` replaces that
default, so add the call back if you want it:

```julia
function agent_step!(agent, model)
    t, llm = abmclock(model), abmllm(model)
    # ... perceive, converse, decide ...
    maybe_reflect!(agent, llm; tick = t, threshold = 80.0)
    return nothing
end
```

## Spatial models

When agents *do* have a position, the only change is the agent type. The
[`GenerativeAgent`](@ref) is non-spatial; for a spatial model define your own
Agents.jl agent with `Agents.@agent` and give it a `mind` field:

```julia
using Agents, GABM

@agent struct Resident(GridAgent{2})
    mind::Mind
end
```

The cognitive functions need no change: they dispatch on the [`mind`](@ref)
accessor, and the fallback `mind(agent) = agent.mind` already covers any agent
with that field. Pass the type and a space to [`generative_abm`](@ref):

```julia
model = generative_abm(; llm = llm,
                          agent_type = Resident,
                          space = GridSpace((20, 20)),
                          agent_step! = agent_step!)

add_agent!(model, Mind(Persona("Resident 1")))   # standard Agents.jl call
```

Inside `agent_step!`, the spatial neighbourhood becomes the agent's perception
— `nearby_agents`, `nearby_positions`, and the rest of the Agents.jl spatial
API are what `perceive` and `situation` are built from. The same pattern
applies to `GraphSpace` for network models and `ContinuousSpace` for
off-lattice ones.

## The shape of every model

However elaborate the world, a generative model has the same four moving
parts:

```
  generative_abm(...)   ── the StandardABM, carrying llm + clock
        │
        ├── agent_type   ── GenerativeAgent, or an @agent with a `mind`
        │
        ├── agent_step!  ── perceive → (reflect) → (plan) → decide → apply
        │
        └── run!(model, n; adata, mdata)   ── ordinary Agents.jl analysis
```

Master those and any GABM.jl model is a variation on them.

## Where to next

- [Models and Simulation](../api/model.md) — the API reference for
  [`generative_abm`](@ref) and the model helpers.
- [The Cognitive Loop](cognition.md) — the operations called inside
  `agent_step!`.
- [References](../references.md) — the modelling literature behind the design.
