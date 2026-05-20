# GABM.jl

*Generative Agent-Based Modeling in Julia.*

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://simoneSantoni.github.io/GABM.jl/dev)
[![CI](https://github.com/simoneSantoni/GABM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/simoneSantoni/GABM.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

GABM.jl couples a language model to [Agents.jl](https://juliadynamics.github.io/Agents.jl/).
It keeps the scheduling, spaces, and data collection of a classical agent-based
model, and replaces the agents' fixed decision rules with **generative
cognition**: agents are given a natural-language identity, accumulate
natural-language memories of what happens to them, and — when they must act —
are *asked*, in natural language, what they would do.

The cognitive architecture follows Park et al. (2023), *Generative Agents:
Interactive Simulacra of Human Behavior* — a **memory stream**, scored
**retrieval**, **reflection**, **planning**, and **decision**.

## Two kinds of agent

A **classical** agent decides with a rule the modeller wrote down:

```julia
decision = payoff(:cooperate, neighbours) > payoff(:defect, neighbours) ?
           :cooperate : :defect
```

A **generative** agent decides by consulting a language model that knows who it
is and what it has lived through:

```julia
observe!(mind, "My neighbour took water from the shared channel during the drought."; tick = 12)

d = decide(mind, llm, "The council asks whether to keep sharing water with the neighbour.";
           tick = 30, options = ["keep sharing", "cut them off"])

d.action     # "cut them off"
d.reasoning  # "They broke the channel agreement during the drought."
```

GABM.jl does not claim one is better. It makes the generative agent a
first-class citizen of an Agents.jl model, so the two can be mixed and
compared.

## Installation

GABM.jl requires Julia 1.10 or newer.

```julia
using Pkg
Pkg.add(url = "https://github.com/simoneSantoni/GABM.jl")
```

## Quick start

The example below runs **offline** against the deterministic `ScriptedLLM`
backend — no API key, no cost, fully reproducible.

```julia
using GABM

# 1. A backend. ScriptedLLM is a deterministic mock; swap in
#    PromptingToolsLLM() for a live model (OpenAI, Anthropic, Ollama, …).
llm = ScriptedLLM(prompt -> occursin("ACTION:", prompt) ?
    "ACTION: host a Valentine's Day party\nREASON: the regulars asked for it" :
    "It sounds wonderful.")

# 2. A step function: each agent perceives, then decides.
function agent_step!(agent, model)
    t = abmclock(model)
    d = decide(agent, abmllm(model), "Day $t — what should $(persona(agent).name) do?";
               tick = t)
    mind(agent).status = d.action
end

# 3. A model — an ordinary Agents.jl StandardABM carrying the backend & a clock.
model = generative_abm(; llm = llm, agent_step! = agent_step!)
add_generative_agent!(model, Persona("Isabella"; occupation = "café owner"))

# 4. Run it, and collect data with the Agents.jl API.
status(a) = mind(a).status
adf, _ = run!(model, 5; adata = [status])
```

## The cognitive loop

Every agent owns a `Mind` — a `Persona`, a `MemoryStream`, a plan, a status.
Four operations read and write that stream:

| Operation | What it does |
| --- | --- |
| `observe!` | record a perception, rating its importance |
| `reflect!` | synthesise higher-level insights from past memories |
| `plan!` | form intentions grounded in those insights |
| `decide` | choose an action from the memories relevant *now* |

All four depend on `retrieve`, which scores each memory by **recency**,
**importance**, and **relevance** and returns the few that matter — the
retrieval function of Park et al. (2023).

## Features

- **Provider-agnostic** — `PromptingToolsLLM` reaches OpenAI, Anthropic,
  Mistral, or a local Ollama model through one interface.
- **Offline and reproducible** — develop and test against `ScriptedLLM`; the
  model definition does not change when you go live.
- **Agents.jl native** — a generative model *is* a `StandardABM`; spaces,
  schedulers, `run!`, and data collection all apply unchanged.
- **Spatial or not** — use the ready-made `GenerativeAgent`, or give any
  `@agent` type a `mind` field.
- **Traceable** — reflections cite the memories they were inferred from.

## Documentation

The full manual — guides and API reference — is at
**<https://simoneSantoni.github.io/GABM.jl/dev>**.

- *Getting Started* — install and run a complete model offline.
- *User Guide* — language models, the memory stream, the cognitive loop,
  building models.
- *API Reference* — every exported type and function.

## References

- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., &
  Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human
  behavior. *UIST '23*.
- Datseris, G., Vahdati, A. R., & DuBois, T. C. (2022). Agents.jl: A performant
  and feature-full agent-based modelling software. *Simulation*, 98(4).
- Epstein, J. M., & Axtell, R. (1996). *Growing Artificial Societies*. MIT
  Press.

A fuller, annotated bibliography is in the
[documentation](https://simoneSantoni.github.io/GABM.jl/dev/references/).

## License

MIT — see [LICENSE](LICENSE).
