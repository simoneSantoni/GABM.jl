# Models and Simulation

A generative model is an Agents.jl `StandardABM` carrying a language-model
backend and a clock. This page documents the functions that build such a
model, populate it, and reach its shared state from inside a step function.
The [Building Models](../guide/models.md) guide assembles them into a complete
simulation.

## Constructing a model

[`generative_abm`](@ref) builds the `StandardABM` and guarantees it carries an
[`abmllm`](@ref) backend and an [`abmclock`](@ref) tick counter.

```@docs
generative_abm
add_generative_agent!
```

## Accessing model state

Inside a step function, these reach the state every generative model shares.

```@docs
abmllm
abmclock
```

## Step functions

[`clock_step!`](@ref) is the default `model_step!`; [`generative_step!`](@ref)
is the default `agent_step!`. A model with custom dynamics replaces the latter
and, if it still wants the clock advanced, calls the former.

```@docs
clock_step!
generative_step!
```

## See also

- [Building Models](../guide/models.md) — a complete worked model.
- [Cognition and LLMs](cognition.md) — the operations a step function calls.
- The [Agents.jl documentation](https://juliadynamics.github.io/Agents.jl/) —
  for spaces, schedulers, `run!`, and data collection, all of which apply to a
  generative model unchanged.
