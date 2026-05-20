# Personas and Agents

A generative agent is built in two layers. The **persona** is its fixed
identity — text that is legible to a language model. The **mind** wraps that
persona around a [memory stream](../api/memory.md), a plan, and a status, and
is what every cognitive function reads and writes. The **agent** is the
Agents.jl body that carries a mind through a simulation.

The split between mind and body is deliberate: the cognitive layer dispatches
on the [`mind`](@ref) accessor, never on a concrete agent type, so the same
machinery serves the ready-made non-spatial [`GenerativeAgent`](@ref) and any
spatial `@agent` type with a `mind` field equally well.

## Personas

A [`Persona`](@ref) is the part of an agent that does not change during a run.
It is rendered into every cognitive prompt by [`describe`](@ref).

```@docs
Persona
describe
```

## Minds

A [`Mind`](@ref) is the cognitive state of an agent: a persona, a
[`MemoryStream`](@ref), a plan, and a status line.

```@docs
Mind
```

## Agents

[`GenerativeAgent`](@ref) is the ready-made body for non-spatial models. For a
spatial model, define an agent with `Agents.@agent` that has a `mind` field;
the [`mind`](@ref) and [`persona`](@ref) accessors then apply to it
automatically.

```@docs
GenerativeAgent
mind
persona
```

## See also

- [Memory and Retrieval](memory.md) — the [`MemoryStream`](@ref) inside every
  mind.
- [Cognition and LLMs](cognition.md) — the operations that act on a mind.
- [Models and Simulation](model.md) — adding agents to a model.
