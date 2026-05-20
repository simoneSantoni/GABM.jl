# Cognition and LLMs

This page documents the two halves of generative cognition: the
[`AbstractLLM`](@ref) backends that supply the reasoning, and the cognitive
operations that use them to perceive, abstract, intend, and act. The
[Language Models](../guide/llm.md) and [Cognitive Loop](../guide/cognition.md)
guides cover both in narrative form.

## The backend interface

Every cognitive call reaches an [`AbstractLLM`](@ref). A backend implements
[`complete`](@ref) and, optionally, [`embed`](@ref).

```@docs
AbstractLLM
complete
embed
supports_embeddings
LLMError
```

## Backends

GABM.jl ships a live backend and a deterministic mock. The mock is the one to
develop and test against; the live backend is a one-line substitution for a
real run.

```@docs
PromptingToolsLLM
ScriptedLLM
GABM.reset!
```

## Importance rating

```@docs
rate_importance
```

## The cognitive loop

The four operations of generative cognition. Each reads or writes the agent's
[memory stream](memory.md); each accepts either a [`Mind`](@ref) or an agent
that has one.

```@docs
observe!
recall
reflect!
maybe_reflect!
plan!
decide
Decision
```

## Conversation

```@docs
converse
```

## See also

- [Language Models](../guide/llm.md) — choosing and configuring a backend.
- [The Cognitive Loop](../guide/cognition.md) — the operations in narrative
  form.
- [Models and Simulation](model.md) — calling these operations inside a step
  function.
