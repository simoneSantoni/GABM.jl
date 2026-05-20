# Memory and Retrieval

The memory stream is the central data structure of a generative agent: an
append-only, timestamped log of natural-language records. This page documents
the types that make it up and the [`retrieve`](@ref) function that selects from
it. The [Memory Stream](../guide/memory.md) guide explains the retrieval
scoring in narrative form and works it through an example.

## Memory entries

A [`MemoryEntry`](@ref) is one record in a stream — an observation, a
reflection, or a plan, as enumerated by [`MEMORY_KINDS`](@ref).

```@docs
MemoryEntry
MEMORY_KINDS
```

## The memory stream

A [`MemoryStream`](@ref) holds an agent's entries together with the parameters
— a recency `decay` and three retrieval `weights` — that govern how they are
recalled.

```@docs
MemoryStream
remember!
```

## Retrieval

[`retrieve`](@ref) is the function of Park et al. (2023): it scores every
candidate memory on recency, importance, and relevance, and returns the
highest-scoring few. [`embed_memories!`](@ref) populates the embeddings that
let the relevance term use cosine similarity rather than the lexical fallback.

```@docs
retrieve
embed_memories!
```

## See also

- [The Memory Stream](../guide/memory.md) — the retrieval formula in detail.
- [Cognition and LLMs](cognition.md) — [`observe!`](@ref), [`reflect!`](@ref),
  and [`plan!`](@ref) all write to the stream.
