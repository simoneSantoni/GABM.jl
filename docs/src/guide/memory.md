# The Memory Stream

The memory stream is the one indispensable data structure of a generative
agent. Everything an agent knows, has concluded, or intends is a timestamped
record in it; every cognitive operation is a read from or a write to it. This
page describes how the stream is built and, in detail, how [`retrieve`](@ref)
selects from it.

## What a memory is

A [`MemoryEntry`](@ref) is a natural-language record with a little structure
around it:

| Field | Meaning |
|-------|---------|
| `content` | The memory itself, as a sentence of text |
| `kind` | `:observation`, `:reflection`, or `:plan` (see [`MEMORY_KINDS`](@ref)) |
| `importance` | Poignancy on the 1–10 scale |
| `created` | The tick at which the memory was formed |
| `last_accessed` | The tick at which it was most recently retrieved |
| `embedding` | An optional dense vector for relevance scoring |
| `citations` | For a reflection, the memories it was inferred from |

The `kind` field draws the distinction Park et al. (2023) make between the raw
perceptual stream and the abstractions built on it. An `:observation` is
something that happened; a `:reflection` is a conclusion the agent reached by
[`reflect!`](@ref); a `:plan` is an intention recorded by [`plan!`](@ref).
Because all three live in the same stream, a reflection can be retrieved as
evidence for a later reflection, and a plan can be recalled when deciding.

## Building a stream

A [`MemoryStream`](@ref) starts empty. The low-level way to add to it is
[`remember!`](@ref):

```julia
stream = MemoryStream()
remember!(stream, "I opened the café at seven."; kind = :observation,
          importance = 3.0, tick = 1)
```

In practice a model rarely calls `remember!` directly. The cognitive
operations — [`observe!`](@ref), [`reflect!`](@ref), [`plan!`](@ref) — call it
with the right `kind` and a sensible `importance`, and that is the interface
the [Cognitive Loop](cognition.md) guide describes.

The stream is configured by two parameters set at construction:

```julia
stream = MemoryStream(decay = 0.99, weights = (1.0, 1.0, 1.0))
```

`decay` is the per-tick base of the recency score, and `weights` are the three
coefficients combined during retrieval. Both are explained below.

## The retrieval problem

After a few hundred ticks an agent's stream holds far more than can be placed
in a single prompt. Retrieval is the act of choosing, for a given query, the
handful of memories worth surfacing. GABM.jl scores each candidate memory on
the three signals of Park et al. (2023) — recency, importance, relevance —
normalises them, combines them with the stream's weights, and returns the
top-scoring few.

For a query ``q`` at tick ``t``, memory ``m`` scores

```math
\text{score}(m \mid q,t) =
   w_{\text{rec}}\,\widetilde r(m,t) +
   w_{\text{imp}}\,\widetilde p(m) +
   w_{\text{rel}}\,\widetilde s(m,q).
```

### Recency

The raw recency of a memory decays exponentially with the time since it was
last touched:

```math
r(m,t) = \delta^{\,t - a(m)},
```

where ``a(m)`` is the `last_accessed` tick and ``\delta`` is the stream's
`decay`. With the default ``\delta = 0.995``, a memory unused for 100 ticks
keeps ``0.995^{100} \approx 0.61`` of its recency weight; with ``\delta = 0.9``
it keeps only ``0.9^{100} \approx 2.7\times10^{-5}`` and is effectively
forgotten. Lowering `decay` gives the agent a shorter memory.

Crucially, retrieving a memory **sets** ``a(m)`` to the current tick. Recall is
therefore self-reinforcing: a memory used today is fresh again tomorrow, while
one never recalled fades, exactly as Park et al. intended.

### Importance

The raw importance is the memory's stored poignancy, ``p(m) \in [1,10]``,
rescaled to ``[0,1]``. It is fixed when the memory is formed — either passed
explicitly or, when a backend is available, rated by [`rate_importance`](@ref).
Importance is what stops a consequential but old memory (a betrayal, a windfall)
from being buried under a heap of trivial recent ones.

### Relevance

The raw relevance ``s(m,q)`` measures how much memory ``m`` bears on query
``q``. GABM.jl computes it two ways:

- **Embeddings.** If the query can be embedded and the memory has a cached
  embedding, relevance is their cosine similarity, mapped to ``[0,1]``. Call
  [`embed_memories!`](@ref) once to populate the embeddings, and pass the
  backend to [`retrieve`](@ref).
- **Lexical fallback.** Otherwise, relevance is the Jaccard overlap of the
  content-word sets of the query and the memory. No backend, no embedding
  endpoint, and no extra API calls are needed — the model still runs.

The fallback is coarse but keeps a model fully functional offline, which is why
the examples in this manual run end-to-end against a [`ScriptedLLM`](@ref).

### Normalisation and weighting

The three raw scores live on different scales, so each is **min–max
normalised** across the candidate set before being combined:

```math
\widetilde x_i = \frac{x_i - \min_j x_j}{\max_j x_j - \min_j x_j}.
```

A component on which every candidate is equal — say, importance when all
memories were rated 5 — collapses to all-zeros and silently drops out of the
sum, so it never distorts the ranking. The normalised components are then
combined with `weights`, and the top `n` memories are returned.

Tilting the weights changes the character of recall. `weights = (0, 0, 1)`
gives pure semantic search; `weights = (2, 1, 1)` produces a present-focused
agent that mostly recalls what just happened; the default `(1, 1, 1)` balances
all three.

## A worked retrieval

Put four memories in a stream and query it:

```julia
stream = MemoryStream()
remember!(stream, "I argued with my brother about money."; tick = 2,  importance = 9.0)
remember!(stream, "I bought milk and bread.";              tick = 40, importance = 2.0)
remember!(stream, "My brother apologised for the argument."; tick = 41, importance = 8.0)
remember!(stream, "It rained in the afternoon.";           tick = 42, importance = 1.0)

top = retrieve(stream, "my relationship with my brother"; tick = 45, n = 2)
```

With the default weights the two brother memories win. Each is recent enough,
the argument scores highest on importance, and both overlap the query on the
words *brother* and (for the second) *argument*. The grocery and weather
memories — recent but trivial and irrelevant — are left behind. Retrieving the
two brother memories also advances their `last_accessed` to tick 45, so the
next query will find them fresher still.

## Reflections cite their evidence

When [`reflect!`](@ref) writes a reflection, it records in the entry's
`citations` field the indices of the memories the inference was drawn from.
This makes a reflection *traceable*: from any conclusion an agent has reached,
a modeller can follow the citations back to the observations that produced it.
For long runs this is the main tool for auditing why an agent behaves as it
does.

## Where to next

- [The Cognitive Loop](cognition.md) — the operations that read and write the
  stream.
- [Memory and Retrieval](../api/memory.md) — the full API reference for the
  types and functions on this page.
