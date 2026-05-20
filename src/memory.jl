# The memory stream and its retrieval function.
#
# The memory stream is the central data structure of a generative agent, taken
# directly from Park et al. (2023). It is an append-only log of timestamped
# natural-language records. The hard problem it poses is *retrieval*: at any
# moment far more is remembered than can fit in a prompt, so a query must
# select the handful of memories that matter now. GABM.jl scores each memory by
# the Park et al. triad — recency, importance, and relevance — and returns the
# top few.

"""
    MEMORY_KINDS

The recognised values of the `kind` field of a [`MemoryEntry`](@ref):

- `:observation` — something the agent perceived or did.
- `:reflection` — a higher-level inference synthesised by [`reflect!`](@ref).
- `:plan` — an intention recorded by [`plan!`](@ref).

The distinction is the one Park et al. (2023) draw between the raw perceptual
stream and the abstractions an agent builds on top of it.
"""
const MEMORY_KINDS = (:observation, :reflection, :plan)

"""
    MemoryEntry(content; kind, importance, created, last_accessed, embedding, citations)

A single record in a [`MemoryStream`](@ref).

# Fields
- `content::String`: the memory itself, in natural language.
- `kind::Symbol`: one of [`MEMORY_KINDS`](@ref).
- `importance::Float64`: poignancy on the Park et al. 1–10 scale, where 1 is
  mundane and 10 is profound. See [`rate_importance`](@ref).
- `created::Int`: the simulation tick at which the memory was formed.
- `last_accessed::Int`: the tick at which the memory was most recently returned
  by [`retrieve`](@ref); recency decay is measured from this value, so
  retrieving a memory rejuvenates it.
- `embedding::Union{Nothing,Vector{Float64}}`: an optional dense vector used
  for relevance scoring.
- `citations::Vector{Int}`: indices of the memories a reflection was derived
  from; empty for observations and plans.
"""
mutable struct MemoryEntry
    content::String
    kind::Symbol
    importance::Float64
    created::Int
    last_accessed::Int
    embedding::Union{Nothing,Vector{Float64}}
    citations::Vector{Int}
end

function MemoryEntry(content::AbstractString;
                     kind::Symbol = :observation,
                     importance::Real = 5.0,
                     created::Integer = 0,
                     last_accessed::Integer = created,
                     embedding::Union{Nothing,AbstractVector} = nothing,
                     citations = Int[])
    kind in MEMORY_KINDS ||
        throw(ArgumentError("unknown memory kind :$kind; expected one of $MEMORY_KINDS"))
    return MemoryEntry(String(content), kind, Float64(importance),
                       Int(created), Int(last_accessed),
                       embedding === nothing ? nothing : Vector{Float64}(embedding),
                       Int[c for c in citations])
end

Base.show(io::IO, e::MemoryEntry) =
    print(io, "MemoryEntry(:", e.kind, ", t=", e.created,
          ", imp=", round(e.importance; digits = 1), ", ",
          repr(first(e.content, 40)), length(e.content) > 40 ? "…)" : ")")

"""
    MemoryStream(; decay = 0.995, weights = (1.0, 1.0, 1.0))

An append-only, time-ordered log of an agent's [`MemoryEntry`](@ref) records.

# Keyword arguments
- `decay::Float64`: the per-tick base of the exponential recency score. With
  the default `0.995`, a memory not accessed for 100 ticks keeps about 60% of
  its recency weight; a smaller value forgets faster.
- `weights::NTuple{3,Float64}`: the `(recency, importance, relevance)`
  coefficients combined by [`retrieve`](@ref). Raising one coordinate makes
  retrieval lean more heavily on that signal.

A fresh stream is empty; add to it with [`remember!`](@ref) and query it with
[`retrieve`](@ref).
"""
mutable struct MemoryStream
    entries::Vector{MemoryEntry}
    decay::Float64
    weights::NTuple{3,Float64}
end

function MemoryStream(; decay::Real = 0.995,
                        weights::NTuple{3,<:Real} = (1.0, 1.0, 1.0))
    return MemoryStream(MemoryEntry[], Float64(decay), Float64.(weights))
end

Base.length(s::MemoryStream) = length(s.entries)
Base.isempty(s::MemoryStream) = isempty(s.entries)
Base.getindex(s::MemoryStream, i) = s.entries[i]
Base.iterate(s::MemoryStream, st...) = iterate(s.entries, st...)
Base.lastindex(s::MemoryStream) = lastindex(s.entries)

Base.show(io::IO, s::MemoryStream) =
    print(io, "MemoryStream(", length(s.entries), " entries, decay=",
          s.decay, ")")

"""
    remember!(stream, content; kind, importance, tick, embedding, citations) -> Int

Append a memory to `stream` and return its index.

`content` is the natural-language record; the keyword arguments map onto the
fields of [`MemoryEntry`](@ref). `tick` is the current simulation time and is
recorded as both the creation and the initial last-accessed time.

Most models do not call `remember!` directly: [`observe!`](@ref),
[`reflect!`](@ref), and [`plan!`](@ref) call it with the appropriate `kind`.
"""
function remember!(stream::MemoryStream, content::AbstractString;
                   kind::Symbol = :observation,
                   importance::Real = 5.0,
                   tick::Integer = 0,
                   embedding::Union{Nothing,AbstractVector} = nothing,
                   citations = Int[])
    entry = MemoryEntry(content; kind = kind, importance = importance,
                        created = tick, last_accessed = tick,
                        embedding = embedding, citations = citations)
    push!(stream.entries, entry)
    return length(stream.entries)
end

# --- Retrieval ------------------------------------------------------------

const _STOPWORDS = Set(split(
    "a an and are as at be but by for from has have he her his i in is it its " *
    "of on or she that the their they to was were will with you your"))

# Tokenise text into a set of lower-cased content words, dropping stopwords and
# punctuation. Used for the lexical relevance fallback.
function _tokens(text::AbstractString)
    words = split(lowercase(text), r"[^a-z0-9]+"; keepempty = false)
    return Set(w for w in words if !(w in _STOPWORDS) && length(w) > 1)
end

# Jaccard overlap of two token sets, in [0, 1].
function _jaccard(a::Set, b::Set)
    (isempty(a) || isempty(b)) && return 0.0
    return length(intersect(a, b)) / length(union(a, b))
end

# Cosine similarity of two vectors, mapped from [-1, 1] to [0, 1].
function _cosine(a::AbstractVector, b::AbstractVector)
    na, nb = norm(a), norm(b)
    (na == 0 || nb == 0) && return 0.0
    return (dot(a, b) / (na * nb) + 1) / 2
end

# Min-max rescale a score vector to [0, 1]. A non-discriminating component
# (every entry equal) collapses to zeros so it drops out of the weighted sum.
function _minmax(v::AbstractVector{<:Real})
    isempty(v) && return Float64[]
    lo, hi = extrema(v)
    hi - lo < 1e-12 && return zeros(Float64, length(v))
    return Float64.((v .- lo) ./ (hi - lo))
end

"""
    embed_memories!(stream, llm) -> MemoryStream

Compute and cache an embedding for every entry in `stream` that lacks one,
using [`embed`](@ref) on the backend `llm`.

Calling this once after a batch of observations lets subsequent
[`retrieve`](@ref) calls score relevance by cosine similarity instead of the
lexical fallback. It is a no-op — and raises nothing — if `llm` does not
[support embeddings](@ref supports_embeddings).
"""
function embed_memories!(stream::MemoryStream, llm::AbstractLLM)
    supports_embeddings(llm) || return stream
    for e in stream.entries
        e.embedding === nothing && (e.embedding = embed(llm, e.content))
    end
    return stream
end

"""
    retrieve(stream, query; tick, n = 5, kinds = nothing, llm = nothing,
             weights = stream.weights) -> Vector{MemoryEntry}

Return the `n` memories from `stream` most worth surfacing for `query`.

This is the retrieval function of Park et al. (2023). Each candidate memory is
scored on three components, each min-max normalised across the candidate set so
the components are comparable:

- **Recency** — `decay ^ (tick - last_accessed)`, an exponential decay that
  favours memories formed or used recently (see [`MemoryStream`](@ref)).
- **Importance** — the memory's stored poignancy, on the 1–10 scale.
- **Relevance** — similarity to `query`: cosine similarity of embeddings when
  both the query and the memory have them, and Jaccard token overlap otherwise.

The three normalised components are combined by `weights` and the top `n`
entries are returned. Retrieving a memory **updates its `last_accessed` tick**
to `tick`, so attention is self-reinforcing: a memory used now is easier to
retrieve next time.

# Keyword arguments
- `tick`: the current simulation time (required).
- `n`: how many memories to return.
- `kinds`: if given, an iterable of [`MEMORY_KINDS`](@ref) to restrict the
  search to (e.g. `kinds = (:reflection,)`).
- `llm`: a backend used to embed the query when embedding-based relevance is
  wanted; pass the same backend that produced the memory embeddings.
- `weights`: override the stream's `(recency, importance, relevance)` weights
  for this query only.
"""
function retrieve(stream::MemoryStream, query::AbstractString;
                  tick::Integer,
                  n::Integer = 5,
                  kinds = nothing,
                  llm::Union{Nothing,AbstractLLM} = nothing,
                  weights::NTuple{3,<:Real} = stream.weights)
    candidates = kinds === nothing ? stream.entries :
        filter(e -> e.kind in kinds, stream.entries)
    isempty(candidates) && return MemoryEntry[]

    query_embedding = (llm !== nothing && supports_embeddings(llm)) ?
        embed(llm, query) : nothing
    query_tokens = _tokens(query)

    recency = [stream.decay ^ max(0, tick - e.last_accessed) for e in candidates]
    importance = [e.importance / 10 for e in candidates]
    relevance = map(candidates) do e
        if query_embedding !== nothing && e.embedding !== nothing
            _cosine(query_embedding, e.embedding)
        else
            _jaccard(query_tokens, _tokens(e.content))
        end
    end

    score = weights[1] .* _minmax(recency) .+
            weights[2] .* _minmax(importance) .+
            weights[3] .* _minmax(relevance)

    order = sortperm(score; rev = true)
    top = candidates[order[1:min(Int(n), length(order))]]
    for e in top
        e.last_accessed = Int(tick)
    end
    return top
end

# Format a list of memories as a numbered block for inclusion in a prompt.
function _format_memories(entries)
    isempty(entries) && return "(no relevant memories)"
    io = IOBuffer()
    for (i, e) in enumerate(entries)
        println(io, i, ". ", e.content)
    end
    return rstrip(String(take!(io)))
end
