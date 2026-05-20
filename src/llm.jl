# Language-model backends.
#
# Every cognitive operation in GABM.jl — rating the importance of a memory,
# synthesising a reflection, choosing an action — is ultimately a call to a
# language model. To keep the simulation logic independent of any particular
# vendor or SDK, all model access goes through the `AbstractLLM` interface
# defined here. Concrete backends supply two methods: `complete` (text in,
# text out) and, optionally, `embed` (text in, vector out).

"""
    AbstractLLM

Supertype for language-model backends.

A backend is anything that can turn a prompt into a string. Concrete subtypes
must implement [`complete`](@ref); they may additionally implement
[`embed`](@ref) and set [`supports_embeddings`](@ref) to `true`.

GABM.jl ships two backends:

- [`PromptingToolsLLM`](@ref) — a live backend that forwards to
  [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl), and
  therefore to OpenAI, Anthropic, Ollama, or any other provider that package
  supports.
- [`ScriptedLLM`](@ref) — a deterministic mock used in tests, documentation,
  and reproducible experiments, where calling a paid API is undesirable.

Keeping the simulation code written against `AbstractLLM` means the *same*
model can be run against a live provider or a scripted mock without changing
a line of the model definition.
"""
abstract type AbstractLLM end

"""
    LLMError(msg)

Exception raised when a language-model backend fails — a network error from a
live provider, or an exhausted response queue in a [`ScriptedLLM`](@ref).
"""
struct LLMError <: Exception
    msg::String
end

Base.showerror(io::IO, e::LLMError) = print(io, "LLMError: ", e.msg)

"""
    complete(llm, prompt; system="", temperature) -> String

Send `prompt` to the backend `llm` and return its completion as a `String`.

`system` is an optional system prompt that frames the request without being
part of the conversation proper. `temperature` controls sampling randomness;
its default is backend-specific.

This is the single primitive that every cognitive function in GABM.jl is built
on. Any subtype of [`AbstractLLM`](@ref) must implement it.
"""
function complete end

"""
    embed(llm, text) -> Vector{Float64}

Return a dense vector embedding of `text`.

Embeddings are optional: a backend that provides them must also set
[`supports_embeddings`](@ref) to `true`. When embeddings are available,
[`retrieve`](@ref) scores the *relevance* component of memory retrieval by
cosine similarity in embedding space; when they are not, it falls back to a
lexical overlap score, so a model still runs end-to-end without an embedding
endpoint.
"""
function embed end

"""
    supports_embeddings(llm) -> Bool

Whether the backend `llm` implements [`embed`](@ref). Defaults to `false`;
backends that provide embeddings override it.
"""
supports_embeddings(::AbstractLLM) = false

# ---------------------------------------------------------------------------
# Live backend: PromptingTools.jl
# ---------------------------------------------------------------------------

"""
    PromptingToolsLLM(; model, embedding_model, temperature, max_tokens)

A live language-model backend that forwards to
[PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl).

Because PromptingTools.jl is itself provider-agnostic, a `PromptingToolsLLM`
can drive OpenAI, Anthropic, Mistral, Ollama, or a locally hosted model — the
choice is made entirely by the `model` string and the API keys present in the
environment (see the PromptingTools.jl documentation for key configuration).

# Keyword arguments
- `model::String = "claudeh"`: the chat model, given as a PromptingTools.jl
  model name or registered alias (e.g. `"claudeh"`, `"claudes"`, `"gpt4om"`).
- `embedding_model::String = "text-embedding-3-small"`: the model used by
  [`embed`](@ref).
- `temperature::Float64 = 0.7`: default sampling temperature.
- `max_tokens::Int = 1024`: maximum tokens per completion.

# Example
```julia
llm = PromptingToolsLLM(model = "claudeh", temperature = 0.9)
complete(llm, "In one sentence, describe a quiet morning.")
```
"""
struct PromptingToolsLLM <: AbstractLLM
    model::String
    embedding_model::String
    temperature::Float64
    max_tokens::Int
end

function PromptingToolsLLM(; model::AbstractString = "claudeh",
                             embedding_model::AbstractString = "text-embedding-3-small",
                             temperature::Real = 0.7,
                             max_tokens::Integer = 1024)
    return PromptingToolsLLM(String(model), String(embedding_model),
                             Float64(temperature), Int(max_tokens))
end

supports_embeddings(::PromptingToolsLLM) = true

function complete(llm::PromptingToolsLLM, prompt::AbstractString;
                  system::AbstractString = "", temperature::Real = llm.temperature)
    conversation = isempty(system) ?
        [PromptingTools.UserMessage(String(prompt))] :
        [PromptingTools.SystemMessage(String(system)),
         PromptingTools.UserMessage(String(prompt))]
    try
        msg = PromptingTools.aigenerate(conversation;
            model = llm.model,
            api_kwargs = (; temperature = Float64(temperature),
                            max_tokens = llm.max_tokens))
        return strip(msg.content)
    catch e
        throw(LLMError("PromptingTools.aigenerate failed for model " *
                       "$(llm.model): $(sprint(showerror, e))"))
    end
end

function embed(llm::PromptingToolsLLM, text::AbstractString)
    try
        msg = PromptingTools.aiembed(String(text); model = llm.embedding_model)
        return Vector{Float64}(msg.content)
    catch e
        throw(LLMError("PromptingTools.aiembed failed for model " *
                       "$(llm.embedding_model): $(sprint(showerror, e))"))
    end
end

Base.show(io::IO, llm::PromptingToolsLLM) =
    print(io, "PromptingToolsLLM(model=\"", llm.model, "\", temperature=",
          llm.temperature, ")")

# ---------------------------------------------------------------------------
# Mock backend: ScriptedLLM
# ---------------------------------------------------------------------------

"""
    ScriptedLLM(responses::AbstractVector)
    ScriptedLLM(responder::Function)
    ScriptedLLM(response::AbstractString)

A deterministic mock backend. `ScriptedLLM` never touches the network, which
makes it the right backend for unit tests, documentation examples, and any
experiment that must be exactly reproducible.

Three construction modes are supported:

- **Queue.** `ScriptedLLM(["yes", "no"])` returns the responses in order, one
  per call to [`complete`](@ref), and raises [`LLMError`](@ref) once the queue
  is exhausted — a missing response is therefore a loud failure, not a silent
  one.
- **Responder.** `ScriptedLLM(prompt -> uppercase(prompt))` computes each
  response from the prompt, which is useful for mocks that must react to what
  the simulation actually asked.
- **Constant.** `ScriptedLLM("ok")` returns the same string for every call.

Every prompt the backend receives is appended to `llm.log`, so a test can
assert on exactly what the cognitive layer asked the model.
"""
mutable struct ScriptedLLM <: AbstractLLM
    responder::Union{Function,Nothing}
    queue::Vector{String}
    cursor::Int
    log::Vector{String}
end

ScriptedLLM(responses::AbstractVector) =
    ScriptedLLM(nothing, String.(collect(responses)), 1, String[])
ScriptedLLM(responder::Function) =
    ScriptedLLM(responder, String[], 1, String[])
ScriptedLLM(response::AbstractString) =
    ScriptedLLM(_ -> String(response), String[], 1, String[])

function complete(llm::ScriptedLLM, prompt::AbstractString;
                  system::AbstractString = "", temperature::Real = 0.0)
    push!(llm.log, String(prompt))
    if llm.responder !== nothing
        return strip(String(llm.responder(String(prompt))))
    end
    if llm.cursor > length(llm.queue)
        throw(LLMError("ScriptedLLM exhausted: a $(length(llm.queue))-response " *
                       "queue received a $(llm.cursor)th call"))
    end
    response = llm.queue[llm.cursor]
    llm.cursor += 1
    return strip(response)
end

"""
    reset!(llm::ScriptedLLM) -> ScriptedLLM

Rewind a queue-backed [`ScriptedLLM`](@ref) to its first response and clear its
prompt log. Useful for re-running a deterministic scenario.
"""
function reset!(llm::ScriptedLLM)
    llm.cursor = 1
    empty!(llm.log)
    return llm
end

Base.show(io::IO, llm::ScriptedLLM) =
    print(io, "ScriptedLLM(", llm.responder === nothing ?
          "$(length(llm.queue)) queued, $(llm.cursor - 1) used" : "responder",
          ", $(length(llm.log)) calls logged)")
