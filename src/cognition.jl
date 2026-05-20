# The cognitive loop.
#
# This file implements the four cognitive operations of a generative agent, in
# the order Park et al. (2023) describe them: perceiving (`observe!`),
# abstracting (`reflect!`), intending (`plan!`), and acting (`decide`). Each is
# a language-model call wrapped around a memory-stream read or write. A fifth
# function, `converse`, composes `decide`-style calls into a dialogue between
# two minds.

# A shared system prompt that frames every cognitive call: the model is asked
# to *be* the agent, not to describe it.
const _COGNITIVE_SYSTEM = """
You are the mind of a person in a social simulation. You think and act only as
that person, using only what they know. Keep responses concise and in
character. Do not break character or mention that this is a simulation.
"""

"""
    Decision(action, reasoning, raw)

The result of a [`decide`](@ref) call.

- `action::String` — what the agent does, as a short imperative phrase.
- `reasoning::String` — the agent's stated justification, if it gave one.
- `raw::String` — the model's unparsed reply, kept for debugging and logging.
"""
struct Decision
    action::String
    reasoning::String
    raw::String
end

Base.show(io::IO, d::Decision) = print(io, "Decision(", repr(d.action), ")")

# --- Parsing helpers ------------------------------------------------------

# Split a model reply into clean lines, stripping list numbering and bullets.
function _parse_lines(text::AbstractString)
    lines = String[]
    for raw in split(text, '\n')
        line = strip(raw)
        isempty(line) && continue
        line = replace(line, r"^\s*(?:\d+[.)]|[-*•])\s*" => "")
        isempty(strip(line)) || push!(lines, strip(line))
    end
    return lines
end

# Pull the first integer out of a model reply, clamped to the 1–10 scale.
function _parse_rating(text::AbstractString)
    m = match(r"\d+", text)
    m === nothing && return 5.0
    return Float64(clamp(parse(Int, m.match), 1, 10))
end

# Parse an "ACTION: … / REASON: …" reply into a Decision. If the model did not
# follow the format, the whole reply is taken as the action.
function _parse_decision(raw::AbstractString)
    action, reasoning = "", ""
    for line in split(raw, '\n')
        s = strip(line)
        if (m = match(r"^ACTION:\s*(.*)$"i, s)) !== nothing
            action = strip(m.captures[1])
        elseif (m = match(r"^REASON(?:ING)?:\s*(.*)$"i, s)) !== nothing
            reasoning = strip(m.captures[1])
        end
    end
    isempty(action) && (action = strip(raw))
    return Decision(String(action), String(reasoning), String(strip(raw)))
end

# --- Importance rating ----------------------------------------------------

"""
    rate_importance(llm, content) -> Float64

Ask the backend `llm` to rate the poignancy of a memory on the Park et al.
1–10 scale, where 1 is wholly mundane (brushing one's teeth) and 10 is
profoundly significant (a bereavement, a break-up).

The rating is the `importance` component of memory retrieval. A reply that
contains no number falls back to a neutral `5.0`.
"""
function rate_importance(llm::AbstractLLM, content::AbstractString)
    prompt = """
    On the scale of 1 to 10, where 1 is purely mundane (e.g. brushing teeth,
    making the bed) and 10 is extremely poignant (e.g. a break-up, a college
    acceptance, a bereavement), rate the likely poignancy of the following
    piece of memory. Respond with a single integer and nothing else.

    Memory: $content
    Rating:"""
    return _parse_rating(complete(llm, prompt; temperature = 0.0))
end

# --- Observation ----------------------------------------------------------

"""
    observe!(mind, content; tick, importance = nothing, llm = nothing,
             kind = :observation) -> Int

Record that `mind` perceived `content`, and return the new memory's index.

If `importance` is given it is used directly. If it is `nothing` and a backend
`llm` is supplied, the importance is rated by [`rate_importance`](@ref);
otherwise it defaults to a neutral `5.0`.

`observe!` also accepts an agent in place of a `Mind`, operating on its
[`mind`](@ref).
"""
function observe!(m::Mind, content::AbstractString;
                  tick::Integer,
                  importance::Union{Real,Nothing} = nothing,
                  llm::Union{Nothing,AbstractLLM} = nothing,
                  kind::Symbol = :observation)
    imp = importance !== nothing ? Float64(importance) :
          llm !== nothing ? rate_importance(llm, content) : 5.0
    return remember!(m.memory, content; kind = kind, importance = imp, tick = tick)
end

observe!(agent::AbstractAgent, content::AbstractString; kwargs...) =
    observe!(mind(agent), content; kwargs...)

# --- Recall ---------------------------------------------------------------

"""
    recall(mind, query; tick, n = 5, kinds = nothing, llm = nothing) -> Vector{MemoryEntry}

Return the memories of `mind` most relevant to `query`.

`recall` is the [`Mind`](@ref)-level wrapper around [`retrieve`](@ref); see
that function for how memories are scored. It also accepts an agent in place of
a `Mind`.
"""
recall(m::Mind, query::AbstractString; tick::Integer, n::Integer = 5,
       kinds = nothing, llm::Union{Nothing,AbstractLLM} = nothing) =
    retrieve(m.memory, query; tick = tick, n = n, kinds = kinds, llm = llm)

recall(agent::AbstractAgent, query::AbstractString; kwargs...) =
    recall(mind(agent), query; kwargs...)

# --- Reflection -----------------------------------------------------------

"""
    reflect!(mind, llm; tick, n_recent = 25, n_questions = 3, n_evidence = 8,
             rate = false) -> Vector{String}

Synthesise higher-level insights from recent experience and store them as
`:reflection` memories. Returns the text of the new reflections.

This is the reflection step of Park et al. (2023), run in three stages:

1. The most recent `n_recent` memories are shown to `llm`, which is asked for
   the `n_questions` most salient questions they raise.
2. For each question, [`retrieve`](@ref) gathers up to `n_evidence` supporting
   memories.
3. `llm` is asked to infer one high-level insight from that evidence; the
   insight is stored with [`citations`](@ref MemoryEntry) pointing back to it.

Because reflections are themselves memories, later reflections can build on
earlier ones, giving the agent a deepening self-model over a long run. With
`rate = true` each reflection's importance is scored by
[`rate_importance`](@ref) rather than fixed at `8.0`.
"""
function reflect!(m::Mind, llm::AbstractLLM;
                  tick::Integer,
                  n_recent::Integer = 25,
                  n_questions::Integer = 3,
                  n_evidence::Integer = 8,
                  rate::Bool = false)
    entries = m.memory.entries
    isempty(entries) && return String[]

    recent = entries[max(1, length(entries) - Int(n_recent) + 1):end]
    question_prompt = """
    $(describe(m.persona))

    Recent observations:
    $(_format_memories(recent))

    Given only the information above, what are the $n_questions most salient
    high-level questions we can answer about the subject? List one question
    per line, with no numbering."""
    questions = _parse_lines(complete(llm, question_prompt;
                                      system = _COGNITIVE_SYSTEM))

    new_reflections = String[]
    for question in first(questions, Int(n_questions))
        evidence = retrieve(m.memory, question; tick = tick,
                            n = Int(n_evidence), llm = llm)
        isempty(evidence) && continue
        insight_prompt = """
        $(describe(m.persona))

        Statements about the subject:
        $(_format_memories(evidence))

        What is one high-level insight you can infer from the statements above
        in answer to the question: "$question"? Reply with a single sentence."""
        insight = complete(llm, insight_prompt; system = _COGNITIVE_SYSTEM)
        isempty(strip(insight)) && continue
        citations = [findfirst(==(e), entries) for e in evidence]
        importance = rate ? rate_importance(llm, insight) : 8.0
        remember!(m.memory, insight; kind = :reflection, importance = importance,
                  tick = tick, citations = filter(!isnothing, citations))
        push!(new_reflections, strip(insight))
    end
    return new_reflections
end

reflect!(agent::AbstractAgent, llm::AbstractLLM; kwargs...) =
    reflect!(mind(agent), llm; kwargs...)

"""
    maybe_reflect!(mind, llm; tick, threshold = 100.0, kwargs...) -> Bool

Trigger [`reflect!`](@ref) only when enough has happened to warrant it, and
report whether it fired.

The summed importance of every `:observation` formed since the agent's last
reflection is compared with `threshold`; if it is exceeded, `reflect!` runs and
`true` is returned. This is the Park et al. trigger that keeps reflection
periodic rather than constant, and it is what the default
[`generative_step!`](@ref) uses. Extra keyword arguments are forwarded to
`reflect!`.
"""
function maybe_reflect!(m::Mind, llm::AbstractLLM;
                        tick::Integer, threshold::Real = 100.0, kwargs...)
    last_reflection = 0
    for e in m.memory.entries
        e.kind === :reflection && (last_reflection = max(last_reflection, e.created))
    end
    accumulated = sum(Float64[e.importance for e in m.memory.entries
                              if e.kind === :observation && e.created >= last_reflection];
                      init = 0.0)
    accumulated < threshold && return false
    reflect!(m, llm; tick = tick, kwargs...)
    return true
end

maybe_reflect!(agent::AbstractAgent, llm::AbstractLLM; kwargs...) =
    maybe_reflect!(mind(agent), llm; kwargs...)

# --- Planning -------------------------------------------------------------

"""
    plan!(mind, llm; tick, horizon = "the day ahead", context = "",
          n_steps = 5) -> Vector{String}

Have the agent draw up a plan, store it on `mind.plan`, record it as a `:plan`
memory, and return it as a list of steps.

The prompt combines the agent's persona, its current status, the `context`
string supplied by the model, and the reflections most relevant to `horizon`,
so the plan is grounded in what the agent has already concluded about itself.
`n_steps` bounds the plan's length.
"""
function plan!(m::Mind, llm::AbstractLLM;
               tick::Integer,
               horizon::AbstractString = "the day ahead",
               context::AbstractString = "",
               n_steps::Integer = 5)
    relevant = retrieve(m.memory, "plans and priorities for $horizon";
                        tick = tick, n = 5, llm = llm)
    prompt = """
    $(describe(m.persona))

    Current status: $(m.status)
    $(isempty(context) ? "" : "Situation: $context\n")
    What the subject has concluded so far:
    $(_format_memories(relevant))

    Write a brief, concrete plan for $horizon — at most $n_steps steps, one per
    line, no numbering, each a short phrase."""
    steps = _parse_lines(complete(llm, prompt; system = _COGNITIVE_SYSTEM))
    steps = first(steps, Int(n_steps))
    m.plan = collect(String, steps)
    remember!(m.memory, "Plan for $horizon: " * join(steps, "; ");
              kind = :plan, importance = 6.0, tick = tick)
    return m.plan
end

plan!(agent::AbstractAgent, llm::AbstractLLM; kwargs...) =
    plan!(mind(agent), llm; kwargs...)

# --- Decision -------------------------------------------------------------

"""
    decide(mind, llm, situation; tick, options = String[],
           remember_situation = true, n_memories = 6) -> Decision

Have the agent decide what to do in `situation`, and return a [`Decision`](@ref).

This is the act step of the cognitive loop. It assembles a prompt from the
agent's persona, status, current plan, and the `n_memories` memories most
relevant to `situation` (via [`retrieve`](@ref)), then asks `llm` for an action
and a one-line reason.

If `options` is non-empty the agent is constrained to choose one of them, which
is what makes `decide` slot cleanly into a classical agent-based model: the
discrete choice set is the model's, the selection is the language model's. When
`remember_situation` is `true` the situation is also stored as an observation,
so the agent will recall having faced it.

`decide` also accepts an agent in place of a `Mind`.
"""
function decide(m::Mind, llm::AbstractLLM, situation::AbstractString;
                tick::Integer,
                options::AbstractVector = String[],
                remember_situation::Bool = true,
                n_memories::Integer = 6)
    relevant = retrieve(m.memory, situation; tick = tick,
                        n = Int(n_memories), llm = llm)
    plan_block = isempty(m.plan) ? "(no plan set)" :
        join(("- " * s for s in m.plan), '\n')
    options_block = isempty(options) ? "" :
        "\nThe subject must choose exactly one of these options:\n" *
        join(("- " * String(o) for o in options), '\n') * "\n"
    prompt = """
    $(describe(m.persona))

    Current status: $(m.status)

    Current plan:
    $plan_block

    Relevant memories:
    $(_format_memories(relevant))

    Situation: $situation
    $options_block
    What does $(m.persona.name) do? Reply on exactly two lines:
    ACTION: <a short imperative phrase$(isempty(options) ? "" : ", one of the options above")>
    REASON: <one sentence>"""
    raw = complete(llm, prompt; system = _COGNITIVE_SYSTEM)
    if remember_situation
        remember!(m.memory, "I faced this situation: $situation";
                  kind = :observation, importance = 4.0, tick = tick)
    end
    return _parse_decision(raw)
end

decide(agent::AbstractAgent, llm::AbstractLLM, situation::AbstractString; kwargs...) =
    decide(mind(agent), llm, situation; kwargs...)

# --- Conversation ---------------------------------------------------------

"""
    converse(a, b, llm; topic = "", turns = 4, tick, remember = true) -> Vector{Pair{String,String}}

Generate a turn-taking dialogue between two minds and return it as a list of
`name => utterance` pairs.

`a` and `b` may be [`Mind`](@ref)s or agents. The two speak alternately for
`turns` utterances each; every utterance is produced by `llm` from the
speaker's persona, the memories most relevant to the conversation, and the
transcript so far. When `remember = true` a summary of the dialogue is written
back into both minds as an observation, so the conversation becomes part of
each agent's history.
"""
function converse(a, b, llm::AbstractLLM;
                  topic::AbstractString = "",
                  turns::Integer = 4,
                  tick::Integer,
                  remember::Bool = true)
    ma, mb = mind(a), mind(b)
    transcript = Pair{String,String}[]
    seed = isempty(topic) ? "a conversation with $(mb.persona.name)" :
           "a conversation about $topic"
    for t in 1:(2 * Int(turns))
        speaker, listener = isodd(t) ? (ma, mb) : (mb, ma)
        relevant = retrieve(speaker.memory, seed; tick = tick, n = 4, llm = llm)
        history = isempty(transcript) ? "(the conversation has not started)" :
            join((p.first * ": " * p.second for p in transcript), '\n')
        prompt = """
        $(describe(speaker.persona))

        $(speaker.persona.name) is talking with $(listener.persona.name).
        $(isempty(topic) ? "" : "The topic is: $topic.")

        Relevant memories:
        $(_format_memories(relevant))

        Conversation so far:
        $history

        What does $(speaker.persona.name) say next? Reply with only the spoken
        line, in the first person, one or two sentences."""
        utterance = complete(llm, prompt; system = _COGNITIVE_SYSTEM)
        push!(transcript, speaker.persona.name => strip(utterance))
    end
    if remember
        summary = "I had a conversation with " *
            (mind(a).persona.name == ma.persona.name ? mb.persona.name : ma.persona.name) *
            (isempty(topic) ? "" : " about $topic") * ":\n" *
            join((p.first * ": " * p.second for p in transcript), '\n')
        remember!(ma.memory, summary; kind = :observation, importance = 6.0, tick = tick)
        remember!(mb.memory, summary; kind = :observation, importance = 6.0, tick = tick)
    end
    return transcript
end
