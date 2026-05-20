# The Cognitive Loop

A generative agent does four things with its [memory stream](memory.md): it
*perceives*, it *abstracts*, it *intends*, and it *acts*. GABM.jl implements
each as one function — [`observe!`](@ref), [`reflect!`](@ref), [`plan!`](@ref),
[`decide`](@ref) — and this page describes them in the order Park et al. (2023)
present them. A fifth function, [`converse`](@ref), composes decision-style
calls into a dialogue.

Every function on this page accepts either a [`Mind`](@ref) or an agent that
has one: they dispatch on the [`mind`](@ref) accessor, so the same call works
on a bare `Mind` during prototyping and on a [`GenerativeAgent`](@ref) inside a
running model.

## Perceive — `observe!`

[`observe!`](@ref) writes a perception into the stream as an `:observation`:

```julia
observe!(agent, "The market square is crowded and noisy."; tick = 12)
```

The only subtlety is importance. If `importance` is given, it is used. If it is
omitted and a backend is supplied, GABM.jl calls [`rate_importance`](@ref),
which asks the language model to score the memory's poignancy on the Park et
al. 1–10 scale:

```julia
observe!(agent, "My closest friend is moving away."; tick = 12, llm = llm)
```

Letting the model rate importance is what makes an agent's sense of *what
matters* emergent rather than hand-coded — but it costs an API call per
observation, so a model with cheap, plentiful perceptions often passes a fixed
`importance` instead and reserves rating for events that might be pivotal.

## Abstract — `reflect!`

Observations alone make a shallow agent: it can recall what happened but has
drawn no conclusions from it. [`reflect!`](@ref) is the step that builds the
higher layer. It runs in three stages, following Park et al. (2023):

1. The most recent memories are shown to the model, which is asked for the few
   most **salient questions** they raise.
2. For each question, [`retrieve`](@ref) gathers supporting **evidence** from
   anywhere in the stream — including earlier reflections.
3. The model is asked to infer one **insight** per question; each insight is
   written back as a `:reflection`, with `citations` pointing to its evidence.

```julia
new_insights = reflect!(agent, llm; tick = 50)
```

Because reflections are themselves memories, the layer compounds: after many
runs of `reflect!` an agent holds reflections drawn from earlier reflections, a
deepening self-model rather than a flat log.

Reflection is expensive — several model calls — so it should be *periodic*, not
constant. [`maybe_reflect!`](@ref) is the trigger Park et al. describe: it sums
the importance of every observation formed since the last reflection and only
reflects when that total crosses a threshold.

```julia
maybe_reflect!(agent, llm; tick = abmclock(model), threshold = 100.0)
```

A quiet agent reflects rarely; one living through eventful, high-importance
ticks reflects often. This is exactly what the default
[`generative_step!`](@ref) does, so agents in a model consolidate their
memories without the modeller wiring it up.

## Intend — `plan!`

[`plan!`](@ref) has the agent draw up a short plan. The prompt is built from
the persona, the current status, an optional `context` string from the model,
and — importantly — the reflections most relevant to the planning horizon, so
the plan is grounded in what the agent has concluded about itself.

```julia
plan!(agent, llm; tick = 0, horizon = "the day ahead")
```

The plan is stored on `mind(agent).plan` as a list of steps and also recorded
as a `:plan` memory, so the agent can later recall what it had intended — and
notice when events forced a departure from it.

## Act — `decide`

[`decide`](@ref) is the step the rest of the simulation waits on. It retrieves
the memories most relevant to a situation, assembles a prompt from the persona,
the status, the current plan, and those memories, and returns a
[`Decision`](@ref):

```julia
d = decide(agent, llm, "A stranger asks to share your table."; tick = 30)
d.action      # a short imperative phrase
d.reasoning   # one sentence of justification
d.raw         # the model's unparsed reply, kept for logging
```

### Constrained choice

Left unconstrained, an agent invents its own action — open-ended behaviour,
which is sometimes the point. But a classical agent-based model is built on a
*discrete choice set*, and `decide` slots into one when given `options`:

```julia
d = decide(agent, llm, "The council votes on the new water tariff.";
           tick = 30, options = ["vote for", "vote against", "abstain"])
```

The agent is now constrained to return one of the options. This is the precise
seam between the two modelling traditions: the **choice set is the modeller's**,
fixed and analysable, while the **selection is the language model's**, grounded
in the agent's remembered history. A model can move an agent along this
spectrum question by question.

### Remembering the situation

By default `decide` also writes the situation into the stream as an
observation, so the agent will recall having faced it. Pass
`remember_situation = false` for hypothetical or counterfactual queries that
should not become part of the agent's history.

## Converse — `converse`

[`converse`](@ref) generates a turn-taking dialogue between two minds:

```julia
dialogue = converse(isabella, tom, llm;
                    topic = "the Valentine's Day party", turns = 3, tick = 40)
```

Each utterance is produced from the speaker's persona, the memories most
relevant to the conversation, and the transcript so far; the speakers
alternate. The result is a vector of `name => utterance` pairs. Unless
`remember = false`, a summary of the dialogue is written back into *both*
minds, so a conversation becomes a shared memory that can later be retrieved,
reflected on, and acted upon — the mechanism by which information spreads
through a population of generative agents.

## The loop, assembled

In a running model the four operations compose into a per-tick cycle inside
`agent_step!`:

```julia
function agent_step!(agent, model)
    llm, t = abmllm(model), abmclock(model)
    observe!(agent, perceive(agent, model); tick = t, llm = llm)   # perceive
    maybe_reflect!(agent, llm; tick = t)                            # abstract
    if t % 24 == 0
        plan!(agent, llm; tick = t)                                 # intend
    end
    d = decide(agent, llm, situation(agent, model); tick = t)       # act
    apply!(agent, model, d.action)
end
```

`perceive`, `situation`, and `apply!` are the modeller's — they are where the
cognitive loop meets the specific world being simulated. The
[Building Models](models.md) guide develops them for a concrete model.

## Where to next

- [Building Models](models.md) — wiring this loop into an Agents.jl model.
- [Cognition and LLMs](../api/cognition.md) — the API reference for every
  function on this page.
