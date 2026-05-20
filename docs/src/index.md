# GABM.jl

**Generative agent-based modeling for Julia.**

An agent-based model is a population of interacting decision-makers, simulated
forward in time to see what collective behaviour their individual rules
produce. For fifty years those rules have been written as equations or
if-then logic: an agent compares two payoffs, samples a probability, follows a
threshold. A *generative* agent-based model replaces that fixed logic with a
language model. The agent is given a natural-language identity, accumulates
natural-language memories of what happens to it, and — when the time comes to
act — is *asked*, in natural language, what it would do.

GABM.jl is the machinery that makes this practical in Julia. It supplies the
cognitive architecture — memory, retrieval, reflection, planning, decision —
and couples it to [Agents.jl](https://juliadynamics.github.io/Agents.jl/) for
everything a simulation also needs: scheduling, spaces, data collection,
reproducibility.

## Two kinds of agent

The contrast is sharpest in an example. Consider an agent deciding whether to
cooperate in a repeated social dilemma.

A **classical** agent decides with a rule the modeller wrote down:

```julia
decision = payoff(:cooperate, neighbours) > payoff(:defect, neighbours) ?
           :cooperate : :defect
```

The rule is transparent, fast, and exactly reproducible. It is also exactly as
rich as the modeller made it: the agent knows nothing the equation does not
encode, remembers nothing, and cannot be surprising.

A **generative** agent decides by consulting a language model that has been
told who the agent is and what it has lived through:

```julia
using GABM

mind = Mind(Persona("Wbishere";
    occupation = "smallholder farmer",
    traits = ["cautious", "values reputation", "long memory"]))

observe!(mind, "My neighbour Adsila took water from the shared channel " *
               "without asking during the drought."; tick = 12)

decision = decide(mind, llm,
    "The irrigation council asks whether to keep sharing water with Adsila.";
    tick = 30, options = ["keep sharing", "cut Adsila off"])

decision.action     # "cut Adsila off"
decision.reasoning  # "Adsila broke the channel agreement during the drought."
```

The generative agent's choice is grounded in a specific remembered grievance
that no payoff matrix contained. It is slower, costs an API call, and is only
*statistically* reproducible — but it can express reciprocity, reputation, and
narrative memory without any of those being formalised in advance.

GABM.jl does not argue that one kind of agent is better. It makes the second
kind a first-class citizen of an Agents.jl model, so the two can be mixed,
compared, and validated against each other.

## The architecture: a memory stream and four operations

GABM.jl implements the cognitive architecture of
[Park et al. (2023)](references.md), *Generative Agents: Interactive Simulacra
of Human Behavior*. Its one essential data structure is the **memory stream**:
an append-only, timestamped log of everything the agent has perceived, every
conclusion it has drawn, and every plan it has formed. Cognition is then four
operations that read and write that stream.

```
                       ┌──────────────────────────────┐
            perceive   │                              │
   world ────────────► │   observe!   →   :observation │
                       │                              │
                       │   reflect!   →   :reflection  │  ── synthesise
                       │      ▲                        │     insight from
                       │      │  retrieve              │     past memories
                       │   ┌──┴────────────────────┐   │
                       │   │     MEMORY STREAM      │   │
                       │   └──┬────────────────────┘   │
                       │      │  retrieve              │
                       │   plan!      →   :plan        │  ── form intentions
                       │                              │
                       │   decide     →   action ─────┼──────────► world
                       │                              │
                       └──────────────────────────────┘
```

- [`observe!`](@ref) writes a perception into the stream as an `:observation`,
  scoring how *poignant* it is on a 1–10 scale.
- [`reflect!`](@ref) reads recent memories, asks the language model what
  high-level questions they raise, answers each from retrieved evidence, and
  writes the answers back as `:reflection` memories. Reflections are memories,
  so later reflections compound earlier ones.
- [`plan!`](@ref) reads the agent's reflections and writes a `:plan`.
- [`decide`](@ref) reads the memories most relevant to the current situation
  and returns an action.

The hard problem common to all four is **retrieval**. After a long run an
agent remembers far more than fits in a single prompt, so each operation must
select the few memories that matter now.

## Retrieving the right memories

GABM.jl scores every candidate memory on the three signals of Park et al.
(2023) and returns the highest-scoring few. For a query ``q`` issued at tick
``t``, a memory ``m`` receives

```math
\text{score}(m \mid q, t) \;=\;
   w_{\text{rec}}\,\widetilde{r}(m,t) \;+\;
   w_{\text{imp}}\,\widetilde{p}(m)   \;+\;
   w_{\text{rel}}\,\widetilde{s}(m,q),
```

a weighted sum of three components, each min–max normalised across the
candidate set so that the weights ``w`` are comparable.

**Recency** decays exponentially in the time since the memory was last used:

```math
r(m,t) \;=\; \delta^{\,t - a(m)},
```

where ``a(m)`` is the memory's last-accessed tick and ``\delta \in (0,1)`` is
the per-tick decay base. Because retrieving a memory resets ``a(m)`` to the
current tick, attention is self-reinforcing — a memory recalled today is
easier to recall tomorrow.

**Importance** is the memory's stored poignancy ``p(m) \in [1,10]``, the rating
[`rate_importance`](@ref) elicits from the language model when the memory is
formed. A mundane observation (the weather) scores low; a consequential one (a
betrayal) scores high and resists being crowded out by sheer recency.

**Relevance** ``s(m,q)`` measures how much the memory bears on the query. When
embeddings are available it is the cosine similarity of the memory and query
embeddings; otherwise GABM.jl falls back to lexical token overlap, so a model
still runs end-to-end without an embedding endpoint.

[`retrieve`](@ref) computes all three, and the [Memory Stream](guide/memory.md)
guide works the formula through on a concrete stream.

## Generative cognition on an Agents.jl model

GABM.jl is deliberately *not* a simulation engine. Scheduling agents, laying
them out in space, stepping the clock, collecting data into a `DataFrame`,
seeding the random number generator for reproducibility — Agents.jl already
does all of this, and does it well. GABM.jl supplies only the layer Agents.jl
lacks: a mind.

```
   ┌─────────────────────────┐        ┌────────────────────────────┐
   │        Agents.jl        │        │           GABM.jl          │
   │                         │        │                            │
   │  StandardABM            │        │  Persona      — identity    │
   │  spaces (grid, graph,   │   +    │  MemoryStream — experience  │
   │     continuous)         │        │  observe! / reflect! /      │
   │  scheduler, run!, step! │        │     plan! / decide          │
   │  data collection        │        │  AbstractLLM  — the backend │
   └─────────────────────────┘        └────────────────────────────┘
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                       a generative agent-based model
```

A generative model is an ordinary Agents.jl `StandardABM`. [`generative_abm`](@ref)
builds one and guarantees it carries the two things the cognitive layer needs:
a shared language-model backend and a tick counter. Each agent is an Agents.jl
agent that owns a [`Mind`](@ref); the ready-made [`GenerativeAgent`](@ref)
covers non-spatial models, and any `@agent` type with a `mind` field works for
spatial ones. The modeller writes a normal `agent_step!` — the only difference
is that, inside it, the agent thinks:

```julia
function agent_step!(agent, model)
    observe!(agent, perceive(agent, model); tick = abmclock(model))
    d = decide(agent, abmllm(model), situation(agent, model);
               tick = abmclock(model))
    apply!(agent, model, d.action)
end
```

The [Building Models](guide/models.md) guide develops this skeleton into a
complete, runnable simulation.

## Live and scripted backends

Every cognitive call ultimately reaches an [`AbstractLLM`](@ref) backend, and
GABM.jl ships two.

[`PromptingToolsLLM`](@ref) is the live backend. It forwards to
[PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) and through
it to any provider that package supports — Anthropic, OpenAI, Mistral, or a
model served locally by Ollama. Switching providers is a one-line change.

[`ScriptedLLM`](@ref) is a deterministic mock. It returns canned or
prompt-computed responses without ever touching the network, which makes it
the right backend for unit tests, for the examples in this manual, and for any
experiment whose results must be bit-for-bit reproducible. Because the whole
simulation is written against `AbstractLLM`, the *same* model runs unchanged on
either backend — developed and tested against `ScriptedLLM`, then run for real
against `PromptingToolsLLM`.

## When a generative model earns its cost

Generative agents are slower and dearer than equations, and their output is a
distribution rather than a number. They repay that cost when a model needs:

- **heterogeneous, legible agents** — populations specified as written
  personas rather than parameter vectors;
- **memory and narrative** — behaviour that depends on a particular
  remembered history, not just an aggregate state variable;
- **open-ended action** — choices the modeller did not enumerate in advance;
- **natural-language environments** — settings (negotiations, deliberations,
  rumour spread) whose very content is text.

When agents are interchangeable and their decision rule is genuinely a short
equation, a classical Agents.jl model remains the better tool — and GABM.jl
leaves it untouched, ready to host generative agents alongside it whenever the
question calls for them.

## Notation used in this manual

| Symbol | Meaning |
|--------|---------|
| ``t`` | The current simulation tick (an integer clock) |
| ``m`` | A single memory in an agent's stream |
| ``q`` | A retrieval query (the current situation or question) |
| ``a(m)`` | The tick at which memory ``m`` was last retrieved |
| ``\delta`` | Per-tick recency decay base, ``\delta \in (0,1)`` |
| ``r,\,p,\,s`` | Raw recency, importance, and relevance scores |
| ``\widetilde{\,\cdot\,}`` | A score after min–max normalisation to ``[0,1]`` |
| ``w_{\text{rec}},w_{\text{imp}},w_{\text{rel}}`` | Retrieval weights on the three components |

## Where to next

- [Getting Started](getting_started.md) — install the package and run a
  complete generative model offline, with no API key.
- [Language Models](guide/llm.md) — the [`AbstractLLM`](@ref) interface, the
  live and scripted backends, and how to choose between them.
- [The Memory Stream](guide/memory.md) — [`MemoryStream`](@ref) and the
  [`retrieve`](@ref) scoring formula, worked through on a concrete example.
- [The Cognitive Loop](guide/cognition.md) — [`observe!`](@ref),
  [`reflect!`](@ref), [`plan!`](@ref), and [`decide`](@ref) in detail.
- [Building Models](guide/models.md) — wiring the cognitive loop into an
  Agents.jl `StandardABM` and running it.
- [References](references.md) — annotated bibliography of the generative-agent
  and agent-based-modelling literature this package builds on.
