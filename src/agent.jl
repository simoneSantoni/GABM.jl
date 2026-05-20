# Minds and agents.
#
# GABM.jl separates the *mind* of a generative agent from its *body*. The mind
# — a persona, a memory stream, a plan, a status line — is what the cognitive
# layer reads and writes. The body is whatever Agents.jl agent the modeller
# chooses, spatial or not. The `GenerativeAgent` type below is a ready-made
# body for the common non-spatial case; for spatial models, a modeller defines
# their own `@agent` and gives it a `mind::Mind` field.

"""
    Mind(persona; memory, plan, status)

The cognitive state of a generative agent.

A `Mind` bundles the four things every cognitive function in GABM.jl needs:

- `persona::Persona` — the agent's fixed identity (see [`Persona`](@ref)).
- `memory::MemoryStream` — its evolving record of experience.
- `plan::Vector{String}` — its current intentions, as set by [`plan!`](@ref).
- `status::String` — a one-line description of what it is doing right now.

The split between `Mind` and the agent *body* is deliberate. A `Mind` carries
everything the language model reasons about; the body carries everything the
spatial or network model needs. Keeping them in separate objects lets the same
cognitive machinery serve a non-spatial [`GenerativeAgent`](@ref), a grid
agent, or a node in a social network without change.
"""
mutable struct Mind
    persona::Persona
    memory::MemoryStream
    plan::Vector{String}
    status::String
end

function Mind(persona::Persona;
              memory::MemoryStream = MemoryStream(),
              plan = String[],
              status::AbstractString = "idle")
    return Mind(persona, memory, String[String(p) for p in plan], String(status))
end

Base.show(io::IO, m::Mind) =
    print(io, "Mind(", m.persona, ", ", length(m.memory),
          " memories, status=", repr(m.status), ")")

"""
    @agent struct GenerativeAgent(NoSpaceAgent)
        mind::Mind
    end

A ready-made, non-spatial generative agent.

`GenerativeAgent` is an [Agents.jl](https://juliadynamics.github.io/Agents.jl/)
agent with two fields: the `id` supplied by `NoSpaceAgent`, and a
[`Mind`](@ref). It is the agent type used by [`generative_abm`](@ref) unless
another is requested, and it covers every model in which agents have no
position — conversations, markets, committees, opinion dynamics.

For a spatial model, define an agent with a `mind` field instead, e.g.

```julia
using Agents, GABM
@agent struct Resident(GridAgent{2})
    mind::Mind
end
```

Every cognitive function ([`observe!`](@ref), [`decide`](@ref), …) accepts such
an agent directly, because they dispatch on the [`mind`](@ref) accessor rather
than on the concrete agent type.
"""
@agent struct GenerativeAgent(NoSpaceAgent)
    mind::Mind
end

"""
    mind(agent) -> Mind

Return the [`Mind`](@ref) of `agent`.

The fallback implementation returns `agent.mind`, so any Agents.jl agent with a
`mind` field works with GABM.jl out of the box. The cognitive layer is written
entirely against this accessor; a modeller whose agent stores its mind
elsewhere need only add a method to `mind`.
"""
mind(agent::AbstractAgent) = agent.mind
mind(m::Mind) = m

"""
    persona(x) -> Persona

Return the [`Persona`](@ref) of a [`Mind`](@ref) or of any agent that has one.
"""
persona(m::Mind) = m.persona
persona(agent::AbstractAgent) = persona(mind(agent))
