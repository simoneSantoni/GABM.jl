# Model construction — the bridge to Agents.jl.
#
# A generative model is an ordinary Agents.jl `StandardABM` with two reserved
# entries in its properties: the language-model backend, and a tick counter.
# Building it through `generative_abm` guarantees those entries exist and that
# the clock advances every step. Everything else — spaces, schedulers, data
# collection, `run!` — is plain Agents.jl.

"""
    abmllm(model) -> AbstractLLM

Return the language-model backend stored in `model`. Inside a step function,
this is how an agent reaches the shared [`AbstractLLM`](@ref).
"""
abmllm(model) = abmproperties(model)[:llm]

"""
    abmclock(model) -> Int

Return the current tick of `model`. The clock starts at `0` and is advanced by
[`clock_step!`](@ref) once per step; pass it as the `tick` argument to
cognitive functions.
"""
abmclock(model) = abmproperties(model)[:clock]

"""
    clock_step!(model)

Advance the model clock by one tick. This is the default `model_step!` used by
[`generative_abm`](@ref); a model that needs its own `model_step!` should call
`clock_step!` from within it so that [`abmclock`](@ref) stays accurate.
"""
function clock_step!(model)
    abmproperties(model)[:clock] += 1
    return nothing
end

"""
    generative_step!(agent, model)

The default per-agent step: it calls [`maybe_reflect!`](@ref) so that each
agent consolidates its memories into reflections once enough salient experience
has accumulated.

This default deliberately does *not* make agents act, because acting is
domain-specific — what an agent can do depends on the model. A real model
supplies its own `agent_step!`, typically one that perceives the environment,
calls [`decide`](@ref), and applies the result. See the
[Building Models](@ref) guide for the pattern.
"""
function generative_step!(agent, model)
    maybe_reflect!(mind(agent), abmllm(model); tick = abmclock(model))
    return nothing
end

"""
    generative_abm(; llm, agent_step!, model_step!, space, properties,
                     agent_type, kwargs...) -> StandardABM

Construct an Agents.jl `StandardABM` set up for generative agents.

The returned model is an ordinary `StandardABM`: it is stepped with
`Agents.step!`, run with `Agents.run!`, and populated with `Agents.add_agent!`
(or the convenience [`add_generative_agent!`](@ref)). `generative_abm` only
guarantees the two things a generative model additionally needs — a backend and
a clock — by injecting `:llm` and `:clock` into the model's properties.

# Keyword arguments
- `llm::AbstractLLM`: the backend shared by every agent; read with
  [`abmllm`](@ref). Required.
- `agent_step!`: the per-agent step function. Defaults to
  [`generative_step!`](@ref).
- `model_step!`: the per-step model function. Defaults to [`clock_step!`](@ref),
  which advances the tick counter.
- `space`: any Agents.jl space, or `nothing` (the default) for a non-spatial
  model.
- `properties`: extra model properties, merged with `:llm` and `:clock`.
- `agent_type`: the agent type. Defaults to [`GenerativeAgent`](@ref); a
  spatial model passes its own `@agent` type with a `mind` field.
- further `kwargs` are forwarded to `StandardABM` (e.g. `scheduler`, `rng`).

# Example
```julia
model = generative_abm(; llm = ScriptedLLM("ACTION: wave\\nREASON: friendly"),
                          agent_step! = my_step!)
add_generative_agent!(model, Persona("Ada"))
step!(model, 3)
```
"""
function generative_abm(; llm::AbstractLLM,
                          agent_step! = generative_step!,
                          model_step! = clock_step!,
                          space = nothing,
                          properties = Dict{Symbol,Any}(),
                          agent_type::Type = GenerativeAgent,
                          kwargs...)
    props = Dict{Symbol,Any}()
    for (k, v) in pairs(properties)
        props[Symbol(k)] = v
    end
    props[:llm] = llm
    get!(props, :clock, 0)
    if space === nothing
        return StandardABM(agent_type; agent_step!, model_step!,
                           properties = props, kwargs...)
    else
        return StandardABM(agent_type, space; agent_step!, model_step!,
                           properties = props, kwargs...)
    end
end

"""
    add_generative_agent!(model, persona; memory, plan, status, kwargs...) -> GenerativeAgent

Build a [`Mind`](@ref) around `persona` and add a [`GenerativeAgent`](@ref)
carrying it to `model`, returning the new agent.

`memory`, `plan`, and `status` are forwarded to the `Mind` constructor; any
further `kwargs` (e.g. a position) are forwarded to `Agents.add_agent!`. This
is a convenience for the common case — a model with a custom agent type adds
agents with `Agents.add_agent!` directly.
"""
function add_generative_agent!(model, persona::Persona;
                               memory::MemoryStream = MemoryStream(),
                               plan = String[],
                               status::AbstractString = "idle",
                               kwargs...)
    m = Mind(persona; memory = memory, plan = plan, status = status)
    return add_agent!(model, m; kwargs...)
end
