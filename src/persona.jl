# Personas — the natural-language identity of a generative agent.
#
# In a classical agent-based model an agent is a tuple of numbers. In a
# generative model the agent must also be *legible to a language model*: its
# identity has to be expressible as text that can be placed in a prompt. The
# `Persona` type holds that text in a lightly structured form.

"""
    Persona(name; age, occupation, traits, background)

The natural-language identity of a generative agent.

A persona is the part of an agent that does not change during a simulation: a
stable description of who the agent *is*, as opposed to the [`MemoryStream`](@ref)
that records what has *happened* to it. It is rendered into every prompt the
agent's mind issues, via [`describe`](@ref).

# Arguments
- `name::AbstractString`: the agent's name (positional, required).
- `age`: an `Integer`, or `nothing` if unspecified.
- `occupation::AbstractString`: a short role description.
- `traits`: an iterable of short trait strings (e.g. `["curious", "frugal"]`).
- `background::AbstractString`: one or more sentences of free-text history.

# Example
```julia
isabella = Persona("Isabella Rodriguez";
    age = 34,
    occupation = "café owner",
    traits = ["warm", "organised", "ambitious"],
    background = "Runs Hobbs Cafe and is planning a Valentine's Day party.")
```
"""
struct Persona
    name::String
    age::Union{Int,Nothing}
    occupation::String
    traits::Vector{String}
    background::String
end

function Persona(name::AbstractString;
                 age::Union{Integer,Nothing} = nothing,
                 occupation::AbstractString = "",
                 traits = String[],
                 background::AbstractString = "")
    return Persona(String(name),
                   age === nothing ? nothing : Int(age),
                   String(occupation),
                   String[String(t) for t in traits],
                   String(background))
end

"""
    describe(p::Persona) -> String

Render a persona as a compact, prompt-ready identity block.

Only the fields that were supplied appear, each on its own line, so the result
stays readable whether a persona was specified richly or minimally. This is the
exact text that the cognitive layer ([`decide`](@ref), [`reflect!`](@ref),
[`plan!`](@ref)) places at the head of every prompt.

# Example
```julia
julia> println(describe(Persona("Ada"; occupation = "engineer")))
Name: Ada
Occupation: engineer
```
"""
function describe(p::Persona)
    io = IOBuffer()
    println(io, "Name: ", p.name)
    p.age === nothing || println(io, "Age: ", p.age)
    isempty(p.occupation) || println(io, "Occupation: ", p.occupation)
    isempty(p.traits) || println(io, "Traits: ", join(p.traits, ", "))
    isempty(p.background) || println(io, "Background: ", p.background)
    return rstrip(String(take!(io)))
end

Base.show(io::IO, p::Persona) = print(io, "Persona(\"", p.name, "\")")
