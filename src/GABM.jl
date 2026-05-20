"""
    GABM

**Generative Agent-Based Modeling in Julia.**

GABM.jl couples a language model to [Agents.jl](https://juliadynamics.github.io/Agents.jl/):
the scheduling, spaces, and data collection of a classical agent-based model,
with agents that perceive, remember, reflect, plan, and decide in natural
language. The cognitive architecture follows Park et al. (2023), *Generative
Agents: Interactive Simulacra of Human Behavior*.

A model is built in four moves:

1. Choose a backend — [`PromptingToolsLLM`](@ref) for a live provider, or
   [`ScriptedLLM`](@ref) for a deterministic, offline run.
2. Give each agent a [`Persona`](@ref) and a [`Mind`](@ref).
3. Assemble the model with [`generative_abm`](@ref).
4. Write an `agent_step!` that calls the cognitive loop — [`observe!`](@ref),
   [`reflect!`](@ref), [`plan!`](@ref), [`decide`](@ref) — and step it with
   `Agents.run!`.

See the [package documentation](https://simoneSantoni.github.io/GABM.jl) for a
guided introduction.
"""
module GABM

using Agents
using LinearAlgebra: dot, norm
using Printf
import PromptingTools

include("llm.jl")
include("persona.jl")
include("memory.jl")
include("agent.jl")
include("cognition.jl")
include("model.jl")

# Language-model backends
export AbstractLLM, PromptingToolsLLM, ScriptedLLM, LLMError
export complete, embed, supports_embeddings

# Personas and minds
export Persona, describe
export Mind, GenerativeAgent, mind, persona

# Memory
export MemoryEntry, MemoryStream, MEMORY_KINDS
export remember!, retrieve, embed_memories!

# Cognition
export Decision
export observe!, recall, reflect!, maybe_reflect!, plan!, decide, converse
export rate_importance

# Models
export generative_abm, add_generative_agent!
export abmllm, abmclock, clock_step!, generative_step!

# Re-exported from Agents.jl so that `using GABM` is enough to build and run a
# model. Use `using Agents` directly for the rest of the Agents.jl API.
export AbstractAgent, NoSpaceAgent, @agent
export StandardABM, add_agent!, step!, run!, nagents, allagents
export abmproperties, abmrng

end # module GABM
