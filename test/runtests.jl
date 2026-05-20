using GABM
using Test

# A deterministic embedding backend used to exercise the embedding-based
# relevance path of `retrieve` without touching a network.
struct FakeEmbedder <: GABM.AbstractLLM end
GABM.supports_embeddings(::FakeEmbedder) = true
GABM.embed(::FakeEmbedder, text::AbstractString) =
    Float64[count(==(c), lowercase(text)) for c in 'a':'j']

# A prompt-aware mock that returns a plausible reply for each kind of
# cognitive call, identified by a marker the prompt is known to contain.
function smart_responder(prompt::AbstractString)
    if occursin("Rating:", prompt)
        return "7"
    elseif occursin("salient", prompt) && occursin("questions", prompt)
        return "What does the subject value?\nHow does the subject spend time?"
    elseif occursin("high-level insight", prompt)
        return "The subject values steady, focused work."
    elseif occursin("brief, concrete plan", prompt)
        return "Review notes\nWrite the report\nTake a break"
    elseif occursin("ACTION:", prompt)
        return "ACTION: keep working\nREASON: the deadline is near"
    else
        return "I think that sounds good."
    end
end

@testset "GABM.jl" begin

    @testset "ScriptedLLM" begin
        q = ScriptedLLM(["one", "two"])
        @test complete(q, "a") == "one"
        @test complete(q, "b") == "two"
        @test_throws LLMError complete(q, "c")
        @test q.log == ["a", "b", "c"]

        r = ScriptedLLM(p -> uppercase(p))
        @test complete(r, "hi") == "HI"

        c = ScriptedLLM("always")
        @test complete(c, "x") == "always"
        @test complete(c, "y") == "always"

        GABM.reset!(q)
        @test complete(q, "again") == "one"
        @test isempty(q.log[1:end-1])

        @test !supports_embeddings(c)
        @test supports_embeddings(FakeEmbedder())
    end

    @testset "PromptingToolsLLM" begin
        llm = PromptingToolsLLM(model = "claudeh", temperature = 0.5)
        @test llm.model == "claudeh"
        @test llm.temperature == 0.5
        @test supports_embeddings(llm)
        @test occursin("claudeh", sprint(show, llm))
    end

    @testset "Persona" begin
        p = Persona("Ada Lovelace"; age = 28, occupation = "mathematician",
                    traits = ["analytical", "visionary"], background = "Works on the Engine.")
        @test p.name == "Ada Lovelace"
        @test p.age == 28
        d = describe(p)
        @test occursin("Name: Ada Lovelace", d)
        @test occursin("Age: 28", d)
        @test occursin("analytical, visionary", d)

        minimal = Persona("Bob")
        dm = describe(minimal)
        @test occursin("Name: Bob", dm)
        @test !occursin("Age", dm)
    end

    @testset "MemoryStream" begin
        s = MemoryStream()
        @test isempty(s)
        i = remember!(s, "I woke up"; tick = 0, importance = 3.0)
        @test i == 1
        remember!(s, "I saw a friend"; tick = 1, importance = 8.0)
        @test length(s) == 2
        @test s[1].kind == :observation
        @test_throws ArgumentError remember!(s, "bad"; kind = :nonsense)
    end

    @testset "retrieve — recency" begin
        s = MemoryStream()
        remember!(s, "alpha event"; tick = 0, importance = 5.0)
        remember!(s, "beta event"; tick = 10, importance = 5.0)
        top = retrieve(s, "completely unrelated query"; tick = 10, n = 1,
                       weights = (1.0, 0.0, 0.0))
        @test length(top) == 1
        @test top[1].content == "beta event"   # newer wins on recency
    end

    @testset "retrieve — importance" begin
        s = MemoryStream()
        remember!(s, "mundane note"; tick = 5, importance = 2.0)
        remember!(s, "major news"; tick = 5, importance = 9.0)
        top = retrieve(s, "xyz"; tick = 5, n = 1, weights = (0.0, 1.0, 0.0))
        @test top[1].content == "major news"
    end

    @testset "retrieve — relevance (lexical)" begin
        s = MemoryStream()
        remember!(s, "the cat sat on the mat"; tick = 0)
        remember!(s, "quantum chromodynamics lecture"; tick = 0)
        top = retrieve(s, "cat"; tick = 1, n = 1, weights = (0.0, 0.0, 1.0))
        @test top[1].content == "the cat sat on the mat"
    end

    @testset "retrieve — relevance (embeddings)" begin
        s = MemoryStream()
        remember!(s, "aaaaa"; tick = 0)
        remember!(s, "jjjjj"; tick = 0)
        emb = FakeEmbedder()
        embed_memories!(s, emb)
        @test all(e -> e.embedding !== nothing, s.entries)
        top = retrieve(s, "aaaa"; tick = 1, n = 1, llm = emb,
                       weights = (0.0, 0.0, 1.0))
        @test top[1].content == "aaaaa"
    end

    @testset "retrieve — updates last_accessed" begin
        s = MemoryStream()
        remember!(s, "remembered thing"; tick = 0)
        retrieve(s, "remembered thing"; tick = 42, n = 1)
        @test s[1].last_accessed == 42
    end

    @testset "Mind & GenerativeAgent" begin
        m = Mind(Persona("Carol"))
        @test m.status == "idle"
        @test isempty(m.plan)
        @test mind(m) === m
        @test persona(m).name == "Carol"
    end

    @testset "rate_importance" begin
        llm = ScriptedLLM(smart_responder)
        @test rate_importance(llm, "I had lunch") == 7.0
        @test rate_importance(ScriptedLLM("no number here"), "x") == 5.0
        @test rate_importance(ScriptedLLM("99"), "x") == 10.0   # clamped
    end

    @testset "observe! & recall" begin
        m = Mind(Persona("Dave"))
        observe!(m, "I met a stranger"; tick = 1, importance = 6.0)
        @test length(m.memory) == 1
        observe!(m, "rated automatically"; tick = 2, llm = ScriptedLLM("8"))
        @test m.memory[2].importance == 8.0
        got = recall(m, "stranger"; tick = 3, n = 2)
        @test any(e -> e.content == "I met a stranger", got)
    end

    @testset "reflect!" begin
        llm = ScriptedLLM(smart_responder)
        m = Mind(Persona("Eve"))
        for k in 1:5
            observe!(m, "observation number $k about work"; tick = k, importance = 5.0)
        end
        reflections = reflect!(m, llm; tick = 6, n_questions = 2)
        @test !isempty(reflections)
        @test any(e -> e.kind == :reflection, m.memory.entries)
        rkind = filter(e -> e.kind == :reflection, m.memory.entries)
        @test all(e -> !isempty(e.citations), rkind)
    end

    @testset "maybe_reflect!" begin
        llm = ScriptedLLM(smart_responder)
        m = Mind(Persona("Frank"))
        observe!(m, "small thing"; tick = 1, importance = 5.0)
        @test maybe_reflect!(m, llm; tick = 2, threshold = 1000.0) == false
        for k in 1:30
            observe!(m, "busy day event $k"; tick = k, importance = 9.0)
        end
        @test maybe_reflect!(m, llm; tick = 31, threshold = 50.0, n_questions = 1) == true
    end

    @testset "plan!" begin
        llm = ScriptedLLM(smart_responder)
        m = Mind(Persona("Grace"))
        steps = plan!(m, llm; tick = 0, horizon = "today")
        @test steps == m.plan
        @test "Review notes" in steps
        @test any(e -> e.kind == :plan, m.memory.entries)
    end

    @testset "decide" begin
        llm = ScriptedLLM(smart_responder)
        m = Mind(Persona("Heidi"))
        observe!(m, "the report is due tomorrow"; tick = 0, importance = 7.0)
        d = decide(m, llm, "It is late in the evening"; tick = 1)
        @test d isa Decision
        @test d.action == "keep working"
        @test occursin("deadline", d.reasoning)
        @test any(e -> occursin("situation", e.content), m.memory.entries)

        # explicit-options form
        opt = ScriptedLLM("ACTION: cooperate\nREASON: trust pays off")
        d2 = decide(Mind(Persona("Ivan")), opt, "A prisoner's dilemma";
                    tick = 0, options = ["cooperate", "defect"])
        @test d2.action == "cooperate"
    end

    @testset "decide — unformatted reply" begin
        m = Mind(Persona("Judy"))
        d = decide(m, ScriptedLLM("She simply leaves the room."),
                   "An awkward moment"; tick = 0)
        @test occursin("leaves the room", d.action)
    end

    @testset "converse" begin
        llm = ScriptedLLM(p -> "Let's work together.")
        a = Mind(Persona("Kim"))
        b = Mind(Persona("Lee"))
        dialogue = converse(a, b, llm; topic = "a joint project", turns = 2, tick = 0)
        @test length(dialogue) == 4
        @test dialogue[1].first == "Kim"
        @test dialogue[2].first == "Lee"
        @test any(e -> occursin("conversation", e.content), a.memory.entries)
        @test any(e -> occursin("conversation", e.content), b.memory.entries)
    end

    @testset "generative_abm" begin
        llm = ScriptedLLM(smart_responder)
        model = generative_abm(; llm = llm)
        @test abmllm(model) === llm
        @test abmclock(model) == 0

        ada = add_generative_agent!(model, Persona("Ada"); status = "thinking")
        @test ada isa GenerativeAgent
        @test persona(ada).name == "Ada"
        @test nagents(model) == 1

        # default model_step! advances the clock
        step!(model, 3)
        @test abmclock(model) == 3
    end

    @testset "generative_abm — custom agent_step!" begin
        llm = ScriptedLLM(smart_responder)
        function act!(agent, model)
            d = decide(agent, abmllm(model), "what to do at tick $(abmclock(model))";
                       tick = abmclock(model))
            mind(agent).status = d.action
            return nothing
        end
        model = generative_abm(; llm = llm, agent_step! = act!)
        agent = add_generative_agent!(model, Persona("Mallory"))
        step!(model, 1)
        @test mind(agent).status == "keep working"
        @test abmclock(model) == 1
    end

end
