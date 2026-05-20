# References

GABM.jl sits at the meeting point of two literatures: the decades-old tradition
of agent-based modelling, and the recent work on language models as simulated
decision-makers. Citations elsewhere in this manual refer to the entries below.

## Generative agents and language-model simulacra

- **Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., &
  Bernstein, M. S.** (2023). Generative agents: Interactive simulacra of human
  behavior. *Proceedings of the 36th Annual ACM Symposium on User Interface
  Software and Technology (UIST '23)*. The originating paper. Introduces the
  *memory stream*, the recency–importance–relevance *retrieval* function,
  *reflection* as the synthesis of higher-level insight, and *planning* as the
  decomposition of intentions. The cognitive architecture of GABM.jl —
  [`MemoryStream`](@ref), [`retrieve`](@ref), [`reflect!`](@ref),
  [`plan!`](@ref) — follows this paper directly.

- **Park, J. S., Popowski, L., Cai, C. J., Morris, M. R., Liang, P., &
  Bernstein, M. S.** (2022). Social simulacra: Creating populated prototypes
  for social computing systems. *UIST '22*. The precursor to the generative-
  agents paper. Shows that a language model can populate a prototype community
  with plausible, heterogeneous members — the first argument that personas, not
  parameter vectors, can specify a synthetic population.

- **Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., &
  Wingate, D.** (2023). Out of one, many: Using language models to simulate
  human samples. *Political Analysis*, 31(3), 337–351. Introduces *algorithmic
  fidelity* — the degree to which a conditioned language model reproduces the
  response distribution of a human subpopulation — and the validity questions
  any generative model of human behaviour must answer.

- **Aher, G., Arriaga, R. I., & Kalai, A. T.** (2023). Using large language
  models to simulate multiple humans and replicate human subject studies.
  *Proceedings of the 40th International Conference on Machine Learning
  (ICML '23)*. Replicates classic behavioural experiments with language-model
  subjects, and catalogues where the simulation succeeds and where it fails.

- **Horton, J. J.** (2023). Large language models as simulated economic agents:
  What can we learn from *homo silicus*? *NBER Working Paper 31122*. Argues
  that language-model agents are a usable substitute for human subjects in
  exploratory economic experiments, and runs several to make the case.

## Foundations of agent-based modelling

- **Schelling, T. C.** (1971). Dynamic models of segregation. *Journal of
  Mathematical Sociology*, 1(2), 143–186. The canonical agent-based model: mild
  individual preferences over neighbours produce stark collective segregation.
  The benchmark against which a generative treatment of the same problem is
  most naturally compared.

- **Axelrod, R.** (1984). *The Evolution of Cooperation*. New York: Basic
  Books. The iterated prisoner's dilemma as an agent-based model, and the
  origin of the repeated-cooperation problem used as a running example in this
  manual.

- **Epstein, J. M., & Axtell, R.** (1996). *Growing Artificial Societies:
  Social Science from the Bottom Up*. Cambridge, MA: MIT Press. The Sugarscape
  model and the *generative* programme for social science — explaining a
  macro-regularity by exhibiting agents whose interaction produces it. GABM.jl
  borrows the word "generative" from this tradition; Park et al. supply the new
  sense of it.

- **Epstein, J. M.** (2006). *Generative Social Science: Studies in Agent-Based
  Computational Modeling*. Princeton: Princeton University Press. A book-length
  statement of the generativist methodology and its standards of explanation.

- **Bonabeau, E.** (2002). Agent-based modeling: Methods and techniques for
  simulating human systems. *Proceedings of the National Academy of Sciences*,
  99(suppl. 3), 7280–7287. A widely cited survey of when an agent-based model
  is the appropriate tool — and when it is not.

- **Macal, C. M., & North, M. J.** (2010). Tutorial on agent-based modelling
  and simulation. *Journal of Simulation*, 4(3), 151–162. A practical
  introduction to the components of an agent-based model: agents, environment,
  scheduling, and data collection — the components Agents.jl supplies.

## Tooling

- **Datseris, G., Vahdati, A. R., & DuBois, T. C.** (2022). Agents.jl: A
  performant and feature-full agent-based modelling software. *Simulation*,
  98(4). The simulation engine GABM.jl builds on. Provides `StandardABM`,
  discrete and continuous spaces, schedulers, `run!`, and data collection;
  GABM.jl adds only the cognitive layer.

- **PromptingTools.jl.** Šplíchal, J., and contributors. A provider-agnostic
  Julia interface to language models — OpenAI, Anthropic, Mistral, Ollama, and
  others. The package behind [`PromptingToolsLLM`](@ref).
  <https://github.com/svilupp/PromptingTools.jl>

## Method and critique

- **Gao, C., Lan, X., Li, N., Yuan, Y., Ding, J., Zhou, Z., Xu, F., & Li, Y.**
  (2024). Large language models empowered agent-based modeling and simulation:
  A survey and perspectives. *Humanities and Social Sciences Communications*,
  11. A survey of the emerging field GABM.jl is part of, organised by
  application domain and by the design choices a generative model must make.

- **Grossmann, I., Feinberg, M., Parker, D. C., Christakis, N. A., Tetlock,
  P. E., & Cunningham, W. A.** (2023). AI and the transformation of social
  science research. *Science*, 380(6650), 1108–1109. A short, pointed statement
  of both the promise and the validity hazards of using language models as
  research subjects — required reading before a generative model is used to
  make a claim about the world.

- **Bail, C. A.** (2024). Can generative AI improve social science?
  *Proceedings of the National Academy of Sciences*, 121(21). Assesses where
  generative agents can and cannot substitute for human data, and the
  reproducibility and bias concerns specific to language-model populations —
  the concerns the deterministic [`ScriptedLLM`](@ref) backend is designed to
  help control.
