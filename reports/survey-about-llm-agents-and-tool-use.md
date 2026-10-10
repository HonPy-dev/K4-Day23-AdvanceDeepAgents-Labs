# LLM Agents and Tool Use: Architectures, Learning, Evaluation, and Safety

## TL;DR
- Tool use involves deciding when to call, selecting a capability, forming arguments, interpreting results, and continuing or stopping; modular systems and interleaved reasoning/action are foundational patterns. [1][2]
- Learning approaches range from eliciting tool calls through prompting to training the model to insert calls, while API scale motivates separate retrieval and routing components. [3][4]
- Strong evaluations increasingly test executable outcomes, policy compliance, and repeatability—not just whether a generated call looks syntactically correct. [5][6][7]
- Benchmark results indicate a persistent gap between single-call competence and reliable completion of long-horizon interactive tasks. [6][8][9]
- Tool access creates an attack surface: retrieved content may manipulate an agent into harmful calls, so permission boundaries and independent authorization checks matter alongside model behavior. [10][11][12]

## Background
An LLM agent can be understood as a model embedded in a control loop: it receives a request and observations, proposes an action (possibly a tool call), obtains an external result, and uses that result to determine what happens next. This changes the system's failure surface. A language-only answer can be wrong; a tool-using system can additionally choose the wrong capability, supply bad arguments, misread a result, violate a policy, repeat a side effect, or act on hostile content. MRKL framed this as a modular combination of language models and specialized modules, while ReAct describes interleaving reasoning and action. [1][2]

The literature and product patterns cover different levels of the stack. Some work trains models to make calls; some supplies a controller, router, or planner around a general model; and some defines a runtime contract for the application to execute calls and return results. These are complementary rather than mutually exclusive choices: even a learned caller needs tool descriptions, an execution environment, result handling, and safeguards. [3][13][14]

## 1. From modular routing to agent loops and orchestration

MRKL offers a modular view that combines language models with specialized symbolic or neural modules, such as calculators or database APIs. ReAct instead interleaves reasoning and task-specific actions, including search and lookup in its tested examples. The contrast is between composing specialized capabilities and structuring action in context. [1][2] [1][2]

The distinction is useful. Routing asks “which capability handles this?”; an action-observation loop asks “what should happen next given what just happened?” Real tool tasks need both, plus a stop decision. HuggingGPT extends this to task decomposition and heterogeneous model orchestration: its controller plans subtasks, selects expert models, executes them, and integrates results. Its authors also note that exposing many candidate models creates a context-limit problem, motivating filtering and ranking. [13]

Recent runtime documentation turns these ideas into explicit operational interfaces. A function call can be a structured request with JSON-Schema arguments, but the application—not the model—executes it and returns a success or error result; the turn then continues. Other documented patterns include deferred tool discovery, programmatic coordination of eligible calls, approval controls, and specialist-agent handoffs. These are implementation patterns, not controlled evidence that one architecture outperforms another. [14][15][16]

## 2. Learning tool use and scaling tool choice

Toolformer treats API use as a behavior a language model can acquire from a small number of demonstrations per API. It samples candidate calls, executes them, and filters for calls whose results improve subsequent-token prediction; the resulting augmented examples are used for fine-tuning. The tested tools include question answering, search, calculation, calendar, and translation. The approach links call decisions to the model's prediction objective, but its premise requires that tool responses be representable in text and useful for prediction. [3]

A different scaling problem is choosing among a large catalog. ToolLLM builds ToolBench from 16,464 RapidAPI REST APIs across 49 categories, with single- and multi-tool instructions. Its pipeline uses searched solution paths and a decision-tree search over reasoning paths; it also trains a dense retriever to shortlist relevant APIs before ToolLLaMA makes multi-round decisions. This decomposes the burden: retrieval narrows the choice set, while the model plans and invokes tools. The paper evaluates generalization to unseen instructions and tools, underscoring that API coverage and transfer are separate issues from success on familiar calls. [4]

Together, these approaches reveal a continuum: inference-time prompting can elicit tool behavior without retraining; fine-tuning can internalize call patterns; retrieval and catalogs can manage large tool spaces; and orchestration can delegate work to specialist models or agents. The sources do not establish that any single approach is universally preferable. The proper design depends on catalog size, whether APIs are dynamic, the need for multi-step planning, latency/context budgets, and how much control the host application must retain. [2][3][13][15][4]

## 3. Evaluation: from call syntax to task outcomes

Tool-use evaluation has several layers. API-Bank evaluates calling, retrieval-plus-calling, and planning-plus-retrieval-plus-calling; its executable setup compares whether predicted calls perform the same database queries or modifications and yield the same results as annotated calls. Its reported performance declines as retrieval and planning demands are added, illustrating why a correctly formatted single call is not equivalent to solving a tool-use task. [5][17]

ToolLLM pairs its model with a learned API retriever that recommends relevant APIs from a large catalog before multi-round decisions. BFCL evaluates serial and parallel calls and uses AST-based checks; its later agentic framing includes abstention and stateful multi-step reasoning. Different metrics expose different properties: structural validity checks whether a call is well-formed; execution checks whether it works; state comparison checks whether the environment reached a target; and conversational judging assesses whether the interaction is acceptable. No one metric captures all of these. [4][7]

Interactive benchmarks stress the complete loop. τ-bench places agents in simulated retail and airline settings with users, APIs, databases, and domain policies; it judges final state against a goal and repeats tasks to measure consistency with pass^k. Its reported results show low success and especially low repeated-trial reliability for tested agents, with errors involving database reasoning, policy adherence, and compound requests. WebArena evaluates long-horizon web tasks in reproducible applications and reports that the best tested GPT-4-based agent achieved 14.41% end-to-end success versus 78.24% for humans. These figures are benchmark-specific, not universal estimates of production performance. [6][9]

AgentBench broadens the interactive settings to eight environments, and reports persistent weaknesses in long-term reasoning, decisions, and instruction-following; it identifies task-limit-exceeded outcomes as the predominant failure category. Collectively, these evaluations shift the unit of analysis from “did the model call a tool?” to “did the agent reliably reach the intended, permitted outcome across a sequence of interactions?” [8]

## 4. Reliability and safety as system properties

Long-horizon tasks compound ordinary errors. An agent may overlook information in an observation, fail to recover from an unproductive action, or exceed the interaction budget. WebArena authors point to weak active exploration and failure recovery, while AgentBench identifies long-term reasoning and instruction following as obstacles. τ-bench additionally highlights policy adherence and complex compound requests. These findings suggest that better tool-call syntax alone will not resolve errors in state tracking, recovery, planning, or policy interpretation. [6][8][9]

Tool outputs also create an indirect prompt-injection path: an agent may encounter malicious instructions embedded in emails, web pages, or reviews and then misuse its tools. INJECAGENT evaluates this threat across 1,054 cases involving 17 user tools and 62 attacker tools; it reports 24% attack success for ReAct-prompted GPT-4 in the base setting and nearly double that with an added “hacking prompt.” These results characterize benchmark conditions, not universal real-world incident rates. [10] [10]

Safety tests likewise expose more than refusal behavior. Agent-SafetyBench evaluates 16 agents across 349 interactive environments and 2,000 cases; none scored above 60% overall safety in its study. The authors characterize robustness and risk awareness as key weaknesses. These results motivate testing permission boundaries and independent safeguards, but the safety scores do not establish which specific mitigation is most effective. [11]

A practical implication is defense in depth. The model can be instructed to act cautiously, but the host should enforce authorization and parameter constraints outside the model; tools should be scoped to the task; untrusted retrieved text should not acquire authority to change permissions; and consequential actions should have explicit authorization and approval controls. OWASP recommends these controls as guidance; the available notes do not establish comparative effectiveness for each mitigation. [12][15]

## Trends and open problems

The field is moving beyond isolated function calls toward multi-step, stateful and multi-agent work: model controllers plan chains, tools can be dynamically surfaced, programmatic coordination can reduce model round trips, and specialist agents may receive handoffs or be invoked as bounded tools. These additions increase expressive power but also enlarge the control and security surface. [13][15][16]

Evaluation is converging on executable and outcome-oriented measures, including state-based correctness, trajectory-aware checks, policy constraints, and repeated-trial consistency. A continuing challenge is balancing strictness with flexibility: exact trajectory matching can reject valid alternative paths, while final-state checks may miss whether required calls or policy steps occurred. Benchmarks also vary in tools, environments, model versions, and simulated users, so comparisons should be interpreted within their stated setups. [5][6][7]

Open research questions include how to robustly recover from failed actions; maintain state and policies over long horizons; generalize to unfamiliar APIs; distinguish untrusted content from authorized instructions; and quantify reliability across repeated attempts, not merely a best run. Equally important is system-level evaluation of safeguards: benchmarks should test whether scoped permissions, external authorization checks, and approvals prevent harmful actions without making benign tasks unusable. Current sources identify these needs, but do not provide a unified answer or universal performance threshold. [4][6][10][11][12]

## References
[1] MRKL Systems A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arxiv. https://arxiv.org/abs/2205.00445 (n.d.)
[2] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[3] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[4] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arxiv. https://arxiv.org/abs/2307.16789 (n.d.)
[5] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. web. https://aclanthology.org/2023.emnlp-main.187.pdf (n.d.)
[6] τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arxiv. https://arxiv.org/abs/2406.12045 (n.d.)
[7] The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. web. https://proceedings.mlr.press/v267/patil25a.html (2025-10-06)
[8] AgentBench: Evaluating LLMs as Agents. arxiv. https://arxiv.org/abs/2308.03688 (2025-10-04)
[9] WebArena: A Realistic Web Environment for Building Autonomous Agents. arxiv. https://arxiv.org/abs/2307.13854 (2023-07-26)
[10] INJECAGENT: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. arxiv. https://arxiv.org/html/2403.02691v1 (2024-03-05)
[11] AGENT-SAFETYBENCH: Evaluating the Safety of LLM Agents. arxiv. https://arxiv.org/pdf/2412.14470 (2024-12-19)
[12] OWASP Top 10 for LLM Applications 2025. web. https://genai.owasp.org/download/43299/ (2024-11-18)
[13] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. arxiv. https://arxiv.org/abs/2303.17580 (2023-03-30)
[14] Functions. web. https://developers.openai.com/api/docs/guides/agents-api/tools/functions (n.d.)
[15] Tools - OpenAI Agents SDK. web. https://openai.github.io/openai-agents-python/tools/ (n.d.)
[16] Orchestration and handoffs. web. https://developers.openai.com/api/docs/guides/agents/orchestration (n.d.)
[17] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. hf-search. https://huggingface.co/papers/2304.08244 (2023-04-14)
