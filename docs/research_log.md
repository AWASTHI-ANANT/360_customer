# Research Log: Agentic Customer 360

This log lists what I read for this project, what I took from each
source, and which decision in the system it shaped. Each entry is
tagged:

- **Applied**: directly shaped a design decision
- **Background**: built understanding, no specific decision
- **Considered, not used**: evaluated and deliberately not adopted

Most reading comes from the Inter IIT training camp resources. The
last two sections list where my reading did not cover a decision, and
how I used AI assistance.

## Decision map

| Decision in the system | Informed by |
|---|---|
| Fixed pipeline of specialised agents instead of one open-ended ReAct loop | [11], [12], [13], [14] |
| Working / episodic / semantic memory split | [11], [12], [19] |
| Agents share structured findings, not full context or reasoning traces | [12], [13] |
| JSON-only outputs and cached LLM responses for reproducibility | [5] |
| No fine-tuning; hosted LLM for text, deterministic code for signals | [6]–[10], [28], [29] |
| LLM used only where text understanding is needed | [3], [28], [29] |
| Policy retrieval with citable chunk IDs | [15], [16], [17], [18], plus AI assistance (see RAG note) |
| Structure-aware chunking of policy documents | [25] |
| Validation and fallbacks around retrieval and LLM output | [20], [21], [22] |
| Live indexing of incoming customer text (future work, not built) | [23] |
| Explicit tool and schema definitions per agent | [14], [26], [27] |

---

## 1. Foundations

### [1] The Annotated Transformer (Harvard NLP)
https://nlp.seas.harvard.edu/annotated-transformer/
**Background.** Line-by-line implementation of the original
Transformer paper. Gave me a working understanding of attention,
which I used later to reason about why context size and structure
matter in prompts.

### [2] The Illustrated Transformer (Jay Alammar)
https://jalammar.github.io/illustrated-transformer/
**Background.** Visual walkthrough of self-attention, multi-head
attention and the encoder-decoder structure.

### [3] Understanding Encoder and Decoder LLMs (Sebastian Raschka)
https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder
**Applied (as an alternative I recorded).** Encoder-only models are
suited to classification; decoder-only models to generation. The
Support Agent's sentiment and intent task is really classification,
so a small encoder classifier would be a cheaper, local alternative
to an LLM call. I kept the LLM for speed of building (one model
handles sentiment, intent and explanation extraction together) and
listed the encoder option as future work.

### [4] Some Intuition on Attention and the Transformer (Eugene Yan)
https://eugeneyan.com/writing/attention/
**Background.** Intuition for how attention weighs parts of the
input.

### [5] How to Generate Text (Hugging Face)
https://huggingface.co/blog/how-to-generate
**Applied.** Covers greedy decoding, beam search, and sampling
(top-k, top-p, temperature). Sampling adds variety, which is useful
for creative text but harmful when outputs must be reproducible.
Decision: every LLM call must return JSON matching a fixed schema,
and responses are cached on disk by a hash of (model, system prompt,
user prompt), so a re-run replays the cached response instead of
sampling again. The same input then always gives the same decision,
which the explainability and traceability requirements depend on.
Temperature 0 is sent only to models that still accept sampling
parameters; the default model (`claude-opus-5`) rejects them, so for
it the cache alone provides reproducibility.
Where: `c360/llm_client.py`, `config/llm.py`.

## 2. Fine-tuning and efficiency (considered, not used)

### [6] Fine-tuning guide (Hugging Face Transformers)
https://huggingface.co/docs/transformers/en/training
### [7] Understanding Parameter-Efficient Finetuning (Lightning AI)
https://lightning.ai/pages/community/article/understanding-llama-adapters/
### [8] LoRA (Hu et al., 2021) and Lightning's LoRA tutorial
https://arxiv.org/abs/2106.09685 ·
https://lightning.ai/pages/community/tutorial/lora-llm/
### [9] Quantization resources
mlabonne's Introduction to Weight Quantization; Hugging Face
quantization overview and blog; a video walkthrough on quantization.
### [10] Mixture of Experts (Hugging Face); LLM training guide
(rentry); Single-GPU training (Hugging Face)
https://huggingface.co/blog/moe · https://rentry.org/llm-training ·
https://huggingface.co/docs/transformers/en/perf_train_gpu_one

**Considered, not used.** These showed how LoRA and quantization make
fine-tuning feasible on limited hardware. I decided against
fine-tuning because there is no labelled training data (three
practice scenarios with about ten checkpoints in total), any model
trained on that would overfit, and prompts plus deterministic
features are faster to iterate on and easier to inspect. Quantization
becomes relevant only if a small model is self-hosted (see [28],
[29]). From the MoE blog I took one idea as an analogy only: a gating
network routes each input to specialised experts, much as our
pipeline routes each event to the agents whose `handles()` accepts
it.

## 3. Agents and reasoning

### [11] LLM Powered Autonomous Agents (Lilian Weng, 2023)
https://lilianweng.github.io/posts/2023-06-23-agent/
**Applied.** Breaks agents into planning, memory and tool use, and
separates short-term memory (what is in the context window) from
long-term memory (an external store queried when needed). This became
our split: working memory (the per-customer state board, overwritten
as state changes) and episodic memory (an append-only history of
episodes, queried by type and time). Where: `c360/memory_store.py`.

### [12] Context Engineering Guide (Prompting Guide)
https://www.promptingguide.ai/guides/context-engineering-guide
**Applied.** The quality of an agent's output depends on what is put
into its context and how it is structured. Decisions: agents pass
each other structured findings (value, confidence, evidence event
IDs), never raw reasoning or full event dumps; each LLM prompt gets
only recent findings plus a few relevant episodes; text is masked
before it enters any prompt.

### [13] Chain-of-Thought prompting (Wei et al., 2022)
https://arxiv.org/abs/2201.11903
**Applied, adapted.** Step-by-step reasoning improves results. I
adapted this: agents write a short rationale that must cite evidence
event IDs, but their reasoning is not passed to other agents, only
their conclusion. That keeps handoffs small and matches the
problem statement's rule that agents share conclusions, not
chain-of-thought.

### [14] ReAct (Yao et al., 2022); Prompting techniques (Prompting Guide);
Inter IIT 12.0 DevRev work on tool planning and function calling
https://arxiv.org/abs/2210.03629 ·
https://www.promptingguide.ai/techniques ·
https://maximus-21.github.io/LLM-Agents-for-Tool-Planning-and-Function-Calling-Part-1/
**Considered, then adapted.** ReAct interleaves reasoning with tool
calls in an open loop. I considered giving each agent a ReAct loop,
but in this problem the steps are known in advance (collect signals,
combine them, decide, check, approve). A fixed pipeline is cheaper,
deterministic, and far easier to trace than an open loop, so I used
the fixed pipeline and kept LLM calls for the steps that need
language understanding. From the DevRev work and the prompting
techniques guide I took the practice of defining each tool and output
format explicitly, and validating the model's output against it
before use.

## 4. Retrieval-augmented generation

### [15] RAG introduction (Prompting Guide)
https://www.promptingguide.ai/research/rag
### [16] LangChain RAG tutorial and video playlist
https://python.langchain.com/v0.2/docs/tutorials/rag/
### [17] Pinecone LangChain series (prompt templates); Pinecone FAISS series
https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates/ ·
https://www.pinecone.io/learn/series/faiss/
### [18] Advanced RAG Techniques (Pinecone blog)
https://www.pinecone.io/learn/advanced-rag-techniques/

**What I understood.** The basic pipeline: split documents into
chunks, index them, retrieve the most relevant chunks for a query,
and give them to the model so its answer is grounded in real text.
The Pinecone blog covered improvements beyond basic RAG, such as
better chunking, combining keyword and vector search, reranking, and
filtering by metadata.

**Honest note.** After reading these, I understood RAG as a general
pipeline, but I could not work out how retrieval should fit into this
particular architecture: what the corpus should be when the dataset
ships no documents, which agent should retrieve, and whether vector
search was needed at all. I asked Claude to handle the retrieval
design and implementation.

**What I understood after reviewing that design:**
- The corpus is policy we wrote ourselves (offer catalogue, cost
  caps, escalation rules), because the dataset contains none. This is
  listed as a limitation.
- Only the Action Agent retrieves, and only when an action is being
  drafted. It may choose an action subtype only from the retrieved
  chunks, and it cites their chunk IDs in its explanation.
- Retrieval uses keyword and section matching, not embeddings. The
  corpus is tiny, the lookups are exact state names, and explanations
  need the same chunk IDs on every run. Of the Pinecone techniques,
  the one that matters here is metadata filtering: chunks are selected
  by the inferred state's section. Hybrid search and reranking would
  add cost without benefit at this size.
- Only the policy corpus is indexed. Customer text (tickets, search
  queries) is not retrieved; it reaches the agents as events and
  findings instead. Indexing it as it arrives is future work (see
  [23]).
Where: `policy/`, `policy/retrieval.py`, `agents/action_agent.py`.

### [19] Improving ChatGPT with Knowledge Graphs (mlabonne)
https://mlabonne.github.io/blog/posts/Article_Improve_ChatGPT_with_Knowledge_Graphs.html
**Applied, loosely.** Structured facts can be retrieved more reliably
than text chunks when the data is relational. Customer history is
structured (event types, dates, amounts), so episodic memory stores
typed records queried by type and time rather than text searched by
similarity. It is not a graph database, but it follows the same idea.

### [20] Self-RAG (Asai et al., 2023)
https://arxiv.org/abs/2310.11511
**Applied, loosely.** The model decides when retrieval is needed and
critiques what it retrieves. Our system retrieves only when drafting
an action, not on every event.

### [21] Corrective RAG (Yan et al., 2024)
https://arxiv.org/abs/2401.15884
**Applied, loosely.** Evaluate retrieval quality and take a
corrective path when it is poor. Our Action Agent's output is
rejected if its chosen subtype is not among the retrieved chunks, and
every LLM step has a deterministic fallback.

### [22] Agentic RAG survey (arXiv 2501.09136)
https://arxiv.org/pdf/2501.09136
**Background, applied loosely.** Surveys ways of combining agents
with retrieval. Our design treats retrieval as a tool owned by one
agent rather than a global step before every call.

## 5. Multi-agent frameworks

### [23] Dynamic Multi-Agent RAG with Pathway (camp team's previous work)
https://github.com/mbappeenjoyer/Dynamic-Multi-Agent-RAG-Pathway
**Considered, not built.** A multi-agent RAG system built on Pathway,
a streaming framework whose index updates as data changes. I took the
principle that freshness should come from ingestion itself rather
than periodic rebuilds. Given the time available I did not build a
live index of incoming customer text; the policy corpus is small and
static, and it is loaded once per process. A Pathway-backed index is
the upgrade path for a production version, and this is listed as a
limitation.

### [24] Agentic framework paper (arXiv 2502.04644)
https://arxiv.org/abs/2502.04644
**Background.** A framework in which a reasoning model delegates to
specialised tool agents and keeps structured memory. Reinforced the
choice of specialised agents with distinct tools over one generalist.

## 6. Chunking by structure (code RAG resources)

### [25] Code RAG resources
https://arxiv.org/pdf/2504.10046 · https://arxiv.org/abs/2408.03910 ·
https://aider.chat/2023/10/22/repomap.html ·
https://blog.lancedb.com/rag-codebase-1/
**Applied.** Aider's repository map uses a parser (tree-sitter) to
split code along its real structure instead of fixed-size windows. I
applied the same principle to policy documents: chunks follow the
markdown's own sections and bullets, and each gets a stable ID such
as `offer_catalog#medical_hardship#2`, so citations stay meaningful.

## 7. Tool protocols

### [26] Model Context Protocol resources
Medium intro · DataCamp tutorial ·
https://modelcontextprotocol.io/docs/getting-started/intro
### [27] Pathway MCP server
https://github.com/pathwaycom/pathway/blob/main/python/pathway/xpacks/llm/mcp_server.py
**Considered, not used; applied in spirit.** MCP standardises how
tools and data are exposed to models, with typed input schemas. The
whole system runs in one process, so a protocol layer would add
overhead without benefit. I kept the idea: each agent file declares
its tools, inputs and output schema at the top. Exposing the event
store and memory through an MCP server is listed as future work.

## 8. Small language models

### [28] Small Language Models overview (Hugging Face)
https://huggingface.co/blog/jjokah/small-language-model
### [29] Small language models for agentic AI (arXiv 2506.02153)
https://arxiv.org/pdf/2506.02153
**Applied.** Many agent subtasks are narrow and repetitive, and small
models or plain code can handle them; large models should be reserved
for what needs them. This supports keeping signal detection,
thresholds and guardrails as deterministic code, with LLM calls only
for text. Future work: replace the Support Agent's LLM call with a
small local model, which would also keep customer text on our own
machine.

---

## Where my reading did not cover a decision

These decisions were made through AI-assisted design discussion and
analysis of the dataset, not from sources I read. They are the least
research-grounded parts of the design:

- **Streaming correctness**: recomputing windows from events sorted
  by event time, so late and duplicate events are handled.
- **Multi-agent coordination patterns**: the debate stage for
  conflicting reads and the critique-plus-one-revision step.
- **Evidence scoring**: combining signals with a noisy-OR formula and
  decaying older evidence.
- **Guardrails and HITL**: deterministic rules that override agents,
  an output veto on every checkpoint, and the append-only approval
  log. Data-layer access control and an executor that needs an
  approval token were designed but not built.
- **PII handling**: deriving signals before masking raw text.

With more time, I would ground these in literature on stream
processing (event time and windowing), multi-agent debate, agent
memory architectures, and LLM application security.

## AI assistance

I used Claude (chat) to understand the problem statement, analyse the
dataset, explore design options, and turn decisions into
implementation specifications, and Claude Code to write and test the
code. The retrieval design and the areas listed above came largely
from those discussions. Decisions I could check against my reading
are marked in the entries above. Thresholds and evidence weights were
calibrated against the three practice scenarios; every change and its
effect is recorded in `docs/tuning_log.md`, and overfitting to those
scenarios is a stated limitation.
