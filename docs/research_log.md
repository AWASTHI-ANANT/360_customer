# Research Log: Agentic Customer 360

What I read for this project, and whether it shaped the system. Tags:

- **Applied**: shaped a design decision
- **Background**: helped me understand, no specific decision
- **Studied, could not apply**: I read it but could not see how to use it here

Most of the reading comes from the Inter IIT training camp resources.

## Decision map

| Decision in the system | Informed by |
|---|---|
| Fixed pipeline of specialised agents instead of one open-ended ReAct loop | [8], [9], [10], [11] |
| Working and episodic memory split | [8], [9] |
| Agents share structured findings, not their full reasoning | [9], [10] |
| JSON-only outputs and cached LLM responses for reproducibility | [5] |
| No fine-tuning; LLM for text, plain code for signals | [6], [7], [14] |
| Policy retrieval with citable chunk IDs | Designed with AI assistance (see [12]) |

---

## 1. Foundations

**[1] The Annotated Transformer** (Harvard NLP):
https://nlp.seas.harvard.edu/annotated-transformer/
**[2] The Illustrated Transformer** (Jay Alammar):
https://jalammar.github.io/illustrated-transformer/
**[4] Some Intuition on Attention and the Transformer** (Eugene Yan):
https://eugeneyan.com/writing/attention/

**Background.** These gave me a basic understanding of attention and how
Transformers work.

**[3] Understanding Encoder and Decoder LLMs** (Sebastian Raschka):
https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder

**Applied (as future work).** Encoder models suit classification, and decoder
models suit generation. The Support Agent's sentiment and intent task is really
classification, so a small encoder model could replace the LLM call. I kept the
LLM because it was quicker to build, and noted the encoder option as future work.

**[5] How to Generate Text** (Hugging Face):
https://huggingface.co/blog/how-to-generate

**Applied.** Sampling adds variety, which is bad when the same input must always
give the same decision. So every LLM call returns JSON in a fixed format, and
responses are cached, so a re-run gives the same answer. Where:
`c360/llm_client.py`, `config/llm.py`.

## 2. Fine-tuning and efficiency

**[6]** Fine-tuning guide (Hugging Face), parameter-efficient fine-tuning and
LoRA (Lightning AI; Hu et al., 2021), quantization resources:
https://huggingface.co/docs/transformers/en/training ·
https://lightning.ai/pages/community/tutorial/lora-llm/ ·
https://arxiv.org/abs/2106.09685

**[7]** Mixture of Experts (Hugging Face), LLM training guide, single-GPU
training: https://huggingface.co/blog/moe · https://rentry.org/llm-training

**Studied, could not apply.** I read these, but there is no labelled data to
fine-tune on (three practice scenarios), so fine-tuning would only overfit.
Prompts and plain code were easier to build and check.

## 3. Agents

**[8] LLM Powered Autonomous Agents** (Lilian Weng):
https://lilianweng.github.io/posts/2023-06-23-agent/

**Applied.** It splits agent memory into short-term and long-term. That became
our working memory (the current state for each customer) and episodic memory (an
append-only history of what happened). Where: `c360/memory_store.py`.

**[9] Context Engineering Guide** (Prompting Guide):
https://www.promptingguide.ai/guides/context-engineering-guide

**Applied.** What goes into the prompt decides the quality of the output. So
agents pass each other short structured findings with evidence event IDs, not
raw data, and customer text is masked before it goes into any prompt.

**[10] Chain-of-Thought prompting** (Wei et al., 2022):
https://arxiv.org/abs/2201.11903

**Applied.** Agents write a short rationale that cites event IDs, but only their
conclusion is passed on to other agents, not their reasoning.

**[11] ReAct** (Yao et al., 2022), and the Inter IIT 12.0 DevRev write-up on tool
planning: https://arxiv.org/abs/2210.03629 ·
https://maximus-21.github.io/LLM-Agents-for-Tool-Planning-and-Function-Calling-Part-1/

**Applied, adapted.** ReAct lets an agent decide its next step in an open loop.
In this problem the steps are known in advance, so I used a fixed pipeline, which
is cheaper and easier to trace. From the DevRev write-up I took the habit of
defining each agent's output format and checking the model's output against it.

## 4. Retrieval (RAG)

**[12]** RAG introduction (Prompting Guide), LangChain RAG tutorial, Pinecone
series and Advanced RAG Techniques, Knowledge Graphs for ChatGPT (mlabonne),
Self-RAG, Corrective RAG, an Agentic RAG survey, and code-RAG resources:
https://www.promptingguide.ai/research/rag ·
https://python.langchain.com/v0.2/docs/tutorials/rag/ ·
https://www.pinecone.io/learn/advanced-rag-techniques/ ·
https://arxiv.org/abs/2310.11511 · https://arxiv.org/abs/2401.15884 ·
https://arxiv.org/pdf/2501.09136

**Studied, could not apply.** I read these and understood the basic idea: split
documents into chunks, find the relevant ones, and give them to the model. But I
could not work out how retrieval should fit into this architecture. The dataset
has no documents to retrieve, and I didn't know which agent should use it or
whether it was needed at all. The retrieval part of the system (a small policy
file the Action Agent looks up) was designed and built with Claude.

## 5. Other

**[13] Agentic framework paper** (arXiv 2502.04644):
https://arxiv.org/abs/2502.04644

**Background.** Supported the idea of several specialised agents rather than one
general one.

**[14] Small Language Models** (Hugging Face; arXiv 2506.02153):
https://huggingface.co/blog/jjokah/small-language-model ·
https://arxiv.org/pdf/2506.02153

**Applied.** Many agent tasks are simple and don't need a large model. So signal
detection, thresholds and guardrails are plain code, and the LLM is used only for
understanding text.

---

## What my reading did not cover

These parts came from discussions with Claude and from looking at the dataset,
not from anything I read:

- handling late and duplicate events by processing events in time order;
- the debate step when two readings conflict, and the critique-and-revise step;
- the scoring formula that combines signals and lets old evidence fade;
- guardrail rules, the output check on every checkpoint, and the approval log;
- masking customer details before text reaches the LLM;
- the retrieval design (see [12]).

## AI assistance

I used Claude (chat) to understand the problem, study the dataset, compare design
options and write specifications, and Claude Code to write and test the code.
Thresholds and weights were tuned on the three practice scenarios. Every change
is recorded in `docs/tuning_log.md`, and overfitting to those scenarios is a known
limitation.
