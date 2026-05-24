# Agents From Scratch

A hands-on learning repository for building AI agents using **LangGraph**, progressing from simple stateless LLM calls to complex human-in-the-loop agentic workflows.

## Tech Stack

- **Python 3.11**
- **LangGraph** — agent orchestration framework
- **LangChain** — LLM integrations (OpenAI, Groq, Google GenAI)
- **ChromaDB** — vector store for RAG
- **SQLite** — persistent checkpointing
- **uv** — fast Python package manager

## Setup

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd agents-from-scratch

# Install dependencies using uv
uv sync

# Copy environment variables and fill in your API keys
cp .env.example .env
```

### Environment Variables

Create a `.env` file in the root with the following keys:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Running Notebooks

```bash
# Activate the virtual environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook
```

---

## Notebooks — LangGraph Intro

| # | Notebook | Concept |
|---|----------|---------|
| 01 | [Simple LLM Agent](#01-simple-llm-agent) | Stateless LangGraph workflow |
| 02 | [Sentiment Review Reply Agent](#02-sentiment-review-reply-agent) | Conditional routing & structured output |
| 03 | [Simple RAG Agent](#03-simple-rag-agent) | Retrieval-Augmented Generation |
| 04 | [X Post Generator Agent](#04-x-post-generator-agent) | Iterative self-improvement loops |
| 05 | [Chatbot with In-Memory Saver](#05-chatbot-with-in-memory-saver) | In-memory checkpointing |
| 06 | [SQLite Persistence](#06-sqlite-persistence) | Long-term SQLite-backed state |
| 07 | [Persistence & Time Travel](#07-persistence--time-travel) | State history & checkpoint replay |
| 08 | [Fault Tolerance](#08-fault-tolerance) | Crash recovery with checkpoints |
| 09 | [Tools](#09-tools) | LLM tool binding & ToolNode |
| 10 | [Human-In-The-Loop (HITL)](#10-human-in-the-loop-hitl) | Interrupt/resume with human approval |

---

## 01 Simple LLM Agent

**File:** `1 - langgraph-intro/01_simple_llm_agent.ipynb`

Builds the most fundamental LangGraph workflow — a single-node stateless agent that sends user questions to an LLM and returns answers. Introduces the core LangGraph paradigm of defining state, registering nodes, and connecting them with edges.

**Key Concepts:**
- `StateGraph` initialization and compilation
- `TypedDict` state schema
- Defining and registering worker nodes
- Connecting nodes with `START` and `END` edges
- Stateless (no memory) LLM invocation via ChatGroq

---

## 02 Sentiment Review Reply Agent

**File:** `1 - langgraph-intro/02_sentiment_review_reply_agent.ipynb`
**PRD:** `1 - langgraph-intro/02_Insurance_Claim_AI_PRD.pdf`

Builds a multi-step agent that classifies customer reviews as positive or negative using structured LLM output, then routes to different response branches. Demonstrates conditional edges and cyclic graphs with iteration failsafes.

**Key Concepts:**
- Conditional edges and routing functions
- Structured output with Pydantic models and `with_structured_output`
- `Literal` type constraints for classification
- `operator.add` reducer for accumulating feedback across cycles
- Cyclic graphs with max-revision limits to prevent infinite loops

---

## 03 Simple RAG Agent

**File:** `1 - langgraph-intro/03_simple_rag_agent.ipynb`

Builds a Retrieval-Augmented Generation agent using Chroma vector store with Google embeddings to answer domain-specific questions. Demonstrates tool-based retrieval and how to distinguish on-topic from off-topic queries.

**Key Concepts:**
- Vector database (ChromaDB) setup and document ingestion
- Google GenAI embeddings for semantic search
- LangChain retriever wrapped as a tool
- Prebuilt `ToolNode` for executing tool calls
- `add_messages` reducer for message history
- Conditional routing based on presence of `tool_calls` in LLM response

---

## 04 X Post Generator Agent

**File:** `1 - langgraph-intro/04_x_post_genetor_agent.ipynb`

Builds an iterative tweet-generation agent that generates, evaluates, and optimizes posts in a loop until they meet strict quality criteria. Shows how to design self-improving agentic workflows with structured evaluation and loop termination.

**Key Concepts:**
- State machine with iteration counting for loop control
- `operator.add` for accumulating generation/feedback history
- Structured evaluation schemas with Pydantic
- Conditional edges with multiple exit paths (approved vs. max iterations)
- Failsafe mechanisms to prevent infinite loops in cyclic graphs

---

## 05 Chatbot with In-Memory Saver

**File:** `1 - langgraph-intro/05_chatbot_with_inmemory_saver.ipynb`

Upgrades the basic chatbot with session-scoped memory using `InMemorySaver`. Demonstrates the contrast between stateless and stateful workflows, and how `thread_id` isolates conversation history per session.

**Key Concepts:**
- `InMemorySaver` as a checkpointer for in-process state storage
- `thread_id` configuration for session isolation
- `add_messages` reducer for maintaining message history
- Stateful vs. stateless workflow comparison
- Passing `config={"configurable": {"thread_id": ...}}` at invocation time
