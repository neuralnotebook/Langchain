# LangChain v1 — Crash Course Notes

> Based on Krishna's 10.5hr YouTube tutorial covering LangChain v1 updates, agents, model integration, streaming, and batch processing.

---

## 1. Project Setup with UV Package Manager

UV is an extremely fast Python package and project manager written in **Rust**.

### Install UV

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Initialize Project

```bash
uv init          # initializes repo, creates pyproject.toml + main.py
uv venv          # creates virtual environment (.venv folder, uses Python 3.13 by default)
```

### Activate Virtual Environment

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Install Libraries

**Option A — from requirements.txt:**
```bash
uv add -r requirements.txt
```

**Option B — individual library:**
```bash
uv add langchain
uv add ipykernel     # for Jupyter notebook kernel
```

### `requirements.txt`

```
langchain
langchain-community
langchain-openai
langchain-groq
langchain-google-genai
python-dotenv
```

> Installed versions are tracked in `pyproject.toml` (e.g. `langchain==1.1.0`).

---

## 2. Environment Setup (API Keys)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

Load in code:

```python
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key  = os.getenv("OPENAI_API_KEY")
groq_api_key    = os.getenv("GROQ_API_KEY")
google_api_key  = os.getenv("GOOGLE_API_KEY")
```

---

## 3. What is an Agent?

A plain LLM has a **training cutoff** — it can't answer questions about today's news, current prices, etc.

An **agent** solves this by:
1. Taking a user input
2. **Deciding** which tool(s) can help answer it
3. Calling the tool → getting **context**
4. Generating the final output using that context

```
User Input → LLM (decision) → Tool (e.g. Google Search) → Context → LLM Output
```

> This decision-making loop is the core of agentic AI.

---

## 4. Creating an Agent (LangChain v1)

### Imports

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
```

### Define a Tool (as a Python function)

```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."
```

> The **docstring** is critical — LangChain uses it to help the LLM decide when to call the tool.

### Create the Agent

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1")

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
```

### Invoke the Agent

```python
# Option A — dict with messages key
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in New York?"}]
})

# Option B — plain string (shorthand)
response = agent.invoke("What is the weather in New York?")

# Get just the last message content
print(response["messages"][-1].content)
```

> When the LLM decides to call a tool, you'll see: **Human Message → AI (tool call) → Tool Message → AI (final output)**

### Check LangChain Version

```python
import langchain
print(langchain.__version__)   # e.g. 1.1.0
```

---

## 5. Model Integration

LangChain v1 supports multiple LLM providers through a unified interface.

### Method A — `init_chat_model` (universal, recommended)

```python
from langchain.chat_models import init_chat_model

# OpenAI
model = init_chat_model("gpt-4.1")

# Google Gemini
model = init_chat_model("google_genai:gemini-2.5-flash")

# Groq
model = init_chat_model("groq:qwen-whatever-model")

response = model.invoke("Hello, how are you?")
print(response.content)
```

### Method B — Provider-specific classes

```python
# OpenAI
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4.1")

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Groq
from langchain_groq import ChatGroq
model = ChatGroq(model="<groq-model-name>")

response = model.invoke("Why do parrots talk?")
print(response.content)
```

> Both methods work. `init_chat_model` is cleaner for switching between providers.

---

## 6. Streaming

By default, `.invoke()` waits for the **entire response** before displaying anything.

**Streaming** displays output as it's being generated — much better UX for long responses.

```python
# Basic streaming
for chunk in model.stream("Write a 200 word paragraph on artificial intelligence"):
    print(chunk.text, end="|", flush=True)
```

- `end="|"` — custom delimiter to visualize token boundaries
- `flush=True` — forces immediate print (don't buffer)

> Use `.stream()` instead of `.invoke()` for chatbots and interactive apps.

---

## 7. Batch Processing

Send **multiple independent requests in parallel** to the model.

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])

# All 3 run in parallel; responses arrive together
for r in responses:
    print(r.content)
```

### Control Parallelism

```python
responses = model.batch(
    ["Question 1", "Question 2", "Question 3", "..."],
    config={"max_concurrency": 5}   # max 5 parallel calls at once
)
```

> Use batch when you need to process many prompts at once — reduces latency and cost.

---

## 8. Quick Reference

| Task | Method |
|---|---|
| Initialize project | `uv init` |
| Create virtualenv | `uv venv` |
| Install from requirements | `uv add -r requirements.txt` |
| Install single library | `uv add <library>` |
| Load any LLM | `init_chat_model("provider:model")` |
| Single call | `model.invoke("prompt")` |
| Stream output | `model.stream("prompt")` |
| Parallel calls | `model.batch(["q1", "q2", ...])` |
| Create agent | `create_agent(model, tools=[...])` |
| Run agent | `agent.invoke({"messages": [...]})` |

---

## Coming Up in the Series

- Message types (HumanMessage, AIMessage, ToolMessage)
- Short-term memory
- Middleware (built-in & custom)
- LangGraph crash course
- RAG (Traditional → Agentic → Vectorless)
- Deep Research Agents
- Guardrails & LLM Evaluation
- LLM Gateways