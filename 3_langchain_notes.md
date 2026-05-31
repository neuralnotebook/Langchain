# LangChain v1 — Part 3: Middleware (Deep Dive)

> Covers Summarization Middleware (3 trigger types) and Human-in-the-Loop Middleware with approve / edit / reject flows.

---

## 1. Middleware — Recap

Middleware adds **hooks** at specific points in the agent lifecycle:

```
Request → [before agent] → Agent → [before model] → Model → [after model] → [tool calls] → [after agent] → Response
```

Each hook can do: logging, validation, summarization, rate limiting, PII detection, human approval, etc.

---

## 2. Summarization Middleware

**Purpose:** Automatically compresses conversation history when it gets too long — prevents hitting token limits in long-running chatbots.

**What it does:**
- Watches the message list grow
- When a trigger condition is met → summarizes older messages using an LLM
- Keeps only recent messages + the summary going forward

### Imports

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
```

### Trigger Type 1 — Message Count

Summarize when the conversation hits N messages.

```python
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,                          # your LLM (e.g. GPT-4o-mini)
    tools=[],
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model=model,                  # LLM used for summarization (use a cheap model)
            trigger={"messages": 10},     # trigger when message count hits 10
            keep_recent=4                 # keep last 4 messages after summarizing
        )
    ]
)
```

```python
# Config — unique thread per user
config = {"configurable": {"thread_id": "user_001"}}

questions = ["What is 2+2?", "What is 10*5?", "What is 100/4?", "What is 15-7?"]

for q in questions:
    response = agent.invoke(
        {"messages": [HumanMessage(content=q)]},
        config=config
    )
    print(response["messages"][-1].content)
    print(f"Message count: {len(response['messages'])}")
    # When count hits 10 → auto-summarizes → count drops back down
```

**Output pattern:**
```
messages: 2 → 4 → 6 → 8 → 10 → [SUMMARIZATION] → 6 → ...
```

---

### Trigger Type 2 — Token Count

Summarize when total tokens exceed a threshold.

```python
def count_tokens(messages):
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4    # ~4 chars per token

agent = create_agent(
    model=model,
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger={"tokens": 550},      # summarize when tokens > 550
            keep_recent_tokens=200        # keep last 200 tokens after summarizing
        )
    ]
)
```

```python
config = {"configurable": {"thread_id": "user_002"}}

cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
        config=config
    )
    token_count = count_tokens(response["messages"])
    print(f"Tokens: {token_count}")
    # 149 → 302 → 456 → [SUMMARIZE at 550] → 396 → [SUMMARIZE] → 232 ...
```

---

### Trigger Type 3 — Fraction of Context Window

Summarize when messages consume X% of the model's total context window.

```python
agent = create_agent(
    model=model,
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger={"fraction": 0.005},  # 0.5% of context window
            keep_recent_tokens=200
        )
    ]
)
# For a 128k token model: 0.005 × 128000 = 640 tokens → triggers at 640 tokens
```

> Use fraction-based when you want trigger to scale automatically with different model context sizes.

---

### Summarization Trigger Comparison

| Trigger | Parameter | Best For |
|---|---|---|
| Message count | `{"messages": N}` | Simple chatbots, fixed conversation length |
| Token count | `{"tokens": N}` | Cost control, precise memory management |
| Context fraction | `{"fraction": 0.X}` | Multi-model setups, auto-scaling |

---

## 3. Human-in-the-Loop Middleware

**Purpose:** Pauses agent execution at critical tool calls and waits for a human to **approve**, **edit**, or **reject** before proceeding.

**Use cases:**
- Financial transactions (buying stocks, transfers)
- Sending emails / messages
- Database writes
- Any compliance or high-stakes operation

### Imports

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langgraph.types import Command   # needed to resume after interrupt
```

### Define Tools

```python
from langchain.tools import tool

@tool
def read_email_tool(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email content for ID: {email_id}"

@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {recipient} with subject: {subject}"
```

### Create Agent with Human-in-the-Loop Middleware

```python
agent = create_agent(
    model=model,                              # e.g. GPT-4o
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {          # pause ONLY on this tool
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "read_email_tool": False      # no interrupt for this tool
            }
        )
    ]
)
```

---

### Flow 1 — Approve ✅

```python
config = {"configurable": {"thread_id": "test-approve"}}

result = agent.invoke(
    {"messages": [HumanMessage(
        content="Send email to john@test.com with subject 'Hello' and body 'How are you?'"
    )]},
    config=config
)

# Agent pauses at send_email_tool → interrupt fires
if "interrupt" in result:
    print("⏸ Paused — awaiting human approval")

    # Human approves → resume execution
    result = agent.invoke(
        Command(resume={"decision_type": "approve"}),
        config=config
    )
    print(result["messages"][-1].content)
    # → "Email has been sent to john@test.com with subject Hello."
```

---

### Flow 2 — Reject ❌

```python
config = {"configurable": {"thread_id": "test-reject"}}

result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com ...")]},
    config=config
)

if "interrupt" in result:
    # Human rejects → abort the tool call
    result = agent.invoke(
        Command(resume={"decision_type": "reject"}),
        config=config
    )
    print(result["messages"][-1].content)
    # → "There was an issue sending the email." (tool call was rejected)
```

---

### Flow 3 — Edit ✏️

```python
config = {"configurable": {"thread_id": "test-edit"}}

# User accidentally gave wrong email
result = agent.invoke(
    {"messages": [HumanMessage(
        content="Send email to wrong@mail.com with subject 'Test' and body 'Hello'"
    )]},
    config=config
)

if "interrupt" in result:
    # Human edits the tool arguments before execution
    result = agent.invoke(
        Command(resume={
            "decision_type": "edit",
            "edited_action": {
                "name": "send_email_tool",
                "args": {
                    "recipient": "correct@gmail.com",   # corrected email
                    "subject": "Test",
                    "body": "Hello — edited by human before sending"
                }
            }
        }),
        config=config
    )
    print(result["messages"][-1].content)
    # → "Email sent to correct@gmail.com"
```

---

### Human-in-the-Loop Decision Summary

| Decision | Effect |
|---|---|
| `"approve"` | Tool executes as-is |
| `"reject"` | Tool call is cancelled; agent informed |
| `"edit"` | Tool runs with human-modified arguments |

---

## 4. Other Built-in Middlewares (Quick Reference)

| Middleware | Purpose |
|---|---|
| `SummarizationMiddleware` | Compress conversation history at token/message limits |
| `HumanInTheLoopMiddleware` | Pause for human approval on tool calls |
| `ModelCallLimitMiddleware` | Cap total model calls — prevents runaway costs/loops |
| `ToolCallLimitMiddleware` | Cap total tool calls per run |
| `ModelFallbackMiddleware` | If primary model fails/errors → fall back to another model |
| `LLMToolSelectorMiddleware` | Filter irrelevant tools per query → reduces token usage |
| `ToolRetryMiddleware` | Auto-retry failed tool calls |

### Adding Multiple Middlewares

```python
agent = create_agent(
    model=model,
    tools=[...],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(model=model, trigger={"messages": 20}, keep_recent=5),
        HumanInTheLoopMiddleware(interrupt_on={"send_email_tool": {...}}),
        # add as many as needed
    ]
)
```

> Middlewares are applied in list order. Think of them as layers — like airport security checks → immigration → boarding gate.