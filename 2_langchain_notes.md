# LangChain v1 — Part 2: Tools, Messages, Structured Output & Middleware

> Continuation of Krishna's crash course. Covers tools, message types, structured output (Pydantic / TypedDict / DataClass), and middleware.

---

## 1. Tools

A **tool** = schema (name + description + args) + a function to execute.

The LLM uses the **docstring** to decide *when* and *which* tool to call.

### Creating a Tool with `@tool` decorator

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather at a location."""
    return f"It's sunny in {location}."
```

> The docstring is the schema — it tells the LLM what the tool does. Always write a clear one.

### Binding a Tool to a Model

```python
from langchain_groq import ChatGroq

model = ChatGroq(model="qwen-32b")

model_with_tools = model.bind_tools([get_weather])
```

### Invoking & Reading Tool Calls

```python
response = model_with_tools.invoke("What's the weather like in Boston?")

# LLM reasons: "I need to call get_weather" → makes a tool call
print(response)               # shows reasoning + tool_calls
print(response.tool_calls)    # [{"name": "get_weather", "args": {"location": "Boston"}}]
```

### Full Tool Execution Loop

```python
from langchain.schema import HumanMessage

messages = [HumanMessage(content="What's the weather in Boston?")]

# Step 1 — LLM decides to call the tool
ai_message = model_with_tools.invoke(messages)
messages.append(ai_message)

# Step 2 — Execute each tool call and collect results
for tool_call in ai_message.tool_calls:
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3 — LLM generates final response using tool context
final_response = model_with_tools.invoke(messages)
print(final_response.content)
# → "The weather in Boston is sunny."
```

> Flow: Human → AI (tool call) → Tool Message (context) → AI (final answer)

---

## 2. Messages

Messages are the **fundamental unit of context** in LangChain. Every input/output is a message object with:
- `role` — identifies message type (system / user / assistant / tool)
- `content` — actual text, audio, or document
- `metadata` — optional (IDs, token counts, tracing info)

### Import

```python
from langchain.schema import SystemMessage, HumanMessage, AIMessage
```

### Message Types

| Type | Role | Purpose |
|---|---|---|
| `SystemMessage` | `system` | Instructions for how the LLM should behave |
| `HumanMessage` | `user` | User's input to the model |
| `AIMessage` | `assistant` | Model's response (text + optional tool calls) |
| `ToolMessage` | `tool` | Output returned by a tool after execution |

### Text Prompt (simplest)

```python
response = model.invoke("What is LangChain?")
# string input → treated as HumanMessage internally
print(response.content)   # AIMessage content
```

### Message List (conversation history)

```python
messages = [
    SystemMessage(content="You are a poetry expert."),
    HumanMessage(content="Write a poem on artificial intelligence.")
]

response = model.invoke(messages)
print(response.content)
```

### Detailed System Message (recommended for production)

```python
messages = [
    SystemMessage(content="""You are a senior Python developer with expertise 
    in web frameworks. Always provide code examples and explain your reasoning. 
    Be concise but thorough in your explanations."""),
    HumanMessage(content="How do I create a REST API?")
]
response = model.invoke(messages)
# Response will be Python-specific with Flask/FastAPI code examples
```

> More detail in the system message = more precise, focused responses.

### HumanMessage with Metadata

```python
msg = HumanMessage(
    content="Hello",
    name="Alice",
    id="message_123"    # useful for tracing
)
response = model.invoke(msg)
```

### Manually Building Conversation History

```python
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Can you help me?"),
    AIMessage(content="I'd be happy to help you with that question."),
    HumanMessage(content="Great! What is 2 + 2?")
]

response = model.invoke(messages)
print(response.content)
```

### Accessing Token Metadata

```python
response = model.invoke(messages)
print(response.usage_metadata)
# → {"input_tokens": 42, "output_tokens": 18, "total_tokens": 60}
```

---

## 3. Structured Output

Force the LLM to return responses matching a **defined schema** — essential when you need to parse LLM output in code.

Three approaches: **Pydantic**, **TypedDict**, **DataClass**

---

### 3a. Pydantic (recommended — has field validation)

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title:    str   = Field(description="Title of the movie")
    year:     int   = Field(description="Year the movie was released")
    director: str   = Field(description="Director of the movie")
    rating:   float = Field(description="Movie rating out of 10")
```

```python
model_with_structure = model.with_structured_output(Movie)

response = model_with_structure.invoke("Provide details about the movie Inception.")

print(response.title)    # "Inception"
print(response.year)     # 2010
print(response.director) # "Christopher Nolan"
print(response.rating)   # 8.8
```

> Pydantic enforces types at runtime — passing an integer to a `str` field raises a validation error.

#### Include Raw Output Alongside Parsed

```python
model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("Details of Inception")

print(response["raw"])     # original AIMessage
print(response["parsed"])  # Movie object
```

#### Nested Pydantic Structures

```python
class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title:   str
    year:    int
    cast:    list[Actor]   # nested model
    genres:  list[str]
    budget:  float = Field(default=None, description="Budget in million USD")
```

```python
model_with_structure = model.with_structured_output(MovieDetails)
response = model_with_structure.invoke("Details of Inception")

# response.cast → [Actor(name="Leonardo DiCaprio", role="Dom Cobb"), ...]
# response.genres → ["Science Fiction", "Action"]
# response.budget → 160.0
```

---

### 3b. TypedDict (no runtime validation)

Use when you just need a schema shape but don't care about strict type enforcement.

```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    title:    Annotated[str,   "Title of the movie"]
    year:     Annotated[int,   "Year released"]
    director: Annotated[str,   "Director name"]
    rating:   Annotated[float, "Rating out of 10"]
```

```python
model_with_typedict = model.with_structured_output(MovieDict)
response = model_with_typedict.invoke("Details of Avengers")

# Returns a plain Python dict
print(response)  # {"title": "Avengers", "year": 2012, "director": "Joss Whedon", ...}
```

> TypedDict = lightweight, no validation. Pydantic = strict, with validation.

---

### 3c. DataClass

Available since Python 3.7. No input validation, but clean class syntax.

```python
from dataclasses import dataclass

@dataclass
class ContactInfo:
    name:  str   # name of the person
    email: str
    phone: str
```

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[],
    response_format=ContactInfo
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact: John Doe, john@example.com, 555-1234"}]
})

print(result["structured_response"])
# ContactInfo(name="John Doe", email="john@example.com", phone="555-1234")
```

---

### Structured Output via `create_agent` (with Pydantic)

You can pass `response_format` directly to `create_agent` instead of using `with_structured_output`:

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    name:  str
    email: str
    phone: str

agent = create_agent(
    model=model,       # e.g. GPT-4.1 or GPT-5
    tools=[],
    response_format=ContactInfo
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract contact info from: John Doe, john@example.com, 555-1234"
    }]
})

print(result["structured_response"].name)   # "John Doe"
print(result["structured_response"].email)  # "john@example.com"
```

---

### Comparison: Pydantic vs TypedDict vs DataClass

| Feature | Pydantic | TypedDict | DataClass |
|---|---|---|---|
| Runtime validation | ✅ Yes | ❌ No | ❌ No |
| Field descriptions | ✅ `Field()` | ✅ `Annotated` | ❌ Limited |
| Nested structures | ✅ Easy | ✅ Possible | ✅ Possible |
| Output type | Object | `dict` | Object |
| Best for | Production APIs | Simple schemas | Quick prototypes |

---

## 4. Middleware

Middleware lets you **hook into the agent's execution** to add logging, transformation, retries, guardrails, PII detection, rate limits, and more — without modifying the core agent logic.

### What Middleware Can Do

- **Logging / analytics / debugging** — track agent behavior
- **Prompt transformation** — modify inputs before they reach the LLM
- **Tool selection / output formatting** — control what tools get called
- **Retries & fallbacks** — handle failures gracefully
- **Early termination** — stop the agent under certain conditions
- **Rate limiting** — throttle requests
- **Guardrails & PII detection** — filter sensitive data

> Middleware sits *between* your application and the agent, intercepting calls in and out.

*(Implementation examples covered in subsequent sections of the course.)*

---

## Quick Reference

```python
# Tool creation
from langchain.tools import tool

@tool
def my_tool(arg: str) -> str:
    """Tool description for the LLM."""
    return "result"

model_with_tools = model.bind_tools([my_tool])

# Message types
from langchain.schema import SystemMessage, HumanMessage, AIMessage
messages = [SystemMessage("..."), HumanMessage("...")]
response = model.invoke(messages)

# Structured output — Pydantic
from pydantic import BaseModel, Field
class Schema(BaseModel):
    field: str = Field(description="...")
model.with_structured_output(Schema).invoke("...")

# Structured output — TypedDict
from typing_extensions import TypedDict, Annotated
class Schema(TypedDict):
    field: Annotated[str, "description"]
model.with_structured_output(Schema).invoke("...")

# Structured output — DataClass
from dataclasses import dataclass
@dataclass
class Schema:
    field: str
create_agent(model=model, tools=[], response_format=Schema)
```