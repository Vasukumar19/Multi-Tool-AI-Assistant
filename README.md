# 🤖 Single Tool Agent

A minimal AI agent that answers questions using web search.

```
Question
   ↓
Agent  (ReAct reasoning loop)
   ↓
Web Search  (DuckDuckGo — free, no API key)
   ↓
LLM  (Google Gemini 2.0 Flash — free tier)
   ↓
Answer
```

## 🛠️ Tech Stack

| Component | Library / Model | Cost |
|-----------|----------------|------|
| **LLM** | `gemini-2.0-flash` via `langchain-google-genai` | Free |
| **Web Search** | DuckDuckGo via `duckduckgo-search` | Free |
| **Agent Framework** | LangChain ReAct Agent | Free / Open Source |

> **Why LangChain (not LangGraph)?**  
> With a single tool, the agent loop is simple: decide → search → answer.  
> LangGraph is designed for complex multi-node graphs with looping states. LangChain's  
> `create_react_agent` is cleaner and more appropriate here.

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your API key
```bash
# Copy the template
copy .env.example .env

# Edit .env and paste your Gemini API key
# Get a free key at: https://aistudio.google.com/app/apikey
```

Your `.env` should look like:
```
GOOGLE_API_KEY=AIza...your_key_here
```

### 3. Run the agent

**Interactive mode (chat loop):**
```bash
python agent.py
```

**Single question via CLI:**
```bash
python agent.py "What is the latest news on artificial intelligence?"
```

---

## 📖 How It Works

The agent uses the ReAct (Reasoning + Acting) pattern:

1. **Thought** — The LLM reasons about what it needs to do
2. **Action** — It decides to call `web_search` with a query
3. **Observation** — DuckDuckGo returns search results
4. **Thought** — The LLM synthesizes the information
5. **Final Answer** — A clear, grounded answer is returned

Example trace:
```
════════════════════════════════════════════════════════════
  Question : What is the current price of gold?
════════════════════════════════════════════════════════════

> Entering new AgentExecutor chain...
  Thought: I need to search for the current gold price.
  Action: web_search
  Action Input: current gold price today 2025
  Observation: Gold is trading at $3,200 per troy ounce...
  Thought: I now know the final answer
  Final Answer: The current price of gold is approximately $3,200 per troy ounce.

──────────────────────────────────────────────────────────
  Answer   : The current price of gold is approximately $3,200 per troy ounce.
──────────────────────────────────────────────────────────
```

---

## 📁 Project Structure

```
single tool agent/
├── agent.py          # Main agent (LLM + tool + executor)
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
├── .env              # Your actual API keys (not committed)
└── README.md         # This file
```
