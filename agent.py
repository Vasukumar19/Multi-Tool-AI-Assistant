"""
Single Tool Agent
=================
Flow: Question -> Agent -> Web Search -> LLM -> Answer

- LLM     : Groq (llama-3.3-70b-versatile) -- free tier, very fast
- Tool    : DuckDuckGo Search              -- free, no API key needed
- Agent   : LangChain ReAct Agent          -- create_react_agent + AgentExecutor
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.prompts import PromptTemplate

# -- Load environment variables ------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Please add it to your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )

# -- 1. LLM -- Groq Llama 3.3 70B (free tier, ultra fast) --------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
)

# -- 2. Tool -- DuckDuckGo Search (free, no key needed) -----------------------
search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Search the web using DuckDuckGo. "
        "Use this when you need current information, facts, news, or anything "
        "that requires real-world knowledge beyond your training data. "
        "Input should be a concise search query string."
    ),
)
tools = [search_tool]

# -- 3. ReAct Prompt -----------------------------------------------------------
REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful AI assistant that can search the web to answer questions accurately.

You have access to the following tools:
{tools}

Use the following format STRICTLY:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (you can repeat Thought/Action/Action Input/Observation if needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# -- 4. Create LangChain ReAct Agent ------------------------------------------
#
# LangChain ReAct agent with:
#   - Groq LLM for lightning-fast reasoning
#   - DuckDuckGo as the single search tool
#   - Classic ReAct prompt pattern
#
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=REACT_PROMPT,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,            # shows Thought / Action / Observation live
    handle_parsing_errors=True,
    max_iterations=5,        # safety cap
    return_intermediate_steps=False,
)

# -- 5. Run function -----------------------------------------------------------
def ask(question: str) -> str:
    """
    Run the single-tool agent pipeline:
        Question -> Agent -> Web Search -> LLM -> Answer

    Args:
        question: The question to answer.

    Returns:
        The agent's final answer as a string.
    """
    print("\n" + "=" * 60)
    print(f"  Question : {question}")
    print("=" * 60)

    result = agent_executor.invoke({"input": question})
    answer = result.get("output", "No answer returned.")

    print("\n" + "-" * 60)
    print(f"  Answer   : {answer}")
    print("-" * 60 + "\n")

    return answer


# -- 6. CLI Entry Point --------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Single question via CLI arg
        # e.g.  python agent.py "What is the latest news on AI?"
        question = " ".join(sys.argv[1:])
        ask(question)
    else:
        # Interactive chat loop
        print("\nSingle Tool Agent  --  Powered by Groq + DuckDuckGo")
        print("Type your question or 'quit' to exit.\n")
        while True:
            try:
                q = input("You: ").strip()
                if not q:
                    continue
                if q.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                ask(q)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
