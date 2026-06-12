
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.prompts import PromptTemplate
# RAG imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter




CHAT_HISTORY_FILE = Path("memory/chat_history.json")


def load_chat_history():

    if not CHAT_HISTORY_FILE.exists():
        return []

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return []
    
def save_chat_history(history):

    CHAT_HISTORY_FILE.parent.mkdir(exist_ok=True)

    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    
def update_chat_history(user_message, assistant_message):

    history = load_chat_history()

    history.append({
        "role": "user",
        "content": user_message
    })

    history.append({
        "role": "assistant",
        "content": assistant_message
    })

    save_chat_history(history)
def get_recent_history(limit=10):

    history = load_chat_history()

    return history[-limit:]
def format_chat_history(history):

    formatted = []

    for msg in history:

        role = msg["role"].capitalize()

        formatted.append(
            f"{role}: {msg['content']}"
        )

    return "\n".join(formatted)

"""
Multi Tool Agent
================
Flow: Question -> Agent -> [Web Search | Calculator | RAG Search] -> LLM -> Answer
- LLM       : Groq (llama-3.3-70b-versatile) -- free tier, very fast
- Tools     : 1. DuckDuckGo Search   -- current news & facts from the web
              2. Calculator          -- math & arithmetic expressions
              3. RAG Search          -- search your own PDF & text documents
- Embeddings: HuggingFace all-MiniLM-L6-v2 (free, runs locally)
- Vector DB : FAISS (local, no server needed)
- Agent     : LangChain ReAct Agent (create_react_agent + AgentExecutor)
"""

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Please add it to your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )
# ---------------------------------------------------------------------------
# 1. LLM  --  Groq Llama 3.3 70B
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
)
# ---------------------------------------------------------------------------
# 2. RAG Setup  --  FAISS + HuggingFace Embeddings
# ---------------------------------------------------------------------------
DOCS_DIR   = Path(__file__).parent / "documents"   # drop your files here
FAISS_DIR  = Path(__file__).parent / "faiss_index" # saved index lives here
DOCS_DIR.mkdir(exist_ok=True)
def build_or_load_vectorstore() -> FAISS | None:
    """
    Loads PDF and text files from the `documents/` folder,
    splits them into chunks, embeds with HuggingFace, and stores in FAISS.
    On subsequent runs it reloads the saved index (fast).
    Returns None if no documents are found.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    # ── Try loading saved index first ───────────────────────────────────────
    if FAISS_DIR.exists():
        print("[RAG] Loading existing FAISS index...")
        return FAISS.load_local(str(FAISS_DIR), embeddings,
                                allow_dangerous_deserialization=True)
    # ── Build fresh index from documents ────────────────────────────────────
    docs = []
    # Load PDFs
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    for pdf_path in pdf_files:
        print(f"[RAG] Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    # Load text files
    txt_files = list(DOCS_DIR.glob("*.txt"))
    for txt_path in txt_files:
        print(f"[RAG] Loading TXT: {txt_path.name}")
        loader = TextLoader(str(txt_path), encoding="utf-8")
        docs.extend(loader.load())
    if not docs:
        print("[RAG] No documents found in 'documents/' folder. RAG tool disabled.")
        return None
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"[RAG] Created {len(chunks)} chunks from {len(docs)} pages.")
    # Embed and store
    print("[RAG] Building FAISS index (this may take a moment)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_DIR))
    print("[RAG] FAISS index saved.")
    return vectorstore
# Build/load on startup
vectorstore = build_or_load_vectorstore()
# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------
import json
from pathlib import Path

MEMORY_FILE = Path("memory/memory.json")

def load_memory():
    """
    Reads memory.json and returns a Python dictionary.
    """

    if not MEMORY_FILE.exists():
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return {}


def save_memory(memory: dict):
    """
    Saves dictionary to memory.json
    """

    MEMORY_FILE.parent.mkdir(exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4)

MEMORY_EXTRACTION_PROMPT = """
You are a memory extraction system.

Your job is to extract ONLY information that may be useful
for future conversations.

Extract these categories if present:

- name
- goal
- interests
- profession
- education
- favorite_technologies
- preferences

Return ONLY valid JSON.

If nothing useful should be remembered return:

{{}}

Examples:

Message:
"My name is Vasu."

Output:
{{
  "name":"Vasu"
}}

Message:
"I want to become an AI Engineer."

Output:
{{
  "goal":"AI Engineer"
}}

Message:
"I love Machine Learning and Python."

Output:
{{
  "interests":["Machine Learning"],
  "favorite_technologies":["Python"]
}}

Message:
"What is LangGraph?"

Output:
{{}}

User Message:
{message}
"""


def extract_memory(message: str):

    prompt = MEMORY_EXTRACTION_PROMPT.format(
        message=message
    )

    response = llm.invoke(prompt)

    try:
        memory_data = json.loads(response.content)
        return memory_data

    except Exception:
        return {}
    

def update_memory(user_message: str):

    extracted_memory = extract_memory(user_message)

    if not extracted_memory:
        return

    memory = load_memory()

    memory.update(extracted_memory)

    save_memory(memory)

# Tool 1: DuckDuckGo Web Search
search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description=(
        "Search the web using DuckDuckGo. "
        "Use this for current events, news, general facts, or anything "
        "that requires real-world knowledge. "
        "Input: a concise search query string."
    ),
)
# Tool 2: Calculator
@tool
def calculator(expression: str) -> str:
    """
    Evaluates arithmetic expressions: +, -, *, /, **, sqrt, etc.
    Use this for any math or calculation question.
    Examples: '2 + 2', '100 * 3.14', 'sqrt(144)', '2 ** 10'
    """
    try:
        import math
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"
    


@tool
def history_lookup(query: str) -> str:
    """
    Retrieve recent conversation history.

    Use this tool for questions like:
    - What did we talk about?
    - What was the last question I asked?
    - Remind me of our recent conversation.
    """

    recent_history = get_recent_history()
    if not recent_history:
        return "No recent conversation history found."

    formatted_history = format_chat_history(recent_history)
    return formatted_history    
# Tool 3: RAG Search (real FAISS-backed implementation)
@tool
def rag_search(query: str) -> str:
    """
    Search through the company's uploaded PDF and text documents using semantic similarity.
    ALWAYS use this tool first for questions about:
    - Company policies (refund, return, cancellation, billing)
    - Company products, pricing, or services
    - Internal reports, revenue, financial data
    - Support contacts, office locations, working hours
    - Any question containing words like 'company', 'our', 'policy', 'report', 'document'
    Examples: 'What is the refund policy?', 'What are the product prices?',
              'Summarize the Q1 revenue report', 'How do I contact support?'
    """
    if vectorstore is None:
        return (
            "No documents found in the 'documents/' folder. "
            "Please add PDF or .txt files there and restart the agent."
        )
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in the documents."
    output = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        label  = f"[Doc {i} | {Path(source).name}" + (f" p.{page+1}]" if page != "" else "]")
        output.append(f"{label}\n{doc.page_content.strip()}")
    return "\n\n".join(output)

@tool
def memory_lookup(query: str) -> str:
    """
    Retrieve stored user memory.

    Use this tool for questions like:
    - What is my name?
    - What is my goal?
    - What technologies do I like?
    - What do you remember about me?
    """

    memory = load_memory()

    if not memory:
        return "No memory stored."

    return json.dumps(memory, indent=2)

# All 3 tools
tools = [search_tool, calculator, rag_search, memory_lookup, history_lookup]
# ---------------------------------------------------------------------------
# 4. ReAct Prompt
# ---------------------------------------------------------------------------
REACT_PROMPT = PromptTemplate.from_template(
"""You are an intelligent AI assistant capable of reasoning and using tools when necessary.

Your primary objective is to answer the user's question accurately, efficiently, and with the minimum number of tool calls required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL REASONING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. First understand the user's request.

2. Determine whether a tool is actually required.

3. If no tool is required, immediately produce a Final Answer without generating an Action step.

4. Use tools only when they provide information that is missing, external, stored, or computational.

5. Use the minimum number of tool calls necessary.

6. After receiving a tool result, evaluate whether the information is sufficient to answer the question.

7. If the information is incomplete, ambiguous, irrelevant, or insufficient, you MAY perform another tool call with a better query.

8. Do not repeatedly call tools without a clear reason.

9. Multiple tools may be used when solving the task requires information from different sources.

10. Never invent tool outputs.

11. Never claim to have searched, retrieved, remembered, or calculated something unless the corresponding tool was actually used.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. memory_lookup

Purpose:
Retrieve long-term stored information about the user.

Use when:

* The user asks about themselves.
* The user asks what you remember.
* The user asks about previously stored goals, preferences, interests, education, profession, or personal facts.

Examples:

* What is my name?
* What do you remember about me?
* What is my goal?
* What technologies do I like?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. history_lookup

Purpose:
Retrieve previous conversation context.

Use when:

* The user refers to something discussed earlier.
* The current question depends on previous conversation context.

Examples:

* Explain it again.
* Continue.
* What were we discussing?
* Summarize our conversation.
* Can you elaborate on that?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. rag_search

Purpose:
Search uploaded documents using semantic retrieval.

Use when:

* The answer may exist in uploaded documents.
* The user asks about company documents, reports, manuals, policies, products, pricing, support information, or internal knowledge.

Examples:

* What is the refund policy?
* Summarize the revenue report.
* What are the product prices?
* What does the company handbook say?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. web_search

Purpose:
Retrieve external or current information.

Use when:

* The question requires recent information.
* The answer may have changed after model training.
* External information is needed.

Examples:

* Latest AI news
* Current gold price
* Recent OpenAI announcements
* Today's weather

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. calculator

Purpose:
Perform mathematical calculations.

Use when:

* Arithmetic or numerical computation is required.

Examples:

* 234 * 567
* sqrt(144)
* 15% of 4500

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL SELECTION PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For stored user information:
→ memory_lookup

For previous conversation references:
→ history_lookup

For uploaded/company documents:
→ rag_search

For recent or external information:
→ web_search

For mathematical calculations:
→ calculator

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-TOOL REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use multiple tools when necessary.

Example:

Question:
Compare our refund policy with industry standards.

Thought:
I need the company refund policy.

Action:
rag_search

Observation:
...

Thought:
I need industry information for comparison.

Action:
web_search

Observation:
...

Thought:
I now have enough information to compare them.

Final Answer:
...

IMPORTANT:

If no tool is required, do NOT output Action.

Instead output:

Thought: I can answer without using any tool.
Final Answer: <your response>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL RETRY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After every Observation:

* Evaluate whether the information is sufficient.
* If sufficient, answer the question.
* If insufficient, ambiguous, incomplete, or irrelevant, you MAY perform another tool call.
* Refine search queries when necessary.
* Stop gathering information once you can answer confidently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT

When a tool is required:

Question: the user's question

Thought: determine which tool is needed

Action: one of [{tool_names}]
Action Input: input for the tool

Observation: result from the tool

(Repeat Thought / Action / Observation if necessary)

Thought: I now know the final answer

Final Answer: the answer for the user


When no tool is required:

Question: the user's question

Thought: I can answer without using any tool.

Final Answer: the answer for the user

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tools}

Begin.

Question: {input}

Thought:{agent_scratchpad}




"""
)
# ---------------------------------------------------------------------------
# 5. Create Agent
# ---------------------------------------------------------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
    return_intermediate_steps=False,
)
# ---------------------------------------------------------------------------
# 6. Ask function
# ---------------------------------------------------------------------------



def ask(question: str) -> str:
    """Run the multi-tool agent: Question -> [Tool] -> Answer."""
    print("\n" + "=" * 60)
    print(f"  Question : {question}")
    print("=" * 60)

    update_memory(question)
    recent_history= get_recent_history()
    history_context = format_chat_history(recent_history)
    full_input = f"""
        Recent Conversation:

        {history_context}

        Current Question:

        {question}
        """

    result = agent_executor.invoke({
        "input": full_input
    })
   
    answer = result.get("output", "No answer returned.")
    update_chat_history(question, answer)
    print("\n" + "-" * 60)
    print(f"  Answer   : {answer}")
    print("-" * 60 + "\n")
    return answer
# ---------------------------------------------------------------------------
# 7. CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        # print("\nMulti Tool Agent  --  Groq + DuckDuckGo + Calculator + RAG (FAISS)")
        # print("Tools: web_search | calculator | rag_search")
        # print(f"Documents folder: {DOCS_DIR.resolve()}")
        # print("Type your question or 'quit' to exit.\n")
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
