import os
import os
import sys
# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Import live tools from Phase 3
from tools.market_data import get_quote, get_history, get_ratios

# 1. Define Retrieval Tool using our ChromaDB index
@tool
def retrieve_filings(query: str, ticker: str, doc_type: str = "10-K") -> list:
    """Search ingested SEC filings and transcripts for relevant qualitative passages."""
    persist_dir = "./chroma_db"
    if not os.path.exists(persist_dir):
        return [{"error": "Vector database index not found. Please run ingestion first."}]
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        
        # Maximal Marginal Relevance (MMR) search with metadata filtering
        docs = vectordb.max_marginal_relevance_search(
            query, 
            k=4, 
            fetch_k=15, 
            filter={"$and": [{"ticker": ticker.upper()}, {"doc_type": doc_type}]}
        )
        return [{"text": d.page_content, "metadata": d.metadata} for d in docs]
    except Exception as e:
        return [{"error": f"Retrieval failed: {str(e)}"}]

# Gather all tools
tools = [get_quote, get_history, get_ratios, retrieve_filings]
tool_node = ToolNode(tools)

# 2. Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda a, b: a + b]
    iteration_count: int

# 3. Initialize LLM with tool binding
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are FinAI Analyst, an expert financial research assistant.
You combine real-time market data (via tools) with SEC document retrieval to answer user questions factually.
RULES:
1. Numbers and prices must come strictly from tool outputs in the current turn. Never guess numbers.
2. For qualitative or risk questions, use `retrieve_filings` and cite your sources explicitly.
3. Decline requests for personal buy/sell investment advice and reframe them as objective financial analysis.
4. Keep answers concise, professional, and transparent."""

# 4. Define Graph Nodes
def call_model(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "iteration_count": state.get("iteration_count", 0) + 1}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        if state.get("iteration_count", 0) > 3:
            return "end"
        return "tools"
    return "end"

# 5. Build LangGraph Workflow State Machine
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

# Compile with SQLite memory checkpointer
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/session_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    print("Agent graph compiled successfully with memory and tools.")