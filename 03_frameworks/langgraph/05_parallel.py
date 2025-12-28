"""
LANGGRAPH PARALLEL PATTERN
==========================
Multiple agents working simultaneously, results combined.

Run: python 05_parallel.py

This pattern is useful when you need multiple perspectives
or when tasks are independent and can run concurrently.

Compare to using asyncio.gather() in custom code.
"""

import os
import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable")
    exit(1)


# =============================================================================
# STATE WITH ACCUMULATION
# =============================================================================

class ParallelState(TypedDict):
    """
    The 'perspectives' field uses operator.add as a reducer.
    This means each node's output gets APPENDED to the list
    rather than replacing it.
    
    This is key for fan-out/fan-in patterns.
    """
    topic: str
    perspectives: Annotated[list[str], operator.add]  # Accumulates results
    synthesis: str


# =============================================================================
# PARALLEL AGENTS - Each analyzes from different angle
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def optimist_agent(state: ParallelState) -> dict:
    """Analyzes topic from optimistic perspective."""
    print(f"\n😊 [Optimist] Analyzing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are an optimist. Analyze the given topic
        focusing on opportunities, benefits, and positive outcomes.
        Keep your response to 2-3 sentences."""),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    
    return {"perspectives": [f"OPTIMIST: {response.content}"]}


def pessimist_agent(state: ParallelState) -> dict:
    """Analyzes topic from pessimistic perspective."""
    print(f"\n😟 [Pessimist] Analyzing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a pessimist. Analyze the given topic
        focusing on risks, challenges, and potential problems.
        Keep your response to 2-3 sentences."""),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    
    return {"perspectives": [f"PESSIMIST: {response.content}"]}


def realist_agent(state: ParallelState) -> dict:
    """Analyzes topic from realistic perspective."""
    print(f"\n🤔 [Realist] Analyzing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a realist. Analyze the given topic
        with a balanced view of both opportunities and challenges.
        Keep your response to 2-3 sentences."""),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    
    return {"perspectives": [f"REALIST: {response.content}"]}


def innovator_agent(state: ParallelState) -> dict:
    """Analyzes topic from innovative perspective."""
    print(f"\n💡 [Innovator] Analyzing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are an innovator. Analyze the given topic
        focusing on creative possibilities and unconventional approaches.
        Keep your response to 2-3 sentences."""),
        HumanMessage(content=f"Topic: {state['topic']}")
    ])
    
    return {"perspectives": [f"INNOVATOR: {response.content}"]}


def synthesizer_agent(state: ParallelState) -> dict:
    """Combines all perspectives into a balanced synthesis."""
    print(f"\n🔄 [Synthesizer] Combining perspectives...")
    
    perspectives_text = "\n\n".join(state["perspectives"])
    
    response = llm.invoke([
        SystemMessage(content="""You are a synthesizer. Given multiple perspectives
        on a topic, create a balanced summary that incorporates insights from all views.
        Produce a cohesive 2-3 paragraph analysis."""),
        HumanMessage(content=f"""
Topic: {state['topic']}

Perspectives:
{perspectives_text}

Synthesize these into a balanced analysis:""")
    ])
    
    return {"synthesis": response.content}


# =============================================================================
# BUILD PARALLEL GRAPH
# =============================================================================

def build_parallel_graph():
    """
    Build a fan-out/fan-in graph:
    
                ┌─→ optimist  ─┐
                ├─→ pessimist ─┤
        START ──┼─→ realist   ─┼──→ synthesizer ──→ END
                └─→ innovator ─┘
    
    All four agents run (conceptually) in parallel.
    Their outputs accumulate in the 'perspectives' list.
    Then synthesizer combines them.
    
    Note: LangGraph handles the fan-out/fan-in automatically
    when you add edges from START to multiple nodes, and from
    multiple nodes to a single node.
    """
    
    graph = StateGraph(ParallelState)
    
    # Add nodes
    graph.add_node("optimist", optimist_agent)
    graph.add_node("pessimist", pessimist_agent)
    graph.add_node("realist", realist_agent)
    graph.add_node("innovator", innovator_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    
    # Fan-out: START to all analysts
    graph.add_edge(START, "optimist")
    graph.add_edge(START, "pessimist")
    graph.add_edge(START, "realist")
    graph.add_edge(START, "innovator")
    
    # Fan-in: All analysts to synthesizer
    graph.add_edge("optimist", "synthesizer")
    graph.add_edge("pessimist", "synthesizer")
    graph.add_edge("realist", "synthesizer")
    graph.add_edge("innovator", "synthesizer")
    
    # End
    graph.add_edge("synthesizer", END)
    
    return graph.compile()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("LANGGRAPH PARALLEL - Multi-Perspective Analysis")
    print("=" * 60)
    
    app = build_parallel_graph()
    
    topic = "The impact of AI agents on the future of software development jobs"
    
    print(f"\nTopic: {topic}")
    print("\nGathering perspectives...")
    
    result = app.invoke({
        "topic": topic,
        "perspectives": [],
        "synthesis": ""
    })
    
    print("\n" + "=" * 60)
    print("INDIVIDUAL PERSPECTIVES")
    print("=" * 60)
    for perspective in result["perspectives"]:
        print(f"\n{perspective}")
    
    print("\n" + "=" * 60)
    print("SYNTHESIZED ANALYSIS")
    print("=" * 60)
    print(result["synthesis"])


def visualize():
    """Show the graph structure."""
    app = build_parallel_graph()
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION (Mermaid)")
    print("=" * 60)
    print("\nPaste this into https://mermaid.live:\n")
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
    visualize()