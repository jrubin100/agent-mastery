"""
LANGGRAPH BASICS
================
Core concepts: State, Nodes, Edges, Compilation

Run: python 01_basics.py

This is the "hello world" of LangGraph. No LLM calls yet - just
understanding how graphs work.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# =============================================================================
# CONCEPT 1: State
# =============================================================================
# State is a TypedDict that flows through your graph.
# Every node receives state, can modify it, and passes it on.

class BasicState(TypedDict):
    """
    This is our shared memory. Every node can read/write to it.
    
    Compare to your custom code where you passed dictionaries around manually.
    LangGraph formalizes this.
    """
    message: str
    step_count: int


# =============================================================================
# CONCEPT 2: Nodes
# =============================================================================
# Nodes are just functions. They take state, return updates to state.
# That's it. No magic.

def step_one(state: BasicState) -> dict:
    """
    First node. Modifies the message and increments counter.
    
    IMPORTANT: Return only the fields you want to UPDATE.
    LangGraph merges this with existing state.
    """
    print(f"[Step 1] Received: {state['message']}")
    return {
        "message": state["message"] + " -> processed by step 1",
        "step_count": state["step_count"] + 1
    }


def step_two(state: BasicState) -> dict:
    """Second node."""
    print(f"[Step 2] Received: {state['message']}")
    return {
        "message": state["message"] + " -> processed by step 2",
        "step_count": state["step_count"] + 1
    }


def step_three(state: BasicState) -> dict:
    """Third node."""
    print(f"[Step 3] Received: {state['message']}")
    return {
        "message": state["message"] + " -> processed by step 3",
        "step_count": state["step_count"] + 1
    }


# =============================================================================
# CONCEPT 3: Building the Graph
# =============================================================================

def build_basic_graph():
    """
    Build a simple sequential graph: step_one -> step_two -> step_three
    
    This is equivalent to:
        result = step_one(input)
        result = step_two(result)
        result = step_three(result)
    
    But LangGraph tracks state, enables persistence, and integrates with LangSmith.
    """
    
    # Create graph with our state schema
    graph = StateGraph(BasicState)
    
    # Add nodes (name -> function)
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_node("step_three", step_three)
    
    # Add edges (from -> to)
    # START is a special node meaning "entry point"
    # END is a special node meaning "we're done"
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", "step_three")
    graph.add_edge("step_three", END)
    
    # Compile the graph (makes it runnable)
    return graph.compile()


# =============================================================================
# CONCEPT 4: Running the Graph
# =============================================================================

def main():
    print("=" * 60)
    print("LANGGRAPH BASICS - Sequential Flow")
    print("=" * 60)
    
    # Build and compile
    app = build_basic_graph()
    
    # Initial state
    initial_state = {
        "message": "Hello",
        "step_count": 0
    }
    
    print(f"\nInitial state: {initial_state}\n")
    
    # Run the graph
    # invoke() runs to completion and returns final state
    final_state = app.invoke(initial_state)
    
    print(f"\nFinal state: {final_state}")
    print(f"Total steps: {final_state['step_count']}")


# =============================================================================
# BONUS: Visualize the Graph
# =============================================================================

def visualize():
    """
    LangGraph can show you what your graph looks like.
    This outputs Mermaid diagram syntax.
    """
    app = build_basic_graph()
    
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION (Mermaid)")
    print("=" * 60)
    print("\nPaste this into https://mermaid.live to see the diagram:\n")
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
    visualize()