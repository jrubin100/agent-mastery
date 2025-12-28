"""
LANGGRAPH HIERARCHICAL PATTERN
==============================
Router/Boss that delegates to specialized worker agents.

Run: python 04_hierarchical.py

Compare to your custom hierarchical_agent.py where you had:
    router_decision = router.classify(query)
    if router_decision == "technical":
        return technical_agent.run(query)
    elif router_decision == "creative":
        return creative_agent.run(query)

LangGraph uses CONDITIONAL EDGES to express the same logic.
"""

import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable")
    exit(1)


# =============================================================================
# STATE
# =============================================================================

class RouterState(TypedDict):
    """
    State includes the routing decision so we can inspect it.
    """
    query: str                      # User's question
    route: str                      # Which agent to use
    response: str                   # Final response


# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def router_agent(state: RouterState) -> dict:
    """
    Router: Analyzes the query and decides which specialist should handle it.
    
    This is your hierarchical_agent.py's classify() function.
    The key difference: we return the route in STATE, then use
    a conditional edge to actually do the routing.
    """
    print(f"\n🎯 [Router] Analyzing query...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a routing agent. Analyze the user's query 
        and decide which specialist should handle it.
        
        Options:
        - "technical" - for coding, debugging, architecture questions
        - "creative" - for writing, brainstorming, creative tasks
        - "analytical" - for math, data analysis, research questions
        - "general" - for everything else
        
        Respond with ONLY the category name, nothing else."""),
        HumanMessage(content=f"Query: {state['query']}")
    ])
    
    route = response.content.strip().lower()
    print(f"   Routed to: {route}")
    
    return {"route": route}


def technical_agent(state: RouterState) -> dict:
    """Technical specialist: Handles coding and architecture questions."""
    print(f"\n💻 [Technical Agent] Processing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a technical expert. Provide clear, 
        accurate technical answers. Include code examples when helpful.
        Be concise but thorough."""),
        HumanMessage(content=state["query"])
    ])
    
    return {"response": response.content}


def creative_agent(state: RouterState) -> dict:
    """Creative specialist: Handles writing and brainstorming."""
    print(f"\n🎨 [Creative Agent] Processing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a creative expert. Provide imaginative,
        engaging responses. Think outside the box. Be inspiring and original."""),
        HumanMessage(content=state["query"])
    ])
    
    return {"response": response.content}


def analytical_agent(state: RouterState) -> dict:
    """Analytical specialist: Handles data and research questions."""
    print(f"\n📊 [Analytical Agent] Processing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are an analytical expert. Provide data-driven,
        logical responses. Break down complex problems. Use structured reasoning."""),
        HumanMessage(content=state["query"])
    ])
    
    return {"response": response.content}


def general_agent(state: RouterState) -> dict:
    """General specialist: Handles everything else."""
    print(f"\n🌐 [General Agent] Processing...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a helpful assistant. Provide clear,
        friendly responses to general questions."""),
        HumanMessage(content=state["query"])
    ])
    
    return {"response": response.content}


# =============================================================================
# ROUTING FUNCTION
# =============================================================================

def route_to_specialist(state: RouterState) -> str:
    """
    Conditional edge function: Returns the NAME of the next node.
    
    This is the key insight! In LangGraph, routing is done via
    conditional edges that return node names.
    
    Your custom code:
        if route == "technical":
            return technical_agent.run(query)
    
    LangGraph:
        def route(state) -> str:
            return state["route"]  # Returns node name
    """
    route = state["route"]
    
    # Map route to node names
    route_map = {
        "technical": "technical",
        "creative": "creative", 
        "analytical": "analytical",
        "general": "general"
    }
    
    # Default to general if unknown route
    return route_map.get(route, "general")


# =============================================================================
# BUILD THE HIERARCHICAL GRAPH
# =============================================================================

def build_hierarchical_graph():
    """
    Build a router -> specialist graph.
    
    Structure:
                    ┌─→ technical ─┐
                    ├─→ creative  ─┤
        router ─────┼─→ analytical ┼──→ END
                    └─→ general   ─┘
    
    The conditional edge from router determines which path to take.
    """
    
    graph = StateGraph(RouterState)
    
    # Add all nodes
    graph.add_node("router", router_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("creative", creative_agent)
    graph.add_node("analytical", analytical_agent)
    graph.add_node("general", general_agent)
    
    # Start with router
    graph.add_edge(START, "router")
    
    # Conditional edge from router to specialists
    # route_to_specialist returns the name of the next node
    graph.add_conditional_edges(
        "router",
        route_to_specialist,
        # Explicit mapping of return values to node names
        # (optional but makes it clearer)
        {
            "technical": "technical",
            "creative": "creative",
            "analytical": "analytical",
            "general": "general"
        }
    )
    
    # All specialists end the graph
    graph.add_edge("technical", END)
    graph.add_edge("creative", END)
    graph.add_edge("analytical", END)
    graph.add_edge("general", END)
    
    return graph.compile()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("LANGGRAPH HIERARCHICAL - Router Pattern")
    print("=" * 60)
    
    app = build_hierarchical_graph()
    
    # Test queries for different routes
    test_queries = [
        "How do I implement a binary search tree in Python?",
        "Write a haiku about artificial intelligence",
        "What's the statistical significance of a p-value of 0.03?",
        "What's a good recipe for chocolate chip cookies?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        
        result = app.invoke({
            "query": query,
            "route": "",
            "response": ""
        })
        
        print(f"\nRoute taken: {result['route']}")
        print(f"\nResponse:\n{result['response'][:500]}...")


def visualize():
    """Show the graph structure."""
    app = build_hierarchical_graph()
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION (Mermaid)")
    print("=" * 60)
    print("\nPaste this into https://mermaid.live:\n")
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
    visualize()