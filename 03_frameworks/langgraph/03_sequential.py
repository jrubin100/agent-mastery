"""
LANGGRAPH SEQUENTIAL PATTERN
============================
Multiple specialized agents in a pipeline.

Run: python 03_sequential.py

Compare to your custom multi_agent.py where you did:
    result = researcher.run(query)
    result = writer.run(result)
    result = editor.run(result)

LangGraph does the same thing but with explicit state management.
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable")
    exit(1)


# =============================================================================
# STATE: What flows through the pipeline
# =============================================================================

class ContentState(TypedDict):
    """
    State for content creation pipeline.
    
    Each agent adds to or transforms the state.
    This is cleaner than passing dictionaries manually.
    """
    topic: str              # Original topic
    research: str           # Researcher output
    draft: str              # Writer output  
    final_content: str      # Editor output
    current_stage: str      # Track where we are


# =============================================================================
# AGENTS: Each node is a specialized agent
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def researcher_agent(state: ContentState) -> dict:
    """
    Research Agent: Gathers key points about the topic.
    
    Your custom code: self.researcher.run(topic)
    LangGraph: This function IS the researcher
    """
    print(f"\n🔍 [Researcher] Researching: {state['topic']}")
    
    response = llm.invoke([
        SystemMessage(content="""You are a research agent. Given a topic, 
        provide 3-5 key facts or points that would be useful for writing 
        an article. Be concise and factual. Output just the bullet points."""),
        HumanMessage(content=f"Research this topic: {state['topic']}")
    ])
    
    research = response.content
    print(f"   Research complete: {len(research)} chars")
    
    return {
        "research": research,
        "current_stage": "researched"
    }


def writer_agent(state: ContentState) -> dict:
    """
    Writer Agent: Creates a draft from research.
    
    Receives research from previous agent via state.
    """
    print(f"\n✍️  [Writer] Creating draft from research...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a writing agent. Given research points,
        write a short, engaging article (2-3 paragraphs). Make it informative
        but accessible. Don't use bullet points - write in prose."""),
        HumanMessage(content=f"""
Topic: {state['topic']}

Research:
{state['research']}

Write the article:""")
    ])
    
    draft = response.content
    print(f"   Draft complete: {len(draft)} chars")
    
    return {
        "draft": draft,
        "current_stage": "drafted"
    }


def editor_agent(state: ContentState) -> dict:
    """
    Editor Agent: Polishes the draft.
    
    Final agent in the pipeline.
    """
    print(f"\n📝 [Editor] Polishing draft...")
    
    response = llm.invoke([
        SystemMessage(content="""You are an editor agent. Review and improve
        the draft article. Fix any issues, improve flow, and ensure quality.
        Output the final polished version only."""),
        HumanMessage(content=f"""
Topic: {state['topic']}

Draft to edit:
{state['draft']}

Provide the polished final version:""")
    ])
    
    final = response.content
    print(f"   Editing complete: {len(final)} chars")
    
    return {
        "final_content": final,
        "current_stage": "complete"
    }


# =============================================================================
# BUILD THE SEQUENTIAL GRAPH
# =============================================================================

def build_sequential_pipeline():
    """
    Create a sequential pipeline: Researcher -> Writer -> Editor
    
    Your custom code equivalent:
        for agent in [researcher, writer, editor]:
            result = agent.run(result)
    
    LangGraph makes dependencies explicit and adds:
    - State validation
    - Automatic tracing to LangSmith
    - Easy visualization
    - Checkpointing (pause/resume)
    """
    
    graph = StateGraph(ContentState)
    
    # Add agents as nodes
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("editor", editor_agent)
    
    # Sequential edges
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)
    
    return graph.compile()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("LANGGRAPH SEQUENTIAL - Content Creation Pipeline")
    print("=" * 60)
    
    app = build_sequential_pipeline()
    
    # Initial state
    initial_state = {
        "topic": "Why AI agents are the future of software development",
        "research": "",
        "draft": "",
        "final_content": "",
        "current_stage": "started"
    }
    
    print(f"\nTopic: {initial_state['topic']}")
    print("\nRunning pipeline...")
    
    # Run the pipeline
    final_state = app.invoke(initial_state)
    
    # Output
    print("\n" + "=" * 60)
    print("FINAL ARTICLE")
    print("=" * 60)
    print(final_state["final_content"])
    
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Topic: {final_state['topic']}")
    print(f"Research length: {len(final_state['research'])} chars")
    print(f"Draft length: {len(final_state['draft'])} chars")
    print(f"Final length: {len(final_state['final_content'])} chars")
    print(f"Stage: {final_state['current_stage']}")


def visualize():
    """Show the graph structure."""
    app = build_sequential_pipeline()
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION (Mermaid)")
    print("=" * 60)
    print("\nPaste this into https://mermaid.live:\n")
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
    visualize()