"""
LANGGRAPH SWARM PATTERN
=======================
Agents collaborating dynamically with cycles (iteration).

Run: python 06_swarm.py

This is the most complex pattern - agents iterate until
they reach consensus or a stopping condition.

Compare to your swarm_agent.py where agents kept iterating
until they agreed on an answer.

KEY INSIGHT: LangGraph handles cycles naturally. You can
have edges that loop back, and the framework manages it.
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
# STATE FOR ITERATIVE COLLABORATION
# =============================================================================

class SwarmState(TypedDict):
    """
    State tracks the evolving solution and iteration count.
    
    - solution: The current best answer
    - feedback: Critique from the last iteration
    - iteration: How many rounds we've done
    - is_complete: Whether we've reached consensus
    - history: Log of all iterations
    """
    problem: str
    solution: str
    feedback: str
    iteration: int
    is_complete: bool
    history: list[str]


# =============================================================================
# SWARM AGENTS
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

MAX_ITERATIONS = 3  # Prevent infinite loops


def solver_agent(state: SwarmState) -> dict:
    """
    Solver: Proposes or improves a solution.
    
    On first iteration: Creates initial solution
    On later iterations: Improves based on feedback
    """
    iteration = state["iteration"]
    print(f"\n🔧 [Solver] Iteration {iteration + 1}...")
    
    if iteration == 0:
        # First iteration - create initial solution
        prompt = f"""Problem: {state['problem']}
        
Propose an initial solution. Be specific and actionable."""
    else:
        # Later iterations - improve based on feedback
        prompt = f"""Problem: {state['problem']}

Current solution:
{state['solution']}

Feedback received:
{state['feedback']}

Improve the solution based on this feedback. Address the specific concerns raised."""
    
    response = llm.invoke([
        SystemMessage(content="""You are a problem solver. Your job is to 
        propose solutions and improve them based on feedback. Be specific
        and practical in your suggestions."""),
        HumanMessage(content=prompt)
    ])
    
    new_solution = response.content
    print(f"   Solution proposed ({len(new_solution)} chars)")
    
    return {
        "solution": new_solution,
        "iteration": iteration + 1,
        "history": state["history"] + [f"ITERATION {iteration + 1} SOLUTION:\n{new_solution}"]
    }


def critic_agent(state: SwarmState) -> dict:
    """
    Critic: Reviews the solution and provides feedback.
    
    Can either:
    - Approve (is_complete = True)
    - Request improvements (is_complete = False)
    """
    print(f"\n🔍 [Critic] Reviewing solution...")
    
    response = llm.invoke([
        SystemMessage(content="""You are a critical reviewer. Evaluate the proposed
        solution for the given problem. 
        
        If the solution is good enough (addresses the core problem, is practical, 
        and reasonably complete), respond with: "APPROVED: [brief praise]"
        
        If improvements are needed, respond with: "NEEDS WORK: [specific feedback]"
        
        Be constructive but don't be a perfectionist - approve good solutions."""),
        HumanMessage(content=f"""
Problem: {state['problem']}

Proposed solution:
{state['solution']}

This is iteration {state['iteration']} of {MAX_ITERATIONS}.
{"This is the last iteration, so be more lenient." if state['iteration'] >= MAX_ITERATIONS else ""}

Evaluate:""")
    ])
    
    feedback = response.content
    is_complete = feedback.strip().upper().startswith("APPROVED")
    
    print(f"   Verdict: {'APPROVED ✓' if is_complete else 'NEEDS WORK'}")
    
    return {
        "feedback": feedback,
        "is_complete": is_complete,
        "history": state["history"] + [f"ITERATION {state['iteration']} FEEDBACK:\n{feedback}"]
    }


# =============================================================================
# ROUTING FUNCTION FOR CYCLE
# =============================================================================

def should_continue(state: SwarmState) -> Literal["solver", "end"]:
    """
    Decide whether to continue iterating or end.
    
    This is where LangGraph's cycle support shines.
    We can loop back to solver if we're not done yet.
    """
    if state["is_complete"]:
        print("\n✅ Consensus reached!")
        return "end"
    
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n⚠️  Max iterations ({MAX_ITERATIONS}) reached")
        return "end"
    
    print("\n🔄 Continuing iteration...")
    return "solver"


# =============================================================================
# BUILD SWARM GRAPH
# =============================================================================

def build_swarm_graph():
    """
    Build an iterative swarm graph:
    
        START ──→ solver ──→ critic ──┬──→ END
                    ↑                 │
                    └─────────────────┘
                    (if not complete)
    
    The cycle between solver and critic continues until:
    - Critic approves (is_complete = True)
    - Max iterations reached
    
    This is EXACTLY what your custom swarm did, but LangGraph
    makes the cycle explicit and manageable.
    """
    
    graph = StateGraph(SwarmState)
    
    # Add nodes
    graph.add_node("solver", solver_agent)
    graph.add_node("critic", critic_agent)
    
    # Start with solver
    graph.add_edge(START, "solver")
    
    # Solver always goes to critic
    graph.add_edge("solver", "critic")
    
    # Critic conditionally loops back or ends
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "solver": "solver",  # Loop back
            "end": END           # We're done
        }
    )
    
    return graph.compile()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("LANGGRAPH SWARM - Iterative Problem Solving")
    print("=" * 60)
    
    app = build_swarm_graph()
    
    problem = """
    Design a system for a small startup to manage customer support tickets.
    They have 3 support agents and get about 50 tickets per day.
    They need to track ticket status, assign agents, and measure response times.
    Budget is limited - prefer simple solutions over complex ones.
    """
    
    print(f"\nProblem:{problem}")
    print("\nStarting swarm collaboration...")
    print("=" * 60)
    
    result = app.invoke({
        "problem": problem,
        "solution": "",
        "feedback": "",
        "iteration": 0,
        "is_complete": False,
        "history": []
    })
    
    print("\n" + "=" * 60)
    print("ITERATION HISTORY")
    print("=" * 60)
    for entry in result["history"]:
        print(f"\n{entry}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("FINAL SOLUTION")
    print("=" * 60)
    print(result["solution"])
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total iterations: {result['iteration']}")
    print(f"Consensus reached: {result['is_complete']}")


def visualize():
    """Show the graph structure."""
    app = build_swarm_graph()
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION (Mermaid)")
    print("=" * 60)
    print("\nPaste this into https://mermaid.live:\n")
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
    visualize()