"""
LANGGRAPH WITH TOOLS
====================
Adding LLM and tool calling to LangGraph.

Run: python 02_tools.py

This is where LangGraph starts looking like your custom agent.py -
an LLM that can call tools in a loop.
"""

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable")
    exit(1)


# =============================================================================
# CONCEPT 1: Message-Based State
# =============================================================================
# For chat agents, state is typically a list of messages.
# The `add_messages` annotation tells LangGraph to APPEND new messages
# rather than replace them.

class AgentState(TypedDict):
    """
    Messages accumulate as the conversation progresses.
    
    add_messages is a "reducer" - it defines HOW to update this field.
    Without it, returning {"messages": [new_msg]} would REPLACE all messages.
    With it, new messages get APPENDED.
    """
    messages: Annotated[list, add_messages]


# =============================================================================
# CONCEPT 2: Define Tools
# =============================================================================
# Tools are just Python functions with docstrings.
# LangChain converts them to the format OpenAI expects.

def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: The city to get weather for
    """
    # Fake implementation - real app would call weather API
    weather_data = {
        "new york": "72°F, Sunny",
        "san francisco": "65°F, Foggy",
        "chicago": "58°F, Windy",
        "miami": "85°F, Humid"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def get_time(city: str) -> str:
    """Get the current time in a city.
    
    Args:
        city: The city to get time for
    """
    # Fake implementation
    times = {
        "new york": "3:30 PM EST",
        "san francisco": "12:30 PM PST",
        "chicago": "2:30 PM CST",
        "miami": "3:30 PM EST"
    }
    return times.get(city.lower(), f"Time data not available for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")
    """
    try:
        # WARNING: eval is dangerous in production! Use a safe math parser.
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"


# List of tools the agent can use
tools = [get_weather, get_time, calculate]


# =============================================================================
# CONCEPT 3: Create the LLM with Tools
# =============================================================================

# Bind tools to the model - this tells the model what tools are available
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# CONCEPT 4: Define Nodes
# =============================================================================

def agent_node(state: AgentState) -> dict:
    """
    The "thinking" node. Calls the LLM to decide what to do.
    
    This is equivalent to your agent.py's main loop where you call
    the LLM and check for tool calls.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Return the response - it gets appended to messages
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Conditional edge: decide what happens after the agent node.
    
    This is the equivalent of your:
        if response.tool_calls:
            # execute tools
        else:
            # we're done
    
    Returns the NAME of the next node to go to.
    """
    last_message = state["messages"][-1]
    
    # If the LLM made tool calls, go to the tools node
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, we're done
    return END


# Use LangGraph's prebuilt ToolNode - it handles tool execution
tool_node = ToolNode(tools)


# =============================================================================
# CONCEPT 5: Build the Graph
# =============================================================================

def build_agent_graph():
    """
    Build a ReAct-style agent graph.
    
    The pattern:
        agent -> (has tool calls?) -> tools -> agent -> (has tool calls?) -> ...
                        |
                        v (no tool calls)
                       END
    
    This is EXACTLY what your custom agent.py did, but expressed as a graph.
    """
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    # Entry point
    graph.add_edge(START, "agent")
    
    # Conditional edge from agent
    # should_continue returns either "tools" or END
    graph.add_conditional_edges("agent", should_continue)
    
    # After tools, always go back to agent
    graph.add_edge("tools", "agent")
    
    return graph.compile()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("LANGGRAPH WITH TOOLS - ReAct Pattern")
    print("=" * 60)
    
    app = build_agent_graph()
    
    # Test queries
    test_queries = [
        "What's the weather in New York?",
        "What time is it in San Francisco and what's the weather there?",
        "Calculate 15 * 7 + 23",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)
        
        # Run the agent
        result = app.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Print final response
        final_message = result["messages"][-1]
        print(f"\nFinal Response: {final_message.content}")
        
        # Show the full conversation
        print(f"\n--- Full Message History ({len(result['messages'])} messages) ---")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            content = msg.content[:100] + "..." if len(str(msg.content)) > 100 else msg.content
            print(f"  {i+1}. [{msg_type}] {content}")


def visualize():
    """Show the graph structure."""
    app = build_agent_graph()
    print("\n" + "=" * 60)
    print("GRAPH VISUALIZATION (Mermaid)")
    print("=" * 60)
    print("\nPaste this into https://mermaid.live:\n")
    print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
    visualize()