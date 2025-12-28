"""
MULTI-AGENT SYSTEM
==================
Two agents working together:
1. Research Agent - Gathers information from the web
2. Writer Agent - Turns research into polished content

This is the SEQUENTIAL pattern: Agent A → Agent B
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
client = OpenAI()
tavily = TavilyClient()


# ============================================================
# THE TOOLS (Shared by agents that need them)
# ============================================================

def search_web(query: str) -> str:
    """Search the web using Tavily."""
    try:
        response = tavily.search(query=query, max_results=5)
        results = []
        for result in response['results']:
            results.append(f"- {result['title']}: {result['content'][:300]}")
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================
# AGENT 1: RESEARCH AGENT
# ============================================================

RESEARCH_AGENT_PROMPT = """You are a Research Agent. Your job is to gather comprehensive information on a topic.

YOUR PROCESS:
1. Search for information on the given topic
2. Search multiple times if needed to get different angles
3. Compile your findings into a clear research summary

OUTPUT FORMAT:
Provide a structured research summary with:
- Key facts and statistics
- Different perspectives or viewpoints
- Recent developments
- Interesting quotes or data points

Be thorough. The Writer Agent will use your research to create content."""


def run_research_agent(topic: str, verbose: bool = True) -> str:
    """Research Agent: Gathers information on a topic."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🔬 RESEARCH AGENT activated")
        print(f"📋 Topic: {topic}")
        print(f"{'='*60}\n")
    
    messages = [
        {"role": "system", "content": RESEARCH_AGENT_PROMPT},
        {"role": "user", "content": f"Research this topic thoroughly: {topic}"}
    ]
    
    tool_functions = {"search_web": search_web}
    max_iterations = 8
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        if verbose:
            print(f"🔄 Research iteration {iteration}")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=SEARCH_TOOLS,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        if assistant_message.tool_calls:
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"   🔍 Searching: {tool_args.get('query', '')[:50]}...")
                
                result = tool_functions[tool_name](**tool_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            research = assistant_message.content
            if verbose:
                print(f"\n✅ Research complete ({iteration} iterations)")
                print(f"📄 Research length: {len(research)} characters\n")
            return research
    
    return "Research incomplete - max iterations reached."


# ============================================================
# AGENT 2: WRITER AGENT
# ============================================================

WRITER_AGENT_PROMPT = """You are a Writer Agent. Your job is to transform research into polished, engaging content.

You will receive research from the Research Agent. Your task:
1. Analyze the research provided
2. Create well-structured, engaging content
3. Use a clear, professional tone
4. Include relevant facts and data from the research
5. Make it readable and interesting

OUTPUT: A polished blog post or article based on the research."""


def run_writer_agent(research: str, content_type: str = "blog post", verbose: bool = True) -> str:
    """Writer Agent: Transforms research into polished content."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✍️  WRITER AGENT activated")
        print(f"📝 Content type: {content_type}")
        print(f"{'='*60}\n")
    
    messages = [
        {"role": "system", "content": WRITER_AGENT_PROMPT},
        {"role": "user", "content": f"""Based on this research, write a {content_type}:

RESEARCH:
{research}

Write an engaging {content_type} based on this research."""}
    ]
    
    if verbose:
        print("📝 Writing content...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    content = response.choices[0].message.content
    
    if verbose:
        print(f"✅ Writing complete")
        print(f"📄 Content length: {len(content)} characters\n")
    
    return content


# ============================================================
# ORCHESTRATOR: Coordinates the agents
# ============================================================

def run_multi_agent(task: str, verbose: bool = True) -> dict:
    """
    Orchestrates the multi-agent workflow.
    
    Flow: User Task → Research Agent → Writer Agent → Final Output
    """
    
    if verbose:
        print("\n" + "="*60)
        print("🚀 MULTI-AGENT SYSTEM")
        print("="*60)
        print(f"📌 Task: {task}")
        print("="*60)
    
    # Step 1: Research Agent gathers information
    research = run_research_agent(task, verbose=verbose)
    
    # Step 2: Writer Agent creates content from research
    final_content = run_writer_agent(research, "blog post", verbose=verbose)
    
    # Return both for transparency
    return {
        "research": research,
        "final_content": final_content
    }


# ============================================================
# RUN IT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 MULTI-AGENT SYSTEM")
    print("="*60)
    print("This system uses two agents:")
    print("  1. Research Agent - Gathers information")
    print("  2. Writer Agent - Creates polished content")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("What should I research and write about?\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = run_multi_agent(user_input, verbose=True)
        
        print("\n" + "="*60)
        print("📰 FINAL BLOG POST")
        print("="*60)
        print(result["final_content"])
        print("="*60 + "\n")