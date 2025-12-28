"""
SIMPLE RESEARCH AGENT
=====================
This is the foundational pattern behind every AI agent.
Once you understand this, frameworks are just convenience layers on top.

The Loop: THINK → ACT → OBSERVE → DECIDE (repeat or respond)
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ============================================================
# PART 1: DEFINE YOUR TOOLS
# ============================================================
# Tools are functions the agent can call. Each tool needs:
# 1. A Python function that does the work
# 2. A schema that tells the AI what the tool does and what inputs it needs

def search_web(query: str) -> str:
    """Simulate a web search. In production, you'd hit a real API."""
    fake_results = {
        "weather": "Current weather in Tel Aviv: 18°C, partly cloudy.",
        "python": "Python is a programming language created by Guido van Rossum in 1991.",
        "agents": "AI agents are autonomous systems that can perceive, decide, and act to achieve goals.",
        "openai": "OpenAI is an AI research company founded in 2015. They created GPT-4 and ChatGPT.",
        "default": f"Search results for '{query}': Found 3 articles discussing this topic with varying perspectives."
    }
    for key in fake_results:
        if key in query.lower():
            return fake_results[key]
    return fake_results["default"]


def read_file(filepath: str) -> str:
    """Read contents of a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and +-*/.() allowed."
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Map tool names to functions
TOOL_FUNCTIONS = {
    "search_web": search_web,
    "read_file": read_file,
    "calculate": calculate,
}

# ============================================================
# PART 2: TOOL SCHEMAS
# ============================================================
# This is the exact format OpenAI expects.
# The AI reads these descriptions to decide WHEN and HOW to use each tool.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Use this when you need facts, news, or information you don't know.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use this when the user references a specific file or document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations. Use this for any math operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate (e.g., '2 + 2', '100 * 0.15')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ============================================================
# PART 3: SYSTEM PROMPT
# ============================================================
# This is where you define HOW the agent behaves.

SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.

YOUR PROCESS:
1. When you receive a question, THINK about what information you need
2. If you need external information, use your tools to get it
3. You can use multiple tools if needed - keep going until you have enough info
4. Once you have sufficient information, provide a clear, helpful answer

IMPORTANT RULES:
- ALWAYS use tools when you need current/external information
- Don't make up facts - if you need to search, search
- Show your reasoning - explain what you're doing and why
- If a tool returns an error, try a different approach
- Be concise but thorough in your final answer

You have access to: search_web, read_file, calculate
"""

# ============================================================
# PART 4: THE AGENT LOOP
# ============================================================

def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    The main agent loop. This is where the magic happens.
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🧑 USER: {user_message}")
        print(f"{'='*60}\n")
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        if verbose:
            print(f"🔄 Loop iteration {iteration}")
            print("-" * 40)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        if assistant_message.tool_calls:
            if verbose:
                print(f"🤖 THINK: I need to use tools...")
            
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"🔧 ACT: Calling {tool_name}({tool_args})")
                
                if tool_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[tool_name](**tool_args)
                else:
                    result = f"Error: Unknown tool '{tool_name}'"
                
                if verbose:
                    print(f"📊 OBSERVE: {result[:200]}...")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            if verbose:
                print(f"🔄 DECIDE: Let me process these results...\n")
            
        else:
            final_response = assistant_message.content
            
            if verbose:
                print(f"✅ DONE: Agent has finished thinking")
                print(f"\n{'='*60}")
                print(f"🤖 AGENT: {final_response}")
                print(f"{'='*60}\n")
            
            return final_response
    
    return "I apologize, but I wasn't able to complete this task within the allowed steps."


# ============================================================
# PART 5: RUN IT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 RESEARCH AGENT - Learning Edition")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        run_agent(user_input, verbose=True)