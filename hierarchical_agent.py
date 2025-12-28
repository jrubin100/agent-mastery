"""
HIERARCHICAL MULTI-AGENT SYSTEM
===============================
A Boss/Router agent that delegates to specialist agents.

Pattern: User → Router Agent → Specialist Agent → Response

Use case: Customer support system that routes to the right department.
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
# SPECIALIST AGENTS
# ============================================================

def run_billing_agent(query: str, verbose: bool = True) -> str:
    """Billing specialist - handles payments, invoices, refunds."""
    
    if verbose:
        print(f"\n   💰 BILLING AGENT activated")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a Billing Specialist agent.
You handle:
- Payment issues
- Invoice questions
- Refund requests
- Subscription changes
- Pricing questions

Be helpful, clear, and professional. If you need to process a refund or change, 
explain what would happen (simulate the action)."""},
            {"role": "user", "content": query}
        ]
    )
    
    return response.choices[0].message.content


def run_technical_agent(query: str, verbose: bool = True) -> str:
    """Technical specialist - handles bugs, errors, how-to questions."""
    
    if verbose:
        print(f"\n   🔧 TECHNICAL AGENT activated")
    
    # Technical agent has access to search for documentation
    messages = [
        {"role": "system", "content": """You are a Technical Support Specialist agent.
You handle:
- Bug reports
- Error messages
- How-to questions
- Integration issues
- Performance problems

You have access to search for technical documentation. Be thorough and precise.
Provide step-by-step solutions when possible."""},
        {"role": "user", "content": query}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search technical documentation and knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    
    # Handle tool calls if any
    if assistant_message.tool_calls:
        messages.append(assistant_message)
        
        for tool_call in assistant_message.tool_calls:
            tool_args = json.loads(tool_call.function.arguments)
            if verbose:
                print(f"      🔍 Searching docs: {tool_args.get('query', '')[:40]}...")
            
            # Use Tavily to search
            try:
                search_result = tavily.search(query=tool_args['query'], max_results=3)
                result_text = "\n".join([f"- {r['title']}: {r['content'][:200]}" for r in search_result['results']])
            except:
                result_text = "Documentation search unavailable."
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text
            })
        
        # Get final response after tool use
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
    
    return response.choices[0].message.content


def run_sales_agent(query: str, verbose: bool = True) -> str:
    """Sales specialist - handles upgrades, enterprise, feature questions."""
    
    if verbose:
        print(f"\n   📈 SALES AGENT activated")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a Sales Specialist agent.
You handle:
- Upgrade inquiries
- Enterprise plans
- Feature comparisons
- Custom solutions
- Partnership opportunities

Be enthusiastic but not pushy. Focus on understanding needs and matching solutions.
Offer to schedule calls for complex discussions."""},
            {"role": "user", "content": query}
        ]
    )
    
    return response.choices[0].message.content


def run_general_agent(query: str, verbose: bool = True) -> str:
    """General support - handles anything that doesn't fit other categories."""
    
    if verbose:
        print(f"\n   📋 GENERAL AGENT activated")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a General Support agent.
You handle inquiries that don't fit into billing, technical, or sales categories.
This includes:
- General questions about the company
- Feedback and suggestions
- Account management
- Anything else

Be helpful and redirect to specialists if needed."""},
            {"role": "user", "content": query}
        ]
    )
    
    return response.choices[0].message.content


# ============================================================
# ROUTER AGENT (The Boss)
# ============================================================

ROUTER_PROMPT = """You are a Router Agent for a customer support system.

Your ONLY job is to analyze incoming messages and decide which specialist should handle them.

SPECIALISTS AVAILABLE:
1. BILLING - payments, invoices, refunds, subscriptions, pricing
2. TECHNICAL - bugs, errors, how-to, integrations, performance
3. SALES - upgrades, enterprise plans, feature questions, partnerships
4. GENERAL - anything else

Respond with ONLY a JSON object:
{
    "department": "BILLING" | "TECHNICAL" | "SALES" | "GENERAL",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}

Examples:
- "I was charged twice" → {"department": "BILLING", "confidence": 0.95, "reasoning": "Payment issue"}
- "App crashes on login" → {"department": "TECHNICAL", "confidence": 0.95, "reasoning": "Bug report"}
- "What's in the enterprise plan?" → {"department": "SALES", "confidence": 0.9, "reasoning": "Plan inquiry"}
- "Who is your CEO?" → {"department": "GENERAL", "confidence": 0.85, "reasoning": "Company info"}
"""


def run_router_agent(query: str, verbose: bool = True) -> dict:
    """Router Agent: Decides which specialist handles the query."""
    
    if verbose:
        print(f"\n🎯 ROUTER AGENT analyzing...")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    if verbose:
        print(f"   📍 Routing to: {result['department']}")
        print(f"   📊 Confidence: {result['confidence']}")
        print(f"   💭 Reason: {result['reasoning']}")
    
    return result


# ============================================================
# ORCHESTRATOR
# ============================================================

SPECIALISTS = {
    "BILLING": run_billing_agent,
    "TECHNICAL": run_technical_agent,
    "SALES": run_sales_agent,
    "GENERAL": run_general_agent
}


def run_hierarchical_system(query: str, verbose: bool = True) -> dict:
    """
    Main orchestrator for the hierarchical system.
    
    Flow: Query → Router → Specialist → Response
    """
    
    if verbose:
        print("\n" + "="*60)
        print("🏢 HIERARCHICAL SUPPORT SYSTEM")
        print("="*60)
        print(f"📩 Query: {query}")
        print("="*60)
    
    # Step 1: Router decides who handles this
    routing = run_router_agent(query, verbose=verbose)
    
    # Step 2: Call the appropriate specialist
    department = routing["department"]
    specialist_fn = SPECIALISTS.get(department, run_general_agent)
    
    specialist_response = specialist_fn(query, verbose=verbose)
    
    if verbose:
        print(f"\n✅ Response generated by {department} specialist")
    
    return {
        "routing": routing,
        "department": department,
        "response": specialist_response
    }


# ============================================================
# RUN IT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏢 HIERARCHICAL CUSTOMER SUPPORT SYSTEM")
    print("="*60)
    print("This system routes your query to the right specialist:")
    print("  💰 Billing - payments, refunds, invoices")
    print("  🔧 Technical - bugs, errors, how-to")
    print("  📈 Sales - upgrades, enterprise, features")
    print("  📋 General - everything else")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Your support request:\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = run_hierarchical_system(user_input, verbose=True)
        
        print("\n" + "="*60)
        print(f"💬 RESPONSE FROM {result['department']}")
        print("="*60)
        print(result["response"])
        print("="*60 + "\n")