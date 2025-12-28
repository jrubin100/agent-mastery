"""
SWARM MULTI-AGENT SYSTEM
========================
Agents collaborate dynamically, share findings, and react to each other.

Pattern: Agents explore → Share findings → React → Converge on answer

Use case: Investigating a complex topic from multiple angles.
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
# SHARED CONTEXT (The Swarm's Collective Memory)
# ============================================================

def create_shared_context(task: str) -> dict:
    """Initialize the shared context that all agents can read/write."""
    return {
        "task": task,
        "findings": [],              # All discoveries go here
        "open_threads": [task],      # Things to investigate (starts with main task)
        "investigated": [],          # Already explored threads
        "iteration": 0,
        "max_iterations": 10,
        "confidence": 0.0,
        "agents_last_round": [],     # Track which agents contributed
    }


# ============================================================
# SWARM AGENTS
# ============================================================

def run_researcher_agent(shared_context: dict, verbose: bool = True) -> dict:
    """Researcher: Searches for factual information."""
    
    if not shared_context["open_threads"]:
        return {"contributed": False}
    
    # Pick a thread to investigate
    thread = shared_context["open_threads"][0]
    
    if verbose:
        print(f"\n   🔬 RESEARCHER investigating: {thread[:50]}...")
    
    try:
        search_results = tavily.search(query=thread, max_results=3)
        findings = []
        for r in search_results['results']:
            findings.append(f"{r['title']}: {r['content'][:200]}")
        result_text = "\n".join(findings) if findings else "No results found."
    except Exception as e:
        result_text = f"Search failed: {str(e)}"
    
    # Ask GPT to extract key insights and new threads
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a Research Agent in a swarm.
Analyze search results and extract:
1. Key findings (facts, data, insights)
2. New questions or threads to investigate

Respond in JSON:
{
    "findings": ["finding 1", "finding 2"],
    "new_threads": ["question 1", "question 2"],
    "confidence_boost": 0.0 to 0.2
}"""},
            {"role": "user", "content": f"Task: {shared_context['task']}\n\nSearch results for '{thread}':\n{result_text}"}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    # Update shared context
    for finding in result.get("findings", []):
        shared_context["findings"].append(f"[RESEARCHER] {finding}")
    
    for new_thread in result.get("new_threads", []):
        if new_thread not in shared_context["investigated"] and new_thread not in shared_context["open_threads"]:
            shared_context["open_threads"].append(new_thread)
    
    shared_context["confidence"] += result.get("confidence_boost", 0.05)
    
    # Mark thread as investigated
    if thread in shared_context["open_threads"]:
        shared_context["open_threads"].remove(thread)
    shared_context["investigated"].append(thread)
    
    if verbose:
        print(f"      Found {len(result.get('findings', []))} insights, {len(result.get('new_threads', []))} new threads")
    
    return {"contributed": True, "findings": result.get("findings", [])}


def run_analyst_agent(shared_context: dict, verbose: bool = True) -> dict:
    """Analyst: Looks for patterns and connections in findings."""
    
    if len(shared_context["findings"]) < 2:
        return {"contributed": False}
    
    if verbose:
        print(f"\n   📊 ANALYST looking for patterns...")
    
    findings_text = "\n".join(shared_context["findings"][-10:])  # Last 10 findings
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are an Analyst Agent in a swarm.
Look at the findings and identify:
1. Patterns or connections between findings
2. Contradictions that need resolution
3. Gaps in knowledge

Respond in JSON:
{
    "patterns": ["pattern 1", "pattern 2"],
    "contradictions": ["contradiction 1"],
    "gaps": ["gap 1", "gap 2"],
    "confidence_boost": 0.0 to 0.15
}"""},
            {"role": "user", "content": f"Task: {shared_context['task']}\n\nCurrent findings:\n{findings_text}"}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    # Add patterns as findings
    for pattern in result.get("patterns", []):
        shared_context["findings"].append(f"[ANALYST] Pattern: {pattern}")
    
    # Add gaps as new threads
    for gap in result.get("gaps", []):
        if gap not in shared_context["investigated"] and gap not in shared_context["open_threads"]:
            shared_context["open_threads"].append(gap)
    
    shared_context["confidence"] += result.get("confidence_boost", 0.05)
    
    if verbose:
        print(f"      Found {len(result.get('patterns', []))} patterns, {len(result.get('gaps', []))} gaps")
    
    return {"contributed": True, "patterns": result.get("patterns", [])}


def run_critic_agent(shared_context: dict, verbose: bool = True) -> dict:
    """Critic: Challenges assumptions and checks for weaknesses."""
    
    if len(shared_context["findings"]) < 3:
        return {"contributed": False}
    
    if verbose:
        print(f"\n   🔍 CRITIC checking for weaknesses...")
    
    findings_text = "\n".join(shared_context["findings"][-8:])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a Critic Agent in a swarm.
Your job is to challenge findings and identify:
1. Weak claims that need more evidence
2. Assumptions being made
3. Alternative explanations

Respond in JSON:
{
    "challenges": ["challenge 1", "challenge 2"],
    "needs_verification": ["claim 1"],
    "confidence_adjustment": -0.1 to 0.1
}"""},
            {"role": "user", "content": f"Task: {shared_context['task']}\n\nFindings to critique:\n{findings_text}"}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    # Add challenges as findings
    for challenge in result.get("challenges", []):
        shared_context["findings"].append(f"[CRITIC] Challenge: {challenge}")
    
    # Add verification needs as threads
    for claim in result.get("needs_verification", []):
        verification_thread = f"Verify: {claim}"
        if verification_thread not in shared_context["investigated"] and verification_thread not in shared_context["open_threads"]:
            shared_context["open_threads"].append(verification_thread)
    
    shared_context["confidence"] += result.get("confidence_adjustment", 0)
    shared_context["confidence"] = max(0, shared_context["confidence"])  # Don't go negative
    
    if verbose:
        print(f"      Raised {len(result.get('challenges', []))} challenges")
    
    return {"contributed": True, "challenges": result.get("challenges", [])}


def run_synthesizer_agent(shared_context: dict, verbose: bool = True) -> str:
    """Synthesizer: Creates the final answer from all findings."""
    
    if verbose:
        print(f"\n   📝 SYNTHESIZER creating final answer...")
    
    findings_text = "\n".join(shared_context["findings"])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a Synthesizer Agent.
Take all the findings from the swarm and create a comprehensive, well-organized answer.

Structure your response:
1. Executive Summary (2-3 sentences)
2. Key Findings (the most important discoveries)
3. Analysis (patterns, connections, implications)
4. Caveats (limitations, challenges raised, uncertainties)
5. Conclusion

Be thorough but concise. This is the final output of the investigation."""},
            {"role": "user", "content": f"Task: {shared_context['task']}\n\nAll findings from the swarm:\n{findings_text}"}
        ]
    )
    
    return response.choices[0].message.content


# ============================================================
# TERMINATION CONDITIONS
# ============================================================

def should_terminate(shared_context: dict, verbose: bool = True) -> tuple[bool, str]:
    """Check if the swarm should stop."""
    
    # Condition 1: Max iterations (safety net)
    if shared_context["iteration"] >= shared_context["max_iterations"]:
        return True, "Max iterations reached"
    
    # Condition 2: High confidence
    if shared_context["confidence"] >= 0.85:
        return True, f"Confidence threshold reached ({shared_context['confidence']:.2f})"
    
    # Condition 3: No more open threads AND we have findings
    if not shared_context["open_threads"] and len(shared_context["findings"]) > 3:
        return True, "All threads investigated"
    
    # Condition 4: No progress (same findings count for 2 rounds)
    # This would require tracking, keeping simple for now
    
    return False, ""


# ============================================================
# SWARM ORCHESTRATOR
# ============================================================

def run_swarm(task: str, verbose: bool = True) -> dict:
    """
    Run the swarm investigation.
    
    Agents collaborate until termination conditions are met.
    """
    
    if verbose:
        print("\n" + "="*60)
        print("🐝 SWARM SYSTEM ACTIVATED")
        print("="*60)
        print(f"📌 Task: {task}")
        print("="*60)
        print("\nAgents: Researcher 🔬 | Analyst 📊 | Critic 🔍")
        print("="*60)
    
    # Initialize shared context
    ctx = create_shared_context(task)
    
    # The swarm loop
    while True:
        ctx["iteration"] += 1
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"🔄 SWARM ITERATION {ctx['iteration']}")
            print(f"   Open threads: {len(ctx['open_threads'])} | Findings: {len(ctx['findings'])} | Confidence: {ctx['confidence']:.2f}")
            print(f"{'─'*60}")
        
        # Check termination BEFORE running agents
        should_stop, reason = should_terminate(ctx, verbose)
        if should_stop:
            if verbose:
                print(f"\n⏹️  TERMINATION: {reason}")
            break
        
        # Run each agent (they all see the same shared context)
        agents = [
            ("Researcher", run_researcher_agent),
            ("Analyst", run_analyst_agent),
            ("Critic", run_critic_agent),
        ]
        
        contributions = 0
        for name, agent_fn in agents:
            result = agent_fn(ctx, verbose=verbose)
            if result.get("contributed"):
                contributions += 1
        
        # If no agent contributed, we might be stuck
        if contributions == 0:
            if verbose:
                print("\n⚠️  No agent contributed this round")
            # Add confidence anyway to eventually terminate
            ctx["confidence"] += 0.1
    
    # Synthesize final answer
    if verbose:
        print("\n" + "="*60)
        print("📝 SYNTHESIZING FINAL ANSWER")
        print("="*60)
    
    final_answer = run_synthesizer_agent(ctx, verbose=verbose)
    
    return {
        "task": task,
        "iterations": ctx["iteration"],
        "total_findings": len(ctx["findings"]),
        "threads_investigated": len(ctx["investigated"]),
        "final_confidence": ctx["confidence"],
        "findings": ctx["findings"],
        "answer": final_answer
    }


# ============================================================
# RUN IT
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🐝 SWARM INVESTIGATION SYSTEM")
    print("="*60)
    print("This system uses multiple agents that collaborate:")
    print("  🔬 Researcher - Finds information")
    print("  📊 Analyst - Identifies patterns")
    print("  🔍 Critic - Challenges assumptions")
    print("  📝 Synthesizer - Creates final answer")
    print("="*60)
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("What should the swarm investigate?\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = run_swarm(user_input, verbose=True)
        
        print("\n" + "="*60)
        print("🎯 SWARM INVESTIGATION COMPLETE")
        print("="*60)
        print(f"Iterations: {result['iterations']}")
        print(f"Findings: {result['total_findings']}")
        print(f"Threads investigated: {result['threads_investigated']}")
        print(f"Final confidence: {result['final_confidence']:.2f}")
        print("="*60)
        print("\n📋 FINAL ANSWER:\n")
        print(result["answer"])
        print("\n" + "="*60 + "\n")