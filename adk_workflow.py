import asyncio
import uuid
import os
from dotenv import load_dotenv
from typing import Dict, Any
import google.adk.agents as adk_agents
import google.adk.sessions.in_memory_session_service as session_service
import google.adk.runners as adk_runners
import google.genai.types as genai_types
from langchain_search_adk_agent import LangchainSearchADKAgent
from llama_index_summarize_adk_agent import LlamaIndexSummarizeADKAgent

# Load environment variables
load_dotenv()

print(f"[ENV] GOOGLE_API_KEY loaded? {os.getenv('GOOGLE_API_KEY') is not None}")
print(f"[ENV] SERPAPI_API_KEY loaded? {os.getenv('SERPAPI_API_KEY') is not None}")

async def run_adk_workflow(initial_query: str) -> Dict[str, Any]:
    print(f"[ADK Workflow] Starting workflow with initial query: '{initial_query}'")

    # Instantiate agents
    print("[ADK Workflow] Instantiating agents...")
    search_agent = LangchainSearchADKAgent(name="LangchainSearchAgent")
    summarize_agent = LlamaIndexSummarizeADKAgent(name="LlamaIndexSummarizeAgent")
    print("[ADK Workflow] Agents instantiated.")

    # Create workflow
    print("[ADK Workflow] Defining sequential workflow...")
    workflow_agent = adk_agents.SequentialAgent(
        name="ResearchAndSummarizeWorkflow",
        description="A workflow that searches for information and then summarizes it.",
        sub_agents=[search_agent, summarize_agent]
    )
    print("[ADK Workflow] Workflow defined.")

    # Setup session
    print("[ADK Workflow] Setting up session service...")
    session_svc = session_service.InMemorySessionService()
    print("[ADK Workflow] Session service set up.")

    # Runner
    print("[ADK Workflow] Setting up runner...")
    runner = adk_runners.Runner(
        agent=workflow_agent,
        app_name="ResearchSummarizeApp",
        session_service=session_svc
    )
    print("[ADK Workflow] Runner set up.")

    # Create session
   # Create session
    print("[ADK Workflow] Creating new session...")
    session_id = str(uuid.uuid4())
    user_id = "user123"

    await session_svc.create_session(
        session_id=session_id,
        app_name="ResearchSummarizeApp",
        user_id=user_id
    )
    print(f"[ADK Workflow] Session created with ID: {session_id} for user {user_id}.")

    # ✅ Manually initialize session state if needed
    session = await session_svc.get_session(
        app_name="ResearchSummarizeApp",
        user_id=user_id,
        session_id=session_id
    )
    session.state["initial_query"] = initial_query


    # Initial user message
    initial_message = genai_types.Content(parts=[genai_types.Part(text=initial_query)])

    # Run workflow
    print("[ADK Workflow] Running workflow...")
    print("-" * 40)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=initial_message
    ):
        event_text = ""
        if event.content and hasattr(event.content, 'parts'):
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if text:
                    event_text += text

        print(f"[EVENT from {event.author if event.author else 'Workflow'}]: Content='{event_text or '[No text content]'}', State Delta={event.actions.state_delta if event.actions else 'None'}")

    print("-" * 40)
    print("[ADK Workflow] Workflow finished.")

    # Final session state
    print("[ADK Workflow] Retrieving final session state...")
    final_session = await session_svc.get_session(
        app_name="ResearchSummarizeApp",
        user_id=user_id,
        session_id=session_id
    )

    print("\n[ADK Workflow] Final Session State:")
    search_result = final_session.state.get("search_result", 'N/A')
    summarized_result = final_session.state.get("summarized_result", 'N/A')

    print(f"  Search Result: {search_result[:500] + '...' if isinstance(search_result, str) and len(search_result) > 500 else search_result}")
    print(f"  Summarized Result: {summarized_result}")

    print("\n[ADK Workflow] Overall Process Summary:")
    if search_result != 'N/A' and summarized_result != 'N/A':
        print("✅ Search and summarization completed successfully.")
    elif search_result != 'N/A':
        print("⚠️ Search succeeded, but summarization failed.")
    elif summarized_result != 'N/A':
        print("⚠️ Summarization succeeded, but search result was not found.")
    else:
        print("❌ Neither search nor summarization succeeded.")

    print("-" * 40)

    results = {
        "initial_query": final_session.state.get("initial_query", initial_query),
        "search_results": search_result,
        "summary": summarized_result,
        "session_id": session_id,
        "status": "completed" if final_session.state.get("summary_completed") else "incomplete",
        "error": None if (search_result != 'N/A' and summarized_result != 'N/A') else "Partial or full failure in multi-agent process."
    }

    return results

def display_results(results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("🔍 MULTI-AGENT SYSTEM RESULTS")
    print("=" * 80)

    if results.get("error"):
        print(f"❌ Error: {results['error']}")
        print("=" * 80)
        return

    print(f"📝 Query: {results.get('initial_query', 'N/A')}")
    print(f"🆔 Session ID: {results.get('session_id', 'N/A')}")
    print(f"📊 Status: {results.get('status', 'N/A')}")
    print("-" * 80)

    print("\n🔍 SEARCH RESULTS:")
    print("-" * 40)
    search_results = results.get('search_results', 'No results')
    if isinstance(search_results, str) and len(search_results) > 500:
        print(f"{search_results[:500]}...")
        print(f"[Results truncated - Total length: {len(search_results)} characters]")
    else:
        print(search_results)

    print("\n📄 SUMMARY:")
    print("-" * 40)
    print(results.get('summary', 'No summary available'))

    print("\n" + "=" * 80)
