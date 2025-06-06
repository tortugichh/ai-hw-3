import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import PromptTemplate
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part  # Optional: remove if no longer using Gemini parts

# Load environment variables
load_dotenv()

# --- Debugging: Check if API keys are loaded ---
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
print(f"[LangchainSearchADKAgent] OPENAI_API_KEY loaded: {bool(openai_api_key)}")
print(f"[LangchainSearchADKAgent] SERPAPI_API_KEY loaded: {bool(serpapi_api_key)}")
# --------------------------------------------------

class LangchainSearchADKAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _run_async_impl(self, ctx: InvocationContext):

        # Safety check
        if not openai_api_key or not serpapi_api_key:
            yield Event(
                content=Content(parts=[Part(text="[LangchainSearchADKAgent] Missing API keys (OpenAI or SerpAPI).")]),
                author=self.name
            )
            return

        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            search_tool = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
        except Exception as e:
            yield Event(
                content=Content(parts=[Part(text=f"[LangchainSearchADKAgent] Failed to initialize tools: {e}")]),
                author=self.name
            )
            return

        def extract_text_from_content(content):
            if not content or not content.parts:
                return None
            return " ".join(part.text for part in content.parts if part.text)

        user_query = extract_text_from_content(ctx.user_content) or ctx.session.state.get("initial_query", "default search query")

        yield Event(
            content=Content(parts=[Part(text=f"[LangchainSearchADKAgent] Searching for: {user_query}")]),
            author=self.name
        )

        try:
            search_result = search_tool.run(user_query)
            ctx.session.state["search_result"] = search_result

            yield Event(
                content=Content(parts=[Part(text="[LangchainSearchADKAgent] Search complete.")]),
                actions=EventActions(state_delta={"search_result": search_result}),
                author=self.name
            )
        except Exception as e:
            yield Event(
                content=Content(parts=[Part(text=f"[LangchainSearchADKAgent] Search failed: {e}")]),
                author=self.name
            )
