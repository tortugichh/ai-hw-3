import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # still fine if used elsewhere
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai.types import Content, Part
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI  # ✅ updated import

# Load environment variables
load_dotenv()

class LlamaIndexSummarizeADKAgent(BaseAgent):
    """
    ADK Agent that uses LlamaIndex for summarizing search results.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _run_async_impl(self, ctx: InvocationContext):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            yield Event(
                content=Content(parts=[Part(text="[LlamaIndexSummarizeADKAgent] Missing OPENAI_API_KEY. Cannot initialize LLM.")]),
                author=self.name
            )
            return

        try:
            # ✅ Correct LLM initialization for llama-index
            Settings.llm = OpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
            
            # ✅ Use local HuggingFace embeddings
            Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

        except Exception as e:
            yield Event(
                content=Content(parts=[Part(text=f"[LlamaIndexSummarizeADKAgent] Failed to initialize LLM: {e}")]),
                author=self.name
            )
            return

        search_result = ctx.session.state.get("search_result")
        if not search_result:
            yield Event(
                content=Content(parts=[Part(text="[LlamaIndexSummarizeADKAgent] Search result not found in session state.")]),
                author=self.name
            )
            return

        yield Event(
            content=Content(parts=[Part(text="[LlamaIndexSummarizeADKAgent] Summarizing search results using LlamaIndex...")]),
            author=self.name
        )

        try:
            documents = [Document(text=search_result)]
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine(response_mode="tree_summarize")

            summary_response = query_engine.query("Provide a concise summary of the content.")
            summarized_result = str(summary_response)

            ctx.session.state["summarized_result"] = summarized_result

            yield Event(
                content=Content(parts=[Part(text="[LlamaIndexSummarizeADKAgent] Summarization complete. Summary stored in session state.")]),
                actions=EventActions(state_delta={"summarized_result": summarized_result}),
                author=self.name
            )
        except Exception as e:
            yield Event(
                content=Content(parts=[Part(text=f"[LlamaIndexSummarizeADKAgent] An error occurred during summarization: {e}")]),
                author=self.name
            )
