from typing import Optional
from src.config.settings import settings
from src.retrieval.retriever import MultiModalRetriever
from src.generation.graph import create_rag_graph
from src.models.schemas import QueryResponse
import logging

logger = logging.getLogger(__name__)


class RAGChain:
    def __init__(self, retriever: MultiModalRetriever):
        self.retriever = retriever
        self.llm = None
        self.graph = create_rag_graph(retriever)

    def _make_config(self, session_id: Optional[str] = None) -> dict:
        return {"configurable": {"thread_id": session_id or "default"}}

    def _make_initial_state(self, question: str, chat_history: Optional[list] = None, session_id: Optional[str] = None) -> dict:
        return {
            "question": question,
            "chat_history": chat_history or [],
            "transformed_queries": [],
            "retrieval_result": None,
            "context_text": "",
            "context_images": [],
            "response": "",
            "session_id": session_id,
        }

    def invoke(self, question: str, chat_history: Optional[list] = None, session_id: Optional[str] = None) -> str:
        result = self.graph.invoke(
            self._make_initial_state(question, chat_history, session_id),
            config=self._make_config(session_id),
        )
        return result["response"]

    def invoke_with_sources(self, question: str, chat_history: Optional[list] = None) -> QueryResponse:
        result = self.graph.invoke(
            self._make_initial_state(question, chat_history),
            config=self._make_config(),
        )

        context_texts = []
        for t in result["retrieval_result"].texts:
            text = t.text if hasattr(t, "text") else (t.page_content if hasattr(t, "page_content") else str(t))
            context_texts.append(text)

        return QueryResponse(
            answer=result["response"],
            context_texts=context_texts,
            context_images=result["context_images"],
        )

    async def astream(self, question: str, chat_history: Optional[list] = None, session_id: Optional[str] = None):
        initial_state = self._make_initial_state(question, chat_history, session_id)
        config = self._make_config(session_id)
        async for event in self.graph.astream_events(initial_state, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", {})
                content = getattr(chunk, "content", "")
                if content:
                    yield content
