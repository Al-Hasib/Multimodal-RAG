from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from src.config.settings import settings
from src.retrieval.retriever import MultiModalRetriever
from src.models.schemas import RetrievalResult, QueryResponse
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
The context includes text, tables, and images. Use all available information to give a thorough answer.
If the context lacks sufficient information, say so clearly. Cite specific parts of the context."""

RESPONSE_TEMPLATE = """
Answer the question based only on the following context, which can include text, tables, and images.
Context: {context_text}
Question: {user_question}
"""


class RAGState(TypedDict):
    question: str
    chat_history: list
    transformed_queries: list[str]
    retrieval_result: Optional[RetrievalResult]
    context_text: str
    context_images: list[str]
    response: str
    session_id: Optional[str]


def transform_queries(state: RAGState) -> dict:
    from src.retrieval.query_transformer import QueryTransformer
    qt = QueryTransformer()
    queries = qt.transform(state["question"])
    return {"transformed_queries": queries}


def retrieve(state: RAGState) -> dict:
    retriever = state.get("_retriever")
    if not retriever:
        raise ValueError("Retriever not set in state")
    result = retriever.retrieve(state["question"])
    return {"retrieval_result": result}


def build_context(state: RAGState) -> dict:
    result = state["retrieval_result"]
    context_text = ""
    if result.texts:
        for t in result.texts:
            text = t.text if hasattr(t, "text") else (t.page_content if hasattr(t, "page_content") else str(t))
            context_text += text + "\n\n"
    return {
        "context_text": context_text,
        "context_images": result.images,
    }


def generate(state: RAGState) -> dict:
    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0)

    context = state["context_text"]
    question = state["question"]
    images = state["context_images"]

    prompt_text = RESPONSE_TEMPLATE.format(context_text=context, user_question=question)
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in state.get("chat_history", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    content = [{"type": "text", "text": prompt_text}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

    messages.append(HumanMessage(content=content))
    response = llm.invoke(messages)
    return {"response": response.content}


class RAGChain:
    def __init__(self, retriever: MultiModalRetriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(RAGState)

        builder.add_node("transform_queries", transform_queries)
        builder.add_node("retrieve", retrieve)
        builder.add_node("build_context", build_context)
        builder.add_node("generate", generate)

        builder.set_entry_point("transform_queries")
        builder.add_edge("transform_queries", "retrieve")
        builder.add_edge("retrieve", "build_context")
        builder.add_edge("build_context", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    def invoke(self, question: str, chat_history: Optional[list] = None, session_id: Optional[str] = None) -> str:
        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "transformed_queries": [],
            "retrieval_result": None,
            "context_text": "",
            "context_images": [],
            "response": "",
            "session_id": session_id,
            "_retriever": self.retriever,
        }
        result = self.graph.invoke(initial_state)
        return result["response"]

    def invoke_with_sources(self, question: str, chat_history: Optional[list] = None, k: int = 5) -> QueryResponse:
        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "transformed_queries": [],
            "retrieval_result": None,
            "context_text": "",
            "context_images": [],
            "response": "",
            "session_id": None,
            "_retriever": self.retriever,
        }
        result = self.graph.invoke(initial_state)

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
        initial_state = {
            "question": question,
            "chat_history": chat_history or [],
            "transformed_queries": [],
            "retrieval_result": None,
            "context_text": "",
            "context_images": [],
            "response": "",
            "session_id": session_id,
            "_retriever": self.retriever,
        }
        async for event in self.graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk", {})
                content = chunk.content if hasattr(chunk, "content") else ""
                if content:
                    yield content
