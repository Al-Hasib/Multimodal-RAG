from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config.settings import settings
from src.retrieval.retriever import MultiModalRetriever
from src.generation.prompts import prompt_manager
from src.models.schemas import RetrievalResult
import logging

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    question: str
    chat_history: list
    transformed_queries: list[str]
    retrieval_result: Optional[RetrievalResult]
    context_text: str
    context_images: list[str]
    response: str
    session_id: Optional[str]


def create_rag_graph(retriever: MultiModalRetriever):
    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0)

    def transform_queries(state: RAGState) -> dict:
        from src.retrieval.query_transformer import QueryTransformer
        qt = QueryTransformer()
        queries = qt.transform(state["question"])
        return {"transformed_queries": queries}

    def retrieve(state: RAGState) -> dict:
        result = retriever.retrieve(state["question"])
        return {"retrieval_result": result}

    def build_context(state: RAGState) -> dict:
        result = state["retrieval_result"]
        context_text = ""
        if result.texts:
            for t in result.texts:
                text = t.text if hasattr(t, "text") else (t.page_content if hasattr(t, "page_content") else str(t))
                context_text += text + "\n\n"
        return {"context_text": context_text, "context_images": result.images}

    def generate(state: RAGState) -> dict:
        context = state["context_text"]
        question = state["question"]
        images = state["context_images"]

        system_prompt = prompt_manager.get_prompt("system")
        response_template = prompt_manager.get_prompt("response")
        prompt_text = response_template.format(context_text=context, user_question=question)

        messages = [SystemMessage(content=system_prompt)]
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

    builder = StateGraph(RAGState)
    builder.add_node("transform_queries", transform_queries)
    builder.add_node("retrieve", retrieve)
    builder.add_node("build_context", build_context)
    builder.add_node("generate", generate)

    builder.add_edge(START, "transform_queries")
    builder.add_edge("transform_queries", "retrieve")
    builder.add_edge("retrieve", "build_context")
    builder.add_edge("build_context", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())
