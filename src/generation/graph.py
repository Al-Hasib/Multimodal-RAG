from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.config.settings import settings
from src.retrieval.retriever import MultiModalRetriever
from src.generation.prompts import prompt_manager
from src.models.schemas import RetrievalResult
from src.core.guardrails import guardrail_manager, GuardrailResult
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
    guardrail_input: Optional[dict]
    guardrail_output: Optional[dict]
    blocked: bool
    block_reason: Optional[str]


def create_rag_graph(retriever: MultiModalRetriever):
    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0)

    def transform_queries(state: RAGState) -> dict:
        from src.retrieval.query_transformer import QueryTransformer
        qt = QueryTransformer()
        queries = qt.transform(state["question"])
        return {"transformed_queries": queries}

    def check_input_guardrail(state: RAGState) -> dict:
        import asyncio
        result = asyncio.run(guardrail_manager.check_input(state["question"]))

        if not result.passed and result.action.value == "block":
            logger.warning(f"Input guardrail blocked: {result.reason}")
            reason = f"I couldn't process that request. Reason: {result.reason}" if settings.guardrail_block_on_input_violation else ""
            return {
                "guardrail_input": {"passed": False, "reason": result.reason, "action": result.action.value},
                "blocked": True,
                "block_reason": reason,
                "response": reason,
            }

        return {
            "guardrail_input": {"passed": True, "flags": result.details.get("flags", [])},
            "blocked": False,
            "block_reason": None,
        }

    def reject_query(state: RAGState) -> dict:
        reason = state.get("block_reason") or "Your request couldn't be processed."
        safe = "I'm sorry, but I can't answer that question based on the available documents."
        return {"response": safe, "blocked": True}

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

    def check_output_guardrail(state: RAGState) -> dict:
        import asyncio
        result = asyncio.run(guardrail_manager.check_output(
            question=state["question"],
            answer=state["response"],
            context=state["context_text"],
        ))

        if not result.passed and result.action.value == "block":
            logger.warning(f"Output guardrail blocked: {result.reason}")
            safe = "I'm sorry, but I couldn't generate a suitable answer for that question."
            return {
                "guardrail_output": {"passed": False, "reason": result.reason, "action": result.action.value},
                "response": safe,
                "blocked": True,
            }

        return {
            "guardrail_output": {"passed": True, "flags": result.details.get("flags", [])},
        }

    def route_after_input_guardrail(state: RAGState) -> Literal["retrieve", "reject_query"]:
        if state.get("blocked"):
            return "reject_query"
        return "retrieve"

    def route_after_generate(state: RAGState) -> Literal["check_output_guardrail", END]:
        if state.get("blocked"):
            return END
        return "check_output_guardrail"

    def route_after_output_guardrail(state: RAGState) -> Literal["generate", END]:
        if state.get("blocked"):
            return "generate"
        return END

    builder = StateGraph(RAGState)

    builder.add_node("transform_queries", transform_queries)
    builder.add_node("check_input_guardrail", check_input_guardrail)
    builder.add_node("reject_query", reject_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("build_context", build_context)
    builder.add_node("generate", generate)
    builder.add_node("check_output_guardrail", check_output_guardrail)

    builder.add_edge(START, "transform_queries")
    builder.add_edge("transform_queries", "check_input_guardrail")
    builder.add_conditional_edges("check_input_guardrail", route_after_input_guardrail)
    builder.add_edge("reject_query", END)
    builder.add_edge("retrieve", "build_context")
    builder.add_edge("build_context", "generate")
    builder.add_conditional_edges("generate", route_after_generate)
    builder.add_conditional_edges("check_output_guardrail", route_after_output_guardrail)

    compiled = builder.compile(checkpointer=MemorySaver())
    logger.info("RAG graph compiled with guardrails")
    return compiled
