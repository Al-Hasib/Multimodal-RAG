from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

HYDE_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant. Given a question, write a concise hypothetical document "
    "that would perfectly answer the question. This document will be used for similarity search.\n\n"
    "Question: {question}\n\nHypothetical document:"
)

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant. Generate {num_queries} different versions of the given question "
    "to retrieve relevant documents from a vector database. "
    "Each query should focus on a different aspect.\n\n"
    "Original question: {question}\n\n"
    "Return one query per line, numbered. No preamble."
)

STEP_BACK_PROMPT = ChatPromptTemplate.from_template(
    "You are an AI assistant. Given a specific question, write a more general, "
    "step-back question that would help answer the specific one.\n\n"
    "Question: {question}\n\nStep-back question:"
)


class QueryTransformer:
    def __init__(self, model: Optional[str] = None):
        model_name = model or settings.groq_chat_model
        self.llm = ChatGroq(model=model_name, temperature=0)
        self.method = settings.query_transformer_method

    def transform(self, question: str) -> list[str]:
        if not settings.query_transformer_enabled:
            return [question]

        if self.method == "hyde":
            return [self._hyde(question)]
        elif self.method == "multi_query":
            return self._multi_query(question)
        elif self.method == "step_back":
            return [question, self._step_back(question)]
        else:
            return [question]

    def _hyde(self, question: str) -> str:
        chain = HYDE_PROMPT | self.llm
        result = chain.invoke({"question": question})
        logger.debug(f"HyDE generated document for: {question}")
        return result.content

    def _multi_query(self, question: str, num_queries: int = 3) -> list[str]:
        chain = MULTI_QUERY_PROMPT | self.llm
        result = chain.invoke({"question": question, "num_queries": num_queries})
        queries = [q.split(". ", 1)[-1] for q in result.content.strip().split("\n") if q.strip()]
        queries.append(question)
        logger.info(f"Generated {len(queries)} query variants")
        return queries

    def _step_back(self, question: str) -> str:
        chain = STEP_BACK_PROMPT | self.llm
        result = chain.invoke({"question": question})
        logger.debug(f"Step-back question: {result.content}")
        return result.content
