from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from src.config.settings import settings
from src.retrieval.retriever import MultiModalRetriever
from src.models.schemas import RetrievalResult, QueryResponse
from typing import Optional
import logging

logger = logging.getLogger(__name__)

RESPONSE_TEMPLATE = """
Answer the question based only on the following context, which can include text, tables, and the below image.
Context: {context_text}
Question: {user_question}
"""


class RAGChain:
    def __init__(self, retriever: MultiModalRetriever, model: Optional[str] = None):
        self.retriever = retriever
        model_name = model or settings.openai_chat_model
        self.llm = ChatOpenAI(model=model_name)
        self.chain = self._build_chain()
        self.chain_with_sources = self._build_chain_with_sources()

    def _parse_docs(self, docs):
        return self.retriever._parse_docs(docs)

    @staticmethod
    def _build_prompt(kwargs: dict) -> ChatPromptTemplate:
        docs_by_type: RetrievalResult = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if docs_by_type.texts:
            for text_element in docs_by_type.texts:
                if hasattr(text_element, "text"):
                    context_text += text_element.text
                else:
                    context_text += str(text_element)

        prompt_text = RESPONSE_TEMPLATE.format(
            context_text=context_text,
            user_question=user_question,
        )

        prompt_content = [{"type": "text", "text": prompt_text}]

        if docs_by_type.images:
            for image in docs_by_type.images:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def _build_chain(self):
        return (
            {
                "context": RunnableLambda(lambda q: self.retriever.retrieve(q)) | RunnableLambda(self._parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._build_prompt)
            | self.llm
            | StrOutputParser()
        )

    def _build_chain_with_sources(self):
        return {
            "context": RunnableLambda(lambda q: self.retriever.retrieve(q)) | RunnableLambda(self._parse_docs),
            "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self._build_prompt)
                | self.llm
                | StrOutputParser()
            )
        )

    def invoke(self, question: str, k: int = 5) -> str:
        logger.info(f"Generating answer for: {question}")
        return self.chain.invoke(question)

    def invoke_with_sources(self, question: str, k: int = 5) -> QueryResponse:
        logger.info(f"Generating answer with sources for: {question}")
        result = self.chain_with_sources.invoke(question)

        context_texts = []
        for t in result["context"].texts:
            if hasattr(t, "text"):
                context_texts.append(t.text)
            else:
                context_texts.append(str(t))

        return QueryResponse(
            answer=result["response"],
            context_texts=context_texts,
            context_images=result["context"].images,
        )
