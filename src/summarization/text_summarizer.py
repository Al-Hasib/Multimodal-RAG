from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}

"""


class TextSummarizer:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.5):
        model_name = model or settings.groq_chat_model
        llm = ChatGroq(temperature=temperature, model=model_name)
        prompt = ChatPromptTemplate.from_template(SUMMARIZE_PROMPT)
        self.chain = {"element": lambda x: x} | prompt | llm | StrOutputParser()

    def summarize(self, element) -> str:
        return self.chain.invoke(element)

    def summarize_batch(self, elements: list, max_concurrency: Optional[int] = None) -> list[str]:
        concurrency = max_concurrency or settings.summarization_max_concurrency
        logger.info(f"Summarizing {len(elements)} elements with concurrency {concurrency}")
        return self.chain.batch(elements, {"max_concurrency": concurrency})
