from typing import Optional
from src.config.settings import settings
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

SUMMARIZE_PROMPT = """You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}"""

IMAGE_PROMPT = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""

HYDE_PROMPT = """You are an AI assistant. Given a question, write a concise hypothetical document
that would perfectly answer the question. This document will be used for similarity search.

Question: {question}

Hypothetical document:"""

MULTI_QUERY_PROMPT = """You are an AI assistant. Generate {num_queries} different versions of the given question
to retrieve relevant documents from a vector database.
Each query should focus on a different aspect.

Original question: {question}

Return one query per line, numbered. No preamble."""

STEP_BACK_PROMPT = """You are an AI assistant. Given a specific question, write a more general,
step-back question that would help answer the specific one.

Question: {question}

Step-back question:"""


class PromptManager:
    def __init__(self):
        self.langfuse_enabled = settings.prompt_use_langfuse and settings.langfuse_enabled
        self._langfuse = None
        if self.langfuse_enabled:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info("PromptManager connected to LangFuse")
            except Exception as e:
                logger.warning(f"LangFuse unavailable, using local prompts: {e}")
                self.langfuse_enabled = False

    def get_prompt(self, name: str, version: Optional[int] = None) -> str:
        if self.langfuse_enabled and self._langfuse:
            try:
                v = version or settings.prompt_default_version
                prompt = self._langfuse.get_prompt(name=name, version=v)
                compiled = prompt.compile()
                logger.debug(f"Fetched prompt '{name}' v{prompt.version} from LangFuse")
                return compiled
            except Exception as e:
                logger.warning(f"Failed to fetch prompt '{name}' from LangFuse: {e}")

        return self._local_prompt(name)

    def _local_prompt(self, name: str) -> str:
        registry = {
            "system": SYSTEM_PROMPT,
            "response": RESPONSE_TEMPLATE,
            "summarize": SUMMARIZE_PROMPT,
            "image_describe": IMAGE_PROMPT,
            "hyde": HYDE_PROMPT,
            "multi_query": MULTI_QUERY_PROMPT,
            "step_back": STEP_BACK_PROMPT,
        }
        if name not in registry:
            raise KeyError(f"Unknown prompt name: {name}")
        return registry[name]


prompt_manager = PromptManager()
