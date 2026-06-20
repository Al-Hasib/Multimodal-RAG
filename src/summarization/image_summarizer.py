from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import settings
from typing import Optional
import logging

logger = logging.getLogger(__name__)

IMAGE_PROMPT = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""

IMAGE_MESSAGES = [
    (
        "user",
        [
            {"type": "text", "text": IMAGE_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]


class ImageSummarizer:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.5):
        model_name = model or settings.openai_chat_model
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages(IMAGE_MESSAGES)
        self.chain = prompt | llm | StrOutputParser()

    def summarize(self, image_base64: str) -> str:
        return self.chain.invoke(image_base64)

    def summarize_batch(self, images: list[str], max_concurrency: Optional[int] = None) -> list[str]:
        concurrency = max_concurrency or settings.summarization_max_concurrency
        logger.info(f"Summarizing {len(images)} images with concurrency {concurrency}")
        return self.chain.batch(images, {"max_concurrency": concurrency})
