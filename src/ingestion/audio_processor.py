import os
import io
import tempfile
from pydub import AudioSegment
from src.config.settings import settings
from src.models.schemas import ExtractedDocument, ExtractedElement, DocumentType, FileFormat
import logging

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}


class AudioProcessor:
    def __init__(self):
        self.provider = settings.audio_transcription_provider
        self.model = settings.audio_transcription_model
        self.language = settings.audio_transcription_language
        self.chunk_seconds = settings.audio_chunk_seconds
        self.max_file_size_mb = settings.audio_max_file_size_mb

    async def extract(self, file_path: str) -> ExtractedDocument:
        logger.info(f"Transcribing audio: {file_path}")
        output_path = self._maybe_convert(file_path)

        segments = self._transcribe(output_path)
        doc = ExtractedDocument(
            filename=os.path.basename(file_path),
            file_format=FileFormat.AUDIO,
        )

        for seg in segments:
            doc.texts.append(ExtractedElement(
                type=DocumentType.TEXT,
                content=seg["text"],
                metadata={"start": seg["start"], "end": seg["end"]},
            ))

        if output_path != file_path:
            os.unlink(output_path)

        logger.info(f"Transcribed {len(segments)} segments from {file_path}")
        return doc

    async def extract_from_bytes(self, filename: str, content: bytes) -> ExtractedDocument:
        suffix = os.path.splitext(filename)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return await self.extract(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _maybe_convert(self, file_path: str) -> str:
        ext = os.path.splitext(file_path.lower())[1]
        if ext == ".mp3":
            return file_path
        out_path = file_path + "_converted.mp3"
        logger.info(f"Converting {ext} to MP3: {out_path}")
        audio = AudioSegment.from_file(file_path)
        audio.export(out_path, format="mp3")
        return out_path

    def _transcribe(self, audio_path: str) -> list[dict]:
        if self.provider == "openai":
            return self._transcribe_openai(audio_path)
        elif self.provider == "local":
            return self._transcribe_local(audio_path)
        else:
            raise ValueError(f"Unknown audio_transcription_provider: {self.provider}")

    def _transcribe_openai(self, audio_path: str) -> list[dict]:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        if file_size_mb > self.max_file_size_mb:
            logger.warning(f"Audio file {file_size_mb:.1f}MB exceeds limit, chunking")
            return self._transcribe_chunked(audio_path)

        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="verbose_json",
                language=self.language,
            )

        segments = []
        for seg in getattr(transcript, "segments", []):
            segments.append({
                "text": seg.get("text", "").strip(),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })

        if not segments:
            full_text = getattr(transcript, "text", transcript) or ""
            segments.append({"text": full_text.strip(), "start": 0, "end": 0})

        return segments

    def _transcribe_chunked(self, audio_path: str) -> list[dict]:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        audio = AudioSegment.from_mp3(audio_path)
        chunk_ms = self.chunk_seconds * 1000
        segments = []

        for i, start_ms in enumerate(range(0, len(audio), chunk_ms)):
            chunk = audio[start_ms:start_ms + chunk_ms]
            chunk_path = f"/tmp/audio_chunk_{i}.mp3"
            chunk.export(chunk_path, format="mp3")
            try:
                with open(chunk_path, "rb") as f:
                    result = client.audio.transcriptions.create(
                        model=self.model,
                        file=f,
                        response_format="verbose_json",
                        language=self.language,
                    )
                for seg in getattr(result, "segments", []):
                    segments.append({
                        "text": seg.get("text", "").strip(),
                        "start": seg.get("start", 0) + start_ms / 1000,
                        "end": seg.get("end", 0) + start_ms / 1000,
                    })
            finally:
                os.unlink(chunk_path)

        return segments

    def _transcribe_local(self, audio_path: str) -> list[dict]:
        try:
            import whisper
        except ImportError:
            raise ImportError("Install openai-whisper for local transcription: pip install openai-whisper")

        model = whisper.load_model(self.model)
        result = model.transcribe(audio_path, language=self.language)

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "text": seg.get("text", "").strip(),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })

        if not segments:
            segments.append({"text": result.get("text", "").strip(), "start": 0, "end": 0})

        return segments
