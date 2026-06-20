import argparse
import uvicorn
from src.config.settings import settings
from src.pipeline.pipeline import RAGPipeline


def run_api(host: str = "0.0.0.0", port: int = 8000):
    from src.api.app import create_app
    pipeline = RAGPipeline()
    app = create_app(pipeline)
    uvicorn.run(app, host=host, port=port)


def run_ingest(pdf_path: str):
    pipeline = RAGPipeline()
    result = pipeline.ingest(pdf_path)
    print(f"Ingested: {pdf_path}")
    print(f"  Texts: {result['texts']}, Tables: {result['tables']}, Images: {result['images']}")


def run_query(question: str, k: int = 5, session_id: str | None = None):
    pipeline = RAGPipeline()
    result = pipeline.query(question, k=k, session_id=session_id)
    print(f"Answer: {result.answer}")
    if result.context_texts:
        print(f"\nSources ({len(result.context_texts)} text chunks):")
        for t in result.context_texts[:3]:
            print(f"  - {t[:200]}...")
    if result.context_images:
        print(f"\nImages retrieved: {len(result.context_images)}")


def run_ui():
    import sys
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", "8501"])


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("api", help="Run the FastAPI server")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF document")
    ingest_parser.add_argument("pdf_path", type=str, help="Path to PDF file")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    query_parser.add_argument("--session-id", type=str, default=None, help="Session ID for chat history")

    subparsers.add_parser("ui", help="Run the Streamlit web UI")

    args = parser.parse_args()

    if args.command == "api":
        run_api()
    elif args.command == "ingest":
        run_ingest(args.pdf_path)
    elif args.command == "query":
        run_query(args.question, args.k, args.session_id)
    elif args.command == "ui":
        run_ui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
