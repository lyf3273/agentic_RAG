"""Evidence Studio — FastAPI UI + SSE trace for the Agentic RAG."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import runlog
from main import (
    engine_info,
    ensure_engine,
    extract_chunk_ids,
    format_user_facing,
    handle_user_question,
    knowledge_packs,
    storage_label,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
INDEX_HTML = WEB_DIR / "index.html"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Engine is already started by `python main.py web`. This only covers `uvicorn app:app`.
    ensure_engine()
    yield


app = FastAPI(title="Agentic RAG Evidence Studio", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


@app.get("/")
def index() -> FileResponse:
    if not INDEX_HTML.is_file():
        raise HTTPException(500, "web/index.html missing")
    return FileResponse(INDEX_HTML)


@app.get("/api/health")
def health() -> dict:
    info = engine_info()
    info["ok"] = True
    return info


@app.get("/api/health/stream")
async def health_stream() -> StreamingResponse:
    async def gen():
        last = None
        while True:
            info = engine_info()
            info["ok"] = True
            stamp = (info.get("storage"), info.get("status"), storage_label())
            if stamp != last:
                last = stamp
                yield f"data: {json.dumps(info, ensure_ascii=False)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/packs")
def packs() -> dict:
    return {"packs": knowledge_packs()}


def _log_web_turn(question: str, answer: str, trace: list) -> None:
    runlog.log_conversation(question, answer, trace, display=format_user_facing(answer))


@app.post("/api/ask")
def ask(body: AskBody) -> dict:
    answer, trace = handle_user_question(body.question)
    _log_web_turn(body.question, answer, trace)
    return {
        "answer": format_user_facing(answer),
        "raw": answer,
        "citations": sorted(extract_chunk_ids(answer)),
        "trace": trace,
        "route": (trace[0] or {}).get("route") if trace else None,
        "backend": engine_info(),
    }


@app.get("/api/ask/stream")
async def ask_stream(q: str) -> StreamingResponse:
    question = (q or "").strip()
    if not question:
        raise HTTPException(400, "q is required")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def on_event(payload: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    def worker() -> None:
        try:
            answer, trace = handle_user_question(question, on_event=on_event)
            _log_web_turn(question, answer, trace)
            on_event({
                "type": "done",
                "answer": format_user_facing(answer),
                "raw": answer,
                "citations": sorted(extract_chunk_ids(answer)),
                "trace": trace,
                "route": (trace[0] or {}).get("route") if trace else None,
            })
        except Exception as exc:
            on_event({"type": "error", "message": str(exc)})

    threading.Thread(target=worker, daemon=True, name="ask-stream").start()

    async def gen():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in {"done", "error"}:
                break

    return StreamingResponse(gen(), media_type="text/event-stream")
