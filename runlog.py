"""每次启动把完整运行日志写成 run.json，方便对照报错。"""

from __future__ import annotations

import atexit
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PATH: Path | None = None
_DATA: dict = {}
_ORIG_STDOUT = sys.stdout


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flush() -> None:
    if _PATH is None:
        return
    _PATH.write_text(
        json.dumps(_DATA, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class _Tee:
    encoding = "utf-8"

    def write(self, s: str) -> int:
        _ORIG_STDOUT.write(s)
        if s:
            _DATA["stdout"] = _DATA.get("stdout", "") + s
            _flush()
        return len(s)

    def flush(self) -> None:
        _ORIG_STDOUT.flush()

    def isatty(self) -> bool:
        return False


def init_run(path: Path) -> None:
    global _PATH, _DATA
    _PATH = path
    _DATA = {
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
        "stdout": "",
        "events": [],
        "sync": {
            "skipped": 0,
            "rebuild": 0,
            "deleted_files": 0,
            "files": [],
        },
        "error": None,
        "conversations": [],
        "log_path": str(path.resolve()),
    }
    _flush()
    sys.stdout = _Tee()
    print(f"[*] 本次会话日志: {path.resolve()}", flush=True)


def log_conversation(
    question: str,
    answer: str,
    trace: list | None = None,
    display: str | None = None,
) -> None:
    _DATA.setdefault("conversations", []).append(
        {
            "ts": _now(),
            "question": question,
            "answer": answer,
            "display": display if display is not None else answer,
            "steps": len(trace or []),
            "trace": trace or [],
        }
    )
    _flush()

    def _hook(etype, value, tb) -> None:
        fail(value)
        sys.__excepthook__(etype, value, tb)

    sys.excepthook = _hook
    atexit.register(_finish_ok)


def event(msg: str, **fields) -> None:
    rec = {"ts": _now(), "msg": msg, **fields}
    _DATA.setdefault("events", []).append(rec)
    _flush()


def file_result(**fields) -> None:
    _DATA.setdefault("sync", {}).setdefault("files", []).append(
        {"ts": _now(), **fields}
    )
    _flush()


def set_sync(**fields) -> None:
    _DATA.setdefault("sync", {}).update(fields)
    _flush()


def fail(exc: BaseException) -> None:
    _DATA["status"] = "error"
    _DATA["finished_at"] = _now()
    if sys.exc_info()[1] is exc:
        tb_text = traceback.format_exc()
    else:
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _DATA["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": tb_text,
    }
    _flush()


def _finish_ok() -> None:
    if _DATA.get("status") == "running":
        _DATA["status"] = "ok"
        _DATA["finished_at"] = _now()
        _flush()
    try:
        sys.stdout = _ORIG_STDOUT
    except Exception:
        pass
