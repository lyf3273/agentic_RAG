"""检索命中配图后的视觉旁路：SVG/PNG -> qwen3-vl-flash 看图，描述写回上下文。"""

from __future__ import annotations

import atexit
import base64
import hashlib
import json
import os
import re
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
RASTER_DIR = CACHE_DIR / "vl_raster"
CAPTION_CACHE = CACHE_DIR / "vl_captions.json"

VISION_MODEL = os.getenv("TONGYI_VISION_MODEL", "qwen3-vl-flash")
VISION_FALLBACKS = [VISION_MODEL, "qwen3-vl-plus", "qwen-vl-plus"]
MAX_IMAGES_PER_QUERY = int(os.getenv("RAG_VISION_MAX_IMAGES", "3"))
VISION_ENABLED = os.getenv("RAG_VISION", "1").lower() in {"1", "true", "yes", "on"}

_FIGURE_BLOCK = re.compile(
    r"【图：(?P<caption>[^】]+)】"
    r"(?:\n配图文件：(?P<file>[^\n]+))?"
    r"(?:\n图中文字：(?P<labels>[^\n]+))?",
)
_CAPTION_ID = re.compile(r"(图\d+-\d+)", re.I)
_VISION_INTENT = re.compile(
    r"(图\d+-\d+|架构图|示意图|流程图|结构图|配图|示意图|看图|这张图|figure|diagram)",
    re.I,
)
_KW = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")

_browser = None
_playwright = None
_page = None
_client = None
_working_model: str | None = None
_caption_store: dict[str, str] | None = None
_figure_index: dict[str, Path] | None = None
_pw_lock = threading.Lock()
_store_lock = threading.Lock()
_PW_ARGS = (
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-extensions",
    "--hide-scrollbars",
)


def _load_caption_store() -> dict[str, str]:
    global _caption_store
    if _caption_store is None:
        if CAPTION_CACHE.is_file():
            _caption_store = json.loads(CAPTION_CACHE.read_text(encoding="utf-8"))
        else:
            _caption_store = {}
    return _caption_store


def _save_caption_store() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _store_lock:
        CAPTION_CACHE.write_text(
            json.dumps(_load_caption_store(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_figure_index(data_dir: Path | None = None) -> dict[str, Path]:
    root = data_dir or DATA_DIR
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".svg", ".png", ".jpg", ".jpeg", ".webp"}:
            continue
        if path.name.startswith("~$"):
            continue
        index[path.name.lower()] = path
        stem = path.stem.lower()
        index[stem] = path
        if stem.startswith("fig"):
            index["图" + stem[3:]] = path
    return index


def _index() -> dict[str, Path]:
    global _figure_index
    if _figure_index is None:
        _figure_index = build_figure_index()
    return _figure_index


def resolve_figure_path(caption: str, file_hint: str | None = None) -> Path | None:
    index = _index()
    if file_hint:
        name = Path(file_hint.strip()).name.lower()
        if name in index:
            return index[name]
        stem = Path(name).stem.lower()
        if stem in index:
            return index[stem]
    cap = (caption or "").strip()
    token = cap.split()[0] if cap else ""
    for key in (token.lower(), token):
        if key in index:
            return index[key]
    m = _CAPTION_ID.search(cap)
    if m:
        fig_id = m.group(1).lower()
        if fig_id in index:
            return index[fig_id]
        alt = "fig" + fig_id[1:]
        if alt in index:
            return index[alt]
    return None


def shutdown_vision() -> None:
    """Close the Chromium singleton. Playwright leaks ~100MB+ per leaked browser."""
    global _browser, _playwright, _page
    with _pw_lock:
        try:
            if _page is not None:
                _page.close()
        except Exception:
            pass
        try:
            if _browser is not None:
                _browser.close()
        except Exception:
            pass
        try:
            if _playwright is not None:
                _playwright.stop()
        except Exception:
            pass
        _page = None
        _browser = None
        _playwright = None


atexit.register(shutdown_vision)


def _ensure_page():
    global _browser, _playwright, _page
    if _page is not None:
        return _page
    from playwright.sync_api import sync_playwright

    if _playwright is None:
        _playwright = sync_playwright().start()
    if _browser is None:
        _browser = _playwright.chromium.launch(headless=True, args=list(_PW_ARGS))
    _page = _browser.new_page(viewport={"width": 1400, "height": 900})
    return _page


def rasterize(path: Path) -> Path:
    """SVG 转 PNG；位图原样返回。结果缓存在 .cache/vl_raster/。"""
    global _page
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    out = RASTER_DIR / f"{path.stem}-{digest}.png"
    if out.is_file():
        return out
    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    svg = path.read_text(encoding="utf-8")
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        "<body style='margin:0;background:#fff'>" + svg + "</body></html>"
    )
    with _pw_lock:
        page = _ensure_page()
        try:
            page.set_content(html, wait_until="load", timeout=15000)
            loc = page.locator("svg").first
            loc.wait_for(timeout=5000)
            loc.screenshot(path=str(out), type="png")
        except Exception:
            try:
                page.close()
            except Exception:
                pass
            _page = None
            page = _ensure_page()
            page.set_content(html, wait_until="load", timeout=15000)
            loc = page.locator("svg").first
            loc.wait_for(timeout=5000)
            loc.screenshot(path=str(out), type="png")
    return out


def _client_openai():
    global _client
    if _client is None:
        from openai import OpenAI

        import httpx
        _client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=httpx.Client(
                trust_env=False,
                timeout=httpx.Timeout(60.0, connect=15.0),
                follow_redirects=True,
            ),
        )
    return _client


def describe_image(path: Path, caption: str = "", labels: str = "") -> str:
    key = hashlib.sha256(
        f"{VISION_MODEL}|{path.resolve()}|{path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    store = _load_caption_store()
    if key in store:
        return store[key]

    png = rasterize(path)
    b64 = base64.b64encode(png.read_bytes()).decode("ascii")
    prompt = (
        f"这是技术书中的架构/流程图。标题：{caption or path.name}\n"
        f"图中已抽出的文字标签：{labels or '无'}\n"
        "请结合图像结构，用中文说明：模块如何划分、箭头表示什么关系、核心结论是什么。"
        "不超过180字，不要重复罗列标签，不要空话。"
    )
    models = list(dict.fromkeys(VISION_FALLBACKS))
    last_error = None
    global _working_model
    if _working_model:
        models = [_working_model] + [m for m in models if m != _working_model]
    for model in models:
        try:
            resp = _client_openai().chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                extra_body={"enable_thinking": False},
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                continue
            _working_model = model
            store[key] = text
            _save_caption_store()
            return text
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"视觉模型调用失败: {last_error}")


def figure_relevant_to_query(query: str, caption: str = "", labels: str = "") -> bool:
    """问句在谈图，或图标题/标签与问题有实质词重叠，才看图。"""
    q = (query or "").strip()
    if not q:
        return False
    if _VISION_INTENT.search(q):
        return True
    blob = f"{caption} {labels}".lower()
    if not blob.strip():
        return False
    q_kw = {w.lower() for w in _KW.findall(q) if w.lower() not in {"什么", "怎么", "如何", "哪些", "一个", "这个", "那个", "以及", "或者"}}
    if not q_kw:
        return False
    return any(w in blob for w in q_kw if len(w) >= 2)


def enrich_snippets(
    snippets: list[str],
    max_images: int | None = None,
    query: str = "",
) -> list[str]:
    """仅当问题与配图相关时，为【图：…】补视觉描述。"""
    if not VISION_ENABLED or not os.getenv("DASHSCOPE_API_KEY"):
        return snippets
    budget = MAX_IMAGES_PER_QUERY if max_images is None else max_images
    described: dict[str, str] = {}
    debug = os.getenv("RAG_DEBUG", "0").lower() in {"1", "true", "yes", "on"}

    def _inject(text: str) -> str:
        nonlocal budget

        def repl(match: re.Match) -> str:
            nonlocal budget
            original = match.group(0)
            caption = (match.group("caption") or "").strip()
            file_hint = (match.group("file") or "").strip()
            labels = (match.group("labels") or "").strip()
            if not figure_relevant_to_query(query, caption, labels):
                if debug:
                    print(f"[视觉] 跳过（与问题不相关）{file_hint or caption[:40]}")
                return original
            path = resolve_figure_path(caption, file_hint)
            if path is None:
                return original
            cache_key = str(path.resolve())
            if cache_key not in described:
                if budget <= 0:
                    return original
                try:
                    described[cache_key] = describe_image(path, caption, labels)
                    budget -= 1
                    print(f"[视觉] {path.name} -> {described[cache_key][:60]}…")
                except Exception as exc:
                    print(f"[!] 看图失败 {path.name}: {exc}")
                    return original
            return original + f"\n图示解读：{described[cache_key]}"

        return _FIGURE_BLOCK.sub(repl, text)

    return [_inject(s) for s in snippets]


def warmup() -> None:
    n = len({p for p in _index().values()})
    if not VISION_ENABLED:
        print("[*] 视觉旁路已关闭（RAG_VISION=0）")
        return
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[!] 未设置 DASHSCOPE_API_KEY，检索命中配图时跳过视觉模型")
        return
    print(f"[+] 视觉旁路就绪（{VISION_MODEL}，配图 {n} 张，每问最多看 {MAX_IMAGES_PER_QUERY} 张）")
