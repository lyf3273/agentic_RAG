from dotenv import load_dotenv
load_dotenv()

import argparse
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

# HuggingFace 镜像，避免直连超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 直连 DashScope，避开系统代理导致的 SSL EOF / Connection error
_NO_PROXY = "dashscope.aliyuncs.com,localhost,127.0.0.1"
os.environ["NO_PROXY"] = ",".join(
    x for x in (os.environ.get("NO_PROXY", ""), _NO_PROXY) if x
)

import chromadb
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from llama_index.core import Document, Settings, VectorStoreIndex
from markdown_chunker import MarkdownParentChildSplitter, expand_retrieved_node
from vision_sidecar import enrich_snippets, warmup as warmup_vision
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from hybrid_retriever import build_hybrid_retriever, rrf_fuse
from query_plan import QueryPlan, looks_like_chitchat, plan_query
from index_sync import (
    CHUNKER_VERSION,
    EMBED_DIM,
    HnswCorruptError,
    hnsw_dir_healthy,
    probe_embedding_dim,
    quarantine_chroma_dir,
    release_chroma_locks,
    stamp_nodes,
    file_fingerprint,
    sync_chroma_index,
    sync_faiss_index,
)
from context_compact import apply_compact_delta, estimate_tokens
from embed_norm import install_l2_norm
import runlog

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_LOG: Path | None = None
DATA_DIR = BASE_DIR / "data"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-large-zh-v1.5")
RERANK_MODEL = os.getenv("OLLAMA_RERANK_MODEL", "bge-reranker-v2-m3")
CHAT_MODEL = os.getenv("TONGYI_CHAT_MODEL", "qwen3.6-flash")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "knowledge_base")
# faiss：默认本地嵌入式，克隆即可跑
# remote：优先云端 Chroma，心跳失败再 FAISS
# chroma-local：本机 PersistentClient
CHROMA_MODE = os.getenv("CHROMA_MODE", "faiss").strip().lower()
CHROMA_LOCAL_DIR = Path(os.getenv("CHROMA_LOCAL_DIR", str(BASE_DIR / "chroma_db")))
RAG_DEBUG = os.getenv("RAG_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
CHROMA_TIMEOUT = int(os.getenv("CHROMA_TIMEOUT", "30"))
CHROMA_RETRY = int(os.getenv("CHROMA_RETRY", "5"))
CHROMA_RETRY_INTERVAL = float(os.getenv("CHROMA_RETRY_INTERVAL", "3"))
CHROMA_PROBE_INTERVAL = float(os.getenv("CHROMA_PROBE_INTERVAL", "5"))
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
AUDIT_LOG_DIR = BASE_DIR / "react_audit_logs"
SUPPORTED_SUFFIXES = {".doc", ".docx", ".txt", ".md", ".pdf"}
_SKIP_KB_NAMES = {"SOURCE.MD", "LICENSE.MD", "README.MD", "NOTICE.MD"}
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "cloud").strip().lower()
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "7860"))

# ReAct 配置
MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "5"))  # 最大循环步数
REACT_LOG_DIR = BASE_DIR / "react_traces"  # ReAct 轨迹日志
KEEP_RECENT_TURNS = int(os.getenv("KEEP_RECENT_TURNS", "3"))  # 跨轮保留完整对话轮数

# 跨轮对话记忆（模块级，整个进程生命周期内持久化）
_conversation_history: list[dict] = []  # [{"question": str, "answer": str}]

# 当前使用的存储后端，供外部查询
STORAGE_BACKEND = "unknown"
_STORAGE_STATUS = ""
_WATCH_STARTED = False
_QUERY_DEPTH = 0
_FAILOVER_STANDBY = False

local_index = None
retriever = None
vector_retriever = None
bm25_retriever = None
reranker = None
reranker_type = RERANK_BACKEND if RERANK_BACKEND in {"cloud", "local"} else "cloud"
llm = None
_raw_client = None
_ENGINE_LOCK = threading.Lock()
_ENGINE_READY = False
_MODELS_READY = False


def start_session_log() -> Path:
    """Start JSON session log on first real run, not at import."""
    global _SESSION_LOG
    if _SESSION_LOG is not None:
        return _SESSION_LOG
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _SESSION_LOG = LOG_DIR / datetime.now().strftime("session-%Y%m%d-%H%M%S.json")
    runlog.init_run(_SESSION_LOG)
    (LOG_DIR / "latest.txt").write_text(str(_SESSION_LOG.resolve()), encoding="utf-8")
    return _SESSION_LOG


def _emit(on_event, payload: dict) -> None:
    if on_event is None:
        return
    try:
        on_event(payload)
    except Exception:
        pass


def chroma_heartbeat(timeout: float = 2.0) -> bool:
    """One heartbeat. Host and port come from the environment, never hardcoded."""
    import urllib.request
    url = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/heartbeat"
    try:
        req = urllib.request.urlopen(url, timeout=timeout)
        return req.status in (200, 204)
    except Exception:
        return False


def _set_storage_status(text: str) -> None:
    global _STORAGE_STATUS
    _STORAGE_STATUS = text


def check_chroma_available() -> bool:
    """Startup probe: up to CHROMA_RETRY heartbeats, CHROMA_RETRY_INTERVAL apart."""
    import time
    for attempt in range(1, CHROMA_RETRY + 1):
        if chroma_heartbeat(2.0):
            print(f"[+] 云端 Chroma 向量数据库已连接", flush=True)
            return True
        if attempt >= CHROMA_RETRY:
            break
        msg = f"RAG connection failed, retrying {attempt}/{CHROMA_RETRY}"
        _set_storage_status(msg)
        print(f"[*] {msg}", flush=True)
        time.sleep(CHROMA_RETRY_INTERVAL)
    return False


def normalize_text(text: str) -> str:
    """Remove Word control characters while preserving readable paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0b", "\n")
    text = "".join(
        char
        if char in "\n\t" or (char.isprintable() and not 0xD800 <= ord(char) <= 0xDFFF)
        else " "
        for char in text
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_console_input(text: str) -> str:
    """Repair UTF-8 bytes decoded as GBK in Windows PowerShell pipes."""
    try:
        raw = text.encode("gb18030", errors="surrogateescape")
        repaired = raw.decode("utf-8")
        return repaired if repaired != text else text
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def clean_ole_fallback_text(text: str) -> str:
    """Keep useful Chinese/ASCII text and deduplicate artifacts from the OLE stream."""
    punctuation = "，。；：！？、（）《》【】""''—"
    filtered = "".join(
        char
        if char in "\n\r\t" + punctuation
        or 0x4E00 <= ord(char) <= 0x9FA5
        or 0x20 <= ord(char) <= 0x7E
        else " "
        for char in text
    )
    normalized = normalize_text(filtered)
    parts = []
    for raw_part in re.split(r"[\n]+", normalized):
        part = raw_part.strip()
        first_phrase = re.search(r"[\u4e00-\u9fa5]{4,}", part)
        if not first_phrase:
            continue
        parts.append(part[first_phrase.start() :])
    unique_parts = list(dict.fromkeys(parts))
    return "\n".join(unique_parts)


def read_legacy_doc(path: Path) -> str:
    """Read a binary .doc through Microsoft Word, with an OLE fallback."""
    lock_file = path.with_name(f"~${path.name}")
    if lock_file.exists():
        print(f"[!] {path.name} 正被 Word 打开，跳过 COM，使用 OLE 读取已保存内容")
        return read_legacy_doc_ole(path)

    try:
        import win32com.client

        word = None
        document = None
        try:
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0
            document = word.Documents.Open(
                str(path.resolve()), ReadOnly=True, AddToRecentFiles=False
            )
            return normalize_text(document.Content.Text)
        finally:
            if document is not None:
                document.Close(False)
            if word is not None:
                word.Quit()
    except Exception as word_error:
        print(f"[!] Word 读取失败，已使用兼容模式解析 {path.name}: {word_error}")
        return read_legacy_doc_ole(path)


def read_legacy_doc_ole(path: Path) -> str:
    """Read the saved WordDocument stream without launching Word."""
    import olefile

    if not olefile.isOleFile(str(path)):
        raise ValueError(f"不是有效的 OLE .doc 文件: {path.name}")
    with olefile.OleFileIO(str(path)) as ole:
        if not ole.exists("WordDocument"):
            raise ValueError(f".doc 中缺少 WordDocument 数据流: {path.name}")
        raw = ole.openstream("WordDocument").read()
    text = clean_ole_fallback_text(raw.decode("utf-16le", errors="ignore"))
    if not text:
        raise ValueError(f"无法从 {path.name} 提取文本")
    return text


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".doc":
        return read_legacy_doc(path)
    if suffix == ".docx":
        from docx import Document as WordDocument
        doc = WordDocument(str(path))
        return normalize_text("\n".join(p.text for p in doc.paragraphs))
    if suffix == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_text("\n".join(pages))
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return normalize_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 {path.name} 的文本编码")


def knowledge_files() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith("~$")
        and path.name.upper() not in _SKIP_KB_NAMES
    )


def index_signature(files: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{EMBED_MODEL}|{CHUNKER_VERSION}|dim{EMBED_DIM}".encode())
    for path in files:
        digest.update(path.relative_to(DATA_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_or_build_index() -> VectorStoreIndex:
    global STORAGE_BACKEND, CHROMA_LOCAL_DIR
    files = knowledge_files()
    if not files:
        raise RuntimeError(f"知识库为空，请把文档文件放入 {DATA_DIR}")
    signature = index_signature(files)

    if CHROMA_MODE in {"faiss"}:
        STORAGE_BACKEND = "faiss"
        return _load_or_build_faiss(files, signature)

    chroma_client = None
    persist_label = "local"
    insert_batch = 64
    if CHROMA_MODE in {"remote", "cloud", "http"}:
        print("[*] 连接云端 Chroma 向量数据库...", flush=True)
        if check_chroma_available():
            print("[+] 远程 ChromaDB 响应正常")
            STORAGE_BACKEND = "chroma-remote"
            persist_label = "remote"
            insert_batch = 8
            chroma_client = _open_remote_chroma()
        else:
            print("[!] 云端 Chroma 向量数据库连接失败，降级本地 FAISS 向量数据库", flush=True)
            STORAGE_BACKEND = "faiss"
            return _load_or_build_faiss(files, signature)

    if chroma_client is None:
        if CHROMA_LOCAL_DIR.exists() and not hnsw_dir_healthy(CHROMA_LOCAL_DIR):
            print(
                "[!] 本地 Chroma 向量数据库不可用，降级本地 FAISS 向量数据库",
                flush=True,
            )
            STORAGE_BACKEND = "faiss"
            return _load_or_build_faiss(files, signature)
        CHROMA_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[*] 使用本地 Chroma：{CHROMA_LOCAL_DIR}", flush=True)
        STORAGE_BACKEND = "chroma-local"
        persist_label = "local"
        insert_batch = 64
        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_LOCAL_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )

    print("[*] 开始同步 collection…", flush=True)

    def _sync(client):
        return sync_chroma_index(
            files=files,
            data_dir=DATA_DIR,
            chroma_client=client,
            collection_name=CHROMA_COLLECTION,
            embed_model_name=EMBED_MODEL,
            read_document=read_document,
            splitter=_make_splitter(),
            persist_label=persist_label,
            insert_batch_size=insert_batch,
        )

    try:
        index = _sync(chroma_client)
    except HnswCorruptError as exc:
        print(f"[!] 本地 Chroma 向量数据库不可用: {exc}", flush=True)
        print("[!] 降级本地 FAISS 向量数据库", flush=True)
        STORAGE_BACKEND = "faiss"
        return _load_or_build_faiss(files, signature)
    except Exception as exc:
        print(f"[!] Chroma 同步失败: {exc}", flush=True)
        print("[!] 降级本地 FAISS 向量数据库", flush=True)
        STORAGE_BACKEND = "faiss"
        return _load_or_build_faiss(files, signature)
    where = "本地 Chroma" if STORAGE_BACKEND == "chroma-local" else "云端 Chroma"
    print(f"[+] 向量索引已同步（{where}，{EMBED_DIM} 维）", flush=True)
    return index


def _load_documents(files: list[Path]) -> list[Document]:
    """读取文件列表，返回 LlamaIndex Document 列表。"""
    documents = []
    print("[*] 读取文件中...")
    for path in files:
        try:
            text = read_document(path)
        except Exception as e:
            print(f"    [!] 跳过 {path.name}: {e}")
            continue
        if text:
            rel = path.relative_to(DATA_DIR).as_posix()
            topic = rel.split("/")[0] if "/" in rel else "default"
            documents.append(Document(
                text=text,
                metadata={
                    "file_name": path.name,
                    "source_path": rel,
                    "topic": topic,
                },
                excluded_llm_metadata_keys=["file_name", "source_path", "topic", "parent_text", "parent_id"],
                excluded_embed_metadata_keys=["file_name", "source_path", "topic", "parent_text", "parent_id", "header_path"],
            ))
    if not documents:
        raise RuntimeError("知识库文件均未提取到有效文字")
    print(f"[*] 已读取 {len(documents)} 个文件")
    return documents


def _load_or_build_faiss(files: list[Path], signature: str) -> VectorStoreIndex:
    """FAISS 增量同步：hash 变了才重算该文档的 embedding。"""
    return sync_faiss_index(
        files=files,
        data_dir=DATA_DIR,
        persist_dir=FAISS_INDEX_DIR,
        read_document=read_document,
        splitter=_make_splitter(),
    )


def _make_splitter() -> MarkdownParentChildSplitter:
    return MarkdownParentChildSplitter(
        child_max_chars=int(os.getenv("CHILD_MAX_CHARS", "1800")),
        parent_max_chars=int(os.getenv("PARENT_MAX_CHARS", "3500")),
    )


EMBED_BACKEND = os.getenv("EMBED_BACKEND", "dashscope").strip().lower()
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))

import httpx as _httpx
_DASHSCOPE_HTTP = _httpx.Client(
    trust_env=False,
    timeout=_httpx.Timeout(60.0, connect=15.0),
    follow_redirects=True,
    limits=_httpx.Limits(max_keepalive_connections=20, max_connections=40, keepalive_expiry=30.0),
)


def configure_models() -> None:
    """Bind embed + chat clients. Cheap, idempotent, no index I/O."""
    global EMBED_BACKEND, llm, _raw_client, _MODELS_READY
    if _MODELS_READY:
        return
    backend = EMBED_BACKEND
    if backend == "dashscope" and not DASHSCOPE_API_KEY:
        print("[!] 未检测到 DASHSCOPE_API_KEY，自动降级为本地 Ollama")
        backend = "ollama"
        EMBED_BACKEND = backend
    if backend == "dashscope":
        print("[*] 初始化 LlamaIndex（云端向量模型: text-embedding-v3, 1024维）...")
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(
            model_name="text-embedding-v3",
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=DASHSCOPE_API_KEY,
            dimensions=1024,
            # DashScope text-embedding-v3 rejects batches larger than 10.
            embed_batch_size=10,
        )
    else:
        print(f"[*] 初始化 LlamaIndex（本地向量模型: {EMBED_MODEL}）...")
        Settings.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_URL,
            request_timeout=120.0,
        )
    Settings.embed_model = install_l2_norm(Settings.embed_model)

    from llama_index.llms.openai_like import OpenAILike as LlamaOpenAI
    try:
        Settings.llm = LlamaOpenAI(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            context_window=131072,
            is_chat_model=True,
            http_client=_DASHSCOPE_HTTP,
        )
    except TypeError:
        Settings.llm = LlamaOpenAI(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            context_window=131072,
            is_chat_model=True,
        )

    from openai import OpenAI as _OpenAI
    _raw_client = _OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=_DASHSCOPE_HTTP,
    )
    try:
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            model_kwargs={"extra_body": {"enable_thinking": False}},
            http_client=_DASHSCOPE_HTTP,
        )
    except TypeError:
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            model_kwargs={"extra_body": {"enable_thinking": False}},
        )
    _MODELS_READY = True


def _init_reranker():
    global reranker, reranker_type
    print("[*] 初始化 Reranker...")
    try:
        if DASHSCOPE_API_KEY and reranker_type == "cloud":
            def dashscope_rerank(query: str, documents: list[str]) -> list[float]:
                response = _DASHSCOPE_HTTP.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
                    headers={
                        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "qwen3-rerank",
                        "input": {"query": query, "documents": documents},
                        "parameters": {"top_n": len(documents)},
                    },
                    timeout=20.0,
                )
                if response.status_code != 200:
                    raise Exception(f"DashScope Rerank API 错误: {response.text}")
                result = response.json()
                results = result.get("output", {}).get("results", [])
                scores = [0.0] * len(documents)
                for item in results:
                    scores[item["index"]] = item["relevance_score"]
                return scores

            reranker = dashscope_rerank
            print("[+] Reranker 就绪（云端 qwen3-rerank API）")
            return
        from sentence_transformers import CrossEncoder
        reranker_model = CrossEncoder(
            f"BAAI/{RERANK_MODEL}",
            max_length=512,
            device="cpu",
        )
        reranker = lambda query, docs: reranker_model.predict([[query, doc] for doc in docs]).tolist()
        reranker_type = "local"
        print(f"[+] Reranker 就绪（本地 {RERANK_MODEL}）")
    except Exception as e:
        print(f"[!] Reranker 初始化失败: {e}")
        print("[!] 将使用向量检索分数，不进行重排序")
        reranker = None


def storage_label() -> str:
    if _STORAGE_STATUS:
        return _STORAGE_STATUS
    if STORAGE_BACKEND == "chroma-remote":
        return "云端 Chroma 向量数据库"
    if STORAGE_BACKEND == "chroma-local":
        return "本地 Chroma 向量数据库"
    return "本地 FAISS 向量数据库"


def _bind_retriever(index) -> None:
    global local_index, retriever, vector_retriever, bm25_retriever
    local_index = index
    retriever, vector_retriever, bm25_retriever = build_hybrid_retriever(
        local_index,
        vector_top_k=VECTOR_TOP_K,
        bm25_top_k=BM25_TOP_K,
        fusion_top_k=FUSION_TOP_K,
        rrf_k=RRF_K,
    )
    print(
        f"[+] 检索器就绪（BM25 top-{BM25_TOP_K} + 向量 top-{VECTOR_TOP_K} "
        f"+ RRF 融合 top-{FUSION_TOP_K} + Reranker 精排 top-{RERANK_TOP_N}）"
    )


def _open_remote_chroma():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        import httpx
        from chromadb.api import ServerAPI
        system = getattr(client, "_system", None)
        if system is not None:
            api = system.instance(ServerAPI)
            if hasattr(api, "_session"):
                api._session.timeout = httpx.Timeout(
                    connect=10.0, read=60.0, write=60.0, pool=10.0
                )
    except Exception as exc:
        print(f"[!] 未能设置 Chroma HTTP 超时: {exc}", flush=True)
    return client


def _load_remote_index():
    files = knowledge_files()
    client = _open_remote_chroma()
    return sync_chroma_index(
        files=files,
        data_dir=DATA_DIR,
        chroma_client=client,
        collection_name=CHROMA_COLLECTION,
        embed_model_name=EMBED_MODEL,
        read_document=read_document,
        splitter=_make_splitter(),
        persist_label="remote",
        insert_batch_size=8,
    )


def failover_to_faiss(reason: str = "") -> None:
    """Keep answering on local FAISS. The probe keeps trying to promote Chroma later."""
    global STORAGE_BACKEND, _FAILOVER_STANDBY
    if STORAGE_BACKEND == "faiss" and _FAILOVER_STANDBY:
        _set_storage_status("")
        return
    with _ENGINE_LOCK:
        if STORAGE_BACKEND == "faiss" and _FAILOVER_STANDBY:
            _set_storage_status("")
            return
        if reason:
            print(f"[!] {reason}", flush=True)
        print("[!] 降级本地 FAISS 向量数据库", flush=True)
        _FAILOVER_STANDBY = CHROMA_MODE in {"remote", "cloud", "http"}
        STORAGE_BACKEND = "faiss"
        _set_storage_status("")
        files = knowledge_files()
        index = _load_or_build_faiss(files, index_signature(files))
        _bind_retriever(index)
        print(f"[+] 使用{storage_label()}", flush=True)


def promote_to_remote() -> None:
    """Background reconnect. In-flight questions stay on FAISS until this swap."""
    global STORAGE_BACKEND, _FAILOVER_STANDBY
    import time
    _set_storage_status("云端 Chroma 探活成功，后台连接中")
    print("[*] 云端 Chroma 探活成功，后台连接中，当前仍使用本地 FAISS", flush=True)
    try:
        index = _load_remote_index()
    except Exception as exc:
        _set_storage_status("")
        print(f"[!] 云端 Chroma 后台连接失败，继续使用本地 FAISS：{exc}", flush=True)
        return
    while True:
        with _ENGINE_LOCK:
            if _QUERY_DEPTH == 0 and STORAGE_BACKEND == "faiss" and _FAILOVER_STANDBY:
                _bind_retriever(index)
                STORAGE_BACKEND = "chroma-remote"
                _FAILOVER_STANDBY = False
                _set_storage_status("")
                print("[+] 已切换云端 Chroma 向量数据库", flush=True)
                return
            if not _FAILOVER_STANDBY:
                _set_storage_status("")
                return
        time.sleep(0.2)


def _watch_remote_chroma() -> None:
    """Probe every CHROMA_PROBE_INTERVAL seconds. Fail over, then promote back in the background."""
    import time
    while True:
        time.sleep(CHROMA_PROBE_INTERVAL)
        if CHROMA_MODE not in {"remote", "cloud", "http"}:
            return
        if STORAGE_BACKEND == "chroma-remote":
            if chroma_heartbeat(2.0):
                if _STORAGE_STATUS:
                    _set_storage_status("")
                continue
            recovered = False
            for attempt in range(1, CHROMA_RETRY + 1):
                msg = f"RAG connection failed, retrying {attempt}/{CHROMA_RETRY}"
                _set_storage_status(msg)
                print(f"[*] {msg}", flush=True)
                time.sleep(CHROMA_RETRY_INTERVAL)
                if chroma_heartbeat(2.0):
                    recovered = True
                    break
            if recovered:
                _set_storage_status("")
                print(f"[+] 云端 Chroma 向量数据库保持连接", flush=True)
                continue
            failover_to_faiss("云端 Chroma 向量数据库连接失败")
            continue
        if STORAGE_BACKEND == "faiss" and _FAILOVER_STANDBY:
            if not chroma_heartbeat(2.0):
                continue
            promote_to_remote()


def start_storage_watch() -> None:
    global _WATCH_STARTED
    if _WATCH_STARTED or CHROMA_MODE not in {"remote", "cloud", "http"}:
        return
    _WATCH_STARTED = True
    threading.Thread(target=_watch_remote_chroma, name="chroma-watch", daemon=True).start()


def ensure_engine() -> None:
    """Load vector index + hybrid retriever on first query, not at import."""
    global local_index, retriever, vector_retriever, bm25_retriever, _ENGINE_READY
    with _ENGINE_LOCK:
        if _ENGINE_READY:
            return
        configure_models()
        try:
            local_index = load_or_build_index()
        except Exception as exc:
            runlog.fail(exc)
            raise
        _init_reranker()
        _bind_retriever(local_index)
        warmup_vision()
        print(f"[+] 使用{storage_label()}", flush=True)
        _ENGINE_READY = True
        start_storage_watch()


@tool
def multiply(a: int, b: int) -> int:
    """当需要计算两个整数相乘时使用。"""
    return a * b


@tool
def read_local_file(file_name: str) -> str:
    """当用户明确要求读取当前目录下某个文本文件时使用。"""
    path = BASE_DIR / Path(file_name).name
    if not path.is_file():
        return f"错误：未找到文件 {path.name}"
    return path.read_text(encoding="utf-8")


def _write_audit_log(entry: dict) -> None:
    """将检索审计记录追加写入按日期分割的 JSONL 文件。"""
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = AUDIT_LOG_DIR / f"retrieval_audit_{today}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _llm_complete(system: str, user: str) -> str:
    configure_models()
    resp = _raw_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content or ""


def retrieve_fused_nodes(queries: list[str], hyde_text: str = "") -> list:
    """多问法混合检索后 RRF。HyDE 文本只走向量路。"""
    lists = []
    seen_q = set()
    for q in queries:
        q = (q or "").strip()
        if not q or q in seen_q:
            continue
        seen_q.add(q)
        lists.append(retriever.retrieve(q))
    if (hyde_text or "").strip():
        lists.append(vector_retriever.retrieve(hyde_text.strip()))
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0]
    return rrf_fuse(lists, top_n=FUSION_TOP_K, rrf_k=RRF_K)


def retrieve_with_plan(plan: QueryPlan) -> list:
    queries = [plan.search_query]
    if plan.original and plan.original != plan.search_query:
        queries.append(plan.original)
    if plan.step_back:
        queries.append(plan.step_back)
    if "multi_query" in plan.strategies:
        queries.extend(plan.multi_queries)
    hyde = plan.hyde if "hyde" in plan.strategies else ""
    nodes = retrieve_fused_nodes(queries, hyde_text=hyde)
    if not nodes and plan.step_back:
        print(f"[Step-back] 原问未命中，改用后退问题：{plan.step_back}", flush=True)
        nodes = retrieve_fused_nodes([plan.step_back])
    return nodes


def _rerank_nodes(query: str, nodes: list) -> list[dict]:
    if not nodes:
        return []
    if reranker is not None:
        try:
            documents = [n.node.get_content(metadata_mode="none").strip() for n in nodes]
            rerank_scores = reranker(query, documents)
            scored = []
            for idx, node in enumerate(nodes):
                scored.append({
                    "node": node,
                    "rerank_score": float(rerank_scores[idx]),
                    "vector_score": node.score or 0.0,
                })
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:RERANK_TOP_N]
        except Exception as exc:
            print(f"[!] 精排失败，改用融合分数: {exc}", flush=True)
    return [{"node": n, "rerank_score": n.score or 0.0, "vector_score": n.score or 0.0} for n in nodes[:RERANK_TOP_N]]


def pack_retrieval_text(query: str, nodes: list, vision_query: str | None = None) -> str:
    """精排、去重父段、带 chunk_id 的检索原文。"""
    if RAG_DEBUG:
        print(f"\n[混合检索 BM25+向量+RRF] {query}")
    if not nodes:
        _write_audit_log({
            "ts": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "vector_top20": [],
            "bm25_top20": [],
            "hybrid_top20": [],
            "rerank_top5": [],
            "context_snippets": [],
        })
        return "知识库中未检索到相关内容。"

    source_hits = {name: hits for name, hits in getattr(retriever, "last_source_results", [])}
    vector_hits = source_hits.get("vector", [])
    bm25_hits = source_hits.get("bm25", [])

    def _hits_audit(hits, score_key: str):
        rows = []
        for hit in hits:
            rows.append({
                "doc_id": hit.node.node_id,
                "chunk_id": hit.node.metadata.get("chunk_id") or hit.node.node_id,
                "file_name": hit.node.metadata.get("file_name", "未知文件"),
                score_key: round(hit.score or 0.0, 6),
            })
        return rows

    vector_top20 = _hits_audit(vector_hits, "cosine_score")
    bm25_top20 = _hits_audit(bm25_hits, "bm25_score")
    hybrid_top20 = _hits_audit(nodes, "rrf_score")
    top_nodes = _rerank_nodes(query, nodes)

    snippets = []
    rerank_top5 = []
    context_snippets = []
    seen_parents: set[str] = set()
    unique_top = []
    for item in top_nodes:
        node_obj = item["node"]
        parent_id = node_obj.node.metadata.get("parent_id") or node_obj.node.node_id
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        unique_top.append(item)
    top_nodes = unique_top

    for rank, item in enumerate(top_nodes, start=1):
        node_obj = item["node"]
        content, header_path = expand_retrieved_node(node_obj.node)
        file_name = node_obj.node.metadata.get("file_name", "未知文件")
        chunk_id = node_obj.node.metadata.get("chunk_id") or node_obj.node.node_id
        rerank_score = item["rerank_score"]
        loc = f"chunk_id={chunk_id}"
        if header_path:
            loc += f" | {header_path}"
        loc += f" | {file_name}"
        snippets.append(
            f"### RAG片段 chunk_id={chunk_id}\n"
            f"来源 {rank}: {loc}, 精排分数 {rerank_score:.4f}\n"
            f"----- 原文开始 -----\n{content}\n----- 原文结束 chunk_id={chunk_id} -----"
        )
        rerank_top5.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "doc_id": node_obj.node.node_id,
            "file_name": file_name,
            "header_path": header_path,
            "rerank_score": round(rerank_score, 6),
            "vector_score": round(item["vector_score"], 6),
        })
        context_snippets.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "file_name": file_name,
            "text": content,
        })

    snippets = enrich_snippets(snippets, query=vision_query or query)
    for item, snippet in zip(context_snippets, snippets):
        parts = snippet.split("\n", 1)
        item["text"] = parts[1] if len(parts) > 1 else snippet

    _write_audit_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "vector_top20": vector_top20,
        "bm25_top20": bm25_top20,
        "hybrid_top20": hybrid_top20,
        "rerank_top5": rerank_top5,
        "context_snippets": context_snippets,
    })
    return (
        "以下是检索命中的原文。每段都被 chunk_id 标记。"
        "推理时每用一处原文，必须在该句末尾写 [chunk_id=...]。"
        "禁止使用未出现在下方的 chunk_id。原文没有的内容回复「知识库中暂无此信息」：\n\n"
        + "\n\n".join(snippets)
    )


@tool
def query_security_knowledge(query: str) -> str:
    """查询知识库。《AI Agent 入门》全书：上下文工程、记忆与 RAG、工具、Coding Agent、评估、后训练、多 Agent。"""
    ensure_engine()
    return pack_retrieval_text(query, retriever.retrieve(query))


# ==================== 长期记忆（ChromaDB） ====================

MEMORY_COLLECTION_NAME = "react_long_term_memory"
_memory_collection = None


def _get_memory_collection():
    """长期记忆走本地 PersistentClient，不依赖远程 Chroma 是否在线。"""
    global _memory_collection
    if _memory_collection is not None:
        return _memory_collection
    try:
        persist = BASE_DIR / ".cache" / "long_term_memory"
        persist.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(
            path=str(persist),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        _memory_collection = chroma_client.get_or_create_collection(
            MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return _memory_collection
    except Exception as e:
        print(f"[长期记忆] 本地存储不可用: {e}")
        return None


def _store_steps_to_memory(blocks: list[str], question: str, session_id: str) -> None:
    """将即将被压缩的原始 Step 块逐条存入长期记忆，每块一条向量记录。"""
    if not blocks:
        return
    collection = _get_memory_collection()
    if collection is None:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        ids, embeddings, documents, metadatas = [], [], [], []
        for idx, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue
            # 用时间戳+序号保证 ID 唯一，避免冒号等特殊字符
            doc_id = f"{session_id}_{now[:19].replace(':', '-')}_{idx}"
            embedding = Settings.embed_model.get_text_embedding(block)
            ids.append(doc_id)
            embeddings.append(embedding)
            documents.append(block)
            metadatas.append({
                "session_id": session_id,
                "question": question[:200],
                "timestamp": now,
                "block_index": idx,
            })
        if ids:
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            print(f"[长期记忆] 已存储 {len(ids)} 个步骤块")
    except Exception as e:
        print(f"[长期记忆] 存储失败: {e}")


@tool
def search_memory(query: str) -> str:
    """搜索长期记忆。当需要回忆之前推理步骤中被压缩掉的历史信息时使用此工具。"""
    collection = _get_memory_collection()
    if collection is None:
        return "长期记忆不可用（ChromaDB 未连接）。"
    try:
        query_embedding = Settings.embed_model.get_text_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if not docs:
            return "长期记忆中未找到相关内容。"
        snippets = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = round(1 - dist, 4)
            snippets.append(
                f"[记忆 {i}] 相似度: {similarity} | 来源问题: {meta.get('question', '')[:80]}\n{doc}"
            )
        return "以下是从长期记忆中检索到的相关历史步骤：\n\n" + "\n\n".join(snippets)
    except Exception as e:
        return f"长期记忆检索失败: {e}"


# ==================== ReAct 框架核心 ====================

TOOLS = [multiply, read_local_file, query_security_knowledge, search_memory]
TOOLS_MAP = {item.name: item for item in TOOLS}

# 意图路由：锚句（代表回溯历史意图的典型中文表达，用于嵌入相似度检测）
_MEMORY_INTENT_ANCHORS = [
    "我需要回忆之前查到的信息",
    "前面已经查过这个内容了",
    "之前步骤中得到了相关结果",
    "历史推理步骤中包含这个信息",
    "刚才已经检索过类似问题",
    "之前的查询结果被压缩了",
    "需要参考历史记录中的内容",
    "前面的摘要中提到过这个",
]
_MEMORY_INTENT_THRESHOLD = 0.75   # 触发长期记忆检索的余弦相似度阈值
_MEMORY_ANCHOR_EMBEDDINGS: "list | None" = None  # 模块级缓存，首次调用时懒加载


def _detect_memory_intent(thought: str) -> bool:
    """
    用轻量嵌入相似度检测 Thought 是否含回溯历史意图。

    首次调用时向量化所有锚句并缓存；后续每步仅向量化当前 Thought（约 50-100ms）。
    任一锚句余弦相似度 >= _MEMORY_INTENT_THRESHOLD 即触发。
    """
    global _MEMORY_ANCHOR_EMBEDDINGS
    if not thought:
        return False

    import numpy as np

    if _MEMORY_ANCHOR_EMBEDDINGS is None:
        _MEMORY_ANCHOR_EMBEDDINGS = [
            np.array(Settings.embed_model.get_text_embedding(s))
            for s in _MEMORY_INTENT_ANCHORS
        ]

    thought_vec = np.array(Settings.embed_model.get_text_embedding(thought))
    norm_t = thought_vec / (np.linalg.norm(thought_vec) + 1e-9)

    for anchor_vec in _MEMORY_ANCHOR_EMBEDDINGS:
        norm_a = anchor_vec / (np.linalg.norm(anchor_vec) + 1e-9)
        if float(np.dot(norm_t, norm_a)) >= _MEMORY_INTENT_THRESHOLD:
            return True
    return False

# 静态系统提示词（模块级常量，内容固定，走显式缓存）
_REACT_SYSTEM_PROMPT = """你是《AI Agent 入门》知识库的 ReAct (Reasoning and Acting) 助手，回答必须依据检索原文。

【可用工具】
1. query_security_knowledge - 查询知识库（Agent 公式、上下文工程、记忆与 RAG、工具、Coding Agent、评估、后训练、多 Agent）
   输入格式: {"query": "具体查询问题"}

2. search_memory - 搜索长期记忆，检索被压缩掉的历史推理步骤
   当发现当前摘要缺失某个关键信息，或需要回忆之前已查到但被压缩的内容时使用
   输入格式: {"query": "要检索的历史信息关键词"}

3. multiply - 计算两个整数的乘积
   输入格式: {"a": 整数1, "b": 整数2}

4. read_local_file - 读取本地文件
   输入格式: {"file_name": "文件名"}

【回复格式】严格按以下格式：

Thought: [详细的推理过程]
- 若 Observation 里已有 RAG片段：每条用到的事实必须带 [chunk_id=...]，只能用 Observation 里出现的 id
- 分析当前已知信息和缺失信息
- 说明为什么需要这个行动
Action: [工具名称]
Action Input: [JSON格式参数]

当收集到足够信息后：
Thought: [逐步推理，每条论据末尾 [chunk_id=文件名::0001]]
Final Answer:
推理：
- 事实…… [chunk_id=...]
答案：
...
引用来源：
- chunk_id | 文件名
（必须使用 Observation 里出现的 chunk_id，禁止编造。原文没有的内容写「知识库中暂无此信息」）

【多步推理指南】
复杂 Agent 问题要分步检索：

✓ 上下文工程 → 分步查询：KV Cache / 压缩 → 记忆 → 知识库
  示例："给长流程 Agent 设计上下文时，压缩、记忆和 RAG 各管什么？"

✓ 工具与 Harness → 分步查询：工具分类 → MCP → 护栏
  示例："生产级 Agent 的 Harness 包含哪些层，工具安全怎么做？"

✓ 多 Agent → 分步查询：分工方式 → 交接 → 共享上下文
  示例："多 Agent 协作从编排到 handoff 的完整流程是什么？"

✓ 评估与进化 → 分步查询：指标 → 环境 → 持续进化
  示例："如何评估 Agent，又如何把评估信号送进持续进化？"

【核心规则】
1. 复杂问题必须拆解成多个子问题，逐步检索
2. 每次查询要聚焦具体知识点，不要笼统查询
3. 根据前一步的 Observation 调整下一步查询策略
4. 严格基于检索结果作答，知识库没有的内容明确说明
5. 最终答案要综合所有步骤的信息，结构化呈现

【判断是否需要多步】
- 问题包含"完整流程"、"从...到..."、"如何...以及..." → 需要多步
- 问题需要关联多个章节才能回答 → 需要多步
- 简单概念查询（"XX是什么"、"XX的定义"） → 单步即可
"""


def _write_react_trace(session_id: str, trace: list[dict]) -> None:
    """保存 ReAct 执行轨迹到 JSONL 文件"""
    REACT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = REACT_LOG_DIR / f"react_trace_{today}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace": trace
        }, ensure_ascii=False) + "\n")


def parse_react_response(response_text: str) -> dict:
    """
    解析 LLM 返回的 ReAct 格式响应
    期望格式:
    Thought: <思考过程>
    Action: <工具名>
    Action Input: <JSON格式参数>

    或者:
    Thought: <思考过程>
    Final Answer: <最终答案>
    """
    result = {
        "thought": "",
        "action": None,
        "action_input": None,
        "final_answer": None
    }

    # 提取 Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|$)", response_text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # 检查是否有 Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", response_text, re.DOTALL | re.IGNORECASE)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    # 提取 Action 和 Action Input
    action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", response_text, re.IGNORECASE)
    if action_match:
        result["action"] = action_match.group(1).strip()

    action_input_match = re.search(r"Action Input:\s*(.+?)(?=\n(?:Thought|Action|Final Answer):|$)", response_text, re.DOTALL | re.IGNORECASE)
    if action_input_match:
        input_text = action_input_match.group(1).strip()
        # 尝试解析 JSON
        try:
            result["action_input"] = json.loads(input_text)
        except json.JSONDecodeError:
            # 如果不是 JSON，作为字符串保存
            result["action_input"] = input_text

    return result


def execute_tool(tool_name: str, tool_input: dict | str) -> str:
    """
    执行工具调用，返回真实环境反馈
    """
    if tool_name not in TOOLS_MAP:
        return f"错误：未找到工具 '{tool_name}'。可用工具: {', '.join(TOOLS_MAP.keys())}"

    tool_func = TOOLS_MAP[tool_name]

    try:
        # 处理不同格式的输入
        if isinstance(tool_input, dict):
            result = tool_func.invoke(tool_input)
        elif isinstance(tool_input, str):
            # 尝试解析为 JSON
            try:
                parsed_input = json.loads(tool_input)
                result = tool_func.invoke(parsed_input)
            except json.JSONDecodeError:
                # 如果不是 JSON，直接作为单参数传入
                result = tool_func.invoke({"query": tool_input} if "query" in tool_func.args else tool_input)
        else:
            return f"错误：工具输入格式不正确 - {type(tool_input)}"

        return str(result)
    except Exception as e:
        return f"工具执行错误: {str(e)}"


def _estimate_tokens(text: str) -> int:
    return estimate_tokens(text)


def _compact_steps(question: str, dynamic_steps: str, session_id: str = "") -> str:
    """
    压缩 dynamic_steps，始终保留最近3步完整内容，压缩其余历史。
    第二次压缩时，已有的 [推理摘要] 块与新增旧步骤一起合并成新摘要。
    [历史对话]...[当前对话步骤] 前缀不参与压缩，压缩后原样拼回。
    system prompt 不在 dynamic_steps 里，不会被压缩。
    """
    print("[压缩] dynamic_steps 过长，触发上下文压缩...")

    # 剥离跨轮历史前缀（不压缩，压缩后原样拼回）
    history_prefix = ""
    raw_steps_text = dynamic_steps.strip()
    history_marker = "[历史对话]"
    steps_marker = "[当前对话步骤]"
    if raw_steps_text.startswith(history_marker):
        marker_pos = raw_steps_text.find(steps_marker)
        if marker_pos != -1:
            # 保留从 [历史对话] 到 [当前对话步骤] 的整段（含标记本身）
            history_prefix = raw_steps_text[:marker_pos + len(steps_marker)].strip() + "\n"
            raw_steps_text = raw_steps_text[marker_pos + len(steps_marker):].strip()

    # 剥离已有摘要块（二次压缩场景）
    prior_summary = ""
    if raw_steps_text.startswith("[推理摘要]"):
        # 找到 [最近步骤] 标记，摘要块在它之前
        marker = "[最近步骤（完整保留）]"
        marker_pos = raw_steps_text.find(marker)
        if marker_pos != -1:
            prior_summary = raw_steps_text[:marker_pos].strip()
            raw_steps_text = raw_steps_text[marker_pos + len(marker):].strip()

    # 按空行切分原始步骤块
    KEEP_RECENT = 3
    blocks = [b.strip() for b in raw_steps_text.split("\n\n") if b.strip()]

    if len(blocks) <= KEEP_RECENT and not prior_summary:
        return dynamic_steps  # 步骤太少且无已有摘要，不压缩（history_prefix 已含在 dynamic_steps 里）

    recent_blocks = blocks[-KEEP_RECENT:] if len(blocks) > KEEP_RECENT else blocks
    older_blocks = blocks[:-KEEP_RECENT] if len(blocks) > KEEP_RECENT else []
    recent_text = "\n\n".join(recent_blocks)

    # 压缩前：将即将被摘要的原始 Step 块持久化到长期记忆
    if older_blocks and session_id:
        _store_steps_to_memory(older_blocks, question, session_id)

    # 组装待压缩内容：已有摘要 + 更早的原始步骤
    parts_to_compress = []
    if prior_summary:
        parts_to_compress.append(prior_summary)
    if older_blocks:
        parts_to_compress.append("\n\n".join(older_blocks))
    older_text = "\n\n".join(parts_to_compress)

    if not older_text:
        # 没有需要压缩的内容（只有 recent），直接返回
        return f"{history_prefix}[最近步骤（完整保留）]\n{recent_text}\n\n"

    history_label = "推理摘要和推理步骤" if prior_summary else "推理步骤"
    compact_prompt = (
        f"以下是一个 Agent 知识问题的{history_label}，请用100字以内的中文总结为一份进度报告：\n"
        f"- 已完成目标：已查询并确认的知识点和关键结论\n"
        f"- 未完成目标：还需要查询或确认的内容\n"
        f"- 关键中间结果：每次查询的核心发现（只保留事实，去掉原文）\n"
        f"不要超过300字，直接输出报告。\n\n"
        f"原始问题：{question}\n\n"
        f"{history_label}：\n{older_text}"
    )
    resp = _raw_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": compact_prompt}],
        temperature=0,
        max_tokens=600,
        extra_body={"enable_thinking": False},
    )
    summary = resp.choices[0].message.content or ""
    compacted = f"{history_prefix}[推理摘要]\n{summary}\n\n[最近步骤（完整保留）]\n{recent_text}\n\n"
    print(f"[压缩] 压缩前 ~{_estimate_tokens(dynamic_steps)} tokens"
          f" → 压缩后 ~{_estimate_tokens(compacted)} tokens")
    return compacted


def react_loop(
    question: str,
    session_id: str | None = None,
    on_event=None,
) -> tuple[str, list[dict]]:
    """
    ReAct 框架核心循环

    Args:
        question: 用户问题
        session_id: 会话ID（用于追踪）
        on_event: 可选回调，供 WebUI / SSE 推送逐步轨迹

    Returns:
        (final_answer, trace): 最终答案和执行轨迹
    """
    ensure_engine()
    if session_id is None:
        session_id = f"react_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 初始化上下文（动态部分单独维护，静态 _REACT_SYSTEM_PROMPT 走缓存）
    # 注入最近 N 轮对话记忆作为前缀
    turns_prefix = ""
    if _conversation_history:
        recent_turns = _conversation_history[-KEEP_RECENT_TURNS:]
        lines = ["[历史对话]"]
        for turn in recent_turns:
            lines.append(f"问题: {turn['question']}")
            lines.append(f"答案: {turn['answer']}")
            lines.append("")
        lines.append("[当前对话步骤]")
        turns_prefix = "\n".join(lines) + "\n"

    dynamic_steps = turns_prefix  # 累积的 Thought/Action/Observation 步骤
    trace = []
    step = 0
    tool_call_count = 0  # 工具调用次数，用于步数触发压缩

    # 模型窗口 128K tokens
    MODEL_CONTEXT_WINDOW = 131072
    SOFT_THRESHOLD = int(MODEL_CONTEXT_WINDOW * 0.70)   # 70%：后台异步预压缩
    HARD_THRESHOLD = int(MODEL_CONTEXT_WINDOW * 0.90)   # 90%：同步阻塞强制压缩
    # 步数触发：工具调用超过此次数且无结果，说明在绕圈，强制压缩帮模型重新定向
    TOOL_CALL_COMPACT_THRESHOLD = 5

    # 后台压缩：快照 + 压缩期间新增步骤必须拼回去，否则会丢掉当前 Observation
    compact_lock = threading.Lock()
    compact_thread: threading.Thread | None = None
    compact_snapshot = ""
    compact_result: list[str] = []

    def _apply_ready_compact(current: str) -> str:
        nonlocal compact_thread, compact_snapshot
        with compact_lock:
            if not compact_result:
                return current
            compacted = compact_result.pop()
            snapshot = compact_snapshot
            compact_thread = None
            compact_snapshot = ""
        merged = apply_compact_delta(current, snapshot, compacted)
        print("[压缩] 后台压缩完成，已合并压缩期间新增步骤")
        _emit(on_event, {"type": "compact", "tokens": _estimate_tokens(merged)})
        return merged

    print(f"\n{'='*60}")
    print(f"[ReAct 循环开始] 问题: {question}")
    print(f"[最大步数] {MAX_REACT_STEPS}")
    print(f"{'='*60}\n")
    _emit(on_event, {"type": "route", "route": "react", "question": question})

    while step < MAX_REACT_STEPS:
        step += 1
        print(f"\n--- Step {step}/{MAX_REACT_STEPS} ---")
        _emit(on_event, {"type": "step", "step": step, "max": MAX_REACT_STEPS})

        if compact_thread is not None and not compact_thread.is_alive():
            dynamic_steps = _apply_ready_compact(dynamic_steps)

        current_tokens = _estimate_tokens(dynamic_steps)

        # 硬阈值（90%）：同步阻塞，强制压缩
        if dynamic_steps and current_tokens > HARD_THRESHOLD:
            print(f"[压缩] 硬阈值触发（~{current_tokens} tokens，>{HARD_THRESHOLD}）")
            _emit(on_event, {"type": "compact", "mode": "hard", "tokens": current_tokens})
            if compact_thread is not None and compact_thread.is_alive():
                print("[压缩] 等待后台压缩线程完成...")
                compact_thread.join()
            if compact_result:
                dynamic_steps = _apply_ready_compact(dynamic_steps)
            else:
                dynamic_steps = _compact_steps(question, dynamic_steps, session_id)
            compact_thread = None

            # 二次检查：极端输入（单条消息自身就超窗口）压缩后仍可能超标
            post_tokens = _estimate_tokens(dynamic_steps)
            if post_tokens > HARD_THRESHOLD:
                error_msg = (
                    f"[错误] 压缩后上下文仍超过硬阈值（~{post_tokens} tokens），"
                    f"当前输入过长无法继续推理。请缩短问题或分多次提问。"
                )
                print(error_msg)
                _write_react_trace(session_id, trace)
                return error_msg, trace

        # 软阈值（70%）：启动后台线程预压缩，本步继续跑
        elif dynamic_steps and current_tokens > SOFT_THRESHOLD:
            if compact_thread is None or not compact_thread.is_alive():
                print(f"[压缩] 软阈值触发（~{current_tokens} tokens，>{SOFT_THRESHOLD}），后台开始预压缩")
                _emit(on_event, {"type": "compact", "mode": "soft", "tokens": current_tokens})
                steps_snapshot = dynamic_steps
                with compact_lock:
                    compact_result.clear()
                    compact_snapshot = steps_snapshot

                def _bg_compact(q: str, snap: str, out: list, sid: str, lock: threading.Lock) -> None:
                    text = _compact_steps(q, snap, sid)
                    with lock:
                        out.append(text)

                compact_thread = threading.Thread(
                    target=_bg_compact,
                    args=(question, steps_snapshot, compact_result, session_id, compact_lock),
                    daemon=True,
                )
                compact_thread.start()

        # 步数触发：工具调用超过阈值仍无答案，说明在绕圈，强制压缩
        elif tool_call_count > TOOL_CALL_COMPACT_THRESHOLD:
            print(f"[压缩] 步数触发（工具调用 {tool_call_count} 次仍未给出答案）")
            if compact_thread is not None and compact_thread.is_alive():
                compact_thread.join()
            if compact_result:
                dynamic_steps = _apply_ready_compact(dynamic_steps)
            else:
                dynamic_steps = _compact_steps(question, dynamic_steps, session_id)
            compact_thread = None
            tool_call_count = 0

        # 本步结束后，若后台压缩已完成则替换（软阈值预压缩结果落地）
        # 注意：此处在步骤执行完成后检查，放在循环末尾

        # 构造消息：静态块加 cache_control，动态块不加
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": _REACT_SYSTEM_PROMPT,  # 静态：工具定义+推理指南，走缓存
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": f"用户问题: {question}\n\n{dynamic_steps}",  # 动态：每步追加
            },
        ]

        # 调用 LLM（使用原生 openai SDK，支持 cache_control）
        api_response = _raw_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            extra_body={"enable_thinking": False},
        )
        response_text = api_response.choices[0].message.content or ""

        # 打印缓存命中情况
        usage = api_response.usage
        if usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
            created = getattr(usage.prompt_tokens_details, "cache_creation_input_tokens", 0) or 0
            if created > 0:
                print(f"[缓存] 写入缓存 {created} tokens")
            elif cached > 0:
                print(f"[缓存] 命中缓存 {cached} tokens（节省 {cached * 0.9:.0f} tokens 费用）")

        print(f"\n[LLM Response]")
        print(response_text)

        parsed = parse_react_response(response_text)
        _emit(on_event, {
            "type": "thought",
            "step": step,
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "has_final": bool(parsed["final_answer"]),
        })

        # 意图路由：嵌入相似度检测到回溯历史意图，且模型未主动选 search_memory
        if (_detect_memory_intent(parsed["thought"])
                and parsed.get("action") != "search_memory"):
            print("[意图路由] 检测到回溯历史意图，自动检索长期记忆...")
            memory_result = search_memory.invoke({"query": question})
            if "未找到" not in memory_result and "不可用" not in memory_result:
                dynamic_steps += (
                    f"[系统自动注入] 检测到回溯历史意图，已自动检索长期记忆：\n"
                    f"{memory_result}\n\n"
                )
                print("[意图路由] 长期记忆已注入当前上下文")

        # 记录到轨迹
        step_trace = {
            "step": step,
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "final_answer": parsed["final_answer"],
            "observation": None
        }

        # 检查是否到达最终答案
        if parsed["final_answer"]:
            final = ensure_reasoning_cites_chunk_id(
                parsed["final_answer"],
                dynamic_steps,
                _SIMPLE_ANSWER_SYSTEM,
                f"用户问题：{question}\n\n{dynamic_steps}\n\n请按要求给出带 [chunk_id=...] 的推理和答案。",
            )
            parsed["final_answer"] = final
            print(f"\n[Final Answer] {parsed['final_answer']}")
            step_trace["observation"] = "已到达最终答案"
            step_trace["final_answer"] = final
            trace.append(step_trace)
            # 写入跨轮对话记忆，供下一次 react_loop() 调用时注入
            _conversation_history.append({"question": question, "answer": parsed["final_answer"]})
            _write_react_trace(session_id, trace)
            _emit(on_event, {"type": "final", "answer": parsed["final_answer"], "trace": trace})
            return parsed["final_answer"], trace

        # 执行工具调用
        if parsed["action"] and parsed["action_input"]:
            print(f"\n[执行工具] {parsed['action']}")
            print(f"[工具参数] {parsed['action_input']}")

            observation = execute_tool(parsed["action"], parsed["action_input"])
            step_trace["observation"] = observation
            trace.append(step_trace)
            tool_call_count += 1

            print(f"\n[Observation]")
            print(observation[:500] + "..." if len(observation) > 500 else observation)
            _emit(on_event, {
                "type": "observation",
                "step": step,
                "action": parsed["action"],
                "preview": observation[:400],
            })

            dynamic_steps += (
                f"Thought: {parsed['thought']}\n"
                f"Action: {parsed['action']}\n"
                f"Action Input: {json.dumps(parsed['action_input'], ensure_ascii=False)}\n"
                f"Observation: {observation}\n\n"
            )

            if compact_thread is not None and not compact_thread.is_alive():
                dynamic_steps = _apply_ready_compact(dynamic_steps)
        else:
            # 如果没有有效的 Action，给出提示
            observation = "错误：你必须选择一个有效的 Action 和 Action Input，或者给出 Final Answer"
            step_trace["observation"] = observation
            trace.append(step_trace)
            print(f"\n[系统提示] {observation}")
            dynamic_steps += f"{observation}\n"

    # 达到最大步数限制
    final_msg = f"已达到最大步数限制({MAX_REACT_STEPS})，无法继续推理。请尝试重新提问或简化问题。"
    print(f"\n[超时] {final_msg}")
    _write_react_trace(session_id, trace)
    return final_msg, trace


_CHUNK_ID_IN_TEXT = re.compile(
    r"chunk_id\s*=\s*([^\s|,\]】]+)|((?:[\w.\-]|[\u4e00-\u9fff])+\.md::\d{4})",
    re.I,
)


def extract_chunk_ids(text: str) -> set[str]:
    found: set[str] = set()
    for match in _CHUNK_ID_IN_TEXT.finditer(text or ""):
        found.add((match.group(1) or match.group(2) or "").strip())
    return {item for item in found if item}


def strip_md_stars(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    return text


def format_user_facing(full: str) -> str:
    """用户只看「答案」；chunk_id 收到文末一行；去掉 **。"""
    text = full or ""
    text = re.sub(r"- <chunk_id>\s*\|\s*<文件名>\s*\n?", "", text)
    answer_m = re.search(
        r"^答案[：:]\s*(.+?)(?=^\s*引用来源[：:]|\Z)",
        text,
        re.S | re.M,
    )
    if answer_m:
        body = answer_m.group(1).strip()
    else:
        body = re.sub(r"^推理[：:][\s\S]*?(?=^答案[：:]|\Z)", "", text, flags=re.M).strip()
        if not body:
            body = text.strip()
    body = re.sub(r"\[chunk_id=[^\]]+\]", "", body)
    body = re.sub(r"^\s*引用来源[：:][\s\S]*", "", body, flags=re.M).strip()
    body = strip_md_stars(body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    cited = sorted(extract_chunk_ids(full))
    cited = [c for c in cited if "<" not in c and ">" not in c]
    if cited:
        body += "\n\n引用: " + "；".join(cited)
    return body


def ensure_reasoning_cites_chunk_id(answer: str, context: str, system: str, user: str) -> str:
    """推理正文必须出现检索里的 chunk_id，否则强制再生成一次。"""
    allowed = extract_chunk_ids(context)
    if not allowed:
        return answer
    cited = extract_chunk_ids(answer)
    if cited & allowed:
        return answer
    print("[校验] 推理未标注 chunk_id，强制补写引用", flush=True)
    allowed_list = "、".join(sorted(allowed))
    retry_user = (
        f"{user}\n\n【系统重写】上一稿推理过程没有标注 chunk_id。"
        f"必须重写：推理每一条用到原文的句子末尾加上 [chunk_id=...]。"
        f"只能使用这些 id：{allowed_list}"
    )
    retry = _llm_complete(system, retry_user)
    if extract_chunk_ids(retry) & allowed:
        return retry
    footer = "\n\n引用来源（系统补全，因模型未在推理中标注）：\n" + "\n".join(
        f"- {cid}" for cid in sorted(allowed)
    )
    return (retry or answer) + footer


_SIMPLE_ANSWER_SYSTEM = """你是知识库问答助手。必须先逐步推理再给答案，以降低幻觉。

检索原文每段都标了 chunk_id。推理时每使用一处原文，必须在该句末尾写 [chunk_id=文件名::0001]。
禁止使用检索原文里没有出现的 chunk_id。

格式：
推理：
- …… [chunk_id=...]
- 原文没覆盖的点写「知识库中暂无此信息」

答案：
（面向用户的完整回答，关键结论也可带 [chunk_id=...]）

引用来源：
- <chunk_id> | <文件名>
"""


def answer_simple_rag(question: str, plan: QueryPlan, on_event=None) -> tuple[str, list[dict]]:
    print(f"[路由] 简单问答，策略={plan.strategies or ['none']}", flush=True)
    _emit(on_event, {"type": "route", "route": "simple_rag", "strategies": plan.strategies})
    if plan.rewritten and plan.rewritten != question:
        print(f"[改写] {plan.rewritten}", flush=True)
    if plan.step_back:
        print(f"[Step-back] {plan.step_back}", flush=True)
    if plan.hyde:
        print("[HyDE] 已生成假想短文，用于向量检索", flush=True)
    if plan.multi_queries:
        print(f"[多Query] {plan.multi_queries}", flush=True)

    _emit(on_event, {"type": "retrieve", "query": plan.search_query})
    nodes = retrieve_with_plan(plan)
    packed = pack_retrieval_text(plan.search_query, nodes, vision_query=question)
    if packed.startswith("知识库中未检索到") and plan.step_back:
        print("[Step-back] 细问题未命中，用后退问题补检索", flush=True)
        extra = retrieve_fused_nodes([plan.step_back])
        packed = pack_retrieval_text(plan.step_back, extra, vision_query=question)

    user = (
        f"用户问题：{question}\n"
        f"规范化问句：{plan.search_query}\n"
        f"后退问题：{plan.step_back or '无'}\n\n"
        f"{packed}\n\n"
        "若细问题在原文中没有直接数字/配置，可依据后退问题命中的原理作答，"
        "但必须标明这是由背景知识推断，且仍要引用 chunk_id。"
    )
    answer = _llm_complete(_SIMPLE_ANSWER_SYSTEM, user)
    answer = ensure_reasoning_cites_chunk_id(answer, packed, _SIMPLE_ANSWER_SYSTEM, user)
    trace = [{
        "step": 1,
        "route": "simple_rag",
        "strategies": plan.strategies,
        "rewritten": plan.search_query,
        "step_back": plan.step_back,
        "thought": "单跳检索 + CoT 作答",
        "action": "retrieve",
        "final_answer": answer,
    }]
    _conversation_history.append({"question": question, "answer": answer})
    _emit(on_event, {"type": "final", "answer": answer, "trace": trace})
    return answer, trace


def answer_chitchat(question: str, on_event=None) -> tuple[str, list[dict]]:
    """寒暄不检索、不贴 chunk_id。"""
    print("[路由] 寒暄/闲聊 → 不检索", flush=True)
    _emit(on_event, {"type": "route", "route": "chitchat"})
    answer = _llm_complete(
        "你是知识库助手。用户只是寒暄或闲聊，不要检索，不要编造 chunk_id。"
        "简短礼貌回应，并邀请提出 AI Agent、上下文工程或 RAG 相关问题。不要输出「推理：」长文。",
        question,
    )
    trace = [{
        "step": 1,
        "route": "chitchat",
        "thought": "无知识需求，跳过检索",
        "action": None,
        "final_answer": answer,
    }]
    _conversation_history.append({"question": question, "answer": answer})
    _emit(on_event, {"type": "final", "answer": answer, "trace": trace})
    return answer, trace


def handle_user_question(question: str, on_event=None) -> tuple[str, list[dict]]:
    """寒暄不检索；知识问答先检索再 CoT；复杂题才 ReAct。"""
    global _QUERY_DEPTH
    ensure_engine()
    with _ENGINE_LOCK:
        _QUERY_DEPTH += 1
    _emit(on_event, {"type": "status", "stage": "planning"})
    try:
        return _dispatch_question(question, on_event)
    finally:
        with _ENGINE_LOCK:
            _QUERY_DEPTH = max(0, _QUERY_DEPTH - 1)


def _dispatch_question(question: str, on_event=None) -> tuple[str, list[dict]]:
    if looks_like_chitchat(question):
        print("[规划] complexity=chitchat strategies=[]", flush=True)
        return answer_chitchat(question, on_event=on_event)
    plan = plan_query(question, _llm_complete, _conversation_history)
    print(
        f"[规划] complexity={plan.complexity} strategies={plan.strategies}",
        flush=True,
    )
    _emit(on_event, {
        "type": "plan",
        "complexity": plan.complexity,
        "strategies": plan.strategies,
        "rewritten": plan.search_query,
    })
    if plan.complexity == "chitchat":
        return answer_chitchat(question, on_event=on_event)
    if plan.complexity == "complex":
        print("[路由] 复杂题 → ReAct", flush=True)
        hint = plan.search_query
        if hint and hint != question:
            return react_loop(
                f"{question}\n（检索可先用：{hint}；后退问题：{plan.step_back}）",
                on_event=on_event,
            )
        return react_loop(question, on_event=on_event)
    return answer_simple_rag(question, plan, on_event=on_event)


def engine_info() -> dict:
    return {
        "storage": STORAGE_BACKEND,
        "storage_label": storage_label() if _ENGINE_READY else "",
        "embed": EMBED_BACKEND,
        "chat_model": CHAT_MODEL,
        "vision_model": os.getenv("TONGYI_VISION_MODEL", "qwen3-vl-flash"),
        "chroma_mode": CHROMA_MODE,
        "packs": knowledge_packs(),
        "files": len(knowledge_files()),
        "ready": _ENGINE_READY,
        "status": _STORAGE_STATUS,
    }


def knowledge_packs() -> list[str]:
    files = knowledge_files()
    packs = []
    for path in files:
        rel = path.relative_to(DATA_DIR).as_posix()
        pack = rel.split("/")[0] if "/" in rel else "(root)"
        if pack not in packs:
            packs.append(pack)
    return packs


def cmd_doctor() -> int:
    print("=" * 60)
    print("Agentic RAG doctor")
    print("=" * 60)
    key_ok = bool(DASHSCOPE_API_KEY)
    print(f"  DASHSCOPE_API_KEY : {'ok' if key_ok else 'MISSING  →  set it in .env'}")
    print(f"  EMBED_BACKEND     : {EMBED_BACKEND}")
    print(f"  CHROMA_MODE       : {CHROMA_MODE}")
    print(f"  CHAT_MODEL        : {CHAT_MODEL}")
    files = knowledge_files()
    packs = knowledge_packs()
    print(f"  knowledge files   : {len(files)}")
    print(f"  knowledge packs   : {', '.join(packs) or '(none)'}")
    env_path = BASE_DIR / ".env"
    print(f"  .env              : {'found' if env_path.is_file() else 'missing  →  copy .env.example'}")
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        print("  playwright        : installed")
    except Exception:
        print("  playwright        : missing  →  pip install playwright && playwright install chromium")
    if EMBED_BACKEND == "ollama":
        print(f"  ollama            : {OLLAMA_URL} / {EMBED_MODEL}")
    print("=" * 60)
    if not files:
        print("Put markdown into data/<pack-name>/ and re-run.")
        return 1
    if not key_ok and EMBED_BACKEND == "dashscope":
        print("No API key. Fill DASHSCOPE_API_KEY or set EMBED_BACKEND=ollama.")
        return 1
    print("Ready. Try: python main.py ask 什么是 ReAct")
    return 0


def cmd_chat() -> None:
    start_session_log()
    ensure_engine()
    print("=" * 60)
    print("ReAct + hybrid retrieval + rerank")
    print(f"向量库: {storage_label()}")
    print(f"max steps: {MAX_REACT_STEPS}")
    print(f"packs: {', '.join(knowledge_packs())}")
    print(f"session: {_SESSION_LOG}")
    print("type exit / quit to leave")
    print("=" * 60)
    while True:
        try:
            user_input = repair_console_input(input("\n用户: ").strip())
            if user_input.startswith("__ROLE__="):
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue
            answer, trace = handle_user_question(user_input)
            display = format_user_facing(answer)
            runlog.log_conversation(user_input, answer, trace, display=display)
            print(f"\n{'='*60}")
            print(display)
            print(f"{'='*60}")
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"\n错误: {exc}")
            import traceback
            traceback.print_exc()
    print("\n再见！")


def cmd_ask(question: str) -> int:
    start_session_log()
    answer, trace = handle_user_question(question)
    display = format_user_facing(answer)
    runlog.log_conversation(question, answer, trace, display=display)
    print(display)
    return 0


def cmd_ingest() -> int:
    start_session_log()
    ensure_engine()
    print(f"[+] 索引同步完成 · {storage_label()} · {len(knowledge_files())} 个文件")
    return 0


def cmd_web(host: str, port: int) -> None:
    start_session_log()
    ensure_engine()
    import uvicorn
    from app import app as web_app

    print(f"[+] Evidence Studio  http://{host}:{port}")
    uvicorn.run(web_app, host=host, port=port, log_level="info")


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multimodal Agentic RAG — chat, one-shot ask, web studio, doctor.",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("chat", help="interactive REPL (default)")
    sub.add_parser("doctor", help="check API key, packs, playwright")
    sub.add_parser("ingest", help="build / sync the vector index only")
    ask_p = sub.add_parser("ask", help="one-shot question")
    ask_p.add_argument("question", nargs="+", help="user question")
    web_p = sub.add_parser("web", help="Evidence Studio (FastAPI)")
    web_p.add_argument("--host", default=WEB_HOST)
    web_p.add_argument("--port", type=int, default=WEB_PORT)
    args = parser.parse_args(argv)
    cmd = args.cmd or "chat"
    if cmd == "doctor":
        return cmd_doctor()
    if cmd == "ingest":
        return cmd_ingest()
    if cmd == "ask":
        return cmd_ask(" ".join(args.question))
    if cmd == "web":
        cmd_web(args.host, args.port)
        return 0
    cmd_chat()
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
