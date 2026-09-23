"""按文档增量同步向量库：1024 维、chunk_id、文档 hash / mtime。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import runlog

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.chroma import ChromaVectorStore
from markdown_chunker import MarkdownParentChildSplitter

EMBED_DIM = 1024
CHUNKER_VERSION = "v6-md-pc"
INSERT_BATCH_SIZE = 8
INSERT_RETRIES = 6
PARENT_STORE_PATH = Path(__file__).resolve().parent / ".cache" / "parent_texts.json"

_META_EXCLUDE_EMBED = [
    "file_name",
    "source_path",
    "topic",
    "parent_text",
    "parent_id",
    "header_path",
    "chunk_id",
    "chunk_index",
    "doc_chunk_count",
    "doc_hash",
    "doc_mtime",
    "vector_dim",
    "chunker",
]
_META_EXCLUDE_LLM = [
    "source_path",
    "topic",
    "parent_text",
    "parent_id",
    "chunk_id",
    "chunk_index",
    "doc_chunk_count",
    "doc_hash",
    "doc_mtime",
    "vector_dim",
    "chunker",
]


def probe_embedding_dim() -> int:
    """用一条短文本探测当前 embedding 维度，必须为 1024。"""
    embed_model = Settings.embed_model
    if embed_model is None:
        raise RuntimeError("Settings.embed_model 未初始化")
    vec = embed_model.get_text_embedding("dimension-check")
    dim = len(vec)
    print(f"[+] embedding 维度: {dim}（模型 {getattr(embed_model, 'model_name', '?')}）")
    if dim != EMBED_DIM:
        raise RuntimeError(
            f"当前向量维度是 {dim}，需要 {EMBED_DIM}（bge-large-zh-v1.5）。"
            "请检查 OLLAMA_EMBED_MODEL。"
        )
    return dim


def file_fingerprint(path: Path) -> tuple[str, str]:
    """文档内容 sha256 + UTC 修改时间。"""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return digest, mtime


def make_chunk_id(file_name: str, index: int) -> str:
    """文件名 + 四位序号，例如 01-AI-Agent入门.md::0001。"""
    return f"{file_name}::{index:04d}"


def make_node_id(source_path: str, index: int) -> str:
    """全局唯一 id，避免同名文件冲突。"""
    return f"{source_path.replace(chr(92), '/')}::{index:04d}"


def _topic_for(rel: str) -> str:
    return rel.split("/")[0] if "/" in rel.replace("\\", "/") else "default"


def stamp_nodes(
    nodes: list[BaseNode],
    *,
    path: Path,
    data_dir: Path,
    doc_hash: str,
    doc_mtime: str,
) -> list[BaseNode]:
    rel = path.relative_to(data_dir).as_posix()
    file_name = path.name
    stamped: list[BaseNode] = []
    total = len(nodes)
    for i, node in enumerate(nodes, start=1):
        chunk_id = make_chunk_id(file_name, i)
        node_id = make_node_id(rel, i)
        node.node_id = node_id
        node.metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": i,
                "doc_chunk_count": total,
                "file_name": file_name,
                "source_path": rel,
                "topic": _topic_for(rel),
                "doc_hash": doc_hash,
                "doc_mtime": doc_mtime,
                "vector_dim": EMBED_DIM,
                "chunker": CHUNKER_VERSION,
            }
        )
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=rel)
        node.excluded_embed_metadata_keys = list(
            set(getattr(node, "excluded_embed_metadata_keys", []) or []) | set(_META_EXCLUDE_EMBED)
        )
        node.excluded_llm_metadata_keys = list(
            set(getattr(node, "excluded_llm_metadata_keys", []) or []) | set(_META_EXCLUDE_LLM)
        )
        stamped.append(node)
    return stamped


def _save_parent_texts(mapping: dict[str, str]) -> None:
    PARENT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, str] = {}
    if PARENT_STORE_PATH.is_file():
        try:
            store = json.loads(PARENT_STORE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            store = {}
    store.update(mapping)
    PARENT_STORE_PATH.write_text(
        json.dumps(store, ensure_ascii=False),
        encoding="utf-8",
    )


def _slim_for_chroma(nodes: list[BaseNode]) -> list[BaseNode]:
    """parent_text 很大，不进 Chroma，改存本地 sidecar，避免 502。"""
    parents: dict[str, str] = {}
    for node in nodes:
        parent = node.metadata.get("parent_text") or ""
        if parent:
            parents[node.node_id] = parent
        node.metadata["parent_text"] = ""
    if parents:
        _save_parent_texts(parents)
    return nodes


def _is_retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in ("502", "503", "504", "bad gateway", "timeout", "timed out", "connection")
    )


def insert_nodes_reliable(
    index: VectorStoreIndex,
    nodes: list[BaseNode],
    rel: str,
    batch_size: int = INSERT_BATCH_SIZE,
) -> None:
    """小批量写入，502 时降批重大试。"""
    slim = _slim_for_chroma(nodes)
    total = len(slim)
    batches = (total + batch_size - 1) // max(batch_size, 1)
    print(f"    [*] 开始写入 {rel}：{total} chunks，约 {batches} 批（每批最多 {batch_size}）", flush=True)
    i = 0
    batch_no = 0
    t0 = time.monotonic()
    while i < len(slim):
        size = min(batch_size, len(slim) - i)
        batch = slim[i : i + size]
        last_exc: BaseException | None = None
        batch_no += 1
        for attempt in range(1, INSERT_RETRIES + 1):
            try:
                t_batch = time.monotonic()
                index.insert_nodes(batch)
                elapsed = time.monotonic() - t_batch
                done = i + size
                print(
                    f"    … [{done}/{total}] {rel}  第{batch_no}批 +{size}  {elapsed:.1f}s",
                    flush=True,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                runlog.event(
                    f"写入失败 {rel} batch={i}-{i+size} attempt={attempt}",
                    error=str(exc),
                    retryable=_is_retryable(exc),
                )
                print(f"    [!] 写入失败 {rel} [{i}:{i+size}] 第{attempt}次: {exc}", flush=True)
                if not _is_retryable(exc) or attempt == INSERT_RETRIES:
                    break
                time.sleep(min(2 ** attempt, 16))
                if size > 1 and _is_retryable(exc):
                    size = max(1, size // 2)
                    batch = slim[i : i + size]
                    print(f"    [*] 降批量到 {size} 后重试", flush=True)
        if last_exc is not None:
            raise last_exc
        i += size
    print(f"    [+] {rel} 写入完成，共 {total} chunks，耗时 {time.monotonic() - t0:.1f}s", flush=True)


SCAN_PAGE_SIZE = 50


class HnswCorruptError(RuntimeError):
    """本地 HNSW 段文件不完整或无法加载。"""


def is_hnsw_corrupt(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "hnsw" in text and (
        "loading" in text or "segment reader" in text or "compactor" in text
    )


def release_chroma_locks() -> None:
    """Windows 上 sqlite 会被 PersistentClient 锁住，挪目录前必须释放。"""
    import gc

    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()
    time.sleep(1.0)


def hnsw_dir_healthy(path: Path) -> bool:
    """sqlite 在、HNSW 段文件不在，视为损坏，不要再 count()。"""
    if not path.exists():
        return True
    sqlite = path / "chroma.sqlite3"
    if not sqlite.is_file():
        return True
    return any(path.rglob("data_level0.bin"))


def _unique_backup_dir(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    bak = path.with_name(f"{path.name}.corrupt-{stamp}")
    n = 0
    while bak.exists():
        n += 1
        bak = path.with_name(f"{path.name}.corrupt-{stamp}-{n}")
    return bak


def quarantine_chroma_dir(path: Path) -> Path | None:
    """把损坏的 persist 目录挪走。占用失败则返回 None，由调用方改用新目录。"""
    if not path.exists():
        return None
    release_chroma_locks()
    last_exc: Exception | None = None
    for _ in range(5):
        bak = _unique_backup_dir(path)
        try:
            os.rename(str(path), str(bak))
            return bak
        except OSError as exc:
            last_exc = exc
            release_chroma_locks()
    print(f"[!] 无法挪走 {path}（仍被占用）: {last_exc}", flush=True)
    return None


def _manifest_path(label: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label) or "default"
    return Path(__file__).resolve().parent / ".cache" / f"chroma_manifest_{safe}.json"


def _load_manifest(collection_name: str, cloud_count: int, label: str = "default") -> dict[str, dict] | None:
    path = _manifest_path(label)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("collection") != collection_name:
        return None
    if int(data.get("count") or -1) != int(cloud_count):
        print(
            f"[*] 本地清单 {data.get('count')} 条，库内 {cloud_count} 条，改为分页扫描",
            flush=True,
        )
        return None
    docs = data.get("docs") or {}
    print(f"[+] 使用本地清单：{cloud_count} chunks / {len(docs)} 个文档", flush=True)
    return docs


def _save_manifest(collection_name: str, collection, docs: dict[str, dict], label: str = "default") -> None:
    path = _manifest_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": collection_name,
        "count": collection.count(),
        "docs": {
            rel: {
                "doc_hash": info.get("doc_hash", ""),
                "chunker": info.get("chunker", ""),
                "doc_chunk_count": int(info.get("doc_chunk_count") or 0),
                "ids": list(info.get("ids") or []),
            }
            for rel, info in docs.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def scan_indexed_docs(collection, collection_name: str = "", label: str = "default") -> dict[str, dict]:
    """source_path -> {doc_hash, chunker, count}。分页拉取，避免一次 get 全库卡住。"""
    print("[*] 统计库内 chunk 数量…", flush=True)
    try:
        total = collection.count()
    except Exception as exc:
        if is_hnsw_corrupt(exc):
            raise HnswCorruptError(str(exc)) from exc
        raise
    print(f"[+] 库内当前 {total} 条", flush=True)
    cached = _load_manifest(collection_name, total, label) if collection_name else None
    if cached is not None:
        return cached

    print("[*] 分页扫描 metadata（每页 "
          f"{SCAN_PAGE_SIZE}，只取 metadatas）…", flush=True)
    docs: dict[str, dict] = {}
    offset = 0
    while True:
        print(f"    … 扫描进度 {offset}/{total}", flush=True)
        result = collection.get(
            include=["metadatas"],
            limit=SCAN_PAGE_SIZE,
            offset=offset,
        )
        ids = result.get("ids") or []
        metas = result.get("metadatas") or []
        if not ids:
            break
        for node_id, meta in zip(ids, metas):
            meta = meta or {}
            rel = meta.get("source_path") or meta.get("document_id") or ""
            if not rel or rel == "None":
                rel = "__legacy__"
            info = docs.setdefault(
                rel,
                {
                    "doc_hash": meta.get("doc_hash", ""),
                    "chunker": meta.get("chunker", ""),
                    "doc_chunk_count": int(meta.get("doc_chunk_count") or 0),
                    "ids": [],
                },
            )
            info["ids"].append(node_id)
            if meta.get("doc_hash"):
                info["doc_hash"] = meta.get("doc_hash")
            if meta.get("chunker"):
                info["chunker"] = meta.get("chunker")
            if meta.get("doc_chunk_count"):
                info["doc_chunk_count"] = int(meta.get("doc_chunk_count"))
        offset += len(ids)
        if len(ids) < SCAN_PAGE_SIZE:
            break
    print(f"[+] 扫描完成：{offset} chunks / {len(docs)} 个文档", flush=True)
    return docs


def delete_doc_chunks(collection, source_path: str, extra_ids: list[str] | None = None) -> int:
    """先按文档删光该文件的全部 chunk，再写入新块。"""
    ids: list[str] = list(extra_ids or [])
    for where in (
        {"source_path": source_path},
        {"document_id": source_path},
        {"ref_doc_id": source_path},
    ):
        try:
            got = collection.get(where=where, include=[])
            ids.extend(got.get("ids") or [])
        except Exception:
            continue
    ids = list(dict.fromkeys(ids))
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)


def collection_needs_reset(collection, embed_model: str) -> str | None:
    """只看 collection.metadata，避免 get(embeddings) 在云端卡住。"""
    meta = collection.metadata or {}
    if str(meta.get("vector_dim")) != str(EMBED_DIM):
        return f"vector_dim={meta.get('vector_dim')} != {EMBED_DIM}"
    if meta.get("embed_model") and meta.get("embed_model") != embed_model:
        return f"embed_model={meta.get('embed_model')} != {embed_model}"
    return None


def sync_chroma_index(
    *,
    files: list[Path],
    data_dir: Path,
    chroma_client,
    collection_name: str,
    embed_model_name: str,
    read_document,
    splitter: MarkdownParentChildSplitter | None = None,
    persist_label: str = "default",
    insert_batch_size: int = INSERT_BATCH_SIZE,
) -> VectorStoreIndex:
    """按文件增量同步 Chroma：未改跳过；改了则删除该文档全部 chunk 再重建。"""
    splitter = splitter or MarkdownParentChildSplitter()
    probe_embedding_dim()

    print("[*] 获取 collection…", flush=True)
    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"[+] 已找到 collection {collection_name}", flush=True)
    except Exception as exc:
        print(f"[*] 无现成 collection（{exc}），稍后创建", flush=True)
        collection = None

    if collection is not None:
        print("[*] 检查 collection 元数据（维度/模型）…", flush=True)
        reason = collection_needs_reset(collection, embed_model_name)
        if reason:
            print(f"[*] collection 与 1024 维/当前模型不兼容（{reason}），整库重建", flush=True)
            chroma_client.delete_collection(collection_name)
            collection = None
        else:
            print("[+] collection 元数据匹配 1024 维", flush=True)

    if collection is None:
        print("[*] 创建 collection…", flush=True)
        collection = chroma_client.get_or_create_collection(
            collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embed_model": embed_model_name,
                "vector_dim": EMBED_DIM,
                "chunker": CHUNKER_VERSION,
            },
        )
        print(f"[+] 已创建 collection {collection_name}（cosine / {EMBED_DIM} 维）", flush=True)

    indexed = scan_indexed_docs(collection, collection_name, persist_label)
    current: dict[str, Path] = {}
    fingerprints: dict[str, tuple[str, str]] = {}
    for path in files:
        rel = path.relative_to(data_dir).as_posix()
        current[rel] = path
        fingerprints[rel] = file_fingerprint(path)

    removed = [rel for rel in indexed if rel not in current]
    for rel in removed:
        n = delete_doc_chunks(collection, rel, extra_ids=indexed[rel].get("ids"))
        print(f"    [-] 文档已删除，清掉 chunk: {rel} ({n})", flush=True)
        indexed.pop(rel, None)

    to_rebuild: list[str] = []
    skipped = 0
    for rel, path in current.items():
        doc_hash, _mtime = fingerprints[rel]
        old = indexed.get(rel)
        expected = int(old.get("doc_chunk_count") or 0) if old else 0
        n_ids = len((old or {}).get("ids") or [])
        if old and expected > 0:
            complete = n_ids >= expected
        else:
            # 旧写入没有 doc_chunk_count：hash 对上且已有块，视为完整
            complete = bool(old) and n_ids > 0
        if (
            old
            and complete
            and old.get("doc_hash") == doc_hash
            and old.get("chunker") == CHUNKER_VERSION
        ):
            skipped += 1
            continue
        to_rebuild.append(rel)

    print(f"[*] 文档同步：未变化 {skipped}，需重建 {len(to_rebuild)}，删除 {len(removed)}", flush=True)
    if to_rebuild:
        preview = "、".join(to_rebuild[:8])
        more = f" 等 {len(to_rebuild)} 个" if len(to_rebuild) > 8 else ""
        print(f"[*] 待重建：{preview}{more}", flush=True)
    runlog.set_sync(
        skipped=skipped,
        rebuild=len(to_rebuild),
        deleted_files=len(removed),
        rebuild_list=to_rebuild,
        deleted_list=removed,
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        insert_batch_size=insert_batch_size,
    )

    for seq, rel in enumerate(to_rebuild, start=1):
        path = current[rel]
        doc_hash, doc_mtime = fingerprints[rel]
        print(f"[*] ({seq}/{len(to_rebuild)}) 处理 {rel} …", flush=True)
        old = indexed.get(rel)
        if old:
            n = delete_doc_chunks(collection, rel, extra_ids=old.get("ids"))
            print(f"    [*] 先删除旧 chunk {rel}: {n} 条", flush=True)
        try:
            text = read_document(path)
        except Exception as exc:
            print(f"    [!] 跳过 {rel}: {exc}", flush=True)
            continue
        if not text or not text.strip():
            print(f"    [!] 空文档 {rel}", flush=True)
            continue
        print(f"    [*] 切分中（正文 {len(text)} 字）…", flush=True)
        document = Document(
            text=text,
            metadata={
                "file_name": path.name,
                "source_path": rel,
                "topic": _topic_for(rel),
                "doc_hash": doc_hash,
                "doc_mtime": doc_mtime,
            },
            excluded_llm_metadata_keys=_META_EXCLUDE_LLM,
            excluded_embed_metadata_keys=_META_EXCLUDE_EMBED,
        )
        raw_nodes = splitter.get_nodes_from_documents([document])
        nodes = stamp_nodes(
            raw_nodes,
            path=path,
            data_dir=data_dir,
            doc_hash=doc_hash,
            doc_mtime=doc_mtime,
        )
        if not nodes:
            print(f"    [!] 未切出 chunk: {rel}", flush=True)
            runlog.file_result(path=rel, status="empty", chunks=0)
            continue
        chars = [len(n.get_content()) for n in nodes]
        print(f"    [*] 切出 {len(nodes)} chunks，开始 embedding + 上传", flush=True)
        insert_nodes_reliable(index, nodes, rel, batch_size=insert_batch_size)
        print(f"    [+] {rel}  →  {len(nodes)} chunks  hash={doc_hash[:8]}  mtime={doc_mtime}", flush=True)
        indexed[rel] = {
            "doc_hash": doc_hash,
            "chunker": CHUNKER_VERSION,
            "doc_chunk_count": len(nodes),
            "ids": [n.node_id for n in nodes],
        }
        runlog.file_result(
            path=rel,
            status="ok",
            chunks=len(nodes),
            doc_hash=doc_hash,
            doc_mtime=doc_mtime,
            chunk_chars_max=max(chars),
            chunk_chars_avg=int(sum(chars) / len(chars)),
        )

    total = collection.count()
    print(f"[+] Chroma 同步完成，当前 {total} 个 chunk（{EMBED_DIM} 维）", flush=True)
    try:
        _save_manifest(collection_name, collection, indexed, persist_label)
        print(f"[+] 已写入本地清单 {_manifest_path(persist_label).name}", flush=True)
    except Exception as exc:
        print(f"[!] 写本地清单失败: {exc}", flush=True)
    runlog.set_sync(total_chunks=total, vector_dim=EMBED_DIM)
    return index


def faiss_write_index(index, dest: Path) -> None:
    """FAISS 的 C fopen 不认 Windows 中文路径，先写到 ASCII 临时文件再搬过去。"""
    import faiss
    import tempfile

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, tmp_path)
        shutil.copy2(tmp_path, str(dest))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def faiss_read_index(src: Path):
    import faiss
    import shutil
    import tempfile

    src = Path(src)
    fd, tmp_path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        shutil.copy2(str(src), tmp_path)
        return faiss.read_index(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _open_existing_faiss(persist_dir: Path, docstore, faiss_path: Path, index_store_path: Path) -> VectorStoreIndex:
    from llama_index.core import load_index_from_storage
    from llama_index.core.storage.index_store import SimpleIndexStore
    from llama_index.vector_stores.faiss import FaissVectorStore

    raw = faiss_read_index(faiss_path)
    vector_store = FaissVectorStore(faiss_index=raw)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
        index_store=SimpleIndexStore.from_persist_path(str(index_store_path)),
    )
    index = load_index_from_storage(storage_context)
    ntotal = getattr(raw, "ntotal", 0)
    print(f"[+] 已加载 FAISS：{ntotal} 条向量，docstore {len(docstore.docs)} 个节点", flush=True)
    runlog.set_sync(total_chunks=len(docstore.docs), vector_dim=EMBED_DIM, loaded_existing=True)
    return index


def _embeddings_from_docstore_json(path: Path) -> dict[str, list]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    inner = data.get("docstore/data") or {}
    out: dict[str, list] = {}
    for nid, rec in inner.items():
        payload = rec.get("__data__") if isinstance(rec, dict) else None
        if not isinstance(payload, dict):
            continue
        emb = payload.get("embedding")
        if emb:
            out[str(nid)] = emb
    return out


def _attach_stored_embeddings(docstore, docstore_path: Path) -> int:
    stored = _embeddings_from_docstore_json(docstore_path)
    n = 0
    for node in docstore.docs.values():
        if getattr(node, "embedding", None):
            continue
        emb = stored.get(node.node_id)
        if emb:
            node.embedding = emb
            n += 1
    return n


def _embed_nodes(nodes: list[BaseNode]) -> None:
    if not nodes:
        return
    texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    embed_model = Settings.embed_model
    if hasattr(embed_model, "get_text_embedding_batch"):
        vecs = embed_model.get_text_embedding_batch(texts, show_progress=True)
    else:
        vecs = [embed_model.get_text_embedding(t) for t in texts]
    for node, vec in zip(nodes, vecs):
        node.embedding = vec


def sync_faiss_index(
    *,
    files: list[Path],
    data_dir: Path,
    persist_dir: Path,
    read_document,
    splitter: MarkdownParentChildSplitter | None = None,
) -> VectorStoreIndex:
    """本地 FAISS 增量同步：hash 未变跳过；变了则删该文档全部 chunk 再 embedding。"""
    import faiss
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.vector_stores.faiss import FaissVectorStore

    splitter = splitter or MarkdownParentChildSplitter()
    persist_dir.mkdir(parents=True, exist_ok=True)
    dim = probe_embedding_dim()
    docstore_path = persist_dir / "docstore.json"
    manifest_path = persist_dir / "manifest.json"
    faiss_path = persist_dir / "index.faiss"

    if docstore_path.is_file():
        docstore = SimpleDocumentStore.from_persist_path(str(docstore_path))
    else:
        docstore = SimpleDocumentStore()
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    current: dict[str, Path] = {}
    fingerprints: dict[str, tuple[str, str]] = {}
    for path in files:
        rel = path.relative_to(data_dir).as_posix()
        current[rel] = path
        fingerprints[rel] = file_fingerprint(path)

    removed = [rel for rel in list(manifest) if rel not in current]
    for rel in removed:
        for nid in manifest[rel].get("node_ids") or []:
            try:
                docstore.delete_document(nid, raise_error=False)
            except Exception:
                pass
        print(f"    [-] 文档已删除，清掉 chunk: {rel}", flush=True)
        manifest.pop(rel, None)

    to_rebuild: list[str] = []
    skipped = 0
    for rel, path in current.items():
        doc_hash, _mtime = fingerprints[rel]
        old = manifest.get(rel) or {}
        expected = int(old.get("doc_chunk_count") or 0)
        n_ids = len(old.get("node_ids") or [])
        complete = (expected > 0 and n_ids >= expected) or (expected == 0 and n_ids > 0)
        if complete and old.get("doc_hash") == doc_hash and old.get("chunker") == CHUNKER_VERSION:
            skipped += 1
            continue
        to_rebuild.append(rel)

    print(f"[*] 索引同步：未变化 {skipped}，需重建 {len(to_rebuild)}，删除 {len(removed)}", flush=True)
    runlog.set_sync(skipped=skipped, rebuild=len(to_rebuild), deleted_files=len(removed), rebuild_list=to_rebuild)

    index_store_path = persist_dir / "index_store.json"
    if not to_rebuild and not removed and faiss_path.is_file() and index_store_path.is_file():
        print("[+] 文档未变化，直接加载已有索引", flush=True)
        return _open_existing_faiss(persist_dir, docstore, faiss_path, index_store_path)

    for seq, rel in enumerate(to_rebuild, start=1):
        path = current[rel]
        doc_hash, doc_mtime = fingerprints[rel]
        print(f"[*] ({seq}/{len(to_rebuild)}) 处理 {rel} …", flush=True)
        old = manifest.get(rel)
        if old:
            for nid in old.get("node_ids") or []:
                try:
                    docstore.delete_document(nid, raise_error=False)
                except Exception:
                    pass
            print(f"    [*] 已删除旧 chunk {len(old.get('node_ids') or [])} 条", flush=True)
        try:
            text = read_document(path)
        except Exception as exc:
            print(f"    [!] 跳过 {rel}: {exc}", flush=True)
            continue
        if not text or not text.strip():
            print(f"    [!] 空文档 {rel}", flush=True)
            continue
        print(f"    [*] 切分中（正文 {len(text)} 字）…", flush=True)
        document = Document(
            text=text,
            metadata={
                "file_name": path.name,
                "source_path": rel,
                "topic": _topic_for(rel),
                "doc_hash": doc_hash,
                "doc_mtime": doc_mtime,
            },
            excluded_llm_metadata_keys=_META_EXCLUDE_LLM,
            excluded_embed_metadata_keys=_META_EXCLUDE_EMBED,
        )
        nodes = stamp_nodes(
            splitter.get_nodes_from_documents([document]),
            path=path,
            data_dir=data_dir,
            doc_hash=doc_hash,
            doc_mtime=doc_mtime,
        )
        if not nodes:
            print(f"    [!] 未切出 chunk: {rel}", flush=True)
            continue
        print(f"    [*] 切出 {len(nodes)} chunks，开始 embedding", flush=True)
        _embed_nodes(nodes)
        for node in nodes:
            docstore.add_documents([node], allow_update=True)
        manifest[rel] = {
            "doc_hash": doc_hash,
            "doc_mtime": doc_mtime,
            "chunker": CHUNKER_VERSION,
            "doc_chunk_count": len(nodes),
            "node_ids": [n.node_id for n in nodes],
        }
        print(f"    [+] {rel} → {len(nodes)} chunks  hash={doc_hash[:8]}  mtime={doc_mtime}", flush=True)
        runlog.file_result(path=rel, status="ok", chunks=len(nodes), doc_hash=doc_hash, doc_mtime=doc_mtime)
        docstore.persist(str(docstore_path))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    restored = _attach_stored_embeddings(docstore, docstore_path)
    if restored:
        print(f"[*] 从 docstore 恢复 {restored} 条已存向量", flush=True)
    all_nodes: list[BaseNode] = []
    for node in docstore.docs.values():
        if getattr(node, "embedding", None):
            all_nodes.append(node)
    if not all_nodes:
        raise RuntimeError("没有带 embedding 的节点，无法构建 FAISS")

    print(f"[*] 用已存向量重建 FAISS（{len(all_nodes)} 条，不重复调用 embedding）…", flush=True)
    # Embeddings are L2-normalized at the embed wrapper, so L2 distance ≈ cosine.
    raw_faiss = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=raw_faiss)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        store_nodes_override=True,
        insert_batch_size=64,
    )
    faiss_write_index(vector_store.client, faiss_path)
    docstore.persist(str(docstore_path))
    if hasattr(storage_context, "index_store") and storage_context.index_store is not None:
        storage_context.index_store.persist(str(persist_dir / "index_store.json"))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] FAISS 同步完成：{len(all_nodes)} chunks，{EMBED_DIM} 维，已写入 {persist_dir}", flush=True)
    runlog.set_sync(total_chunks=len(all_nodes), vector_dim=EMBED_DIM)
    return index

