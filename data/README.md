# 知识库

默认语料是 `ai-agent-book/`：《AI Agent 入门》开源书（Apache-2.0，保留署名）。Agent 只根据这里的 Markdown 作答。

启动时系统会递归扫描 `data/` 下全部 `.md` / `.txt` / `.pdf` / `.docx`，按文件内容 SHA-256 增量同步向量索引。`SOURCE.md` / `LICENSE.md` / `README.md` 不进向量库。配图（`images/*.svg`）不向量化，由视觉旁路按需渲染。

## 可选：再加自己的 Markdown

1. 新建目录，例如 `data/my-notes/`
2. 放入带 `#` / `##` / `###` 标题的 Markdown
3. 可选配图放到 `data/my-notes/images/`，正文用：

```markdown
【图：图1-1 系统架构】
配图文件：images/arch.svg
图中文字：Gateway；Retriever；Reranker
```

4. 执行 `python main.py ingest`，只有 hash 变化的文件会重新 embedding。
