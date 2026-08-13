from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .contracts import EvidenceItem


@dataclass(frozen=True)
class KnowledgeChunk:
    source: str
    title: str
    text: str


class KnowledgeBase:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.chunks = self._load()

    def _load(self) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for path in sorted(self.root.rglob("*.md")):
            raw = path.read_text(encoding="utf-8")
            title = path.stem
            current_title = title
            body: list[str] = []
            for line in raw.splitlines():
                if line.startswith("## "):
                    if body:
                        chunks.append(
                            KnowledgeChunk(
                                source=str(path.relative_to(self.root)),
                                title=current_title,
                                text="\n".join(body).strip(),
                            )
                        )
                    current_title = line[3:].strip()
                    body = []
                elif not line.startswith("# "):
                    body.append(line)
            if body:
                chunks.append(
                    KnowledgeChunk(
                        source=str(path.relative_to(self.root)),
                        title=current_title,
                        text="\n".join(body).strip(),
                    )
                )
        return [chunk for chunk in chunks if chunk.text]

    @staticmethod
    def _terms(query: str) -> set[str]:
        latin = re.findall(r"[a-zA-Z0-9:_-]{2,}", query.lower())
        chinese = [
            token
            for token in (
                "故障",
                "封路",
                "封闭",
                "检修",
                "死锁",
                "倒退",
                "紧急",
                "任务",
                "审批",
                "安全",
                "通道",
                "拥堵",
                "取货",
                "放货",
            )
            if token in query
        ]
        return set(latin + chinese)

    def search(self, query: str, limit: int = 3) -> list[EvidenceItem]:
        terms = self._terms(query)
        ranked: list[tuple[int, KnowledgeChunk]] = []
        for chunk in self.chunks:
            haystack = f"{chunk.title}\n{chunk.text}".lower()
            score = sum(3 if term in chunk.title.lower() else 1 for term in terms if term in haystack)
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].source, item[1].title))
        return [
            EvidenceItem(
                source=chunk.source,
                title=chunk.title,
                detail=chunk.text[:360],
            )
            for _, chunk in ranked[:limit]
        ]

