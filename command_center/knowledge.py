from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .contracts import EvidenceItem


DOMAIN_TERMS = (
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
    "停车",
    "重派",
    "资源预约",
    "数字孪生",
)

QUERY_EXPANSIONS = {
    "坏了": "车辆故障 安全停车",
    "坏车": "车辆故障 安全停车",
    "堵车": "通道拥堵 资源预约",
    "封路": "通道封闭",
    "插单": "紧急任务 调度",
    "卡住": "等待 死锁 拥堵",
    "恢复": "故障恢复 死锁恢复 安全停车",
    "改路线": "路径规划 资源预约",
}


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    text: str


class KnowledgeBase:
    """Small-corpus hybrid retrieval with deterministic sparse and vector scores."""

    method = "hybrid-bm25-char-vector-v1"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.chunks = self._load()
        self._tokens = [Counter(self._terms(self._document(chunk))) for chunk in self.chunks]
        self._lengths = [sum(row.values()) for row in self._tokens]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._document_frequency = Counter(
            term for row in self._tokens for term in row.keys()
        )
        self._vectors = [self._vector(self._document(chunk)) for chunk in self.chunks]

    def _load(self) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for path in sorted(self.root.rglob("*.md")):
            raw = path.read_text(encoding="utf-8")
            source = path.relative_to(self.root).as_posix()
            title = path.stem
            current_title = title
            body: list[str] = []
            for line in raw.splitlines():
                if line.startswith("## "):
                    if body:
                        chunks.append(
                            self._chunk(source, current_title, "\n".join(body).strip())
                        )
                    current_title = line[3:].strip()
                    body = []
                elif not line.startswith("# "):
                    body.append(line)
            if body:
                chunks.append(
                    self._chunk(source, current_title, "\n".join(body).strip())
                )
        return [chunk for chunk in chunks if chunk.text]

    @staticmethod
    def _chunk(source: str, title: str, text: str) -> KnowledgeChunk:
        digest = hashlib.sha256(f"{source}\0{title}".encode("utf-8")).hexdigest()[:16]
        return KnowledgeChunk(
            chunk_id=f"kb-{digest}", source=source, title=title, text=text
        )

    @staticmethod
    def _document(chunk: KnowledgeChunk) -> str:
        return f"{chunk.title}\n{chunk.text}"

    @staticmethod
    def _expanded_query(query: str) -> str:
        expansions = [value for key, value in QUERY_EXPANSIONS.items() if key in query]
        return " ".join([query, *expansions]).strip()

    @staticmethod
    def _terms(text: str) -> list[str]:
        lowered = text.lower()
        terms = re.findall(r"[a-z0-9:_-]{2,}", lowered)
        for sequence in re.findall(r"[\u4e00-\u9fff]+", lowered):
            if len(sequence) == 1:
                terms.append(sequence)
                continue
            terms.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
            if len(sequence) >= 3:
                terms.extend(
                    sequence[index : index + 3] for index in range(len(sequence) - 2)
                )
        terms.extend(term for term in DOMAIN_TERMS if term in lowered)
        return terms

    @staticmethod
    def _vector(text: str, dimensions: int = 384) -> dict[int, float]:
        normalized = re.sub(r"\s+", "", text.lower())
        features: list[str] = []
        for width in (2, 3):
            features.extend(
                normalized[index : index + width]
                for index in range(max(0, len(normalized) - width + 1))
            )
        counts: Counter[int] = Counter()
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            counts[int.from_bytes(digest, "big") % dimensions] += 1
        norm = math.sqrt(sum(value * value for value in counts.values()))
        return {
            index: value / norm for index, value in counts.items()
        } if norm else {}

    def _bm25(self, query_terms: list[str], index: int) -> float:
        if not query_terms or not self.chunks:
            return 0.0
        frequencies = self._tokens[index]
        document_length = self._lengths[index]
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in set(query_terms):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequency.get(term, 0)
            inverse_frequency = math.log(
                1
                + (len(self.chunks) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(self._average_length, 1)
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        return score

    @staticmethod
    def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return max(
            0.0,
            sum(value * right.get(index, 0.0) for index, value in left.items()),
        )

    def search(self, query: str, limit: int = 3) -> list[EvidenceItem]:
        expanded = self._expanded_query(query)
        query_terms = self._terms(expanded)
        query_vector = self._vector(expanded)
        sparse_scores = [
            self._bm25(query_terms, index) for index in range(len(self.chunks))
        ]
        sparse_peak = max(sparse_scores, default=0.0)
        ranked: list[tuple[float, float, float, KnowledgeChunk]] = []
        for index, chunk in enumerate(self.chunks):
            sparse = sparse_scores[index] / sparse_peak if sparse_peak else 0.0
            vector = self._cosine(query_vector, self._vectors[index])
            title_terms = set(self._terms(chunk.title))
            title_overlap = len(title_terms.intersection(query_terms)) / max(
                len(title_terms), 1
            )
            score = min(
                1.0, 0.68 * sparse + 0.24 * vector + 0.08 * title_overlap
            )
            if score >= 0.04:
                ranked.append((score, sparse, vector, chunk))
        ranked.sort(
            key=lambda item: (-item[0], -item[1], -item[2], item[3].source)
        )
        return [
            EvidenceItem(
                source=chunk.source,
                title=chunk.title,
                detail=chunk.text[:360],
                chunkId=chunk.chunk_id,
                score=round(score, 6),
                retrievalMethod=self.method,
            )
            for score, _, _, chunk in ranked[: max(1, limit)]
        ]

    def stats(self) -> dict[str, object]:
        return {
            "chunkCount": len(self.chunks),
            "sourceCount": len({chunk.source for chunk in self.chunks}),
            "retrievalMethod": self.method,
        }
