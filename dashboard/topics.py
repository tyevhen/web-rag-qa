"""Topic discovery over stored chunk embeddings — local, no LLM calls."""
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

_NGRAM = 4  # words; at a 30% share only verbatim repeats (banners, footers, menus) reach it


def _ngrams(words: list[str]) -> list[str]:
    return [" ".join(words[i : i + _NGRAM]) for i in range(len(words) - _NGRAM + 1)]


def _repeated_spans(words: list[str], frequent: set[str]) -> set[str]:
    """Merge runs of consecutive frequent n-grams back into the full repeated phrase."""
    spans: set[str] = set()
    grams = _ngrams(words)
    i = 0
    while i < len(grams):
        if grams[i] not in frequent:
            i += 1
            continue
        j = i
        while j + 1 < len(grams) and grams[j + 1] in frequent:
            j += 1
        spans.add(" ".join(words[i : j + _NGRAM]))
        i = j + 1
    return spans


def boilerplate_phrases(
    contents: list[str], sources: list[str], min_share: float = 0.3, top: int = 5
) -> dict[str, list[tuple[str, float]]]:
    """Phrases repeated verbatim in at least min_share of one source's chunks.

    Phrase-level rather than line-level: the chunker's word-level fallback joins lines with spaces,
    so a page banner usually sits mid-line.
    """
    by_source: dict[str, list[list[str]]] = defaultdict(list)
    for text, source in zip(contents, sources):
        by_source[source].append(text.split())
    result: dict[str, list[tuple[str, float]]] = {}
    for source, docs in by_source.items():
        gram_counts = Counter(g for words in docs for g in set(_ngrams(words)))
        frequent = {g for g, n in gram_counts.items() if n / len(docs) >= min_share}
        span_counts = Counter(span for words in docs for span in _repeated_spans(words, frequent))
        result[source] = [
            (span, n / len(docs)) for span, n in span_counts.most_common(top) if n / len(docs) >= min_share
        ]
    return result


def cluster(embeddings: np.ndarray, k: int) -> np.ndarray:
    # Embeddings are L2-normalized, so euclidean k-means ranks like cosine similarity
    return KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(embeddings)


def label_clusters(contents: list[str], labels: np.ndarray, top_n: int = 4) -> dict[int, str]:
    """Name each cluster by its highest mean TF-IDF terms."""
    # max_df drops terms present in most chunks (e.g. repeated page banners) so they never become labels
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.4, min_df=2, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    matrix: Any = vectorizer.fit_transform(contents)  # scipy sparse; sklearn's stub types it loosely
    terms = vectorizer.get_feature_names_out()
    names: dict[int, str] = {}
    for cid in sorted(set(labels.tolist())):
        mean = np.asarray(matrix[labels == cid].mean(axis=0)).ravel()
        names[cid] = ", ".join(str(terms[i]) for i in mean.argsort()[::-1][:top_n])
    return names


def project_2d(embeddings: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=0).fit_transform(embeddings)
