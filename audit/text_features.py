from __future__ import annotations

import re
from collections import Counter

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    TfidfVectorizer = None
    cosine_similarity = None


TOKEN_PATTERN = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b")


def _clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _valid_texts(texts: list[str]) -> list[str]:
    return [text for text in (_clean_text(value) for value in texts) if text]


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _ngram_features(text: str) -> list[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    features = list(tokens)
    features.extend(f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1))
    return features


def _build_vectorizer() -> TfidfVectorizer:
    if TfidfVectorizer is None:
        raise ModuleNotFoundError("scikit-learn is not installed")
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1024,
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",
    )


def _fallback_caption_cohesion(texts: list[str]) -> float:
    docs = [set(_ngram_features(text)) for text in texts]
    vals: list[float] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            left = docs[i]
            right = docs[j]
            if not left or not right:
                continue
            union = left | right
            if not union:
                continue
            vals.append(len(left & right) / len(union))
    return float(np.mean(vals)) if vals else 0.0


def _fallback_top_phrases(texts: list[str], top_n: int) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(set(_ngram_features(text)))

    phrases: list[str] = []
    seen: set[str] = set()
    for phrase, _ in counts.most_common():
        phrase = phrase.strip()
        if not phrase or phrase in seen or len(phrase) < 3:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if len(phrases) >= top_n:
            break
    return phrases


def caption_cohesion(texts: list[str]) -> float:
    texts = _valid_texts(texts)
    if len(texts) <= 1:
        return 0.0
    if TfidfVectorizer is None or cosine_similarity is None:
        return _fallback_caption_cohesion(texts)
    try:
        vec = _build_vectorizer().fit_transform(texts)
    except ValueError:
        return 0.0
    sims = cosine_similarity(vec)
    n = sims.shape[0]
    if n <= 1:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    vals = sims[mask]
    return float(np.mean(vals)) if vals.size else 0.0


def top_phrases(texts: list[str], top_n: int = 8) -> list[str]:
    texts = _valid_texts(texts)
    if not texts:
        return []
    if TfidfVectorizer is None:
        return _fallback_top_phrases(texts, top_n)
    try:
        veczr = _build_vectorizer()
        x = veczr.fit_transform(texts)
    except ValueError:
        return []
    scores = np.asarray(x.mean(axis=0)).ravel()
    names = np.asarray(veczr.get_feature_names_out())
    order = np.argsort(-scores)
    phrases: list[str] = []
    seen: set[str] = set()
    for idx in order:
        phrase = names[idx].strip()
        if not phrase or phrase in seen or len(phrase) < 3:
            continue
        seen.add(phrase)
        phrases.append(phrase)
        if len(phrases) >= top_n:
            break
    return phrases
