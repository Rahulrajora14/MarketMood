"""Item-level sentiment scores.

Two scorers:
* FinBERT (ProsusAI/finbert): transformer pretrained on financial text,
  giving P(positive), P(negative), P(neutral) per item.
* Loughran-McDonald word lists: counts of finance-specific positive and
  negative words. Cheap and fully interpretable; used as the comparator.

FinBERT inference is the only expensive step in the project, so results are
cached by a hash of the input text and never recomputed.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LM_PATH = Path(__file__).resolve().parents[1] / "resources" / "lm_dictionary.csv"
_TOKEN = re.compile(r"[a-z]+")


def text_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=10).hexdigest()


def compose_text(title: pd.Series, lead: pd.Series | None, use_lead: bool) -> pd.Series:
    title = title.fillna("").astype(str).str.strip()
    if not use_lead or lead is None:
        return title
    lead = lead.fillna("").astype(str).str.strip()
    return (title + ". " + lead).where(lead.str.len() > 0, title)


# ------------------------------------------------------------------ LM lexicon
class LMScorer:
    def __init__(self, path: Path = LM_PATH):
        d = pd.read_csv(path)
        self.pos = set(d.loc[d.positive == 1, "word"])
        self.neg = set(d.loc[d.negative == 1, "word"])
        self.unc = set(d.loc[d.uncertainty == 1, "word"])

    def score(self, texts: pd.Series) -> pd.DataFrame:
        pos, neg, unc, n = [], [], [], []
        for t in texts.fillna("").str.lower():
            toks = _TOKEN.findall(t)
            p = sum(w in self.pos for w in toks)
            q = sum(w in self.neg for w in toks)
            pos.append(p); neg.append(q)
            unc.append(sum(w in self.unc for w in toks)); n.append(len(toks))
        pos, neg = np.array(pos), np.array(neg)
        return pd.DataFrame({
            "lm_pos": pos, "lm_neg": neg, "lm_unc": unc, "lm_tokens": n,
            # net tone in [-1, 1]; zero when no sentiment words are present
            "lm_tone": (pos - neg) / np.maximum(pos + neg, 1),
        }, index=texts.index)


# ------------------------------------------------------------------ FinBERT
class FinBERTScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", max_length: int = 64,
                 batch_size: int = 64, device: str | None = None, quantize: bool = False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if quantize and self.device == "cpu":
            # int8 weights for the linear layers: ~2-3x faster on CPU, scores move by ~0.01
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.to(self.device)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.max_length, self.batch_size = max_length, batch_size
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.idx = [labels["positive"], labels["negative"], labels["neutral"]]

    def score_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3))
        # batches of similar length waste far less compute on padding
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 3), dtype=np.float32)
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                idx = order[i:i + self.batch_size]
                enc = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                out[idx] = probs[:, self.idx]
        return out


def score_with_cache(texts: pd.Series, cache_path: Path, scorer_factory,
                     chunk: int = 4096) -> pd.DataFrame:
    """Score texts with FinBERT, reusing and extending a Parquet cache.

    The cache is flushed after every chunk so an interrupted run resumes
    where it stopped instead of starting over.
    """
    keys = texts.map(text_key)
    cache = pd.read_parquet(cache_path) if cache_path.exists() else \
        pd.DataFrame(columns=["key", "fb_pos", "fb_neg", "fb_neu"])
    done = set(cache["key"])
    todo = pd.DataFrame({"key": keys, "text": texts}).drop_duplicates("key")
    todo = todo[~todo["key"].isin(done)]
    log.info("FinBERT: %d unique texts, %d cached, %d to score",
             keys.nunique(), len(done), len(todo))
    if len(todo):
        scorer = scorer_factory()
        parts = [cache]
        for i in range(0, len(todo), chunk):
            sl = todo.iloc[i:i + chunk]
            p = scorer.score_texts(sl["text"].tolist())
            parts.append(pd.DataFrame({"key": sl["key"].values, "fb_pos": p[:, 0],
                                       "fb_neg": p[:, 1], "fb_neu": p[:, 2]}))
            cache = pd.concat(parts, ignore_index=True)
            parts = [cache]
            cache.to_parquet(cache_path, index=False)
            log.info("FinBERT: scored %d / %d", min(i + chunk, len(todo)), len(todo))
    scores = cache.set_index("key").reindex(keys.values)
    scores.index = texts.index
    scores["fb_polarity"] = scores["fb_pos"] - scores["fb_neg"]
    return scores
