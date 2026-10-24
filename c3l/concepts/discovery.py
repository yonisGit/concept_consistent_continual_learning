"""
Automatic concept generation (Sec. 3.2), following Label-free CBM
(Oikarinen et al., 2023) with GPT-4o-mini and the original LF-CBM few-shot
prompts *without* the "around" prompt (Appendix A).

Pipeline per class: query the LLM with each prompt -> parse "-item" lines ->
LF-CBM text filters (length, similarity to class names, near-duplicates).
Results are cached to JSON so discovery runs once, offline, before training -
the targets are therefore independent of any model state.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence

from c3l.utils import get_logger

logger = get_logger()

# Prompts from https://github.com/Trustworthy-ML-Lab/Label-free-CBM (GPT_initial_concepts.ipynb)
LFCBM_PROMPTS: Dict[str, str] = {
    "important": (
        'List the most important features for recognizing something as a "goldfish":\n\n'
        "-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\n"
        'List the most important features for recognizing something as a "beerglass":\n\n'
        "-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\n"
        'List the most important features for recognizing something as a "{}":'
    ),
    "superclass": (
        'Give superclasses for the word "tench":\n\n-fish\n-vertebrate\n-animal\n\n'
        'Give superclasses for the word "beer glass":\n\n-glass\n-container\n-object\n\n'
        'Give superclasses for the word "{}":'
    ),
    # "around" is intentionally excluded (paper, Appendix A). Kept for completeness / future work.
    "around": (
        'List the things most commonly seen around a "tench":\n\n-a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\n'
        'List the things most commonly seen around a "beer glass":\n\n-beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\n'
        'List the things most commonly seen around a "{}":'
    ),
}


def parse_items(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] in "-*•":
            line = line[1:].strip()
        elif line[0].isdigit():
            line = line.lstrip("0123456789").lstrip(".)- ").strip()
        else:
            continue
        line = line.strip(" .\"'").lower()
        if line:
            out.append(line)
    return out


class LLMConceptGenerator:
    """Queries an OpenAI chat model (default gpt-4o-mini)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 256, n_samples: int = 2, sleep: float = 0.2):
        from openai import OpenAI  # imported lazily: only needed for discovery

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_samples = n_samples
        self.sleep = sleep

    def query(self, prompt: str) -> List[str]:
        items: List[str] = []
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.n_samples,
                )
                for choice in resp.choices:
                    items += parse_items(choice.message.content or "")
                return items
            except Exception as e:  # rate limits / transient errors
                wait = 2 ** attempt
                logger.warning(f"LLM query failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        return items

    def concepts_for_class(self, class_name: str, prompts: Sequence[str]) -> List[str]:
        concepts: List[str] = []
        for p in prompts:
            concepts += self.query(LFCBM_PROMPTS[p].format(class_name))
            time.sleep(self.sleep)
        seen, uniq = set(), []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq


# --------------------------------------------------------------------------- #
# LF-CBM text filters
# --------------------------------------------------------------------------- #
class _TextSim:
    """Sentence-embedding similarity (all-mpnet-base-v2, as in LF-CBM); difflib fallback."""

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-mpnet-base-v2")
        except Exception:
            logger.warning("sentence-transformers not available; using string similarity for concept filtering.")
            self.model = None

    def matrix(self, a: List[str], b: List[str]):
        if not a or not b:
            return [[0.0] * len(b) for _ in a]
        if self.model is None:
            return [[difflib.SequenceMatcher(None, x, y).ratio() for y in b] for x in a]
        ea = self.model.encode(a, normalize_embeddings=True, convert_to_tensor=True)
        eb = self.model.encode(b, normalize_embeddings=True, convert_to_tensor=True)
        return (ea @ eb.T).cpu().tolist()


def filter_concepts(bank: Dict[str, Dict], class_names: List[str], max_len: int = 30,
                    class_sim_cutoff: float = 0.85, concept_sim_cutoff: float = 0.9) -> Dict[str, Dict]:
    """
    1) drop concepts longer than ``max_len`` characters,
    2) drop concepts too similar to *any* class name (they would leak the label),
    3) drop near-duplicate concepts within a class.
    """
    sim = _TextSim()
    lowered = [c.lower() for c in class_names]
    for key, entry in bank.items():
        cands = [c for c in entry["raw"] if 0 < len(c) <= max_len]
        if cands:
            s_cls = sim.matrix(cands, lowered)
            cands = [c for c, row in zip(cands, s_cls) if max(row) < class_sim_cutoff and c not in lowered]
        kept: List[str] = []
        if cands:
            s_cc = sim.matrix(cands, cands)
            for i, c in enumerate(cands):
                if all(s_cc[i][cands.index(k)] < concept_sim_cutoff for k in kept):
                    kept.append(c)
        entry["concepts"] = kept
    return bank


def discover_concept_bank(class_names: List[str], cfg, cache_path: str, overwrite: bool = False) -> Dict[str, Dict]:
    """
    Returns ``{str(orig_label): {"name": class_name, "raw": [...], "concepts": [...]}}``.
    """
    if os.path.exists(cache_path) and not overwrite:
        with open(cache_path) as f:
            return json.load(f)
    gen = LLMConceptGenerator(model=cfg.concepts.llm_model)
    bank: Dict[str, Dict] = {}
    for i, name in enumerate(class_names):
        raw = gen.concepts_for_class(name, cfg.concepts.prompts)
        bank[str(i)] = {"name": name, "raw": raw, "concepts": raw}
        logger.info(f"[{i + 1}/{len(class_names)}] {name}: {len(raw)} raw concepts")
    bank = filter_concepts(bank, class_names, cfg.concepts.max_concept_length,
                           cfg.concepts.class_sim_cutoff, cfg.concepts.concept_sim_cutoff)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(bank, f, indent=2)
    return bank


def load_concept_bank(path: str) -> Dict[str, Dict]:
    with open(path) as f:
        bank = json.load(f)
    # Accept the simple {class_name: [concepts]} format of the original sketch as well.
    if bank and isinstance(next(iter(bank.values())), list):
        bank = {k: {"name": k, "raw": v, "concepts": v} for k, v in bank.items()}
    return bank


class ConceptLookup:
    """Maps an *original* dataset label to its concept set Psi_y (or Psi_y^* if validated)."""

    def __init__(self, bank: Dict[str, Dict], class_names: List[str], field: str = "concepts"):
        self.class_names = class_names
        self.field = field
        self._by_key = bank
        self._by_name = {_norm(v["name"]): v for v in bank.values()}

    def entry(self, orig_label: int) -> Dict:
        e = self._by_key.get(str(orig_label))
        if e is not None and _norm(e["name"]) == _norm(self.class_names[orig_label]):
            return e
        e = self._by_name.get(_norm(self.class_names[orig_label]))
        if e is None:
            raise KeyError(f"No concepts for class {orig_label} ({self.class_names[orig_label]})")
        return e

    def __call__(self, orig_label: int) -> List[str]:
        e = self.entry(orig_label)
        return list(e.get(self.field) or e.get("concepts") or [])


def _norm(s: str) -> str:
    """Case/underscore-insensitive key; also strips WordNet suffixes like 'tench.n.01'."""
    s = re.sub(r"\.[nvasr]\.\d+$", "", s.strip())
    return " ".join(s.lower().replace("_", " ").split())
