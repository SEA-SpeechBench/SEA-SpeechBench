"""
Base classes and utilities for SpeechBench metrics system.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

# =========================
# Optional dependencies & probes
# =========================

# jiwer (WER/CER)
try:
    from jiwer import compute_measures, wer  # keep 'wer' for compatibility
    JIWER_AVAILABLE = True
    try:
        _probe = compute_measures("a b", "a b")
        # Accept both dict-like and object-like results
        if not isinstance(_probe, dict) and not hasattr(_probe, "as_dict") and not hasattr(_probe, "to_dict"):
            logger.warning("jiwer available but returned an unknown type; enabling fallback path")
    except Exception as e:
        logger.warning(f"jiwer available but test call failed: {e}; enabling fallback path when needed")
except ImportError:
    JIWER_AVAILABLE = False
    logger.warning("jiwer not available, WER/CER metrics will use fallback approximations")

# nltk (BLEU)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using simplified BLEU calculation")

# sacrebleu
try:
    from sacrebleu import BLEU
    SACREBLEU_AVAILABLE = True
except Exception:
    SACREBLEU_AVAILABLE = False
    logger.warning("SacreBLEU not available, using fallback BLEU calculation")

# rouge_score
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    ROUGE_AVAILABLE = False
    logger.warning("Rouge-score not available, ROUGE metrics will not be available")


# =========================
# Helpers for jiwer compatibility
# =========================

def _measures_to_dict(measures: Any) -> Dict[str, Any]:
    """
    Normalize jiwer measures to a dict across versions:
    - v2: dict
    - v3+: Measure-like object with .as_dict() or .to_dict()
    """
    if isinstance(measures, dict):
        return measures
    for attr in ("as_dict", "to_dict"):
        fn = getattr(measures, attr, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    # Last resort: try attribute access
    out = {}
    for key in ("substitutions", "deletions", "insertions", "hits", "correct", "wer", "mer", "wil"):
        if hasattr(measures, key):
            out[key] = getattr(measures, key)
    return out


def _extract_counts_from_measures(measures_dict: Dict[str, Any],
                                  ref_len_fallback: int) -> Tuple[int, int, int, int]:
    """
    Return (S, I, D, N) where N is reference length.
    Prefer N = S + D + H (hits/correct) when available; otherwise fall back.
    """
    S = int(measures_dict.get("substitutions", 0) or 0)
    I = int(measures_dict.get("insertions", 0) or 0)
    D = int(measures_dict.get("deletions", 0) or 0)

    H = measures_dict.get("hits", measures_dict.get("correct", None))
    if H is not None:
        try:
            H = int(H or 0)
        except Exception:
            H = 0

    if H is not None:
        N = S + D + H
    else:
        N = ref_len_fallback

    return S, I, D, N


# =========================
# Metric base
# =========================

class BaseMetric(ABC):
    """Base class for all evaluation metrics"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def compute(self, predictions: List[str], references: List[str],
                questions: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the metric score.

        Args:
            predictions: List of model predictions
            references: List of reference answers
            questions: Optional list of questions (for some metrics)

        Returns:
            Tuple of (score, details_dict)
        """
        pass

    def __str__(self):
        return f"{self.name}: {self.description}"
