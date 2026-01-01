#!/usr/bin/env python3
# haze/__init__.py — package initialization

from .haze import (
    Vocab,
    PostGPT,
    RRPRAMHead,
    ReweightHead,  # backwards compat alias
    ContentHead,
    HybridHead,
    Block,
    load_corpus,
    build_model_from_text,
)

# Import co-occurrence field
from .cooccur import CooccurField

# Import RRPRAM tokenizer if sentencepiece available
try:
    from .rrpram import RRPRAMVocab, analyze_vocab, demo_tokenization
    HAS_RRPRAM = True
except ImportError:
    HAS_RRPRAM = False

# Backwards compatibility aliases
Haze = PostGPT
ReweightGPT = PostGPT

__all__ = [
    'Vocab',
    'PostGPT',
    'Haze',  # alias
    'ReweightGPT',  # backwards compat
    'RRPRAMHead',
    'ReweightHead',  # backwards compat alias for RRPRAMHead
    'ContentHead',
    'HybridHead',
    'Block',
    'load_corpus',
    'build_model_from_text',
    # Co-occurrence field
    'CooccurField',
    # RRPRAM tokenizer (if available)
    'RRPRAMVocab',
    'HAS_RRPRAM',
]
