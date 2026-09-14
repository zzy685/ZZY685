# -*- coding: utf-8 -*-
"""
论文查重核心模块
提供文本预处理、分词、相似度计算等功能
"""

import math
import re
from html.parser import HTMLParser
from collections import Counter
from typing import List

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


class _HTMLTextExtractor(HTMLParser):
    """从HTML中提取纯文本的解析器"""
    def __init__(self):
        super().__init__()
        self._text_parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head', 'noscript'):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head', 'noscript'):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._text_parts.append(data)

    def get_text(self) -> str:
        return ''.join(self._text_parts)


def extract_text_from_html(html: str) -> str:
    """
    从HTML内容中提取纯文本
    如果输入不是HTML，则原样返回
    """
    if not html:
        return html
    # 简单检测是否是HTML
    if not re.search(r'<!DOCTYPE\s+html|<html[\s>]', html, re.IGNORECASE):
        return html
    try:
        parser = _HTMLTextExtractor()
        parser.feed(html)
        parser.close()
        return parser.get_text()
    except Exception:
        return html


# 中文标点符号集合
PUNCTUATION = set(
    '，。！？、；：""''（）【】《》—…·'
    ',.!?;:()[]{}<>-_\'\"`~@#$%^&*+=|\\/'
    '\n\r\t '
)


def clean_text(text: str) -> str:
    """
    文本预处理：去除标点符号、空白字符，统一为小写
    """
    cleaned = ''.join(ch for ch in text if ch not in PUNCTUATION)
    return cleaned.lower()


def segment_words(text: str) -> List[str]:
    """
    对文本进行分词
    优先使用jieba分词，若不可用则退化为单字分词
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if JIEBA_AVAILABLE:
        words = list(jieba.cut(cleaned, cut_all=False))
        words = [w for w in words if w.strip()]
        return words
    else:
        return list(cleaned)


def get_char_ngrams(text: str, n: int = 2) -> List[str]:
    """
    获取字符级n-gram列表
    """
    cleaned = clean_text(text)
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)]


def cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    """
    计算两个词频向量的余弦相似度
    优化：在一次遍历中同时计算点积和模长，减少遍历次数
    """
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1

    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for key, val in vec1.items():
        norm1_sq += val * val
        if key in vec2:
            dot_product += val * vec2[key]

    for val in vec2.values():
        norm2_sq += val * val

    norm1 = math.sqrt(norm1_sq)
    norm2 = math.sqrt(norm2_sq)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    计算两个集合的Jaccard相似度
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def word_level_similarity(text1: str, text2: str) -> float:
    """
    词级相似度：分词后计算词频向量的余弦相似度
    """
    words1 = segment_words(text1)
    words2 = segment_words(text2)

    vec1 = Counter(words1)
    vec2 = Counter(words2)

    return cosine_similarity(vec1, vec2)


def ngram_level_similarity(text1: str, text2: str, n: int = 2) -> float:
    """
    n-gram级相似度：字符级n-gram的余弦相似度
    """
    ngrams1 = get_char_ngrams(text1, n)
    ngrams2 = get_char_ngrams(text2, n)

    vec1 = Counter(ngrams1)
    vec2 = Counter(ngrams2)

    return cosine_similarity(vec1, vec2)


def _segment_words_cleaned(cleaned: str) -> List[str]:
    """对已预处理的文本进行分词（内部函数，避免重复clean_text）"""
    if not cleaned:
        return []
    if JIEBA_AVAILABLE:
        words = list(jieba.cut(cleaned, cut_all=False))
        return [w for w in words if w.strip()]
    return list(cleaned)


def _get_char_ngrams_cleaned(cleaned: str, n: int = 2) -> List[str]:
    """对已预处理的文本生成n-gram（内部函数，避免重复clean_text）"""
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)]


def calculate_similarity(orig_text: str, plagiarized_text: str) -> float:
    """
    综合计算两篇论文的重复率
    采用词级余弦相似度与字符级bigram相似度加权平均

    参数:
        orig_text: 原文文本
        plagiarized_text: 抄袭版论文文本

    返回:
        重复率（0.0 ~ 1.0之间的浮点数）
    """
    if not orig_text.strip() or not plagiarized_text.strip():
        return 0.0

    # 提取HTML纯文本（处理从网页保存的测试文件）
    orig_text = extract_text_from_html(orig_text)
    plagiarized_text = extract_text_from_html(plagiarized_text)

    # 统一预处理一次，避免在词级和n-gram计算中重复调用clean_text
    cleaned1 = clean_text(orig_text)
    cleaned2 = clean_text(plagiarized_text)

    if not cleaned1 or not cleaned2:
        return 0.0

    # 词级相似度（权重0.6）
    words1 = _segment_words_cleaned(cleaned1)
    words2 = _segment_words_cleaned(cleaned2)
    word_sim = cosine_similarity(Counter(words1), Counter(words2))

    # 字符级bigram相似度（权重0.4）
    ngrams1 = _get_char_ngrams_cleaned(cleaned1, n=2)
    ngrams2 = _get_char_ngrams_cleaned(cleaned2, n=2)
    bigram_sim = cosine_similarity(Counter(ngrams1), Counter(ngrams2))

    # 加权综合
    final_similarity = 0.6 * word_sim + 0.4 * bigram_sim

    # 确保在[0, 1]范围内
    final_similarity = max(0.0, min(1.0, final_similarity))

    return final_similarity
