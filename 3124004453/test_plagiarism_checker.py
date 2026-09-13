# -*- coding: utf-8 -*-
"""
论文查重模块单元测试
至少10个测试用例，覆盖各种边界情况和正常场景
"""

import unittest
import os
import sys
import tempfile

# 添加项目目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plagiarism_checker import (
    clean_text,
    segment_words,
    get_char_ngrams,
    cosine_similarity,
    jaccard_similarity,
    word_level_similarity,
    ngram_level_similarity,
    calculate_similarity,
)
from main import read_file, write_result


class TestCleanText(unittest.TestCase):
    """测试文本预处理功能"""

    def test_remove_punctuation(self):
        """测试1：去除中文标点符号"""
        text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        result = clean_text(text)
        self.assertNotIn("，", result)
        self.assertNotIn("。", result)
        self.assertEqual(result, "今天是星期天天气晴今天晚上我要去看电影")

    def test_remove_whitespace(self):
        """测试2：去除空白字符"""
        text = "hello  world\n\ttest"
        result = clean_text(text)
        self.assertNotIn(" ", result)
        self.assertNotIn("\n", result)
        self.assertNotIn("\t", result)

    def test_empty_input(self):
        """测试3：空字符串输入"""
        self.assertEqual(clean_text(""), "")

    def test_only_punctuation(self):
        """测试4：只有标点符号的输入"""
        self.assertEqual(clean_text("，。！？"), "")


class TestSegmentWords(unittest.TestCase):
    """测试分词功能"""

    def test_normal_segmentation(self):
        """测试5：正常中文分词"""
        text = "今天是星期天天气晴"
        words = segment_words(text)
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)
        # 验证分词结果能还原原始字符（不含标点）
        reconstructed = ''.join(words)
        self.assertEqual(reconstructed, clean_text(text))

    def test_empty_text(self):
        """测试6：空文本分词"""
        self.assertEqual(segment_words(""), [])
        self.assertEqual(segment_words("   "), [])


class TestCharNgrams(unittest.TestCase):
    """测试n-gram生成功能"""

    def test_bigram_generation(self):
        """测试7：bigram生成"""
        text = "abcde"
        ngrams = get_char_ngrams(text, n=2)
        self.assertEqual(ngrams, ["ab", "bc", "cd", "de"])

    def test_short_text(self):
        """测试8：短文本n-gram（长度小于n）"""
        text = "ab"
        ngrams = get_char_ngrams(text, n=3)
        self.assertEqual(ngrams, ["ab"])

    def test_empty_text(self):
        """测试9：空文本n-gram"""
        self.assertEqual(get_char_ngrams("", n=2), [])


class TestCosineSimilarity(unittest.TestCase):
    """测试余弦相似度计算"""

    def test_identical_vectors(self):
        """测试10：完全相同的向量"""
        from collections import Counter
        vec = Counter({"a": 2, "b": 3})
        self.assertAlmostEqual(cosine_similarity(vec, vec), 1.0, places=5)

    def test_orthogonal_vectors(self):
        """测试11：完全正交的向量"""
        from collections import Counter
        vec1 = Counter({"a": 1})
        vec2 = Counter({"b": 1})
        self.assertEqual(cosine_similarity(vec1, vec2), 0.0)

    def test_empty_vector(self):
        """测试12：空向量"""
        from collections import Counter
        self.assertEqual(cosine_similarity(Counter(), Counter({"a": 1})), 0.0)


class TestCalculateSimilarity(unittest.TestCase):
    """测试综合相似度计算（核心功能）"""

    def test_identical_texts(self):
        """测试13：完全相同的文本，重复率应为1.0"""
        text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        sim = calculate_similarity(text, text)
        self.assertAlmostEqual(sim, 1.0, places=2)

    def test_completely_different(self):
        """测试14：完全不同的文本，重复率应很低"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "人工智能是计算机科学的一个分支，它企图了解智能的实质。"
        sim = calculate_similarity(text1, text2)
        self.assertLess(sim, 0.3)

    def test_homework_example(self):
        """测试15：作业示例中的原文与抄袭版"""
        orig = "今天是星期天，天气晴，今天晚上我要去看电影。"
        plagiarized = "今天是周天，天气晴朗，我晚上要去看电影。"
        sim = calculate_similarity(orig, plagiarized)
        # 两者有较高相似度，应该在0.5以上
        self.assertGreater(sim, 0.5)
        # 但不是完全相同，应该小于1.0
        self.assertLess(sim, 1.0)

    def test_empty_input(self):
        """测试16：空文本输入"""
        self.assertEqual(calculate_similarity("", "测试"), 0.0)
        self.assertEqual(calculate_similarity("测试", ""), 0.0)
        self.assertEqual(calculate_similarity("", ""), 0.0)

    def test_only_whitespace(self):
        """测试17：只有空白的文本"""
        self.assertEqual(calculate_similarity("   \n\t", "测试"), 0.0)

    def test_partial_overlap(self):
        """测试18：部分重叠的文本"""
        text1 = "机器学习是人工智能的核心技术之一，广泛应用于各个领域。"
        text2 = "机器学习是人工智能的核心技术之一，在金融领域应用广泛。"
        sim = calculate_similarity(text1, text2)
        # 大部分内容相同，相似度应该较高
        self.assertGreater(sim, 0.6)

    def test_word_order_change(self):
        """测试19：语序调整的文本"""
        text1 = "我喜欢吃苹果和香蕉。"
        text2 = "香蕉和苹果我喜欢吃。"
        sim = calculate_similarity(text1, text2)
        # 词语相同但语序不同，词级相似度仍应较高
        self.assertGreater(sim, 0.4)

    def test_long_texts(self):
        """测试20：长文本相似度"""
        base = "软件工程是一门研究用工程化方法构建和维护有效的、实用的和高质量的软件的学科。" * 10
        modified = base.replace("工程化", "工程化的").replace("高质量", "高品质")
        sim = calculate_similarity(base, modified)
        self.assertGreater(sim, 0.8)

    def test_result_range(self):
        """测试21：结果始终在[0, 1]范围内"""
        test_cases = [
            ("", ""),
            ("a", "b"),
            ("测试文本", "测试文本"),
            ("今天天气好", "今天天气不好"),
        ]
        for t1, t2 in test_cases:
            sim = calculate_similarity(t1, t2)
            self.assertGreaterEqual(sim, 0.0)
            self.assertLessEqual(sim, 1.0)


class TestFileOperations(unittest.TestCase):
    """测试文件读写功能"""

    def test_write_and_read_result(self):
        """测试22：结果写入与读取"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "ans.txt")
            write_result(output_path, 0.85678)
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertEqual(content, "0.86")

    def test_read_file_utf8(self):
        """测试23：读取UTF-8编码文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("测试文本内容")
            content = read_file(file_path)
            self.assertEqual(content, "测试文本内容")


class TestMainIntegration(unittest.TestCase):
    """测试主程序集成功能"""

    def test_full_pipeline(self):
        """测试24：完整的文件输入输出流程"""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_path = os.path.join(tmpdir, "orig.txt")
            plag_path = os.path.join(tmpdir, "plag.txt")
            out_path = os.path.join(tmpdir, "ans.txt")

            with open(orig_path, 'w', encoding='utf-8') as f:
                f.write("今天是星期天，天气晴，今天晚上我要去看电影。")
            with open(plag_path, 'w', encoding='utf-8') as f:
                f.write("今天是周天，天气晴朗，我晚上要去看电影。")

            # 模拟命令行参数
            original_argv = sys.argv
            sys.argv = ["main.py", orig_path, plag_path, out_path]

            try:
                from main import main
                main()
            finally:
                sys.argv = original_argv

            # 验证输出文件存在且内容正确
            self.assertTrue(os.path.isfile(out_path))
            with open(out_path, 'r', encoding='utf-8') as f:
                result = float(f.read().strip())
            self.assertGreater(result, 0.5)
            self.assertLess(result, 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
