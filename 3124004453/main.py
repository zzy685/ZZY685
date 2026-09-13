# -*- coding: utf-8 -*-
"""
论文查重主程序
用法: python main.py [原文文件] [抄袭版论文的文件] [答案文件]
"""

import sys
import os

from plagiarism_checker import calculate_similarity


def read_file(file_path: str) -> str:
    """
    读取文件内容，支持多种编码
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 最后尝试用errors='ignore'读取
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def write_result(file_path: str, result: float) -> None:
    """
    将结果写入答案文件，精确到小数点后两位
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"{result:.2f}")


def main():
    # 检查命令行参数数量
    if len(sys.argv) != 4:
        print("用法: python main.py [原文文件] [抄袭版论文的文件] [答案文件]")
        print(f"示例: python main.py orig.txt orig_add.txt ans.txt")
        sys.exit(1)

    orig_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    output_path = sys.argv[3]

    # 检查输入文件是否存在
    if not os.path.isfile(orig_path):
        print(f"错误: 原文文件不存在 - {orig_path}")
        sys.exit(1)

    if not os.path.isfile(plagiarized_path):
        print(f"错误: 抄袭版论文文件不存在 - {plagiarized_path}")
        sys.exit(1)

    # 读取文件内容
    orig_text = read_file(orig_path)
    plagiarized_text = read_file(plagiarized_path)

    # 计算重复率
    similarity = calculate_similarity(orig_text, plagiarized_text)

    # 写入结果
    write_result(output_path, similarity)

    print(f"重复率: {similarity:.2f}")
    print(f"结果已写入: {output_path}")


if __name__ == '__main__':
    main()
