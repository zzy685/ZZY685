# 论文查重程序

广东工业大学计算机学院第一次个人编程作业

## 功能

设计论文查重算法，给出原文文件和抄袭版论文文件，输出重复率（精确到小数点后两位）。

## 运行环境

- Python 3.x
- 依赖：jieba（中文分词）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python main.py [原文文件路径] [抄袭版论文文件路径] [答案文件路径]
```

示例：

```bash
python main.py orig.txt orig_add.txt ans.txt
```

## 项目结构

```
├── main.py                    # 主程序入口
├── plagiarism_checker.py      # 查重算法核心模块
├── test_plagiarism_checker.py # 单元测试（24个测试用例）
├── requirements.txt           # 依赖声明
├── 博客作业.md                 # 作业博客文章
└── test_data/                 # 测试样例
    ├── orig.txt
    ├── orig_add.txt
    ├── long_orig.txt
    └── long_plag.txt
```

## 运行单元测试

```bash
python -m unittest test_plagiarism_checker -v
```

## 算法说明

采用词级余弦相似度（权重0.6）与字符级bigram相似度（权重0.4）加权融合的算法。

1. 文本预处理：去除标点、空白，统一小写
2. jieba分词后统计词频向量，计算余弦相似度
3. 字符级bigram统计频率向量，计算余弦相似度
4. 加权融合得到最终重复率
