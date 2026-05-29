# PDF to EPUB

使用 PaddleOCR 或 MinerU 云端 API 将扫描版 PDF 书籍转换为 EPUB — 无需本地 GPU。

## 安装

```bash
pip install pymupdf ebooklib requests python-dotenv markdown
```

## 配置

复制 `.env.example` 为 `.env`，填入 API token：

```
PADDLE_API_TOKEN=你的token   # --api paddle（默认）
MINERU_API_TOKEN=你的token   # --api mineru
MINERU_LANGUAGE=ch_tra      # 繁体中文 OCR
MINERU_ENABLE_TABLE=0       # 避免把竖排文字误判成表格
PADDLE_LAYOUT_THRESHOLD=0.35  # 保留较窄的竖排文字块
```

在 [百度 AIStudio](https://aistudio.baidu.com/) 获取 PaddleOCR 版面解析 API token。
在 [MinerU](https://mineru.net/) 获取 MinerU v4 API token。

## 使用

```bash
# PaddleOCR（默认）
python pdf2epub_paddle.py book.pdf --title "书名" --author "作者" --auto-toc

# MinerU（更快，20页/块）
python pdf2epub_paddle.py book.pdf --title "书名" --author "作者" --auto-toc --api mineru

# 基本用法 — 交互式输入书名和作者
python pdf2epub_paddle.py book.pdf

# 不拆分章节，整本合为一个章节
python pdf2epub_paddle.py book.pdf --title "书名" --no-toc

# 严格模式 — 残留 OCR 噪声时报错
python pdf2epub_paddle.py book.pdf --title "书名" --strict-ocr-noise
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--output`, `-o` | 输出 EPUB 路径（默认：`<输入文件名>.epub`） |
| `--title` | 书名（跳过交互式提示） |
| `--author` | 作者名 |
| `--language` | EPUB 语言标签（默认：`zh-Hant`） |
| `--api` | OCR 后端：`paddle`（默认）或 `mineru` |
| `--auto-toc` | 跳过目录审查，直接使用自动检测的章节 |
| `--no-toc` | 不拆分章节 — 整本书作为一个章节 |
| `--strict-ocr-noise` | 输出中残留 OCR 噪声时中断报错 |
| `--cover-max-edge` | 封面图片最大边长像素（默认：2000） |
| `--cover-quality` | JPEG 封面质量 1–100（默认：82） |

### MinerU 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MINERU_API_TOKEN` | — | MinerU v4 API Bearer token |
| `MINERU_CHUNK_SIZE` | `20` | 每块页数（最大 200） |
| `MINERU_POLL_INTERVAL` | `5` | 结果轮询间隔（秒） |
| `MINERU_MAX_POLL_TIME` | `1800` | 单批次最大等待时间（秒） |
| `MINERU_LANGUAGE` | `ch_tra` | 繁体中文书籍的 OCR 语言 |
| `MINERU_ENABLE_TABLE` | `0` | 仅在确实需要表格提取时设为 `1` |
| `MINERU_VERIFY_SSL` | `1` | 设为 `0` 关闭 SSL 验证 |

### PaddleOCR 调优

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `PADDLE_LAYOUT_THRESHOLD` | `0.35` | 降低阈值以保留较窄的竖排文字块 |
| `PADDLE_TEMPERATURE` | `0.1` | 降低解码随机性，减少繁体误识别 |
| `PADDLE_REPETITION_PENALTY` | `1.05` | 保留文言文中合理的重复字 |
| `PADDLE_TOP_P` | `0.75` | 收窄候选范围，减少乱码 |

### 断点续传

每个 chunk 的处理进度保存在 `paddle_epub_work_<hash>/` 目录中。中断后重新运行相同命令即可从上次位置继续。

## 功能

- **版面解析**：调用 PaddleOCR API 检测段落、标题、图片和表格
- **章节检测**：从 OCR 结果中自动检测章节标题，支持交互式审查
- **封面提取**：将 PDF 第一页渲染为 EPUB 封面图片
- **图片嵌入**：保留原 PDF 中的插图、照片和图表
- **脚注处理**：提取并链接 OCR 检测到的脚注
- **OCR 噪声清理**：过滤常见的 PaddleOCR 识别噪声（纯数字表格、伪 LaTeX 等）
- **目录修复**：修正 TOC 目录链接指向正确的章节标题

## 许可证

MIT
