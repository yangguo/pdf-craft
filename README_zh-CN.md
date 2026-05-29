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
```

在 [百度 AIStudio](https://aistudio.baidu.com/) 获取 PaddleOCR 版面解析 API token。
在 [MinerU](https://mineru.net/) 获取 MinerU v4 API token。

## 使用

```bash
# PaddleOCR（默认）
python pdf2epub_paddle.py book.pdf --title "书名" --author "作者" --auto-toc

# MinerU（更快，20页/块）
python pdf2epub_paddle.py book.pdf --title "书名" --author "作者" --auto-toc --api mineru

# 竖排繁体推荐：预设（同时影响 Paddle/MinerU 的默认调参）
python pdf2epub_paddle.py book.pdf --title "书名" --auto-toc --ocr-preset vertical-zh-hant

# 竖排繁体推荐：MinerU 指定繁体语言模型
python pdf2epub_paddle.py book.pdf --title "书名" --auto-toc --api mineru --mineru-language chinese_cht

# 竖排繁体推荐：Paddle 强制旋转（尝试 90 或 270）
python pdf2epub_paddle.py book.pdf --title "书名" --auto-toc --paddle-force-rotate 90

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
| `--ocr-preset` | OCR 调参预设：`default` 或 `vertical-zh-hant` |
| `--auto-toc` | 跳过目录审查，直接使用自动检测的章节 |
| `--no-toc` | 不拆分章节 — 整本书作为一个章节 |
| `--strict-ocr-noise` | 输出中残留 OCR 噪声时中断报错 |
| `--cover-max-edge` | 封面图片最大边长像素（默认：2000） |
| `--cover-quality` | JPEG 封面质量 1–100（默认：82） |
| `--mineru-language` | MinerU OCR 语言（如 `chinese_cht`、`ch_server`） |
| `--paddle-force-rotate` | Paddle 送 OCR 前旋转页面（`0/90/180/270`） |
| `--paddle-padding-x` | Paddle 页面左右 padding 比例（装订侧裁切可调大） |
| `--paddle-padding-y` | Paddle 页面上下 padding 比例 |
| `--paddle-use-layout-detection` | Paddle 版面区域检测与排序（true/false） |
| `--paddle-temperature` | Paddle 识别稳定性（越低越保守） |
| `--paddle-top-p` | Paddle 结果可信范围（越低越保守） |

### MinerU 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MINERU_API_TOKEN` | — | MinerU v4 API Bearer token |
| `MINERU_CHUNK_SIZE` | `20` | 每块页数（最大 200） |
| `MINERU_POLL_INTERVAL` | `5` | 结果轮询间隔（秒） |
| `MINERU_MAX_POLL_TIME` | `600` | 单批次最大等待时间（秒） |
| `MINERU_VERIFY_SSL` | `1` | 设为 `0` 关闭 SSL 验证 |
| `MINERU_LANGUAGE` | `ch_server` | OCR 语言（竖排繁体可试 `chinese_cht`） |

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
