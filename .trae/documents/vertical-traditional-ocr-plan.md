# 竖排繁体 OCR 精度提升（计划）

## Summary

目标：在不引入“自动重试/二次识别”的前提下，通过**参数化 + 推荐预设**，提升本项目对**竖排繁体中文书籍**的识别稳定性与准确率，显著减少“识别字形错误/胡乱码字”。

范围：同时覆盖两条后端链路：
- PaddleOCR（AIStudio PaddleOCR-VL Jobs API，当前默认）  
- MinerU v4 API（项目已有备选后端）

非目标：
- 不做自动检测乱码后重跑（用户选择“只做可调参数”）
- 不做人工校对工作流/标注平台接入
- 不做本地 OCR 引擎替换（识别仍然在云端完成）

## Current State Analysis

### PaddleOCR（AIStudio）

- 入口与编排在 [main.py](file:///workspace/paddle_pipeline/main.py#L38-L220)，默认后端为 `--api paddle`。
- OCR 请求在 [paddle_api.py](file:///workspace/paddle_pipeline/paddle_api.py#L87-L186)：
  - 目前 `optionalPayload` 被硬编码为：
    - `useDocOrientationClassify=True`
    - `useDocUnwarping=True`
    - `useChartRecognition=False`
    - `layoutThreshold=0.5`
    - `temperature=0.2`, `topP=0.85`, `repetitionPenalty=1.2`
  - 未显式提供（但 AIStudio 文档存在的）关键参数：`useLayoutDetection`、`layoutShapeMode`、`layoutMergeBboxesMode`、`minPixels/maxPixels` 等（见 AIStudio 文档“请求参数说明”）。
- 预处理仅在 [split_pdf](file:///workspace/paddle_pipeline/paddle_api.py#L42-L84) 做了“底部 5% 白边 padding”，用于缓解扫描件页底裁切；对竖排古籍常见的“装订侧/页边裁切”没有针对性控制。

### MinerU

- MinerU v4 API 调用在 [mineru_api.py](file:///workspace/paddle_pipeline/mineru_api.py#L86-L239)。
- 请求参数中 `language` 当前硬编码为 `"ch_server"`（见 [mineru_api.py](file:///workspace/paddle_pipeline/mineru_api.py#L113-L123)）。
- 对竖排繁体古籍，MinerU 开源文档中存在更贴近场景的语言选项（如 `chinese_cht`），但项目目前无法配置。

### “乱码”与现有防护

- 项目已具备 OCR 噪声清理与校验：
  - 清理： [clean_ocr_noise](file:///workspace/paddle_pipeline/ocr_noise.py#L212-L237)
  - EPUB 校验： [validate_epub_no_ocr_noise](file:///workspace/paddle_pipeline/epub_validate.py#L69-L89)，包含“自校准 CJK bigram”检测潜在乱码段落（见 [ocr_noise.py](file:///workspace/paddle_pipeline/ocr_noise.py#L16-L142)）
- 但这些机制主要用于“发现/阻断/清理格式噪声”，并不能提升云端 OCR 的识别质量；要减少“字形错误”，核心仍需从**云端 OCR 参数 + 输入预处理**入手。

## Proposed Changes

### 1) 增加“OCR 可调参数 + 预设”配置层（不改变默认行为）

**目标**：让竖排繁体场景可以快速试验并收敛到最优参数组合，同时保证现有默认参数不被破坏。

#### 1.1 config.py：新增 Paddle/MinerU 的可配置项（env 级）

在 [config.py](file:///workspace/paddle_pipeline/config.py) 增加（沿用现有 `_env_int` 风格）：
- `_env_bool(name, default)`：解析 `0/1/true/false`（忽略非法值回退默认）
- `_env_float(name, default, minimum/maximum 可选)`
- `_env_str(name, default, allowed 可选)`

新增配置常量（建议命名，最终以实现为准）：
- Paddle：
  - `PADDLE_USE_DOC_ORIENTATION_CLASSIFY`（默认 `True`，对应 `useDocOrientationClassify`）
  - `PADDLE_USE_DOC_UNWARPING`（默认 `True`，对应 `useDocUnwarping`）
  - `PADDLE_USE_LAYOUT_DETECTION`（默认：保持“当前行为等效”。实现上可先设为 `None` 表示不传；或显式 `True` 但需确认不引起行为突变）
  - `PADDLE_LAYOUT_THRESHOLD`（默认 `0.5`，已存在但改为可配置）
  - `PADDLE_LAYOUT_SHAPE_MODE`（默认不传；可选 `rect/quad/poly/auto`，对应 `layoutShapeMode`）
  - `PADDLE_LAYOUT_MERGE_BBOXES_MODE`（默认不传；可选 `large/small/union`，对应 `layoutMergeBboxesMode`）
  - `PADDLE_LAYOUT_NMS`（默认不传；对应 `layoutNms`）
  - `PADDLE_LAYOUT_UNCLIP_RATIO`（默认不传；对应 `layoutUnclipRatio`）
  - `PADDLE_TEMPERATURE`（默认 `0.2`）
  - `PADDLE_TOP_P`（默认 `0.85`）
  - `PADDLE_REPETITION_PENALTY`（默认 `1.2`）
  - `PADDLE_MIN_PIXELS` / `PADDLE_MAX_PIXELS`（默认不传；对应 `minPixels/maxPixels`）
  - `PADDLE_PROMPT_LABEL`（默认不传；对应 `promptLabel`，如 `ocr/formula/table/chart`）
  - `PADDLE_PRETTIFY_MARKDOWN`（默认不传；对应 `prettifyMarkdown`）
  - `PADDLE_VISUALIZE`（默认不传；对应 `visualize`，用于调试）
- MinerU：
  - `MINERU_LANGUAGE`（默认仍为 `"ch_server"`，但允许改为 `chinese_cht` 等）

#### 1.2 main.py：新增 CLI 参数（CLI 优先于 env）

在 [main.py](file:///workspace/paddle_pipeline/main.py) 增加参数（仅在对应后端启用时生效）：
- 通用：
  - `--ocr-preset`：`default` / `vertical-zh-hant`（推荐预设：竖排繁体）
- Paddle：
  - `--paddle-use-layout-detection [true|false]`
  - `--paddle-layout-shape-mode [rect|quad|poly|auto]`
  - `--paddle-layout-merge-bboxes-mode [large|small|union]`
  - `--paddle-temperature <float>`
  - `--paddle-top-p <float>`
  - `--paddle-repetition-penalty <float>`
  - `--paddle-min-pixels <int>` / `--paddle-max-pixels <int>`
  - `--paddle-prettify-markdown [true|false]`
  - `--paddle-visualize [true|false]`
  - `--paddle-force-rotate [0|90|180|270]`（竖排古籍常用：尝试 `90/270`，用于“把竖排强制转为横排”再识别）
  - `--paddle-padding-x <float>` / `--paddle-padding-y <float>`（单位：比例，例如 `0.05` = 5%）
- MinerU：
  - `--mineru-language <lang>`（默认 `ch_server`，竖排繁体推荐尝试 `chinese_cht`）

**预设策略（vertical-zh-hant）**
- 不做激进猜测；采用“更保守、更可解释”的变化：
  - 降低生成随机性：`temperature` 下调、`topP` 下调（减少胡乱码字）
  - 允许显式开启 `useLayoutDetection` 以改善多栏/区域排序（是否默认开启取决于“保持现状等效”的实现选择）
  - 其他复杂参数（`layoutMergeBboxesMode/layoutShapeMode/minPixels`）只暴露不强设，避免造成不可控退化

### 2) Paddle PDF 预处理增强：可选旋转 + 四边 padding

修改 [split_pdf](file:///workspace/paddle_pipeline/paddle_api.py#L42-L84)：
- 将当前“仅底部 padding”扩展为：
  - 支持左右 padding（装订侧常被裁切）
  - 支持顶部 padding（页眉、页码）
  - 保留默认值为“只做底部 5%”以避免回归
- 增加 `--paddle-force-rotate`：
  - 在切 chunk 时对页进行统一旋转（90/270）并保存到 chunk PDF
  - 目的：把竖排版面变成横排，从而让 OCR 更接近其强项方向（由用户试验选择）

### 3) PaddleOCR 请求参数构造：从“硬编码”改为“可配置 + 兼容不传”

修改 [parse_pdf_chunk](file:///workspace/paddle_pipeline/paddle_api.py#L87-L186)：
- 将 `optional_payload` 的构造改为：
  - 基础字段：沿用当前默认值（保证默认行为一致）
  - 可选字段：当用户通过 CLI/env 指定时才加入 payload（避免服务端对未知字段/默认值差异造成行为突变）
- 参数名对齐 AIStudio 文档（PaddleOCR-VL-1.5/1.6 的请求参数说明）：
  - `useDocOrientationClassify`
  - `useDocUnwarping`
  - `useLayoutDetection`
  - `layoutThreshold`
  - `layoutNms`
  - `layoutUnclipRatio`
  - `layoutMergeBboxesMode`
  - `layoutShapeMode`
  - `promptLabel`
  - `temperature`
  - `topP`
  - `repetitionPenalty`
  - `minPixels`
  - `maxPixels`
  - `prettifyMarkdown`
  - `visualize`

同时补齐“模型版本可配置”的通道（可选）：
- [config.py](file:///workspace/paddle_pipeline/config.py#L24-L28) 的 `MODEL_VERSION` 支持 env 覆盖（例如 `PADDLE_MODEL_VERSION`），以便对比 `PaddleOCR-VL-1.5` 与 `1.6` 在竖排繁体上的表现。

### 4) MinerU：language 参数可配置（对繁体最关键）

修改 [mineru_api.py](file:///workspace/paddle_pipeline/mineru_api.py#L86-L239)：
- 将 `language: "ch_server"` 替换为可配置项：
  - CLI：`--mineru-language`
  - env：`MINERU_LANGUAGE`
- 文档提示：
  - 竖排繁体建议优先尝试 `chinese_cht`
  - 如果存在中英日繁混排/手写，则尝试 `ch_server`

### 5) 使用说明与推荐试跑矩阵（README）

更新 [README_zh-CN.md](file:///workspace/README_zh-CN.md)（以及必要时 README.md）：
- 增加“竖排繁体”推荐参数组合示例（不给出绝对最优，只给可复现的试跑矩阵）：
  - Paddle（更保守生成）：
    - `--ocr-preset vertical-zh-hant`
    - 或显式：`--paddle-temperature ... --paddle-top-p ...`
  - Paddle（强制旋转）：
    - `--paddle-force-rotate 90` 与 `270` 各试一次
  - MinerU（语言对齐繁体）：
    - `--api mineru --mineru-language chinese_cht`
- 明确“乱码”主要来自 OCR 识别本身，而非 EPUB 编码（本项目输出为 UTF-8，且 EPUB 校验会扫描异常 token）。

### 6) 测试与回归保护

新增/调整单元测试（复用现有 unittest 风格）：
- 新增 `tests/test_ocr_tuning_config.py`（建议）：
  - env/CLI 覆盖优先级测试（CLI > env > 默认）
  - 可选字段“不传入 payload”时保持现状（例如 `useLayoutDetection` 不设置则不出现在 `optionalPayload`）
- 为 Paddle payload 构造增加“纯函数”入口（建议在 `paddle_api.py` 内新增一个小函数返回 dict），便于测试而无需发起网络请求。

## Assumptions & Decisions

- 识别乱码 = 云端 OCR 字符识别错误（用户已确认），因此优先从：
  1) 降低 VLM 生成随机性（temperature/topP）
  2) 版面排序/区域检测开关（useLayoutDetection 等）
  3) 输入预处理（旋转、padding）
  入手。
- 同时优化两条后端：Paddle 与 MinerU（用户已确认）。
- 仅做可调参数与预设，不做自动重试、也不引入额外成本控制逻辑（用户已确认）。
- 竖排繁体场景最关键的“选择项”交给使用者试跑：是否强制旋转、MinerU language 取值、以及是否开启 layout detection。

## Verification Steps

本地验证（实现后执行）：
- 单元测试：
  - `python -m unittest`（或仓库现有 test 命令）
- 手工试跑（建议取同一本书的 3 个代表性 chunk，观察“乱码率”）：
  - Paddle：默认参数 vs `--ocr-preset vertical-zh-hant`
  - Paddle：`--paddle-force-rotate 90` 与 `270`
  - MinerU：`--mineru-language chinese_cht` vs `ch_server`
- 质量检查：
  - 开启 `--strict-ocr-noise`，确保生成 EPUB 未残留明显 OCR 噪声（并关注“garbled CJK span”数量变化）

