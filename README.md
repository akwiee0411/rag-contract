# 合約問答 RAG 系統

以自然語言對中文合約進行問答的 RAG（Retrieval-Augmented Generation）系統。上傳 PDF 或 DOCX 合約後，系統會自動解析條款階層結構、建立向量索引，並支援語意搜尋與 LLM 問答。

---

## 功能特色

- **合約文件處理**：支援 PDF 與 DOCX 格式，自動判斷使用 PDFPlumber 直接提取或 OCRFlux-3B 進行 AI OCR
- **階層式條款解析**：自動識別中文合約的條款編號（第一條、一、、(一)、1.1 等），建立完整階層結構
- **向量索引與快取**：使用 LlamaIndex + Yuan-embedding-2.0-zh 建立索引，並將 Embedding 結果快取至磁碟，避免重複計算
- **QA 問答快取**：歷史問答自動儲存，支援語意相似度搜尋，相同問題直接從快取回答
- **Gradio 操作介面**：全程圖形化操作，無需撰寫程式碼

---

## 系統架構

```
app.py                  # Gradio 主介面
ui_helpers.py           # UI 元件與互動邏輯
contract_parser.py      # 合約解析器 + RAG 查詢引擎
doc_processor.py        # PDF/DOCX 處理、OCR、索引建立
cache_manager.py        # Embedding 快取 / QA 快取管理
config.py               # 路徑設定（本機環境）
```

### 查詢流程

```
使用者上傳合約
    ↓
PDF/DOCX 轉文字（PDFPlumber 或 OCRFlux-3B）
    ↓
HierarchicalContractNodeParser 階層式條款切割
    ↓
Yuan-embedding-2.0-zh 計算 Embedding → 存入磁碟快取
    ↓
使用者提問
    ↓
Vector Retrieve（快）→ 抓取完整章節 → LLM 生成答案（慢，只做一次）
    ↓
QA 結果存入問答快取
```

---

## 安裝

### 需求

- Python 3.10+
- 建議使用 GPU（OCRFlux-3B 推論需要，Embedding 計算亦受益）

### 安裝依賴

```bash
pip install gradio llama-index pdfplumber pdf2image pymupdf pillow \
            transformers torch numpy
```

> OCRFlux-3B 需另行下載模型權重，請參考 [OCRFlux 官方說明](https://github.com/chatdoc-com/OCRFlux)。

### 路徑設定

複製並修改設定檔：

```bash
cp config.example.py config.py
```

`config.py` 內容：

```python
from pathlib import Path

BASE_DIR        = Path.home() / "rag_contract"   # 修改為你的實際路徑
TEXT_RESULT_DIR = str(BASE_DIR / "text_result")
TEMP_SPLIT_DIR  = str(BASE_DIR / "temp_split_docs")
```

---

## 使用方式

```bash
python app.py
```

瀏覽器開啟後：

1. **上傳合約**：支援 PDF / DOCX
2. **建立索引**：系統自動選擇 OCR 或直接提取文字，並計算 Embedding
3. **開始問答**：輸入自然語言問題，系統返回答案與對應條款來源
4. **查看快取**：可瀏覽、刪除歷史問答與 Embedding 快取

---

## 模型

| 用途 | 模型 |
|------|------|
| OCR（掃描 PDF）| OCRFlux-3B |
| 文字 Embedding | Yuan-embedding-2.0-zh |
| 問答生成 | 透過 LlamaIndex `Settings.llm` 設定 |

---

## 快取說明

- **Embedding 快取**：儲存於 `text_result/<合約名稱>/`，包含 `docstore.json`、`vector_store.json` 等 LlamaIndex 標準格式
- **QA 快取**：儲存於 `text_result/<合約名稱>/qa_cache.json`，每筆問答含問題、答案、時間戳與相關條款

重複上傳相同合約時，若快取存在則直接載入，跳過 Embedding 計算。

---

## .gitignore 建議

```
__pycache__/
*.pyc
config.py
text_result/
temp_split_docs/
*.jpg
```

---

## License

MIT
