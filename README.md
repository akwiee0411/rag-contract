# 📄 合約問答 RAG 系統

以自然語言對中文合約進行問答的 RAG（Retrieval-Augmented Generation）系統。上傳 PDF 或 DOCX 合約後，系統會自動解析條款階層結構、建立向量索引，並支援語意搜尋與 LLM 問答。

---

## 目錄

- [功能特色](#功能特色)
- [系統架構](#系統架構)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [設定說明](#設定說明)
- [啟動方式](#啟動方式)
- [操作說明](#操作說明)
- [模型說明](#模型說明)
- [快取說明](#快取說明)
- [專案結構](#專案結構)
- [常見問題](#常見問題)

---

## 功能特色

| 功能 | 說明 |
|------|------|
| **多格式支援** | 接受 PDF 與 DOCX，DOCX 自動透過 LibreOffice 轉換為 PDF |
| **智慧 OCR 選擇** | 系統自動判斷：文字型 PDF 使用 PDFPlumber 直接提取；掃描型 PDF 使用 OCRFlux-3B 進行 AI OCR |
| **階層式條款解析** | 自動識別中文合約條款編號格式（第一條、一、㈠、(一)、1.1 等），建立完整父子階層結構 |
| **向量索引與磁碟快取** | 使用 LlamaIndex + Yuan-embedding-2.0-zh 建立向量索引，並將 Embedding 結果快取至磁碟，避免重複計算 |
| **QA 問答快取** | 歷史問答自動儲存，支援語意相似度搜尋；相同或相近問題直接從快取回答，大幅降低 LLM 呼叫成本 |
| **智能文件搜尋** | 以自然語言描述合約，系統自動比對最相似的已處理文件並載入 |
| **Gradio 圖形介面** | 全程圖形化操作，無需撰寫程式碼，並提供完整的快取管理介面 |

---

## 系統架構

```
app.py                  # Gradio 主介面、模型載入、事件綁定
ui_helpers.py           # UI 元件邏輯（問答、歷史載入、智能搜尋）
parsetool.py            # 合約解析器（HierarchicalContractNodeParser）+ RAG 查詢引擎
doc_processor.py        # PDF/DOCX 處理、OCR 推論、LlamaIndex 索引建立
cache_manager.py        # Embedding 快取 / QA 問答快取的讀寫管理
```

### 完整查詢流程

```
使用者上傳 PDF / DOCX
        │
        ▼
  格式判斷與轉換
  ├─ DOCX → LibreOffice 轉 PDF
  └─ PDF  → 保留原檔
        │
        ▼
  文字提取（自動選擇）
  ├─ 文字型 PDF → PDFPlumber 直接提取
  └─ 掃描型 PDF → OCRFlux-3B AI OCR（GPU）
        │
        ▼
  HierarchicalContractNodeParser
  條款階層切割（第X條 → 第X款 → 細項）
        │
        ▼
  Yuan-embedding-2.0-zh 計算 Embedding
  └─ 存入磁碟快取（text_result/<合約名稱>/）
        │
        ▼
  使用者輸入問題
        │
        ├─ QA 快取命中？→ 直接回傳快取答案（最快）
        │
        └─ 否 → VectorRetriever 語意搜尋
                  └─ 抓取完整章節內容
                        └─ Ollama LLM（qwen3:4b）生成答案
                              └─ 儲存至 QA 快取
```

---

## 環境需求

- **Python** 3.10+
- **GPU**（強烈建議）：OCRFlux-3B 推論必須使用 GPU；Embedding 計算使用 GPU 速度大幅提升
- **LibreOffice**：DOCX → PDF 轉換需要（僅上傳 DOCX 時才需要）
- **Ollama**：本地 LLM 推論服務，需預先下載 `qwen3:4b` 模型

---

## 安裝步驟

### 1. 建立 Conda 環境

```bash
conda create -n contract python=3.10 -y
conda activate contract
```

### 2. 安裝 Python 依賴

```bash
pip install gradio ollama pdfplumber pdf2image pymupdf pillow \
            transformers torch numpy \
            llama-index llama-index-llms-ollama \
            llama-index-embeddings-huggingface
```

> 若需使用 DOCX 上傳功能，請安裝 [LibreOffice](https://www.libreoffice.org/download/download/) 並確認 `soffice` 可從命令列執行。

### 3. 下載 OCRFlux-3B 模型

請參考 [OCRFlux 官方說明](https://github.com/chatdoc-com/OCRFlux) 下載模型權重，並將路徑記錄下來供後續設定使用。

### 4. 安裝並啟動 Ollama，下載 LLM

```bash
# 安裝 Ollama（請依官網說明）
ollama pull qwen3:4b
ollama serve
```

---

## 設定說明

在 `app.py` 頂端修改以下兩個路徑常數：

```python
# OCRFlux-3B 模型的本地權重目錄
OCR_MODEL_PATH = r"C:\Users\user\Documents\rag_contract\ocrmodel"

# 所有合約的 OCR 結果與向量索引儲存根目錄
TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"
```

若想更換 LLM 或 Embedding 模型，修改 `app.py` 中的以下區塊：

```python
Settings.llm = Ollama(model="qwen3:4b", request_timeout=1200.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="IEITYuan/Yuan-embedding-2.0-zh", device="cuda"
)
```

---

## 啟動方式

```bash
conda activate contract
python app.py
```

啟動後，開啟瀏覽器前往：

```
http://localhost:7777
```

---

## 操作說明

系統介面分為四個頁籤：

---

### 🔍 頁籤一：智能檢索回答

適合已有多份合約並想快速找到目標文件並發問的情境。

1. **輸入檔案描述**：以自然語言描述你要找的合約（例如「租賃合約2024」、「採購合約甲方」）
2. **輸入問題**：輸入你想問的問題
3. **點擊「開始智能搜尋與問答」**：系統從已處理的合約中找出最相似的文件並顯示確認訊息
4. **確認文件**：
   - 點擊「是的，就是這個文件」→ 系統自動載入並回答問題
   - 點擊「不是，顯示其他選項」→ 顯示相似度次高的候選文件
5. **後續追問**：在下方輸入框繼續提問，支援多輪對話
6. **儲存答案**：點擊「💾 儲存此答案」將本次答案存入 QA 快取

---

### 📁 頁籤二：檔案上傳與手動搜尋

適合單次處理新合約，或手動選擇歷史已處理的合約進行問答。

#### 子頁籤：🆕 上傳新文件（四步驟流程）

**Step 1 — 上傳檔案**
- 拖曳或選擇 PDF / DOCX 檔案
- 系統讀取頁數並預覽，若偵測到同名歷史紀錄會給予提示

**Step 2 — 預覽與切割**
- 瀏覽文件預覽圖，選擇要處理的頁碼範圍（起始頁～結束頁）
- 點擊「執行切割」，產生切割後的 PDF

**Step 3 — OCR / 文字提取**
- 點擊「執行 OCR / 文字提取」
- 系統自動判斷文件類型：
  - 文字型 PDF → PDFPlumber 直接提取（快速）
  - 掃描型 PDF → OCRFlux-3B AI OCR（需 GPU，較耗時）
- 提取完成後點擊「下一步：智能問答」

**Step 4 — 智能合約問答**
- 選擇 AI 模型（預設 `qwen3:4b`）與輸出語言（預設中文）
- 輸入問題並點擊「發送問題」
- 若有 QA 快取命中，系統會顯示候選答案供選擇；亦可選擇重新呼叫 AI 生成
- 勾選「包含完整章節」可讓 LLM 參考更多原文（生成較慢但更完整）
- 點擊「👁️ 顯示/隱藏 檢索相關條款」查看系統所參考的原始條款內容

#### 子頁籤：📂 選擇歷史紀錄
- 從下拉選單選擇已處理的合約資料夾
- 可預覽文件圖片，確認後點擊「載入此專案並開始問答」

---

### 🗃️ 頁籤三：Embedding 快取管理

- **查看所有快取**：列出所有已建立向量索引的合約
- **刪除全部快取**：清除所有向量索引（下次使用需重新計算）
- **刪除單一文件快取**：從下拉選單選擇特定合約後刪除其向量索引

---

### 💬 頁籤四：QA 問答快取管理

- **選擇文件資料夾**：查看該合約的所有已儲存問答
- **刪除單筆問答**：從下拉選單選擇後刪除
- **刪除資料夾的全部問答**：清除指定合約的 QA 快取
- **刪除所有 QA 快取**：清空全部合約的 QA 快取

---

## 模型說明

| 用途 | 模型 | 說明 |
|------|------|------|
| OCR（掃描 PDF） | [OCRFlux-3B](https://github.com/chatdoc-com/OCRFlux) | 本地推論，需 GPU，支援中文掃描文件 |
| 文字 Embedding | [Yuan-embedding-2.0-zh](https://huggingface.co/IEITYuan/Yuan-embedding-2.0-zh) | 中文向量模型，用於語意搜尋與問答快取比對 |
| LLM 問答生成 | [qwen3:4b](https://ollama.com/library/qwen3) via Ollama | 本地 LLM，透過 LlamaIndex `Settings.llm` 設定，可替換其他 Ollama 模型 |

> 所有模型皆在本地執行，不需要網路連線或 API Key。

---

## 快取說明

所有快取儲存在 `TEXT_RESULT_DIR/<合約檔名>/` 目錄下：

### Embedding 快取

| 檔案 | 說明 |
|------|------|
| `docstore.json` | LlamaIndex 文件節點資料 |
| `vector_store.json` | 向量索引資料 |
| `index_store.json` | 索引元資料 |

重複上傳同名合約時，若快取存在則直接載入，**跳過 Embedding 計算**，大幅縮短等待時間。

### QA 問答快取

| 檔案 | 說明 |
|------|------|
| `qa_cache.json` | 每筆問答含問題、答案、時間戳與相關條款清單 |

發問時系統會以語意相似度比對快取，若找到相近問題，直接顯示快取答案供使用者選用或重新生成。

---

## 專案結構

```
rag-contract/
├── app.py              # 主程式：模型載入、Gradio 介面、事件綁定
├── ui_helpers.py       # UI 互動邏輯：歷史載入、問答、智能搜尋、格式化輸出
├── parsetool.py        # 合約解析器（階層條款切割）+ RAG VectorQueryEngine
├── doc_processor.py    # PDF/DOCX 前處理、OCR 推論、LlamaIndex 索引建立
├── cache_manager.py    # Embedding 快取 & QA 快取的讀寫與刪除
└── README.md
```

執行後會自動產生：

```
text_result/
└── <合約檔名>/
    ├── <合約檔名>.txt      # OCR / 文字提取結果
    ├── docstore.json        # Embedding 快取
    ├── vector_store.json
    ├── index_store.json
    └── qa_cache.json        # QA 問答快取
```

---

## 常見問題

**Q：OCR 模型載入失敗怎麼辦？**  
A：確認 `OCR_MODEL_PATH` 路徑正確，且模型目錄中包含 `config.json`、`tokenizer_config.json` 等 HuggingFace 模型標準檔案。OCR 功能載入失敗時，系統仍可正常使用 PDFPlumber 處理文字型 PDF。

**Q：LLM 無法回應或逾時？**  
A：確認 Ollama 服務已啟動（`ollama serve`）且已下載 `qwen3:4b` 模型（`ollama pull qwen3:4b`）。可在 `app.py` 中調整 `request_timeout` 參數。

**Q：DOCX 轉換失敗？**  
A：確認 LibreOffice 已安裝，且 `soffice` 指令可在終端機執行。Windows 環境請確認 LibreOffice 已加入 `PATH`。

**Q：如何更換 LLM 模型？**  
A：於介面右上角的「AI 模型」下拉選單選擇其他已下載的 Ollama 模型，或修改 `app.py` 中 `Settings.llm` 的預設值。

**Q：Embedding 計算很慢？**  
A：建議使用 NVIDIA GPU，並確認 PyTorch 已安裝 CUDA 版本。在 `app.py` 中 `HuggingFaceEmbedding` 的 `device` 參數設為 `"cuda"`（預設即是）。

---

## License

本專案為內部使用，未指定開放授權。
