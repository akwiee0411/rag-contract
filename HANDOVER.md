# 智能合約處理系統 — 工程師交接文件

> **閱讀對象**：接手維護或擴充本系統的工程師。
> 本文件涵蓋：路徑設定、關鍵參數調整、提示語（Prompt）說明、模組架構、常見坑點與除錯方法。

---

## 目錄

1. [專案概覽與模組關係](#1-專案概覽與模組關係)
2. [路徑設定（換機器必看）](#2-路徑設定換機器必看)
3. [模型設定與切換](#3-模型設定與切換)
4. [關鍵參數彙整表](#4-關鍵參數彙整表)
5. [提示語（Prompt）詳解](#5-提示語prompt詳解)
6. [OCR 相關參數](#6-ocr-相關參數)
7. [RAG 索引與查詢參數](#7-rag-索引與查詢參數)
8. [快取機制說明](#8-快取機制說明)
9. [合約條款解析邏輯](#9-合約條款解析邏輯)
10. [智能搜尋機制](#10-智能搜尋機制)
11. [State 管理與資料流](#11-state-管理與資料流)
12. [檔案更名注意事項](#12-檔案更名注意事項)
13. [已知限制與坑點](#13-已知限制與坑點)
14. [除錯工具與 Log 說明](#14-除錯工具與-log-說明)

---

## 1. 專案概覽與模組關係

### 檔案結構

```
專案根目錄/
├── app.py              ← 主程式：UI 定義、模型載入、所有事件綁定
├── doc_processor.py    ← 文件處理：PDF工具、OCR、RAG 索引建立
├── parsetool.py        ← 合約解析：階層式條款切割、查詢引擎
├── cache_manager.py    ← 快取管理：Embedding 快取 + QA 問答快取
└── ui_helpers.py       ← UI 輔助：格式化輸出、智能搜尋、問答流程
```

### 模組依賴關係

```
app.py
 ├── doc_processor.py
 │    ├── cache_manager.py
 │    └── parsetool.py
 │         └── ui_helpers.py（只取 _check_abort）
 └── ui_helpers.py
      └── cache_manager.py
```

**重點**：`parsetool.py` 反向依賴 `ui_helpers.py` 的 `_check_abort`（截斷機制），
這是唯一的反向依賴。若 `ui_helpers.py` 改名，需要同步修改 `parsetool.py` 第 997 行。

### 系統 Tab 對應功能

系統共有 **8 個 Tab**，對應 `app.py` 中的 UI 區塊：

| Tab | 標題 | 主要邏輯位置 |
|-----|------|------------|
| 1 | 🔍 智能檢索回答 | `ui_helpers.py` `search_folders_by_description` |
| 2 | 📁 檔案上傳與手動搜尋 | `app.py` `process_upload` / `split_document` / `run_ocr_extraction` / `go_to_step4` |
| 3 | 🗃️ Embedding 快取管理 | `cache_manager.py` `list_cached_collections` / `delete_cache_for_txt` |
| 4 | 💬 問答快取管理 | `cache_manager.py` `load_qa_cache` / `delete_qa_entry_by_id` |
| 5 | 🧠 模型記憶體管理 | `app.py` `ocr_unload_model` / `emb_unload_model` / `ollama_unload_model` |
| 6 | 🗑️ 刪除資料夾 | `ui_helpers.py` `delete_folder` |
| 7 | ⚡ 快速問答（不存檔） | `app.py` `_temp_*` 系列函式 + `doc_processor.py` `pdf_process_temp` / `build_temp_rag_index` |
| 8 | 📖 使用者說明 | 靜態 HTML，無後端邏輯 |

### Tab 7 臨時模式的資料流

Tab 7（快速問答）與 Tab 2 的核心差異：

- **文字提取**：呼叫 `doc_processor.pdf_process_temp`，結果存入系統 `tempfile.mkdtemp()` 暫存目錄，**不寫** `TEXT_RESULT_DIR`
- **RAG 索引**：呼叫 `doc_processor.build_temp_rag_index`，索引完全在記憶體中建立，**不讀也不寫**任何快取
- **重設**：`_temp_reset` 函式會呼叫 `shutil.rmtree` 清除 `tmpdir`（只清除路徑含 `sinbon_temp_` 的目錄，避免誤刪）

---

## 2. 路徑設定（換機器必看）

系統有 **5 個路徑常數**，分散在 4 個檔案中，**換機器時全部都要改**，漏改任一個都會出錯。

### 路徑常數彙整表

| 常數名稱 | 所在檔案 | 大約行號 | 說明 |
|---------|---------|---------|------|
| `OCR_MODEL_PATH` | `app.py` | 第 30 行 | OCRFlux-3B 模型的本機目錄 |
| `TEXT_RESULT_DIR` | `app.py` | 第 34 行 | 所有合約 OCR 結果的根目錄 |
| `TEXT_RESULT_DIR` | `doc_processor.py` | 第 63 行 | ⚠️ 與 app.py 相同，需同步 |
| `TEMP_SPLIT_DIR` | `doc_processor.py` | 第 68 行 | PDF 切割暫存目錄 |
| `TEXT_RESULT_DIR` | `cache_manager.py` | 第 70 行 | ⚠️ 同上，需同步 |
| `TEXT_RESULT_DIR` | `ui_helpers.py` | 第 29 行 | ⚠️ 同上，需同步 |
| `ERRORLOG_DIR` | `ui_helpers.py` | 第 32 行 | 問答錯誤回報存放目錄 |
| `DAILY_LOG_DIR` | `ui_helpers.py` | 第 33 行 | 每日問答自動紀錄目錄 |

### 目錄結構建議

換機器後建議先手動建立這些目錄，避免程式啟動時報錯：

```
C:\YourPath\rag_contract\
├── ocrmodel\           ← OCR_MODEL_PATH：放 OCRFlux-3B 模型檔案
├── text_result\        ← TEXT_RESULT_DIR：系統自動建立合約子資料夾
├── temp_split_docs\    ← TEMP_SPLIT_DIR：暫存（程式啟動時自動清空）
├── ERRORLOG\           ← ERRORLOG_DIR：回報錯誤的 JSON 存放
└── DAILY_LOG\          ← DAILY_LOG_DIR：每日滾動問答紀錄
```

> `text_result` 和 `ERRORLOG` 要提前建立，`DAILY_LOG` 第一次使用時系統會自動建立。

### Gradio 啟動 Port

`app.py` 最後一行：

```python
demo.launch(server_name="0.0.0.0", server_port=8080)
```

- `server_name="0.0.0.0"`：允許區域網路內的其他裝置連線
- `server_port=8080`：若該 Port 已被佔用，改成其他數字（如 7860、7777）

---

## 3. 模型設定與切換

### OCR 模型（OCRFlux-3B）

**位置**：`app.py` 第 84~95 行

```python
ocr_model = AutoModelForImageTextToText.from_pretrained(
    ocr_model_path, torch_dtype="auto", device_map="auto"
)
```

| 參數 | 目前值 | 影響 |
|------|-------|------|
| `torch_dtype="auto"` | auto（通常是 float16）| 改成 `"float32"` 精度更高但 VRAM 用量翻倍 |
| `device_map="auto"` | 自動選 GPU | 無 GPU 時自動退到 CPU（極慢，單頁可能跑幾分鐘）|

**⚠️ 懶載入機制**：OCR 模型**不在程式啟動時載入**，而是在使用者**第一次點「執行 OCR」時**才自動呼叫 `load_ocr_model()`。這樣系統待機可省下 6–7 GB VRAM 給 Ollama LLM 使用。若要改回啟動時載入，在 `app.py` 約第 197 行的 `load_embedding_model()` 後面加上 `load_ocr_model()` 即可。

卸載後可透過 Tab 5（模型記憶體管理）或在使用者下次執行 OCR 時自動重新載入。

### Embedding 模型

**位置**：`app.py` 第 177 行

```python
Settings.embed_model = HuggingFaceEmbedding(
    model_name="IEITYuan/Yuan-embedding-2.0-zh", device="cpu"
)
```

| 參數 | 目前值 | 影響 |
|------|-------|------|
| `model_name` | `IEITYuan/Yuan-embedding-2.0-zh` | 換模型後**必須**清除所有 Embedding 快取，否則向量維度不匹配會報錯 |
| `device="cpu"` | CPU | 節省 VRAM 給 Ollama LLM 使用；改成 `"cuda"` 可加快 Embedding 計算速度，但會多佔 1–2 GB VRAM |

> ⚠️ **換 Embedding 模型是高風險操作**：舊的 Embedding 快取（docstore.json 等）全部作廢，需要進 Tab 3 手動清除，或直接刪除 `text_result` 下所有子資料夾中的 JSON 快取檔案。

### LLM 模型（Ollama）

**預設值**位置：`app.py` 第 176 行、第 292 行

```python
# 啟動時的初始設定
Settings.llm = Ollama(model="qwen3:4b", request_timeout=1200.0)

# go_to_step4 的預設選項
default_model = "qwen3:4b" if "qwen3:4b" in current_models else current_models[0]
```

- `request_timeout=1200.0`：LLM 超時上限 20 分鐘，context 非常長（大型合約）時可能需要
- 改預設模型只需把 `"qwen3:4b"` 改成想要的模型名稱，但該模型必須先 `ollama pull` 下載
- UI 右上角的下拉選單會自動列出 Ollama 所有已下載的模型，**不需改程式碼就能切換模型**
- 模型清單會依 qwen 系列的版本號與參數量降冪排序（如 qwen3:32b > qwen3:14b > qwen3:4b），非 qwen 模型附在尾端

---

## 4. 關鍵參數彙整表

以下是所有散落在各檔案中、改了會有明顯效果的參數：

### app.py

| 參數/常數 | 位置 | 預設值 | 調整建議 |
|----------|------|-------|---------|
| `request_timeout` | `load_embedding_model` → Ollama 初始化 | **1200.0 秒** | 若合約超長（數百頁）且出現 timeout，可進一步提高 |
| Tab 7 tmpdir 前綴 | `_temp_reset` 第 2021 行 | `"sinbon_temp_"` | 若部署環境不同，修改此字串前綴以匹配暫存目錄命名規則 |

### doc_processor.py

| 參數/常數 | 位置 | 預設值 | 調整建議 |
|----------|------|-------|---------|
| `dpi=120`（預覽圖） | `generate_pdf_preview` | 120 | 提高到 150~200 可讓預覽更清晰，但記憶體占用增加 |
| OCR `max_new_tokens` | `pdf_process_and_save` → `ocr_page(...)` | **15000** | 一般合約頁面 4096 就夠，只有圖表密集時才需要 15000；降低可加速 OCR |
| `_pdf_word_count > 50` | `_pdf_word_count` | **50 字元** | 決定用 PDFPlumber 還是 OCR 的閾值。若發現某類合約被誤判為掃描檔，可提高此值 |
| DOCX 轉 PDF `timeout=60` | `docx_to_pdf_libreoffice` | 60 秒 | 複雜大型 DOCX 可能超時，可提高到 120 或 180 |

### parsetool.py

| 參數/常數 | 位置 | 預設值 | 調整建議 |
|----------|------|-------|---------|
| `similarity_top_k=3` | `HierarchicalQueryEngine.__init__` | **3** | 提高到 5 可獲得更多 context，回答更完整，但 LLM 推理時間增加；降低到 2 可加速 |
| `text_preview[:200]` | `query_with_complete_sections` | **200 字元** | 右側面板「相關條款」的預覽長度，不影響問答品質 |

### cache_manager.py

| 參數/常數 | 位置 | 預設值 | 調整建議 |
|----------|------|-------|---------|
| `top_k=3` | `retrieve_similar_qa` | **3** | QA 快取候選顯示筆數，改大讓使用者有更多選擇，但 UI 空間有限 |
| `CACHE_MARKER = "docstore.json"` | 模組頂部 | `docstore.json` | 不建議修改，這是判斷快取完整性的標記 |

### ui_helpers.py

| 參數/常數 | 位置 | 預設值 | 調整建議 |
|----------|------|-------|---------|
| `SIMILARITY_THRESHOLD` | 模組頂部，第 42 行 | **0.8** | QA 快取自動套用的相似度門檻。最高候選 ≥ 此值時直接投放快取答案，不呼叫 LLM。調高（如 0.9）可減少誤用快取；調低（如 0.7）可讓更多問題直接使用快取 |

---

## 5. 提示語（Prompt）詳解

系統有 **兩個主要 Prompt**，修改它們會直接影響 AI 的回答行為。

---

### 5-1. OCR 提示語（EXTRACT_PROMPT）

**位置**：`doc_processor.py` 第 249~257 行

```python
EXTRACT_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "ALL tables should be presented in HTML format.\n"
    'If there are images or figures in the page, present them as "", '
    "(left,top,right,bottom) are the coordinates of the top-left and bottom-right corners.\n"
    "Present all titles and headings as H1 headings.\n"
    "Do not hallucinate.\n"
)
```

**各行的作用與影響：**

| 指令 | 影響 | 建議修改情境 |
|------|------|------------|
| `"as if you were reading it naturally"` | 讓 OCR 以閱讀順序輸出，不跳行 | 通常不用改 |
| `"ALL tables should be presented in HTML format"` | 表格輸出為 `<table>` HTML | 若下游不需要 HTML 表格（如直接給 LLM），可改成「以純文字呈現表格」 |
| `'present them as "<IMAGE ...>"'` | 圖片輸出為佔位符號 + 座標 | 若不需要圖片資訊，可刪除這行（讓 OCR 忽略圖片） |
| `"Present all titles and headings as H1 headings"` | 標題輸出為 `# ` Markdown H1 | 若發現條款標題被誤標為 H1，可移除此行，改由 parsetool.py 的 regex 識別 |
| `"Do not hallucinate"` | 抑制幻覺（模型不確定時不要捏造） | **不建議移除**，OCR 模型在遇到模糊文字時容易捏造 |

**要加入的指令範例：**

```python
# 如果合約有大量直式文字或特殊排版，可加入：
"If the text is vertical (top-to-bottom), read it left column first, then right.\n"

# 如果需要保留頁碼資訊：
"If you see a page number at the bottom, output it as [PAGE: N].\n"
```

---

### 5-2. RAG 問答提示語（prompt）

**位置**：`parsetool.py` 第 979~983 行（`query_with_complete_sections` 方法內）

```python
prompt = (
    f"你是一個專業的合約助手。請根據以下提供的參考資料回答問題。\n"
    f"如果資料中沒有答案，請直接說明。\n\n"
    f"=== 參考資料 ===\n{context_text}\n\n"
    f"=== 使用者問題 ===\n{query}"
)
```

**注意**：`query` 這裡已經包含了語言指示，由 `rag_chat_response` 在傳入前附加：

```python
# ui_helpers.py 第 875 行
combined_prompt = f"{message}\n\n(請注意：請務必使用「{lang}」回答我。)"
```

所以最終送給模型的問題形如：
```
付款條件是什麼？

(請注意：請務必使用「繁體中文」回答我。)
```

**Prompt 各段的影響：**

| 段落 | 說明 | 修改影響 |
|------|------|---------|
| `"你是一個專業的合約助手"` | 角色設定（System Persona） | 改成「法律顧問」或「工程合約專家」可讓回答用語更專業；但太過具體的角色設定有時會讓模型過度推理超出合約內容 |
| `"請根據以下提供的參考資料回答問題"` | 限制 LLM 只看提供的資料 | **這行很重要**，移除後 LLM 可能會從自身訓練資料補充合約內沒有的條款 |
| `"如果資料中沒有答案，請直接說明"` | 防止 LLM 猜答案 | 移除後 LLM 傾向於「發揮創意」補充答案，尤其是 qwen3 系列模型 |
| `"=== 參考資料 ===" 分隔符` | 讓 LLM 清楚識別資料和問題的邊界 | 改成 XML tag 格式（`<context>...</context>`）對某些模型效果更好 |
| `"=== 使用者問題 ==="` | 問題的開始標記 | 改成 `[Question]` 或 `<question>` 也可以 |

**如何加入系統級指令（System Prompt）：**

目前 Prompt 用的是 `messages=[{"role": "user", "content": prompt}]` 單輪對話。
如果要加 System Message，修改 `parsetool.py` 第 999~1002 行：

```python
# 目前
stream = _ollama.chat(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)

# 加入 system message 的版本
stream = _ollama.chat(
    model=model_name,
    messages=[
        {"role": "system", "content": "你是一位台灣法律專業人士，熟悉工程合約與採購法規。"},
        {"role": "user",   "content": prompt}
    ],
    stream=True,
)
```

---

### 5-3. Context 組合格式

**位置**：`parsetool.py` 第 956~960 行

```python
context_text = "\n\n".join(
    f"---章節 {s['section_number']} ({s['section_title']})---\n"
    f"{s['full_content']}"
    for s in results['complete_sections']
)
```

每個章節的分隔標記格式 `---章節 N (標題)---` 是給 LLM 識別章節邊界用的。
若改成 XML 格式（`<section id="N" title="標題">...</section>`），某些模型的結構理解會更好：

```python
context_text = "\n\n".join(
    f"<section id=\"{s['section_number']}\" title=\"{s['section_title']}\">\n"
    f"{s['full_content']}\n</section>"
    for s in results['complete_sections']
)
```

---

## 6. OCR 相關參數

（同第 4 節 doc_processor.py 的參數表，此處補充說明判斷邏輯）

### PDFPlumber vs OCR 的判斷邏輯

**位置**：`doc_processor.py` `_pdf_word_count` 函式

系統用 PDFPlumber 嘗試提取每頁文字，若提取到的字元數 > 50，判定為「原生文字 PDF」，使用 PDFPlumber；否則判定為「掃描檔」，改用 OCRFlux-3B。

**常見誤判情境：**

| 情況 | 現象 | 解法 |
|------|------|------|
| PDF 有浮水印文字但主體是掃描 | 因浮水印字元 > 50 而誤用 PDFPlumber | 提高閾值到 200~500 |
| 原生 PDF 但文字稀少（圖表頁） | 被誤判為掃描檔，啟用 OCR | 接受，OCR 仍可提取；或提高閾值 |

---

## 7. RAG 索引與查詢參數

### similarity_top_k

**位置**：`parsetool.py` `HierarchicalQueryEngine.__init__`

| 數值 | 效果 |
|------|------|
| 2 | 速度最快，context 少，適合條文明確的問題 |
| 3（預設）| 平衡 |
| 5 | 更多 context，適合跨多條款的問題 | LLM prompt 更長，推理時間增加 |
| 8+ | 幾乎涵蓋所有相關條款 | Context 過長可能超過模型的 context window |

> **注意**：改這個值後，如果有 Embedding 快取存在，不需要重建快取。快取儲存的是向量，top-k 是查詢時的參數。

### 完整章節 vs. 只用片段

**UI 選項**：「使用完整章節」的 Checkbox（`use_full_context_chk`）

**程式碼位置**：`parsetool.py` 第 945~972 行

- **勾選（True）**：向量搜尋找到片段後，再去 `section_index` 取該章節的完整文字（主條款 + 所有子條款）。Context 更完整，但 prompt 更長。
- **不勾選（False）**：只用向量搜尋結果的片段。速度快，適合只問單一條款的問題。

---

## 8. 快取機制說明

系統有兩套完全獨立的快取，容易搞混：

### 快取類型對比

| 項目 | Embedding 快取 | QA 問答快取 |
|------|--------------|-----------|
| 存放位置 | `text_result/<合約名>/docstore.json` 等 | `text_result/<合約名>/qa_cache.json` |
| 內容 | 向量索引（LlamaIndex 格式） | 問答記錄（JSON 陣列） |
| 生成時機 | 第一次建立 RAG 索引時 | 使用者點「儲存此問答」時 |
| 失效條件 | 換 Embedding 模型後必須刪除 | 永遠有效（除非手動刪除）|
| 管理 Tab | Tab 3（Embedding 快取管理） | Tab 4（問答快取管理）|

> **Tab 7（快速問答）不產生任何快取**，不管是 Embedding 還是 QA 問答快取都不寫入。

### QA 快取相似度搜尋

**位置**：`cache_manager.py` `retrieve_similar_qa` 函式

每次使用者輸入問題時，系統先在 `qa_cache.json` 裡找相似的歷史問答，優先展示給使用者選擇。

有兩種比對方式（自動選擇）：

```python
# 方式 1：向量餘弦相似度（RAG 模型可用時）
q_emb = np.array(Settings.embed_model.get_text_embedding(question))
c_emb = np.array(Settings.embed_model.get_text_embedding(entry["question"]))
score = float(np.dot(q_emb, c_emb) / norm)

# 方式 2：字元集合交集（Fallback，不需 Embedding）
overlap = len(set(a) & set(b)) / max(len(set(a)), len(set(b)))
```

**自動套用邏輯**：當最高候選相似度 ≥ `SIMILARITY_THRESHOLD`（預設 **0.8**，`ui_helpers.py` 第 42 行），系統直接投放快取答案，不呼叫 LLM，也不顯示選擇面板。

**已知限制**：字元交集 Fallback 的準確度低，對「同義不同字」的問題（如「付款方式」vs「支付條件」）效果差。若要改善，可考慮：
- 增加 BM25 關鍵字搜尋作為 Fallback
- 或直接停用 Fallback，當 Embedding 模型不可用時回傳空列表

---

## 9. 合約條款解析邏輯

### 7 種條款格式（clause_number_patterns）

**位置**：`parsetool.py` 第 93~109 行

```python
clause_number_patterns = [
    re.compile(r"^(?:#\s*)?第\s*[一二三四五六七八九十百千萬\d]+\s*[條款項章節]", ...),  # 0. 第X條
    re.compile(r"^(?:#\s*)?[IVXLCDMivxlcdm]+\s*[、．.]", ...),                          # 1. 羅馬數字
    re.compile(r"^(?:#\s*)?[一二三四五六七八九十百千萬]+\s*[、．.]", ...),               # 2. 中文數字
    re.compile(r"^(?:#\s*)?[\(（]\s*[一二三四五六七八九十]+\s*[\)）]", ...),             # 3. (中文數字)
    re.compile(r"^(?:#\s*)?\d+\s*(?:[、．]|\.(?!\d))", ...),                             # 4. 阿拉伯數字
    re.compile(r"^(?:#\s*)?\(\d+\)", ...),                                                # 5. (數字)
    re.compile(r"^(?:#\s*)?\d+\.\d+", ...),                                              # 6. X.Y
]
```

**動態主條款識別邏輯**：
- 系統不是固定認定哪種格式是主條款
- 而是掃描合約，找出「第一個出現的條款所屬格式」作為主層級（`primary_level`）
- 比主層級索引數字更大的格式，自動視為子條款

**常見問題情境：**

| 問題 | 原因 | 解法 |
|------|------|------|
| 工程規格書的 `1.1`、`1.2` 被當主條款，其上層的 `第一條` 被忽略 | `第一條` 若排在所有 `1.X` 之後，系統會把 `1.X` 識別為主條款 | 確保合約原文的 `第一條` 出現在第一個 `1.X` 之前 |
| 附件頁被切割成很多零散節點 | 附件通常沒有標準條款格式，但可能有數字列表被誤判 | 上傳前只取合約本文頁碼，略過附件 |
| 某條款沒有被識別 | 條款格式不符合 7 種 pattern | 用 `parser.debug_potential_clauses(合約文字)` 診斷 |

**除錯方法**（在 Python 互動模式）：

```python
from parsetool import HierarchicalContractNodeParser

parser = HierarchicalContractNodeParser()
with open("你的合約.txt", encoding="utf-8") as f:
    text = f.read()

clauses = parser.debug_potential_clauses(text)
# 會印出所有偵測到的條款，以及它們的格式索引和優先級
```

---

## 10. 智能搜尋機制

**位置**：`ui_helpers.py` `build_folder_metadata_index` 和 `search_folders_by_description`

### 索引建立原則

智能搜尋的向量索引**只用資料夾名稱**，不讀合約內容。

```python
# ui_helpers.py 第 541 行
documents.append(Document(text=folder.name, metadata=metadata))
```

**影響**：搜尋效果完全取決於資料夾命名。

| 命名好壞 | 範例 | 搜尋效果 |
|---------|------|---------|
| ✅ 包含關鍵資訊 | `2024_大成建設_工程承攬合約_A棟` | 搜「大成工程合約」可找到 |
| ❌ 沒有語意 | `合約001`、`final_v3` | 任何描述都可能匹配到，結果不可靠 |

### 搜尋 top-k

**位置**：`ui_helpers.py` 第 581 行

```python
nodes = index.as_retriever(similarity_top_k=top_k).retrieve(description)
```

UI 有 5 個快速切換按鈕，預設 `top_k=5`。若想顯示更多候選，需要同時：
1. 把 `search_folders_by_description` 的 `top_k` 改大
2. 在 `app.py` 的 UI 定義區增加更多按鈕（Gradio 需要在初始化時固定數量）

---

## 11. State 管理與資料流

### file_state 的生命週期

`file_state` 是一個 Gradio State 字典，貫穿整個四步驟處理流程：

```python
# 初始結構（process_upload 建立）
state_data = {
    "original_file":  "/path/to/上傳的檔案.pdf",
    "file_ext":       ".pdf",          # 或 ".docx"
    "converted_pdf":  None,            # DOCX 轉換後的 PDF（DOCX 才有）
    "split_pdf_path": None,            # split_document 後填入
    "txt_path":       None,            # run_ocr_extraction 後填入
}
```

各步驟填入的欄位：

```
Step 1（process_upload）   → 填入 original_file, file_ext, converted_pdf
Step 2（split_document）   → 填入 split_pdf_path
Step 3（run_ocr_extraction）→ 填入 txt_path
Step 4（go_to_step4）      → 使用 txt_path 建立 RAG 索引
```

**從歷史紀錄載入時**，系統使用 fake_state：

```python
fake_state = {"original_file": "History Load", "txt_path": txt_path}
```

**從智能搜尋載入時**：

```python
{"original_file": "Smart Search", "txt_path": txt_path}
```

### Tab 7（快速問答）的 temp_file_state

Tab 7 使用獨立的 `temp_file_state`，結構與 Tab 2 的 `file_state` 相同，但**多了一個 `tmpdir` 欄位**：

```python
state_data = {
    "original_file":  "/path/to/上傳的檔案.pdf",
    "file_ext":       ".pdf",
    "converted_pdf":  None,
    "split_pdf_path": None,
    "txt_path":       None,
    "tmpdir":         "/tmp/sinbon_temp_xxxx",   # ← Tab 7 專屬：系統暫存目錄
}
```

`tmpdir` 在 `_temp_process_upload` 時用 `tempfile.mkdtemp(prefix="sinbon_temp_")` 建立，所有文字提取結果和切割暫存 PDF 都寫進這個目錄，而非 `TEXT_RESULT_DIR` 或 `TEMP_SPLIT_DIR`。點「重設」時 `_temp_reset` 會安全刪除此目錄（僅刪除路徑含 `sinbon_temp_` 的目錄）。

### rag_engine_state

`rag_engine_state` 是 `HierarchicalQueryEngine` 物件，一旦建立就常駐在 Gradio State 中，不需要每次問答重新建立。

**可能出問題的情境**：使用者在問答途中切換到另一份合約，`rag_engine_state` 會被覆蓋。若切換失敗（build index 報錯），`rag_engine_state` 可能變成 `None`，此時問答區會顯示「⚠️ 知識庫未就緒」。

---

## 12. 檔案更名注意事項

改任何檔名前，請先確認以下所有位置：

### 若要改 `parsetool.py` 的檔名

| 位置 | 行號 | 原本 | 需改成 |
|------|------|------|-------|
| `doc_processor.py` | 第 525 行 | `from parsetool import ...` | `from 新檔名 import ...` |

### 若要改 `ui_helpers.py` 的檔名

| 位置 | 行號 | 原本 | 需改成 |
|------|------|------|-------|
| `parsetool.py` | 第 997 行 | `from ui_helpers import _check_abort` | `from 新檔名 import _check_abort` |

### 若要改 `cache_manager.py` 的檔名

| 位置 | 行號 | 需修改 |
|------|------|-------|
| `doc_processor.py` | 第 54~58 行 | `from cache_manager import ...` |
| `ui_helpers.py` | 第 22~25 行 | `from cache_manager import ...` |
| `app.py` | 第 37~40 行 | `from cache_manager import ...` |

### 若要改 `doc_processor.py` 的檔名

| 位置 | 行號 | 需修改 |
|------|------|-------|
| `app.py` | 第 41~46 行 | `from doc_processor import ...` |

---

## 13. 已知限制與坑點

### A. Gradio 固定 5 個搜尋按鈕

`app.py` 的智能搜尋結果切換按鈕（`smart_doc_btn_0` ~ `smart_doc_btn_4`）在初始化時固定建立 5 個，透過 `visible` 控制顯示哪些。**Gradio 不支援動態增減元件數量**，若要增加到 8 個，需要在 UI 初始化的地方新增對應元件，並更新 `_DOC_SWITCHER_OUTPUTS` 列表。

### B. TEXT_RESULT_DIR 若路徑含中文

LibreOffice 的 `--outdir` 參數在 Windows 上處理含中文的路徑時，偶爾會輸出到意外位置。建議路徑全部使用英文或數字。

### C. OCR 處理中途關閉程式

OCR 進行中強制關閉程式，`text_result` 下的合約資料夾可能只有 `page0.jpg` 而沒有 `.txt` 檔案。下次啟動後這個資料夾仍會出現在歷史清單，但點「載入此專案」時會因為找不到 `.txt` 而報錯。需手動刪除此資料夾，或透過 Tab 6（刪除資料夾）刪除。

### D. Embedding 快取損壞

如果寫入 Embedding 快取時（`index.storage_context.persist(...)`）程式異常中斷，`docstore.json` 可能不完整。下次載入時 `load_index_from_storage` 會報 JSON 解析錯誤。

系統已有自動修復邏輯（`doc_processor.py` 第 552~554 行）：

```python
except Exception as e:
    print(f"⚠️ 快取讀取失敗，將重新建立：{e}")
    for f in _get_cache_json_files(cache):
        f.unlink()   # 刪掉損壞的快取，重新建立
```

但如果自動修復失敗，手動刪除對應資料夾下的 `docstore.json`、`index_store.json`、`vector_store.json` 即可。

### E. 每日滾動紀錄（DAILY_LOG）僅保留 24 小時

`_auto_save_daily_rolling_log` 每次寫入時會過濾掉超過 24 小時的舊紀錄，只保留最近一天。若需要長期保存問答紀錄，應使用「儲存此問答」按鈕寫入 `qa_cache.json`，而不是依賴 DAILY_LOG。

### F. LLM 模型名稱包含 `:` 冒號

Ollama 的模型名稱格式為 `qwen3:4b`（有冒號），此字串若被用在路徑或 JSON key，需注意轉義。目前程式碼中都是把它當字串傳遞，沒有直接用在路徑，尚無問題。

### G. 截斷機制的競爭條件

`_abort_flag` 是全域旗標，由 `threading.Lock` 保護。但如果使用者在上一個問答截斷後立刻送出新問題，`clear_abort()` 可能在舊的 streaming loop 還沒完全結束前就被呼叫，導致截斷旗標提前被清除。目前沒有 workaround，實際影響極小。

### H. Tab 7 暫存目錄在特定情境下殘留

`_temp_reset` 只會在使用者**主動點「重設」**時清除 `tmpdir`。若使用者直接關閉瀏覽器或程式崩潰，`tmpdir`（位於系統 `%TEMP%` 目錄，前綴 `sinbon_temp_`）不會自動清除，需手動刪除。大型合約的暫存目錄可能達數十 MB（頁面圖片）。

---

## 14. 除錯工具與 Log 說明

### 終端機輸出的時間計測

`parsetool.py` 的 `query_with_complete_sections` 在每個步驟都有計時輸出：

```
⏱️ [Start] 開始查詢: 付款條件是什麼？
⏱️ [Step 1] Vector Retrieval 耗時: 0.1230 秒 | 找到 3 個片段
⏱️ [Step 2] Metadata 解析耗時: 0.0005 秒
⏱️ [Step 4] LLM 生成耗時: 8.4210 秒
🏁 總共耗時: 8.5451 秒
```

用這個可以快速判斷瓶頸在哪（通常是 Step 4 LLM 生成）。

### 動態層級識別 Debug

`parsetool.py` 的 `_analyze_dynamic_hierarchy_structure` 在識別層級時會輸出：

```
動態識別主層級: 第一個條款模式索引 = 0
層級使用統計: {0: 15, 2: 43}
確定的主層級: 0 (第X條)
📊 解析完成，共 174 個節點
✅ 建立章節索引緩存：15 個章節
```

若「節點數」異常少（如只有 2~3 個），代表解析失敗，合約文字被當成單一 `full_document` 節點，問答效果會變差。用 `debug_potential_clauses` 方法排查。

### ERRORLOG 格式

每個回報錯誤的問答會存在：

```
ERRORLOG/<合約名稱>/YYYYMMDD_HHMMSS_error.json
```

內容範例：

```json
{
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "reported_at": "2025-06-01 14:32:05",
  "contract_folder": "2024_採購合約_甲方",
  "question": "保固期限是幾年？",
  "answer": "合約中未提到保固期限。",
  "model": "qwen3:14b",
  "lang": "繁體中文",
  "note": "使用者回報此答案有誤"
}
```

直接用文字編輯器或 JSON viewer 開啟即可，不需透過程式。

### 模型記憶體管理 Debug

若卸載 OCR 模型後 `nvidia-smi` 顯示 VRAM 未釋放，通常是 accelerate 的 dispatch hook 未正確移除。確認 `accelerate` 套件已安裝，並查看 `ocr_unload_model` 函式中 `remove_hook_from_module` 的例外輸出。

---

*以上資訊以本次交接版本為準，如有疑問請聯繫原開發者。*
