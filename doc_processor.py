# doc_processor.py
"""
文件處理模組
============
本模組負責「原始文件 → RAG 可查詢索引」的完整處理流程：

  A. PDF 工具
     - 讀取頁數（get_pdf_pages）
     - 生成預覽圖（generate_pdf_preview）
     - DOCX 轉 PDF（docx_to_pdf_libreoffice，需安裝 LibreOffice）
     - 切割 PDF 頁碼範圍（split_pdf）

  B. OCR / 文字提取
     - 自動選擇 PDFPlumber（原生文字 PDF）或 OCR 模型（掃描檔）
     - OCR 使用本地 OCRFlux-3B 視覺語言模型（ocr_page）
     - 將提取結果儲存為 .txt 檔案（pdf_process_and_save）

  C. RAG 索引建立 / 載入
     - 快取存在時直接從磁碟載入，跳過 Embedding 計算（節省時間）
     - 快取不存在時呼叫 parsetool 解析合約結構，
       再透過 LlamaIndex 建立向量索引（build_or_load_index）
     - build_rag_index 為 app.py 呼叫的入口函式

【處理流程全覽】
  使用者上傳 PDF/DOCX
    → process_upload（app.py）: 格式偵測、DOCX 轉 PDF、頁數讀取
    → split_document（app.py）: 依頁碼切割到暫存目錄
    → run_ocr_extraction（app.py）→ pdf_process_and_save（本模組）: 文字提取
    → go_to_step4（app.py）→ build_rag_index（本模組）: 建立向量索引
    → HierarchicalQueryEngine（parsetool.py）: 問答

【相依關係】
  - 引用 cache_manager.py：判斷快取是否存在、讀取快取目錄路徑
  - 引用 parsetool.py（HierarchicalContractNodeParser, HierarchicalQueryEngine）：
    合約結構解析和問答引擎
    ⚠️ 若要更改 parsetool.py 的檔名，需修改本檔第 15 行的 import 陳述式
  - 被 app.py 引用：作為核心處理流程的執行者

【路徑設定注意事項】
  TEXT_RESULT_DIR 和 TEMP_SPLIT_DIR 在本模組頂部定義，
  換機器時需同步修改 cache_manager.py / ui_helpers.py / app.py 中的同名常數。
"""

import os
import json
import shutil
import subprocess
import pdfplumber
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF：用於頁數讀取、PDF 切割

from cache_manager import (
    _cache_dir,                     # 取得快取目錄（.txt 的 parent）
    _cache_marker,                  # 取得快取標記檔案路徑（docstore.json）
    _get_cache_json_files,          # 列出所有快取 JSON 檔案
    collection_exists_and_has_data, # 判斷快取是否存在且有效
)

# 所有合約 OCR 結果的根儲存目錄
# ⚠️ 換機器或搬移路徑時，需同步修改 cache_manager.py / ui_helpers.py / app.py
TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"

# 切割後的暫存 PDF 儲存目錄
# 每次啟動程式時清空（見下方 TEMP_DIR 初始化邏輯）
# 不影響 TEXT_RESULT_DIR 的已處理資料
TEMP_SPLIT_DIR = r"C:\Users\user\Documents\rag_contract\temp_split_docs"

# ──────────────────────────────────────────────────────────────
# 暫存目錄初始化
# ──────────────────────────────────────────────────────────────
# 每次程式啟動時清空 TEMP_DIR，確保不殘留前次的切割暫存檔。
# 若 TEMP_DIR 不存在則建立；若已存在則整個清空後重建。
TEMP_DIR = Path(TEMP_SPLIT_DIR)
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(exist_ok=True)


def cleanup_temp_dir():
    """
    手動清空暫存目錄（TEMP_DIR）中的所有檔案與子資料夾。
    在使用者上傳新檔案時（process_upload）呼叫，
    避免前次切割的 PDF 殘留影響當前操作。

    與程式啟動時的整體清空不同：
      - 程式啟動時：整個 TEMP_DIR 刪除重建
      - cleanup_temp_dir：保留目錄本身，只清空內容
    """
    if TEMP_DIR.exists():
        for item in TEMP_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


# ==========================================
# A. PDF 工具
# ==========================================

def get_pdf_pages(pdf_path: str) -> int:
    """
    使用 PyMuPDF (fitz) 讀取 PDF 的總頁數。
    讀取完立即關閉，不佔用檔案鎖。

    選用 fitz 而非 pdfplumber 是因為 fitz 速度快，
    只需讀取 PDF header 不需解析全文。

    Parameters
    ----------
    pdf_path : PDF 檔案的完整路徑字串

    Returns
    -------
    int：PDF 總頁數
    """
    doc   = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count


def generate_pdf_preview(pdf_path: str) -> list:
    """
    將 PDF 每頁轉成預覽圖片（PIL Image 物件列表），
    供 Gradio Gallery 元件顯示。

    dpi=120：預覽圖解析度（比一般文字辨識低），平衡顯示品質和記憶體使用量。
      - dpi=72：低品質，頁數多時速度快
      - dpi=120：適合預覽，清晰但不佔太多記憶體（目前設定）
      - dpi=300：高品質，適合 OCR，但記憶體消耗大

    Parameters
    ----------
    pdf_path : PDF 檔案的完整路徑字串

    Returns
    -------
    list[PIL.Image]：每頁圖片的列表；轉換失敗時回傳空列表
    """
    try:
        pages = get_pdf_pages(pdf_path)
        return convert_from_path(pdf_path, dpi=120, first_page=1, last_page=pages)
    except Exception as e:
        print(f"Preview Error: {e}")
        return []


def docx_to_pdf_libreoffice(docx_path: str, output_dir: Path) -> Path:
    """
    使用 LibreOffice headless 模式將 DOCX 轉換為 PDF。
    轉換後的 PDF 儲存到 output_dir，檔名與 .docx 相同（只換副檔名）。

    【前提條件】
    - LibreOffice 必須安裝且可透過 soffice 指令呼叫
    - Windows 上通常需要將 LibreOffice 加入 PATH，或使用完整路徑
    - 轉換 timeout 設定為 60 秒，複雜 DOCX 可能需要調高

    【LibreOffice 指令說明】
      soffice --headless          : 不開啟 GUI
      --convert-to pdf            : 指定輸出格式
      --outdir <dir>              : 輸出目錄
      <docx_path>                 : 輸入 DOCX 路徑

    Parameters
    ----------
    docx_path  : DOCX 檔案的完整路徑字串
    output_dir : 輸出 PDF 的目標目錄 Path

    Returns
    -------
    Path：轉換後 PDF 的完整路徑

    Raises
    ------
    RuntimeError：LibreOffice 未安裝、轉換失敗或超時時拋出
    """
    pdf_name    = Path(docx_path).stem + ".pdf"
    output_path = output_dir / pdf_name
    try:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf",
             "--outdir", str(output_dir), str(docx_path)],
            check=True, timeout=60,
        )
        if output_path.exists():
            return output_path
        raise FileNotFoundError("Conversion failed")
    except Exception as e:
        raise RuntimeError(f"DOCX to PDF Error: {e}")


def split_pdf(source_path: str, start: int, end: int, out_stem: str) -> Path:
    """
    從 source_path 切割出 [start, end] 頁（1-based 頁碼），
    儲存為新 PDF 到暫存目錄（TEMP_DIR）。

    【為什麼需要切割？】
    合約有時是整本文件，使用者只需要特定幾頁（如某個章節）。
    切割後的 PDF 較小，可加快後續 OCR 處理速度。

    【檔名格式】
    split_<原始檔名>_<起始頁>-<結束頁>.pdf
    例：split_2024工程合約_3-15.pdf

    【end 頁碼自動修正】
    若 end 超過 PDF 總頁數，自動降至最後一頁，不會拋出例外。

    Parameters
    ----------
    source_path : 來源 PDF 的完整路徑字串
    start       : 切割起始頁（1-based，包含）
    end         : 切割結束頁（1-based，包含；超出時自動修正）
    out_stem    : 輸出檔名的主體部分（通常為原始檔名去掉副檔名）

    Returns
    -------
    Path：切割後 PDF 的暫存路徑
    """
    doc      = fitz.open(source_path)
    total    = doc.page_count
    end      = min(end, total)          # 防止 end 超過總頁數
    out_name = f"split_{out_stem}_{start}-{end}.pdf"
    out_path = TEMP_DIR / out_name
    new_doc  = fitz.open()
    # fitz 使用 0-based 頁碼，因此 start-1 和 end-1
    for i in range(start - 1, end):
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
    new_doc.save(str(out_path))
    doc.close()
    new_doc.close()
    return out_path


# ==========================================
# B. OCR / 文字提取
# ==========================================

# EXTRACT_PROMPT：傳給 OCR 視覺模型的指令提示。
# 定義模型應如何呈現各種元素：
#   - 一般文字：直接輸出純文字
#   - 表格：以 HTML <table> 格式呈現（保留結構）
#   - 圖片/圖形：輸出 <IMAGE> 標籤加座標
#   - 標題：以 H1 標題格式呈現
#   - 嚴禁幻覺（do not hallucinate）
# 若要調整 OCR 輸出格式，修改此常數即可，不需改動 ocr_page 函式。
EXTRACT_PROMPT = (
    "Below is the image of one page of a document. "
    "Just return the plain text representation of this document as if you were reading it naturally.\n"
    "ALL tables should be presented in HTML format.\n"
    'If there are images or figures in the page, present them as "", '
    "(left,top,right,bottom) are the coordinates of the top-left and bottom-right corners.\n"
    "Present all titles and headings as H1 headings.\n"
    "Do not hallucinate.\n"
)


def ocr_page(image_path: str, model, processor, max_new_tokens: int = 4096) -> str:
    """
    使用本地 OCRFlux-3B 視覺語言模型，對單頁圖片執行 OCR。
    輸入一張頁面圖片，輸出該頁的文字內容（含 HTML 表格）。

    【模型推理完整流程】
    1. PIL.Image.open 載入圖片（RGB 格式）
    2. 組合多模態訊息（system prompt + 頁面圖片 + EXTRACT_PROMPT 文字指令）
    3. processor.apply_chat_template → 將對話格式轉成模型的文字輸入格式
    4. processor 對圖片和文字做 tokenization，產生 input_ids / pixel_values
    5. inputs.to(model.device) → 確保張量在正確的裝置（GPU/CPU）
    6. model.generate：temperature=0.0（確定性輸出）、do_sample=False（停用隨機取樣）
    7. 只截取「新生成的 token」（去掉輸入部分），decoder 解碼為文字

    【max_new_tokens 設定建議】
    - 一般文字頁（段落、合約條款）：4096（預設值）
    - 表格密集頁（工程規格、報價單）：8192~15000
    - pdf_process_and_save 呼叫時傳入 15000，以應對複雜合約全頁表格

    【temperature=0.0 的意義】
    確定性輸出，相同頁面每次 OCR 結果完全一致。
    避免因隨機取樣導致不同執行結果，方便除錯與快取複用。

    Parameters
    ----------
    image_path     : 頁面圖片的完整路徑（JPEG 或 PNG）
    model          : OCRFlux-3B 模型物件（AutoModelForImageTextToText）
    processor      : 對應的 AutoProcessor（負責多模態輸入的 tokenization）
    max_new_tokens : 輸出最大 token 數（預設 4096，複雜頁面請傳入更大值）

    Returns
    -------
    str：OCR 辨識出的頁面文字，表格部分為 HTML 格式（如 <table><tr><td>...</td></tr></table>）
    """
    image    = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text":  EXTRACT_PROMPT},
        ]},
    ]
    # apply_chat_template：將對話格式轉換成模型可接受的輸入文字格式
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    # 將輸入張量移到與模型相同的裝置（GPU 或 CPU）
    inputs = inputs.to(model.device)
    output_ids    = model.generate(**inputs, temperature=0.0,
                                   max_new_tokens=max_new_tokens, do_sample=False)
    # 只取「新生成的部分」（去掉輸入 tokens），避免 prompt 文字混入輸出
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text   = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


def _pdf_word_count(path: str) -> int:
    """
    快速偵測 PDF 第一頁是否含有足夠的原生文字，
    用來決定要用 PDFPlumber 還是 OCR。

    【判斷邏輯】
    只讀第一頁（代表性頁面），計算文字字元數：
      - > 50 字元 → 判定為「有原生文字的 PDF」→ 使用 PDFPlumber（快）
      - ≤ 50 字元 → 判定為「掃描 PDF」→ 使用 OCR（慢但準）

    【為什麼只看第一頁？】
    合約第一頁通常是標題/前言，若第一頁就有文字，
    後續頁面大概率也有（掃描檔通常全部頁面都沒有）。
    避免讀取整份文件，提升速度。

    【閾值 50 的選擇依據】
    合約封面頁通常有 50 字以上；
    掃描 PDF 即使有少量雜訊字元也不超過 50。
    若發現某類合約被誤判，可在此調整閾值。

    【此函式為私有函式（名稱以 _ 開頭）】
    僅供 pdf_process_and_save 內部呼叫，
    外部模組不應直接使用。

    Parameters
    ----------
    path : PDF 檔案路徑字串

    Returns
    -------
    int：第一頁的文字字元數（pdfplumber 無法提取時回傳 0）
    """
    with pdfplumber.open(path) as pdf:
        if pdf.pages:
            text = pdf.pages[0].extract_text()
            return len(text) if text else 0
    return 0


def pdf_process_and_save(pdf_path: str, ocr_model, ocr_processor,
                         progress=None) -> tuple[str, str]:
    """
    完整的 PDF 文字提取流程，自動選擇最佳提取方式並儲存結果。
    這是 app.py 呼叫的核心函式，對應 UI 的「Step 3：執行 OCR / 文字提取」。

    【自動選擇邏輯】
    _pdf_word_count > 50：PDFPlumber（原生文字直接提取，速度快）
    _pdf_word_count ≤ 50：OCR（頁面轉圖片後用 OCRFlux-3B 辨識，速度慢但準確）

    【輸出目錄結構】
    TEXT_RESULT_DIR/<合約名稱>/
      ├── <合約名稱>.txt   ← 提取的純文字，後續 RAG 使用
      └── page0.jpg, page1.jpg, ... ← 各頁圖片

    【檔名處理邏輯（fname 變數）】
    原始 PDF 可能由 split_pdf 產生，檔名格式為 "split_<原始名>_<頁碼>.pdf"。
    此函式會嘗試去掉「split_」前綴和「_頁碼」後綴，只保留原始合約名稱，
    作為輸出目錄名稱：
      split_2024工程合約_3-15.pdf → fname = "2024工程合約"

    【進度回調（progress）】
    progress 是 Gradio gr.Progress 物件，呼叫 progress(0.0~1.0, desc="...")
    可在 UI 顯示進度條。傳入 None 時忽略所有進度更新。

    Parameters
    ----------
    pdf_path      : 要處理的 PDF 路徑（通常是 TEMP_DIR 中的切割暫存 PDF）
    ocr_model     : OCRFlux-3B 模型物件（若為 None 且需要 OCR 則回傳錯誤訊息）
    ocr_processor : 對應的 AutoProcessor
    progress      : Gradio progress 物件（可選）

    Returns
    -------
    tuple[str, str]：
      [0] 狀態訊息（"完成 (PDFPlumber)" 或 "完成 (OCR)"）
      [1] 輸出 .txt 檔案的完整路徑字串
    """
    if progress:
        progress(0.1, desc="初始化...")

    # ── 檔名處理：從 split_<名稱>_<頁碼>.pdf 提取合約名稱 ──
    fname = os.path.basename(pdf_path).split(".")[0]
    parts = fname.split("_")
    if len(parts) > 2:
        # split_合約A_3-15 → parts = ["split", "合約A", "3-15"] → 取中間部分
        fname = "_".join(parts[1:-1])

    # ── 建立輸出目錄（若已存在則清空重建，避免舊資料干擾）──
    ocrpath = str(Path(TEXT_RESULT_DIR) / fname)
    if os.path.exists(ocrpath):
        shutil.rmtree(ocrpath)
    os.makedirs(ocrpath, exist_ok=True)

    if progress:
        progress(0.2, desc="PDF 轉圖片...")

    # ── 將 PDF 所有頁面轉成 JPEG 圖片（無論用哪種提取方式都需要）──
    # 圖片供後續 UI 的「查看原稿」功能使用
    pages    = convert_from_path(pdf_path)
    pagelist = []
    for i, p in enumerate(pages):
        p_path = f"{ocrpath}/page{i}.jpg"
        pagelist.append(p_path)
        p.save(p_path, "JPEG")

    full_text = ""

    if progress:
        progress(0.4, desc="分析文字量...")

    # ── 自動選擇提取方式 ──────────────────────────────────────
    use_plumber = _pdf_word_count(pdf_path) > 50
    if use_plumber:
        # PDFPlumber：直接從 PDF 結構提取文字，速度快
        method = "PDFPlumber"
        if progress:
            progress(0.5, desc="提取原生文字...")
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    full_text += t + "\n"
    else:
        # OCR：頁面圖片 → OCRFlux-3B → 文字
        method = "OCR"
        if progress:
            progress(0.5, desc="執行 AI OCR...")
        if ocr_model is None:
            # OCR 模型未載入（啟動時載入失敗），無法處理掃描 PDF
            full_text = "❌ 模型未載入"
        else:
            for i, p_path in enumerate(pagelist):
                if progress:
                    progress(0.5 + 0.4 * (i / len(pagelist)),
                             desc=f"辨識第 {i+1} 頁...")
                try:
                    # max_new_tokens=15000：合約頁面可能有大量文字/表格
                    res = ocr_page(p_path, ocr_model, ocr_processor, 15000)
                    try:
                        # 部分頁面 OCR 結果可能是 JSON 格式，嘗試解析取 natural_text 欄位
                        j = json.loads(res)
                        full_text += j.get("natural_text", str(j))
                    except Exception:
                        # 非 JSON 格式（一般文字），直接使用
                        full_text += res
                    full_text += "\n"
                except Exception:
                    full_text += f"\n[Error Page {i}]\n"

    # ── 儲存提取結果為 .txt 檔案 ──────────────────────────────
    txt_path = f"{ocrpath}/{fname}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    if progress:
        progress(1.0, desc="完成")
    return f"完成 ({method})", txt_path


# ==========================================
# C. RAG 索引建立 / 載入
# ==========================================

def build_or_load_index(txt_path: str, rag_available: bool, progress=None):
    """
    建立或從快取載入 RAG 向量索引，回傳可供問答的 HierarchicalQueryEngine。

    【快取機制（優先載入快取）】
    若本地已有 Embedding 快取（docstore.json 存在）：
      → 直接從磁碟 StorageContext 載入，跳過所有 Embedding 計算
      → 適合反覆問同一份合約的情境

    若快取不存在或載入失敗：
      → 重新解析合約文字（HierarchicalContractNodeParser）
      → 重新計算所有節點的 Embedding（VectorStoreIndex）
      → 儲存到磁碟（作為下次的快取）

    【parsetool.py 的角色】
    - HierarchicalContractNodeParser：將合約 .txt 文字解析成階層式節點
      （辨識「第X條」「X、」等條款結構，建立父子關係）
    - HierarchicalQueryEngine：組合向量索引和節點，提供 query_with_complete_sections 介面
    ⚠️ 若要更改 parsetool.py 的檔名，此處的 from ... import 需同步修改

    【LlamaIndex Document 結構】
    每份合約建立一個 Document 物件，metadata 包含：
      file_name   : .txt 檔案名稱（供 RAG 回答時標注來源）
      file_path   : .txt 完整路徑
      contract_id : .txt 的 stem（不含副檔名），作為合約的唯一識別
      orig_path   : 與 file_path 相同，供 node metadata 備用

    Parameters
    ----------
    txt_path      : 合約 .txt 檔案路徑
    rag_available : 系統是否有可用的 LlamaIndex + Embedding 模型
                    False 時直接回傳錯誤，不嘗試建立索引
    progress      : Gradio progress 物件（可選）

    Returns
    -------
    tuple[HierarchicalQueryEngine | None, str]：
      [0] 問答引擎物件（失敗時為 None）
      [1] 狀態訊息（成功/失敗原因）
    """
    from llama_index.core import (
        Document, StorageContext, VectorStoreIndex, load_index_from_storage,
    )
    # ⚠️ 以下 import 依賴 parsetooltestspeedfix.py 的檔名
    # 若要更改 parsetool.py 的檔名，此行必須同步修改
    from parsetool import HierarchicalContractNodeParser, HierarchicalQueryEngine

    if not rag_available:
        return None, "❌ 系統缺少 RAG 組件，無法建立索引。"
    if not txt_path or not os.path.exists(txt_path):
        return None, "❌ 找不到文字檔案，請先執行 OCR。"

    cache = _cache_dir(txt_path)

    # ── 嘗試從快取載入 ──────────────────────────────────────────
    if collection_exists_and_has_data(txt_path):
        if progress:
            progress(0.2, desc="📂 發現本地快取，直接載入 Embedding...")
        try:
            # StorageContext.from_defaults：從磁碟目錄載入所有 JSON 快取檔案
            storage_context = StorageContext.from_defaults(persist_dir=str(cache))
            index           = load_index_from_storage(storage_context)
            if progress:
                progress(0.8, desc="🔗 組裝查詢引擎...")
            # docstore.docs.values()：取出所有已儲存的 TextNode 物件
            nodes  = list(index.docstore.docs.values())
            engine = HierarchicalQueryEngine(index, nodes)
            if progress:
                progress(1.0, desc="✅ 完成！")
            return engine, f"✅ 已從本地快取載入（{len(nodes)} 個節點）：{cache.name}"
        except Exception as e:
            # 快取損壞（如寫入中斷）：刪除損壞的快取，重新建立
            print(f"⚠️ 快取讀取失敗，將重新建立：{e}")
            for f in _get_cache_json_files(cache):
                f.unlink()

    # ── 重新建立索引 ────────────────────────────────────────────
    if progress:
        progress(0.1, desc="📖 讀取文件內容...")
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    except Exception as e:
        return None, f"❌ 讀取文件失敗：{e}"

    if not text_content.strip():
        return None, "❌ 文件內容為空"

    if progress:
        progress(0.2, desc="🔍 解析合約結構（階層式切割）...")

    # 建立 LlamaIndex Document 物件，附加合約 metadata
    doc = Document(
        text=text_content,
        metadata={
            "file_name":   Path(txt_path).name,
            "file_path":   txt_path,
            "contract_id": Path(txt_path).stem,  # 不含副檔名的合約名稱，作為唯一識別
            "orig_path":   txt_path,
        },
    )

    # HierarchicalContractNodeParser：識別條款結構，將文字切割成有階層關係的節點
    # 節點類型：header / main_clause / sub_clause / complete_section
    parser = HierarchicalContractNodeParser()
    nodes  = parser.get_nodes_from_documents([doc])
    print(f"📊 解析完成，共 {len(nodes)} 個節點")

    if not nodes:
        return None, "❌ 文件解析失敗，未能提取任何節點"

    if progress:
        progress(0.4, desc=f"⚙️ 計算 Embedding（{len(nodes)} 個節點）...")

    try:
        # VectorStoreIndex：對所有節點計算 Embedding，建立向量搜尋索引
        # show_progress=False：不在終端顯示進度（避免和 Gradio 進度條衝突）
        storage_context = StorageContext.from_defaults()
        index           = VectorStoreIndex(nodes, storage_context=storage_context,
                                           show_progress=False)
    except BaseException as e:
        # 使用 BaseException 而非 Exception，捕捉 KeyboardInterrupt / MemoryError 等
        import traceback
        print("[ERROR] 建立向量索引失敗：")
        print(traceback.format_exc())
        return None, f"❌ 建立向量索引失敗：{type(e).__name__}: {e}"

    if progress:
        progress(0.85, desc="💾 儲存 Embedding 至磁碟...")

    # ── 儲存索引到磁碟作為快取 ──────────────────────────────────
    saved_ok = False
    try:
        cache.mkdir(parents=True, exist_ok=True)
        # persist：將 docstore / vector_store / index_store 等 JSON 寫入 cache 目錄
        index.storage_context.persist(persist_dir=str(cache))
        # 確認標記檔案（docstore.json）是否成功寫入，作為快取有效的標誌
        saved_ok = _cache_marker(cache).exists()
    except Exception as e:
        print(f"⚠️ 快取儲存失敗：{e}")

    if progress:
        progress(0.95, desc="🔗 組裝查詢引擎...")

    # HierarchicalQueryEngine：組合 index 和 nodes，提供 query_with_complete_sections 方法
    engine = HierarchicalQueryEngine(index, nodes)

    if progress:
        progress(1.0, desc="✅ 完成！")

    status = (
        f"✅ 索引建立完成（{len(nodes)} 個節點），已存入快取：{cache.name}"
        if saved_ok else
        f"⚠️ 索引建立完成（{len(nodes)} 個節點），但快取儲存失敗"
    )
    return engine, status


def build_rag_index(state_data: dict, rag_available: bool, progress=None):
    """
    app.py 呼叫的入口函式，從 state_data 提取 txt_path 後呼叫 build_or_load_index。

    【state_data 結構】
    {
        "original_file":  原始上傳檔案路徑,
        "file_ext":       副檔名（.pdf 或 .docx）,
        "converted_pdf":  DOCX 轉換後的 PDF 路徑（DOCX 才有）,
        "split_pdf_path": 切割後的暫存 PDF 路徑,
        "txt_path":       OCR 完成後的 .txt 路徑 ← 本函式使用此欄位
    }

    特殊情境：
    - 從歷史紀錄載入時，load_history_and_go_to_step4 會傳入
      fake_state = {"original_file": "History Load", "txt_path": txt_path}
    - 從智能搜尋載入時，confirm_yes_and_load_doc 會傳入
      {"original_file": "Smart Search", "txt_path": txt_path}

    Parameters
    ----------
    state_data    : Gradio State 字典（必須包含 "txt_path" 鍵）
    rag_available : 系統是否有可用的 RAG 組件
    progress      : Gradio progress 物件（可選）

    Returns
    -------
    tuple[HierarchicalQueryEngine | None, str]：
      同 build_or_load_index 的回傳值
    """
    txt_path = state_data.get("txt_path")
    if not txt_path or not os.path.exists(txt_path):
        return None, "❌ 找不到文字檔案，請先回上一步執行 OCR。"
    return build_or_load_index(txt_path, rag_available, progress=progress)
