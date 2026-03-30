# doc_processor.py
"""
文件處理模組
包含：PDF/DOCX 處理、OCR 執行、RAG 索引建立與載入
"""

import os
import json
import shutil
import subprocess
import pdfplumber
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF

from cache_manager import (
    _cache_dir, _cache_marker, _get_cache_json_files,
    collection_exists_and_has_data,
)
TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"
TEMP_SPLIT_DIR  = r"C:\Users\user\Documents\rag_contract\temp_split_docs"
# ──────────────────────────────────────────────────────────────
# 臨時目錄
# ──────────────────────────────────────────────────────────────
TEMP_DIR = Path(TEMP_SPLIT_DIR)
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(exist_ok=True)

def cleanup_temp_dir():
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
    doc   = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count

def generate_pdf_preview(pdf_path: str) -> list:
    try:
        pages = get_pdf_pages(pdf_path)
        return convert_from_path(pdf_path, dpi=120, first_page=1, last_page=pages)
    except Exception as e:
        print(f"Preview Error: {e}")
        return []

def docx_to_pdf_libreoffice(docx_path: str, output_dir: Path) -> Path:
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
    """從 source_path 切割 [start, end] 頁（1-based）回傳輸出路徑"""
    doc        = fitz.open(source_path)
    total      = doc.page_count
    end        = min(end, total)
    out_name   = f"split_{out_stem}_{start}-{end}.pdf"
    out_path   = TEMP_DIR / out_name
    new_doc    = fitz.open()
    for i in range(start - 1, end):
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
    new_doc.save(str(out_path))
    doc.close()
    new_doc.close()
    return out_path


# ==========================================
# B. OCR / 文字提取
# ==========================================

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
    image    = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text",  "text":  EXTRACT_PROMPT},
        ]},
    ]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    output_ids     = model.generate(**inputs, temperature=0.0,
                                    max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids  = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text    = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]

def _pdf_word_count(path: str) -> int:
    with pdfplumber.open(path) as pdf:
        if pdf.pages:
            text = pdf.pages[0].extract_text()
            return len(text) if text else 0
    return 0

def pdf_process_and_save(pdf_path: str, ocr_model, ocr_processor,
                         progress=None) -> tuple[str, str]:
    """
    處理 PDF：自動選擇 PDFPlumber 或 OCR，
    回傳 (狀態訊息, txt 檔路徑)
    """
    if progress:
        progress(0.1, desc="初始化...")

    fname = os.path.basename(pdf_path).split(".")[0]
    parts = fname.split("_")
    if len(parts) > 2:
        fname = "_".join(parts[1:-1])

    ocrpath = str(Path(TEXT_RESULT_DIR) / fname)
    if os.path.exists(ocrpath):
        shutil.rmtree(ocrpath)
    os.makedirs(ocrpath, exist_ok=True)

    if progress:
        progress(0.2, desc="PDF 轉圖片...")
    pages    = convert_from_path(pdf_path)
    pagelist = []
    for i, p in enumerate(pages):
        p_path = f"{ocrpath}/page{i}.jpg"
        pagelist.append(p_path)
        p.save(p_path, "JPEG")

    full_text = ""
    if progress:
        progress(0.4, desc="分析文字量...")

    use_plumber = _pdf_word_count(pdf_path) > 50
    if use_plumber:
        method = "PDFPlumber"
        if progress:
            progress(0.5, desc="提取原生文字...")
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    full_text += t + "\n"
    else:
        method = "OCR"
        if progress:
            progress(0.5, desc="執行 AI OCR...")
        if ocr_model is None:
            full_text = "❌ 模型未載入"
        else:
            for i, p_path in enumerate(pagelist):
                if progress:
                    progress(0.5 + 0.4 * (i / len(pagelist)),
                             desc=f"辨識第 {i+1} 頁...")
                try:
                    res = ocr_page(p_path, ocr_model, ocr_processor, 15000)
                    try:
                        j = json.loads(res)
                        full_text += j.get("natural_text", str(j))
                    except Exception:
                        full_text += res
                    full_text += "\n"
                except Exception:
                    full_text += f"\n[Error Page {i}]\n"

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
    若本地有快取則直接載入，否則重新建立向量索引。
    回傳 (engine, 狀態訊息)
    """
    from llama_index.core import (
        Document, StorageContext, VectorStoreIndex, load_index_from_storage,
    )
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
            storage_context = StorageContext.from_defaults(persist_dir=str(cache))
            index           = load_index_from_storage(storage_context)
            if progress:
                progress(0.8, desc="🔗 組裝查詢引擎...")
            nodes  = list(index.docstore.docs.values())
            engine = HierarchicalQueryEngine(index, nodes)
            if progress:
                progress(1.0, desc="✅ 完成！")
            return engine, f"✅ 已從本地快取載入（{len(nodes)} 個節點）：{cache.name}"
        except Exception as e:
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

    doc = Document(
        text=text_content,
        metadata={
            "file_name":   Path(txt_path).name,
            "file_path":   txt_path,
            "contract_id": Path(txt_path).stem,
            "orig_path":   txt_path,
        },
    )

    parser = HierarchicalContractNodeParser()
    nodes  = parser.get_nodes_from_documents([doc])
    print(f"📊 解析完成，共 {len(nodes)} 個節點")

    if not nodes:
        return None, "❌ 文件解析失敗，未能提取任何節點"

    if progress:
        progress(0.4, desc=f"⚙️ 計算 Embedding（{len(nodes)} 個節點）...")

    try:
        storage_context = StorageContext.from_defaults()
        index           = VectorStoreIndex(nodes, storage_context=storage_context,
                                           show_progress=False)
    except BaseException as e:
        import traceback
        print("[ERROR] 建立向量索引失敗：")
        print(traceback.format_exc())
        return None, f"❌ 建立向量索引失敗：{type(e).__name__}: {e}"

    if progress:
        progress(0.85, desc="💾 儲存 Embedding 至磁碟...")

    saved_ok = False
    try:
        cache.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(cache))
        saved_ok = _cache_marker(cache).exists()
    except Exception as e:
        print(f"⚠️ 快取儲存失敗：{e}")

    if progress:
        progress(0.95, desc="🔗 組裝查詢引擎...")

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
    txt_path = state_data.get("txt_path")
    if not txt_path or not os.path.exists(txt_path):
        return None, "❌ 找不到文字檔案，請先回上一步執行 OCR。"
    return build_or_load_index(txt_path, rag_available, progress=progress)
