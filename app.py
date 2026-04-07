# app.py
"""
智能合約處理系統 — 主程式

負責：
  1. 模型載入（OCRFlux-3B / Yuan-embedding-2.0-zh / Ollama LLM）
  2. Gradio 介面定義（四個 Tab、所有 UI 元件）
  3. 所有 UI 事件的邏輯函式（上傳、切割、OCR、RAG 問答、智能搜尋）
  4. Gradio State 管理（file_state 貫穿整個處理流程）

【四個 Tab】
  Tab 1：🔍 智能檢索回答  — 輸入描述文字自動找合約並問答
  Tab 2：📁 檔案上傳與手動搜尋 — 四步驟手動處理新合約
  Tab 3：🗃️ Embedding 快取管理 — 查看 / 刪除向量索引快取
  Tab 4：💬 問答快取管理 — 查看 / 刪除歷史問答快取

【模型載入策略】
  OCR 模型和 RAG 模型（Embedding）在啟動時載入到記憶體。
  LLM（Ollama）在每次問答時重新初始化（確保 UI 切換模型立即生效）。
  任一模型載入失敗都不會讓整個程式崩潰，只影響對應功能。
"""

import gradio as gr
import ollama
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# OCRFlux-3B 模型的本地目錄路徑
# ⚠️ 換機器或搬移模型時需修改此路徑
OCR_MODEL_PATH = r"C:\Users\user\Documents\rag_contract\ocrmodel"

# 所有合約結果的根儲存目錄
# ⚠️ 換路徑時需同步修改 doc_processor.py / ui_helpers.py / cache_manager.py
TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"

# ── 本地模組 ──────────────────────────────────────────────────
from cache_manager import (
    list_cached_collections, delete_cache_for_txt, delete_all_cache,
    delete_folder_qa_cache, delete_all_qa_cache,
)
from doc_processor import (
    TEMP_DIR, cleanup_temp_dir,
    get_pdf_pages, generate_pdf_preview, docx_to_pdf_libreoffice,
    split_pdf, pdf_process_and_save, build_rag_index,
)
from ui_helpers import (
    get_history_folders, preview_history_folder,
    get_doc_images, toggle_doc_images,
    fetch_qa_candidates, apply_cached_answer,
    render_qa_html, get_qa_entry_choices,
    save_current_qa, report_error_qa, request_abort, rag_chat_response,
    format_related_clauses, format_complete_sections,
    build_folder_metadata_index, search_folders_by_description,
    update_smart_doc_switcher,
)


# ==========================================
# 模型載入
# ==========================================

def get_model_list():
    """
    從 Ollama 取得本機已下載的所有模型名稱，供 UI 下拉選單使用。
    呼叫 ollama.list()["models"]，取每個模型的 "model" 欄位。
    失敗時回傳 ["error"]，UI 仍可顯示但無法正常問答。

    【不需要改程式碼切換模型】
    只需執行 ollama pull <模型名稱> 下載新模型，
    再從 UI 的「AI 模型」下拉選單選擇即可。
    """
    try:
        return [i["model"] for i in ollama.list()["models"]]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["error"]


# ── OCR 模型載入 ─────────────────────────────────────────────
# OCRFlux-3B：用於掃描型 PDF 的 AI OCR
# torch_dtype="auto"：自動選精度（通常 float16），節省 VRAM
# device_map="auto"：自動分配到 GPU；無 GPU 則退到 CPU（極慢）
# 載入失敗時設 ocr_model = None，系統仍可運作，只是無法處理掃描 PDF
ocr_model_path = OCR_MODEL_PATH
try:
    print("正在載入 OCR 模型...")
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        ocr_model_path, torch_dtype="auto", device_map="auto"
    )
    ocr_model.eval()   # 切換到推論模式，關閉 dropout 等訓練模式行為
    ocr_tokenizer  = AutoTokenizer.from_pretrained(ocr_model_path)
    ocr_processor  = AutoProcessor.from_pretrained(ocr_model_path)
except Exception as e:
    print(f"⚠️ OCR 模型載入失敗: {e}")
    ocr_model = ocr_tokenizer = ocr_processor = None


# ── RAG 模型載入 ─────────────────────────────────────────────
# LLM：透過 Ollama 在本機跑，這裡設定預設模型和 Embedding 模型
# model="qwen3:4b"：只是預設值，UI 下拉選單可隨時切換，不需改程式碼
# request_timeout=1200.0：LLM 逾時上限（20 分鐘），context 很大時需要
# device="cuda"：Embedding 計算使用 GPU；無 GPU 改為 "cpu"
try:
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print("正在載入 RAG 模型 (LLM + Embedding)...")
    Settings.llm = Ollama(model="qwen3:4b", request_timeout=1200.0)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="IEITYuan/Yuan-embedding-2.0-zh", device="cuda"
    )
    RAG_AVAILABLE = True
    print("✅ RAG 模型載入完成")
except ImportError:
    print("❌ 缺少 llama_index，Step 4 將無法使用。")
    RAG_AVAILABLE = False
except Exception as e:
    print(f"❌ RAG 模型初始化失敗: {e}")
    RAG_AVAILABLE = False


# ==========================================
# 核心邏輯（上傳、切割、OCR、Step 4 導航）
# ==========================================

def process_upload(file, state_data):
    """
    使用者點「▶️ 下一步：轉換與預覽」時觸發。
    負責：格式轉換（DOCX→PDF）、頁數讀取、預覽圖產生、同名資料夾偵測。

    【state_data 結構】
    state_data 是 Gradio State，貫穿整個四步驟處理流程：
    {
        "original_file": 原始上傳檔案路徑,
        "file_ext":      副檔名（.pdf 或 .docx）,
        "converted_pdf": DOCX 轉換後的 PDF 路徑（DOCX 時才有值）,
        "split_pdf_path": 切割後的暫存 PDF 路徑,
        "txt_path":      OCR 完成後的 .txt 路徑,
    }

    【同名資料夾偵測】
    若 TEXT_RESULT_DIR 下已有同名資料夾，顯示警告訊息（但不阻止繼續）。
    讓使用者決定是否要覆蓋舊資料。
    """
    cleanup_temp_dir()   # 清空暫存目錄，確保不殘留前次的暫存 PDF

    # 失敗時的預設回傳值（顯示 Step 1，隱藏後續步驟）
    fail = [
        None, None, None, "請上傳檔案", 1, 1,
        gr.update(visible=True), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), "",
    ]

    if file is None:
        return [gr.update(value="請先選擇檔案"), *fail[1:]]

    file_path = Path(file)
    ext       = file_path.suffix.lower()

    new_state = {
        "original_file":  str(file_path),
        "file_ext":       ext,
        "converted_pdf":  None,
        "split_pdf_path": None,
        "txt_path":       None,
    }

    try:
        if ext == ".pdf":
            pdf_path = str(file_path)
        elif ext == ".docx":
            # DOCX 需要先轉 PDF 才能進行後續處理
            pdf_path = str(docx_to_pdf_libreoffice(str(file_path), TEMP_DIR))
            new_state["converted_pdf"] = pdf_path
        else:
            return ["❌ 格式錯誤", *fail[1:]]

        cnt  = get_pdf_pages(pdf_path)
        prev = generate_pdf_preview(pdf_path)

        # 偵測是否有同名歷史資料夾（避免使用者不知情地蓋掉舊資料）
        result_dir  = Path(TEXT_RESULT_DIR)
        found_records = []
        if result_dir.exists():
            for folder in result_dir.iterdir():
                if folder.is_dir() and folder.name == file_path.stem:
                    found_records.append("已有處理紀錄")

        if found_records:
            return [
                f"⚠️ 偵測到歷史處理紀錄。推薦改名或確認範圍。",
                new_state, prev, f"共 {cnt} 頁", 1, cnt,
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), "",
            ]

        return [
            f"✅ 讀取成功 (共 {cnt} 頁)", new_state, prev,
            f"文件共 {cnt} 頁，請選擇切割範圍：", 1, cnt,
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=False), "",
        ]

    except Exception as e:
        return [f"❌ 錯誤: {e}", *fail[1:]]


def split_document(state_data, start_page, end_page):
    """
    使用者點「🔪 執行切割」時觸發。
    依指定頁碼範圍切割 PDF，結果存至暫存目錄。

    切割完成後顯示 Step 3，並顯示「執行 OCR / 文字提取」按鈕。
    範圍錯誤（start > end）時回傳錯誤訊息，不執行切割。

    txt_path 清空，確保後續不會誤用前次的文字結果。
    """
    fail = [
        "", None, state_data,
        gr.update(visible=True), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), "", None,
        gr.update(visible=False),
    ]

    if not state_data or not state_data.get("original_file"):
        return "❌ 請重傳", *fail[1:]

    state_data["txt_path"] = None   # 清除前次的 txt_path

    # 決定要切割的 PDF（DOCX 已轉換的版本，或原始 PDF）
    target = (state_data["converted_pdf"]
              if state_data["file_ext"] == ".docx"
              else state_data["original_file"])

    try:
        start = max(1, int(start_page))
        end   = int(end_page)
        if start > end:
            return f"❌ 範圍錯誤：起始頁 ({start}) 不能大於結束頁 ({end})", *fail[1:]

        out_path = split_pdf(target, start, end, Path(state_data["original_file"]).stem)
        state_data["split_pdf_path"] = str(out_path)

        return [
            f"✅ 切割完成: {out_path.name}", str(out_path), state_data,
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),
            "", None, gr.update(visible=False),
        ]

    except Exception as e:
        return f"❌ 失敗: {e}", *fail[1:]


def run_ocr_extraction(state_data, progress=gr.Progress()):
    """
    使用者點「🔍 執行 OCR / 文字提取」時觸發。
    呼叫 doc_processor.pdf_process_and_save，自動選擇 PDFPlumber 或 OCR。
    完成後將 txt_path 存入 state_data，並顯示「下一步：智能問答」按鈕。
    """
    split_path = state_data.get("split_pdf_path")
    if not split_path:
        return "❌ 無切割檔", None, state_data, gr.update(visible=True), gr.update(visible=False)

    try:
        msg, txt_path = pdf_process_and_save(
            split_path, ocr_model, ocr_processor, progress=progress
        )
        state_data["txt_path"] = txt_path   # 儲存 txt 路徑供後續 RAG 使用
        return [f"✅ {msg}", txt_path, state_data,
                gr.update(visible=False), gr.update(visible=True)]
    except Exception as e:
        return f"❌ 失敗: {e}", None, state_data, gr.update(visible=True), gr.update(visible=False)


def go_to_step4(state_data, progress=gr.Progress()):
    """
    使用者點「▶️ 下一步：智能問答 (RAG)」時觸發（新文件流程）。
    自動建立（或從快取載入）RAG 索引，切換到 Step 4 問答介面。

    使用 yield（generator 函式），讓 Gradio 先顯示「⏳ 正在建立知識庫...」
    再回傳最終結果，提供進度反饋。

    預設 LLM 模型：下拉選單預設顯示 qwen3:4b，若不存在則顯示列表第一個。
    """
    txt_path = state_data.get("txt_path")
    if not txt_path:
        yield [gr.update(visible=True), gr.update(visible=False),
               gr.update(value=[]), "❌ 找不到文字檔", None, gr.update(), None]
        return

    current_models = get_model_list()
    default_model  = "qwen3:4b" if "qwen3:4b" in current_models else current_models[0]

    # 第一次 yield：顯示「正在建立知識庫」的等待訊息
    yield (gr.update(visible=False), gr.update(visible=True),
           gr.update(value=[]), "⏳ 正在自動建立知識庫...", None,
           gr.update(choices=current_models, value=default_model), None)

    try:
        engine, status = build_rag_index(state_data, RAG_AVAILABLE, progress=progress)
        yield (gr.update(visible=False), gr.update(visible=True),
               gr.update(value=[]), status, engine,
               gr.update(choices=current_models), txt_path)
    except Exception as e:
        yield (gr.update(visible=False), gr.update(visible=True),
               gr.update(value=[]), f"❌ 失敗: {str(e)}",
               None, gr.update(choices=current_models), None)


def load_history_and_go_to_step4(folder_name, progress=gr.Progress()):
    """
    使用者從歷史紀錄 Tab 選擇資料夾後點「🚀 載入此專案」時觸發。
    與 go_to_step4 功能相同，差別在於 txt_path 從歷史資料夾取得。

    從指定資料夾掃描 .txt 檔案，取第一個作為來源。
    建立 fake_state（模擬 state_data 結構）傳入 build_rag_index。
    """
    current_models = get_model_list()
    default_model  = "qwen3:4b" if "qwen3:4b" in current_models else current_models[0]

    if not folder_name:
        yield (gr.update(visible=True), gr.update(visible=False),
               gr.update(value=[]), "❌ 請先選擇一個資料夾",
               None, gr.update(choices=current_models), False,
               gr.update(visible=False), None)
        return

    txt_files = list((Path(TEXT_RESULT_DIR) / folder_name).glob("*.txt"))
    if not txt_files:
        yield (gr.update(visible=True), gr.update(visible=False),
               gr.update(value=[]), "❌ 找不到 .txt 文字檔",
               None, gr.update(choices=current_models), False,
               gr.update(visible=False), None)
        return

    txt_path   = str(txt_files[0])
    fake_state = {"original_file": "History Load", "txt_path": txt_path}

    yield (gr.update(visible=False), gr.update(visible=True),
           gr.update(value=[]), f"⏳ 正在載入：{folder_name}...",
           None, gr.update(choices=current_models, value=default_model),
           False, gr.update(visible=False), None)

    try:
        engine, status = build_rag_index(fake_state, RAG_AVAILABLE, progress=progress)
        yield (gr.update(visible=False), gr.update(visible=True),
               gr.update(value=[]), status, engine,
               gr.update(choices=current_models), False,
               gr.update(visible=False), txt_path)
    except Exception as e:
        yield (gr.update(visible=False), gr.update(visible=True),
               gr.update(value=[]), f"❌ 載入失敗: {str(e)}",
               None, gr.update(choices=current_models), False,
               gr.update(visible=False), None)


# ==========================================
# 智能搜尋流程
# ==========================================

def smart_search_and_confirm(doc_description, user_question, model, lang,
                              use_context, folder_index, progress=gr.Progress()):
    """
    使用者在「🔍 智能檢索回答」Tab 點「🚀 開始智能搜尋與問答」時觸發。
    依使用者輸入的檔案描述，在資料夾索引中搜尋最相似的合約。

    【流程】
    1. 檢查描述和問題是否有輸入（都不能為空）
    2. 檢查 folder_index 是否已建立
    3. search_folders_by_description 搜尋 top 5 相似資料夾
    4. 顯示最相似的資料夾並詢問確認

    【folder_index 來源】
    ui_helpers.build_folder_metadata_index() 在 app 啟動時執行，
    也在使用者點「🔄 更新檢索範圍」時重新建立。
    以資料夾名稱為文字做向量索引，所以資料夾命名要語意清晰。

    【確認畫面】
    顯示最相似資料夾的名稱、相似度分數、路徑，
    要求使用者點「✅ 是的」或「❌ 不是」確認。
    """
    empty = (None, gr.update(visible=False), gr.update(visible=False),
             gr.update(visible=False), None, [], None, gr.update(visible=False))

    if not doc_description or not doc_description.strip():
        return (None, "<div style='color:orange;padding:20px;'><h3>⚠️ 請輸入檔案描述</h3></div>",
                *empty[1:], "請先輸入檔案描述")

    if not user_question or not user_question.strip():
        return (None, "<div style='color:orange;padding:20px;'><h3>⚠️ 請輸入您的問題</h3></div>",
                *empty[1:], "請先輸入問題")

    if folder_index is None:
        return (None, "<div style='color:red;padding:20px;'><h3>❌ 系統錯誤：文件索引未建立</h3></div>",
                *empty[1:], "索引未建立")

    progress(0.3, desc="🔍 正在檢索相關文件...")
    results = search_folders_by_description(doc_description, folder_index, top_k=5)

    if not results:
        return (
            None,
            f"<div style='background:#fff3cd;border:2px solid #ffc107;"
            f"padding:20px;border-radius:8px;'>"
            f"<h3>😔 未找到相關文件</h3><p>描述：{doc_description}</p></div>",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            None, "未找到文件", [], None, gr.update(visible=False),
        )

    top = results[0]
    confirm_html = f"""
<div style='background:#a7c4db;border:2px solid #0c5460;padding:20px;border-radius:8px;'>
  <h3>🎯 找到最相似的文件</h3>
  <div style='padding:15px;margin:10px 0;border-radius:5px;'>
    <h4>📄 {top["folder_name"]}</h4>
    <p><strong>相似度分數：</strong>{top["score"]:.3f}</p>
    <p><strong>檔案路徑：</strong>{top["txt_path"]}</p>
  </div>
  <p style='font-size:1.1em;margin-top:15px;'><strong>❓ 這是您要找的文件嗎？</strong></p>
</div>"""

    progress(1.0, desc="等待使用者確認...")

    return (
        None, confirm_html,
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=False),
        None, f"找到文件：{top['folder_name']}，等待確認...",
        results, None, gr.update(visible=False),
    )


def confirm_yes_and_load_doc(search_results, user_question, use_context,
                              progress=gr.Progress()):
    """
    使用者點「✅ 是的，就是這個文件」後觸發。
    載入 search_results[0] 對應的合約 RAG 索引，並預先搜尋 QA 快取候選。

    【完整回傳值對應 _CONFIRM_OUTPUTS】
    confirm_yes_btn, confirm_no_btn, smart_qa_area,
    smart_rag_engine, smart_status, smart_retrieval_info, smart_info_panel,
    smart_txt_path_state,
    smart_cand_group, smart_cand_html, smart_cand_radio, smart_use_cache_btn,
    smart_qa_candidates, smart_pending_q
    """
    def _err(msg):
        return (
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True), None, f"❌ {msg}", "", gr.update(visible=False),
            None, gr.update(visible=False), "", gr.update(visible=False),
            gr.update(visible=False), [], user_question,
        )

    if not search_results:
        return _err("沒有選中的文件")

    top         = search_results[0]
    txt_path    = top["txt_path"]
    folder_name = top["folder_name"]

    progress(0.2, desc=f"📂 正在載入文件：{folder_name}...")

    try:
        engine, status = build_rag_index(
            {"original_file": "Smart Search", "txt_path": txt_path},
            RAG_AVAILABLE, progress=progress,
        )
        if engine is None:
            return _err(status)

        progress(0.85, desc="🔍 檢索快取問答...")
        # 預先搜尋 QA 快取，讓使用者可以選用歷史答案
        candidates, cand_grp, radio_upd, use_btn_upd, cand_html = \
            fetch_qa_candidates(user_question, txt_path, rag_available=RAG_AVAILABLE)

        progress(1.0, desc="完成！")

        return (
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True), engine,
            f"✅ 已載入：{folder_name}", "", gr.update(visible=False),
            txt_path,
            cand_grp, cand_html, radio_upd, use_btn_upd,
            candidates, user_question,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _err(str(e))


def confirm_no_show_alternatives(search_results):
    """
    使用者點「❌ 不是，顯示其他選項」後觸發。
    顯示 search_results[1:] 的備選合約（相似度次高的候選）。

    【UI 流程說明】
    按下「不是」後：
      - 隱藏「✅ 是的」和「❌ 不是」確認按鈕（避免重複點擊）
      - 顯示備選合約列表（HTML 格式）
      - 顯示 alternative_area（包含 Radio + 確認按鈕）

    【備選數量限制】
    search_folders_by_description 最多回傳 top_k=5 個結果，
    第一個已在確認畫面顯示，備選最多 4 個（search_results[1:]）。
    只有 1 個結果時（無備選），顯示「沒有其他候選文件」的訊息。

    Parameters
    ----------
    search_results : smart_search_and_confirm 中的 search_results_state

    Returns
    -------
    tuple：(alternative_html, alternative_area_update, confirm_yes_update, confirm_no_update)
    """
    if not search_results or len(search_results) < 2:
        return (
            "<div style='background:#f8d7da;border:2px solid #721c24;"
            "padding:20px;border-radius:8px;'><h3>😔 沒有其他候選文件</h3></div>",
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        )

    html = "<div style='padding:15px;'><h3>📋 其他候選文件</h3>"
    for i, result in enumerate(search_results[1:], 2):
        html += (f"<div style='border:1px solid #007bff;margin:10px 0;"
                 f"padding:15px;border-radius:5px;'>"
                 f"<h4>候選 {i}: {result['folder_name']}</h4>"
                 f"<p><strong>相似度分數：</strong>{result['score']:.3f}</p>"
                 f"<p><strong>路徑：</strong>{result['txt_path']}</p></div>")
    html += "</div>"

    return (html, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))


def select_alternative_and_load_doc(search_results, selected_idx,
                                     user_question, use_context, progress=gr.Progress()):
    """
    使用者從備選列表選了「候選 N」並點「✔️ 確認選擇」時觸發。

    【selected_idx 解析邏輯】
    Radio 選項 label 為「候選 2」「候選 3」...，
    取第二個 token 解析為整數 N，轉換為 search_results 的 0-based 索引（N-1）。
    解析失敗時預設 idx=1（第二個結果），並以 min 確保不超出邊界。

    Parameters
    ----------
    search_results : 搜尋結果列表（來自 search_results_state）
    selected_idx   : Radio 選取的 label 字串（如「候選 2」）
    user_question  : 使用者的原始問題（傳入 confirm_yes_and_load_doc）
    use_context    : 是否使用完整章節
    progress       : Gradio progress 物件

    Returns
    -------
    同 confirm_yes_and_load_doc 的回傳值（_CONFIRM_OUTPUTS 對應）
    """
    try:
        idx = int(selected_idx.split()[1]) - 1
    except Exception:
        idx = 1
    idx = min(idx, len(search_results) - 1)
    return confirm_yes_and_load_doc([search_results[idx]], user_question, use_context, progress)


def _load_doc_by_index(idx, search_results, user_question, use_context, progress=gr.Progress()):
    """
    依索引從 search_results 中取出指定合約並載入。
    供 smart_doc_switcher 的 5 個快速切換按鈕使用。

    【為什麼用 lambda 綁定 idx】
    Gradio 按鈕點擊事件無法直接傳遞常數，
    在 app.py 的事件綁定處使用：
      lambda sr, uq, uc, idx=_idx: _load_doc_by_index(idx, sr, uq, uc)
    透過 Python 的預設參數捕捉（capture）每個按鈕對應的 idx 值（0~4）。

    【錯誤處理】
    idx 超出範圍（如搜尋結果只有 2 個，但按了第 3 個按鈕）時，
    傳入空列表到 confirm_yes_and_load_doc，該函式會回傳「沒有選中的文件」錯誤。

    Parameters
    ----------
    idx            : 要載入的結果索引（0~4）
    search_results : 搜尋結果列表
    user_question  : 使用者的原始問題
    use_context    : 是否使用完整章節

    Returns
    -------
    同 confirm_yes_and_load_doc 的回傳值（_CONFIRM_OUTPUTS 對應）
    """
    if not search_results or idx >= len(search_results):
        return confirm_yes_and_load_doc([], user_question, use_context, progress)
    return confirm_yes_and_load_doc([search_results[idx]], user_question, use_context, progress)


# ==========================================
# 導航與重設
# ==========================================

def back_to_step1():
    """
    返回 Step 1（顯示 step1 群組，隱藏其餘三個步驟群組）。
    用於 Step 2 的「◀️ 返回」按鈕。
    不清空已上傳的檔案或 state_data，只切換顯示。
    """
    return [gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False)]

def back_to_step2():
    """
    返回 Step 2（預覽與切割）。
    用於 Step 3 的「◀️ 返回切割」按鈕。
    """
    return [gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=False), gr.update(visible=False)]

def back_to_step3():
    """
    返回 Step 3（切割結果與 OCR）。
    用於 Step 4 的「◀️ 返回上一步」按鈕。
    回到 Step 3 後使用者可以重新執行 OCR，不需要重新上傳檔案。
    """
    return [gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True), gr.update(visible=False)]

def reset_all():
    """
    點「🔄 重新開始」時觸發，將整個 Tab 2 介面完全重設到初始狀態。

    【reset_all 的職責】
    1. 清空暫存目錄（cleanup_temp_dir）：清除上次切割的 PDF
    2. 重設 file_state 到空字典（防止殘留舊路徑被後續函式誤用）
    3. 重設所有 UI 元件到初始值（file_input 清空、各步驟隱藏等）

    【回傳值的順序很重要】
    reset_btn.click 的 outputs 列表順序必須和回傳的列表順序完全一致，
    否則 Gradio 會將回傳值設定到錯誤的元件。
    若未來新增 UI 元件到 reset_btn 的 outputs，
    需在此函式的回傳列表中同步新增對應的初始值。

    【為什麼不用 gr.ClearButton？】
    Gradio 的 ClearButton 只能清空單一元件，
    無法同時重設 State 物件和清空暫存目錄，
    因此改用自訂 reset_all 函式統一處理。
    """
    cleanup_temp_dir()
    empty_state = {
        "original_file": None, "file_ext": None,
        "converted_pdf": None, "split_pdf_path": None, "txt_path": None,
    }
    return [
        gr.update(value=None, interactive=True), empty_state, "",
        1, 1,
        gr.update(visible=True), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False),
        None, "", "", None, "", None,
        gr.update(visible=True), "尚未建立索引", None, [],
        "", gr.update(visible=False), "中文", False, True,
        None, False,
        gr.update(visible=False), gr.update(value=[]),
        None, "",
    ]


# ==========================================
# Gradio 介面
# ==========================================

with gr.Blocks(title="智能文件處理系統") as demo:
    gr.Markdown("# 📄 智能合約處理系統")

    # ── 全域 State ─────────────────────────────────────────────
    # Gradio State 物件的特性：
    #   - 每個 browser session 有獨立的 State，不同使用者互不干擾
    #   - 同一個 session 中跨函式呼叫持久存在
    #   - 只能透過函式的 inputs/outputs 列表讀寫，不能直接賦值
    #   - 值為 None 代表尚未設定，函式應做 None 判斷再使用
    file_state          = gr.State({"original_file": None, "file_ext": None,
                                    "converted_pdf": None, "split_pdf_path": None,
                                    "txt_path": None})
    rag_engine_state    = gr.State(None)    # 當前載入的 HierarchicalQueryEngine 物件
    info_visible_state  = gr.State(False)   # 右側「🔍 檢索原始條款」面板的顯示狀態
    step4_txt_path_state = gr.State(None)   # Step 4 當前合約的 .txt 完整路徑
    doc_img_visible_4   = gr.State(False)   # Step 4「查看原稿圖片」的顯示狀態
    step4_pending_qa    = gr.State(None)    # 最近一次問答的完整資料（供儲存 / 回報使用）
    step4_qa_candidates = gr.State([])      # QA 快取候選列表（fetch_qa_candidates 的結果）
    step4_pending_q     = gr.State("")      # 使用者目前等待回答的問題（傳遞給 regen 按鈕）

    # 智能搜尋 Tab 專用的 State（與 Step 4 的 State 完全分開，避免互相干擾）
    smart_txt_path_state = gr.State(None)   # 智能搜尋 Tab 當前載入的合約 .txt 路徑
    smart_img_visible    = gr.State(False)  # 智能搜尋 Tab 原稿圖片的顯示狀態
    smart_pending_qa     = gr.State(None)   # 智能搜尋 Tab 最近一次的問答資料
    smart_qa_candidates  = gr.State([])     # 智能搜尋 Tab 的 QA 快取候選列表
    smart_pending_q      = gr.State("")     # 智能搜尋 Tab 等待回答的問題

    with gr.Tabs():

        # ══════════════════════════════════════════════════════
        # Tab 1: 智能檢索回答
        # ══════════════════════════════════════════════════════
        with gr.Tab("🔍 智能檢索回答"):
            gr.Markdown("## 智能文件問答")
            # 智能搜尋 Tab 專用的 State（放在 Tab 內部以便閱讀，但功能上與位置無關）
            folder_index_state  = gr.State(None)   # build_folder_metadata_index 建立的向量索引
            search_results_state = gr.State([])    # search_folders_by_description 的搜尋結果列表
            smart_rag_engine     = gr.State(None)  # 智能搜尋 Tab 當前載入的 RAG 引擎
            smart_info_visible   = gr.State(False) # 智能搜尋 Tab 檢索詳情面板的顯示狀態
            smart_user_question  = gr.State("")    # 儲存使用者的問題，供確認後載入合約時使用

            with gr.Row():
                # 模型選擇下拉（由 get_model_list 動態填入）
                smart_model_selector = gr.Dropdown(choices=get_model_list(), value="qwen3:4b",
                                                   label="🤖 AI 模型", scale=1)
                smart_lang_input = gr.Textbox(label="🌐 輸出語言", value="中文", scale=1)
                smart_status = gr.Textbox(label="📡 系統狀態", value="準備就緒",
                                          interactive=False, scale=2)
                # 點此重新掃描 TEXT_RESULT_DIR，更新 folder_index_state
                refresh_index_btn = gr.Button("🔄 更新檢索範圍", variant="secondary", scale=1)

            with gr.Group():
                gr.Markdown("### 📝 請輸入檔案描述與您的問題")
                with gr.Row():
                    doc_description_input = gr.Textbox(label="📁 檔案描述",
                                                       placeholder="例如：合約1、租賃合約...", scale=1)
                    smart_use_context_chk = gr.Checkbox(label="包含完整章節(生成時間較長)",
                                                        value=True, scale=0, min_width=150)
                user_question_input = gr.Textbox(label="❓ 您的問題",
                                                 placeholder="請輸入關於此合約的問題...", lines=3)
                smart_search_btn = gr.Button("🚀 開始智能搜尋與問答", variant="primary", size="lg")

            # 確認畫面（搜尋後顯示，確認前隱藏問答區）
            with gr.Group() as confirm_group:
                confirm_display = gr.HTML(value="")
                with gr.Row(visible=False) as confirm_buttons:
                    confirm_yes_btn = gr.Button("✅ 是的，就是這個文件", variant="primary", scale=1)
                    confirm_no_btn  = gr.Button("❌ 不是，顯示其他選項", variant="secondary", scale=1)

            # 備選文件列表（點「不是」後顯示）
            with gr.Group(visible=False) as alternative_area:
                alternative_display  = gr.HTML(value="")
                with gr.Row():
                    alternative_selector = gr.Radio(
                        choices=["候選 2","候選 3","候選 4","候選 5"],
                        value="候選 2", label="選擇其他文件")
                    select_alternative_btn = gr.Button("✔️ 確認選擇", variant="primary")

            gr.Markdown("---")

            # 問答區（確認文件後顯示）
            with gr.Group(visible=False) as smart_qa_area:
                gr.Markdown("## 💬 問答結果")
                with gr.Row():
                    smart_show_img_btn = gr.Button("📸 顯示/隱藏文件圖片", variant="secondary", scale=0)

                with gr.Group(visible=False) as smart_img_group:
                    smart_img_gallery = gr.Gallery(label="文件原稿圖片", columns=4,
                                                   height=350, object_fit="contain")

                # QA 快取候選面板（問題送出後顯示）
                with gr.Group(visible=False) as smart_cand_group:
                    smart_cand_html  = gr.HTML(value="")
                    smart_cand_radio = gr.Radio(choices=[], label="選擇快取答案", visible=False)
                    with gr.Row():
                        smart_use_cache_btn = gr.Button("✅ 使用此快取答案", variant="primary",
                                                        visible=False, scale=1)
                        smart_regen_btn     = gr.Button("呼叫AI 生成", variant="secondary", scale=1)

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        smart_chatbot    = gr.Chatbot(type="messages", label="智能助手", height=500)
                        smart_followup_msg = gr.Textbox(placeholder="繼續提問...",
                                                        show_label=False, container=False)
                        with gr.Row():
                            smart_info_toggle = gr.Button("👁️ 顯示/隱藏 檢索詳情", variant="secondary")
                            smart_send_btn    = gr.Button("尋找快取答案", variant="primary")
                        with gr.Row():
                            smart_abort_btn     = gr.Button("⛔ 截斷生成", variant="stop", scale=1)
                            save_qa_btn_smart   = gr.Button("💾 儲存此答案", variant="secondary", scale=1)
                            report_qa_btn_smart = gr.Button("⚠️ 回報問題答案", variant="secondary", scale=1)
                        with gr.Row():
                            save_qa_msg_smart  = gr.Textbox(value="", interactive=False,
                                                            show_label=False, container=False)

                    with gr.Column(scale=2, visible=False) as smart_info_panel:
                        gr.Markdown("### 🔍 檢索原始條款")
                        smart_retrieval_info = gr.HTML(value="<p style='color:gray;'>等待查詢...</p>")

                # 搜尋結果快速切換按鈕（顯示 top 5 相似合約）
                with gr.Group(visible=False) as smart_doc_switcher_group:
                    gr.Markdown("---")
                    gr.Markdown("### 📂 搜尋結果文件 — 點擊切換")
                    gr.Markdown("<p style='font-size:0.85em;color:#888;margin-top:-8px;'>"
                                "以下為本次搜尋找到的相似文件，點擊可直接切換載入並重新問答</p>")
                    with gr.Row():
                        smart_doc_btn_0 = gr.Button("", visible=False, variant="secondary")
                        smart_doc_btn_1 = gr.Button("", visible=False, variant="secondary")
                        smart_doc_btn_2 = gr.Button("", visible=False, variant="secondary")
                        smart_doc_btn_3 = gr.Button("", visible=False, variant="secondary")
                        smart_doc_btn_4 = gr.Button("", visible=False, variant="secondary")

        # ══════════════════════════════════════════════════════
        # Tab 2: 檔案上傳與手動搜尋（四步驟流程）
        # ══════════════════════════════════════════════════════
        with gr.Tab("📁 檔案上傳與手動搜尋"):
            with gr.Group(visible=True) as step1:
                with gr.Tabs():
                    with gr.Tab("🆕 上傳新文件"):
                        gr.Markdown("## 1️⃣ 上傳檔案")
                        file_input  = gr.File(label="PDF / DOCX", file_types=[".pdf",".docx"],
                                              type="filepath")
                        upload_msg  = gr.Textbox(interactive=False, show_label=False)
                        next_step_btn = gr.Button("▶️ 下一步：轉換與預覽", variant="primary")

                    with gr.Tab("📂 選擇歷史紀錄"):
                        gr.Markdown("## 讀取已處理的 OCR 結果")
                        with gr.Row():
                            history_dropdown = gr.Dropdown(label="選擇歷史專案",
                                                           choices=get_history_folders(), interactive=True)
                            refresh_hist_btn = gr.Button("🔄 重新整理清單", size="sm")
                        history_msg     = gr.Textbox(label="資料夾資訊", lines=3, interactive=False)
                        gr.Markdown("### 圖片預覽")
                        history_gallery = gr.Gallery(label="內容預覽", columns=6, height=200,
                                                     object_fit="contain")
                        load_history_btn = gr.Button("🚀 載入此專案並開始問答 (跳至 Step 4)",
                                                     variant="primary")

            with gr.Group(visible=False) as step2:
                gr.Markdown("## 2️⃣ 預覽與切割")
                preview_gallery = gr.Gallery(label="預覽", columns=5, height=250, object_fit="contain")
                page_hint = gr.Markdown("頁數資訊")
                with gr.Row():
                    page_start = gr.Number(label="起始頁數", value=1, precision=0, minimum=1, step=1)
                    page_end   = gr.Number(label="結束頁數", value=1, precision=0, minimum=1, step=1)
                with gr.Row():
                    back_btn_2_1 = gr.Button("◀️ 返回", variant="secondary")
                    split_btn    = gr.Button("🔪 執行切割", variant="primary")

            with gr.Group(visible=False) as step3:
                gr.Markdown("## 3️⃣ 切割結果與 OCR")
                result_msg  = gr.Textbox(label="系統訊息", interactive=False)
                with gr.Row():
                    result_file    = gr.File(label="切割後的 PDF")
                    ocr_result_file = gr.File(label="文字提取結果 (.txt)")
                with gr.Row():
                    back_btn_3_2 = gr.Button("◀️ 返回切割", variant="secondary")
                    ocr_btn      = gr.Button("🔍 執行 OCR / 文字提取", variant="primary")
                    to_rag_btn   = gr.Button("▶️ 下一步：智能問答 (RAG)", variant="primary", visible=False)

            with gr.Group(visible=False) as step4:
                gr.Markdown("## 4️⃣ 智能合約問答 (RAG)")
                with gr.Row():
                    rag_status     = gr.Textbox(label="📡 知識庫狀態", value="尚未建立索引",
                                                interactive=False, scale=2)
                    model_selector = gr.Dropdown(choices=get_model_list(), value="qwen3:4b",
                                                 label="🤖 選擇 AI 模型", scale=1)
                    lang_input     = gr.Textbox(label="🌐 輸出語言", value="中文", scale=1)

                with gr.Row():
                    show_img_btn_4 = gr.Button("📸 顯示/隱藏文件圖片", variant="secondary", scale=0)

                with gr.Group(visible=False) as doc_img_group_4:
                    doc_img_gallery_4 = gr.Gallery(label="文件原稿圖片", columns=4,
                                                   height=350, object_fit="contain")

                # QA 快取候選面板
                with gr.Group(visible=False) as step4_cand_group:
                    step4_cand_html  = gr.HTML(value="")
                    step4_cand_radio = gr.Radio(choices=[], label="選擇快取答案", visible=False)
                    with gr.Row():
                        step4_use_cache_btn = gr.Button("✅ 使用此快取答案", variant="primary",
                                                        visible=False, scale=1)
                        step4_regen_btn  = gr.Button("呼叫 AI 生成", variant="secondary", scale=1)

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(type="messages", label="智能合約助手",
                                             height=550, show_label=False)
                        msg     = gr.Textbox(placeholder="請輸入關於此合約的問題...",
                                             show_label=False, container=False)
                        with gr.Row(variant="panel"):
                            use_full_context_chk = gr.Checkbox(label="包含完整章節(生成時間較長)",
                                                                value=True, interactive=True)
                        with gr.Row():
                            back_btn_4_3   = gr.Button("◀️ 返回上一步", variant="secondary", scale=1)
                            info_toggle_btn = gr.Button("👁️ 顯示/隱藏 檢索相關條款", variant="secondary", scale=1)
                            send_btn        = gr.Button("比對快取", variant="primary", scale=2)
                        with gr.Row():
                            step4_abort_btn  = gr.Button("⛔ 截斷生成", variant="stop", scale=1)
                            save_qa_btn_4    = gr.Button("💾 儲存此答案", variant="secondary", scale=1)
                            report_qa_btn_4  = gr.Button("⚠️ 回報問題答案", variant="secondary", scale=1)
                        with gr.Row():
                            save_qa_msg_4 = gr.Textbox(value="", interactive=False,
                                                       show_label=False, container=False)

                    with gr.Column(scale=2, visible=False) as side_info_panel:
                        gr.Markdown("### 🔍 檢索原始條款")
                        retrieval_info_area = gr.HTML(
                            value="<div style='color:#a7c4db;padding:20px;'>"
                                  "詢問問題後，相關的合約原文將顯示於此...</div>")

                gr.Markdown("---")
                reset_btn = gr.Button("🔄 重新開始", variant="secondary")

        # ══════════════════════════════════════════════════════
        # Tab 3: Embedding 快取管理
        # ══════════════════════════════════════════════════════
        with gr.Tab("🗃️ Embedding 快取管理"):
            gr.Markdown("## 本地 Embedding 快取管理")
            with gr.Row():
                refresh_cache_btn    = gr.Button("🔄 查看所有快取", variant="secondary")
                delete_all_cache_btn = gr.Button("🗑️ 刪除全部快取", variant="stop")
            cache_list_display = gr.JSON(label="目前快取清單", value=[])
            cache_op_msg       = gr.Textbox(label="操作結果", interactive=False)
            with gr.Row():
                cache_folder_dropdown  = gr.Dropdown(label="選擇要刪除快取的文件",
                                                     choices=get_history_folders(), interactive=True)
                delete_single_cache_btn = gr.Button("🗑️ 刪除此文件的快取", variant="secondary")

        # ══════════════════════════════════════════════════════
        # Tab 4: QA 問答快取管理
        # ══════════════════════════════════════════════════════
        with gr.Tab("💬 問答快取管理"):
            gr.Markdown("## 已儲存問答快取管理")
            with gr.Row():
                qa_refresh_btn    = gr.Button("🔄 重新整理", variant="secondary")
                qa_delete_all_btn = gr.Button("🗑️ 刪除所有快取", variant="stop")
            qa_op_msg = gr.Textbox(label="操作結果", interactive=False)
            gr.Markdown("### 選擇文件資料夾查看 / 刪除")
            with gr.Row():
                qa_folder_dropdown    = gr.Dropdown(label="選擇文件資料夾",
                                                    choices=get_history_folders(), interactive=True, scale=3)
                qa_delete_folder_btn  = gr.Button("🗑️ 刪除此資料夾的快取", variant="secondary", scale=1)
            qa_entries_html = gr.HTML(value="<p style='color:gray;padding:20px;'>請先選擇文件資料夾</p>")
            gr.Markdown("### 刪除單筆答案")
            with gr.Row():
                qa_entry_dropdown    = gr.Dropdown(label="選擇要刪除的問答條目",
                                                   choices=[], value=None, interactive=True, scale=3)
                qa_delete_entry_btn  = gr.Button("🗑️ 刪除此筆答案", variant="secondary", scale=1)

        # ══════════════════════════════════════════════════════
        # Tab 5: 使用者說明
        # ══════════════════════════════════════════════════════
        with gr.Tab("📖 使用者說明"):
            gr.Markdown("## 使用者說明")
            gr.Markdown("### 請參考簡報")
            gr.HTML(
                "<div style='padding:16px;'>"
                "<a href='https://docs.google.com/presentation/d/1snh-0PudB6ykapWTr_aoIbpB7kP2t9oVC_JWT6ru_4I/edit?usp=sharing' "
                "target='_blank' "
                "style='font-size:1.05em;word-break:break-all;'>"
                "https://docs.google.com/presentation/d/1snh-0PudB6ykapWTr_aoIbpB7kP2t9oVC_JWT6ru_4I/edit?usp=sharing"
                "</a></div>"
            )



    # Gradio 的事件綁定採 .click() / .change() / .submit() 等方法，
    # inputs 對應函式的參數，outputs 對應函式的回傳值（需順序一致）。
    # 說明：inputs 列表中的 State 物件會自動讀取目前的值傳入函式；
    #       函式的回傳值依 outputs 列表的順序，逐一設定到對應元件。

    # ── Step 1 → 2 ────────────────────────────────────────────
    # 使用者上傳檔案後點「下一步」，觸發格式偵測、轉換、預覽
    next_step_btn.click(process_upload, [file_input, file_state],
                        [upload_msg, file_state, preview_gallery, page_hint,
                         page_start, page_end, step1, step2, step3, step4, result_msg])

    # ── Step 2 → 3 ────────────────────────────────────────────
    # 使用者設定頁碼範圍後點「執行切割」，切割 PDF 並顯示 OCR 按鈕
    split_btn.click(split_document, [file_state, page_start, page_end],
                    [result_msg, result_file, file_state, step2, step3,
                     ocr_btn, result_msg, ocr_result_file, to_rag_btn])

    # 執行 OCR / PDFPlumber 文字提取，完成後顯示「下一步：智能問答」按鈕
    ocr_btn.click(run_ocr_extraction, [file_state],
                  [result_msg, ocr_result_file, file_state, ocr_btn, to_rag_btn])

    # 點「下一步：智能問答」，建立 RAG 索引，切換到 Step 4 問答介面
    to_rag_btn.click(go_to_step4, [file_state],
                     [step3, step4, chatbot, rag_status, rag_engine_state,
                      model_selector, step4_txt_path_state])

    # ── Step 4 工具 ────────────────────────────────────────────
    # 切換文件原稿圖片顯示（toggle_doc_images 取反 doc_img_visible_4）
    show_img_btn_4.click(toggle_doc_images, [step4_txt_path_state, doc_img_visible_4],
                         [doc_img_group_4, doc_img_gallery_4, doc_img_visible_4])

    # 切換右側檢索詳情面板（lambda 取反 info_visible_state）
    info_toggle_btn.click(
        lambda v: (gr.update(visible=not v), not v),
        [info_visible_state], [side_info_panel, info_visible_state])

    # ── 導航返回 ───────────────────────────────────────────────
    back_btn_2_1.click(back_to_step1, [], [step1, step2, step3, step4])
    back_btn_3_2.click(back_to_step2, [], [step1, step2, step3, step4])
    back_btn_4_3.click(back_to_step3, [], [step1, step2, step3, step4])
    reset_btn.click(reset_all, [],
                    [file_input, file_state, upload_msg, page_start, page_end,
                     step1, step2, step3, step4, preview_gallery, page_hint,
                     result_msg, result_file, result_msg, ocr_result_file, ocr_btn,
                     rag_status, rag_engine_state, chatbot, retrieval_info_area,
                     side_info_panel, lang_input, info_visible_state, use_full_context_chk,
                     step4_txt_path_state, doc_img_visible_4, doc_img_group_4, doc_img_gallery_4,
                     step4_pending_qa, save_qa_msg_4])

    # ── Step 4 問答（fetch 候選 → 使用快取 / 重新生成）─────────
    def _fetch_step4(message, txt_path):
        """
        Step 4 的問題送出處理函式（內部函式）。
        先搜尋 QA 快取（fetch_qa_candidates），
        回傳候選列表供使用者選擇使用快取或呼叫 AI 重新生成。
        同時將 message 傳入 step4_pending_q State，
        供後續「呼叫 AI 生成」按鈕（step4_regen_btn）使用。
        """
        cands, grp, radio_upd, use_btn_upd, html = \
            fetch_qa_candidates(message, txt_path, rag_available=RAG_AVAILABLE)
        return cands, grp, html, radio_upd, use_btn_upd, message

    def _fetch_smart(message, txt_path):
        """
        智能搜尋 Tab 的問題送出處理函式（內部函式）。
        邏輯與 _fetch_step4 完全相同，
        分開定義是為了讓 Gradio 能獨立追蹤兩個 Tab 的事件鏈。
        """
        cands, grp, radio_upd, use_btn_upd, html = \
            fetch_qa_candidates(message, txt_path, rag_available=RAG_AVAILABLE)
        return cands, grp, html, radio_upd, use_btn_upd, message

    # 同時綁定 Textbox submit（按 Enter）和 Button click，提升使用體驗
    # .then(lambda: "", None, msg) 在問答完成後清空輸入框
    for trigger in [msg.submit, send_btn.click]:
        trigger(_fetch_step4, [msg, step4_txt_path_state],
                [step4_qa_candidates, step4_cand_group, step4_cand_html,
                 step4_cand_radio, step4_use_cache_btn, step4_pending_q]
                ).then(lambda: "", None, msg)   # 清空輸入框

    step4_use_cache_btn.click(apply_cached_answer,
        [step4_qa_candidates, step4_cand_radio, step4_pending_q, chatbot, use_full_context_chk],
        [chatbot, retrieval_info_area, side_info_panel, step4_pending_qa, step4_cand_group])

    step4_regen_btn.click(rag_chat_response,
        [step4_pending_q, chatbot, rag_engine_state, model_selector,
         lang_input, use_full_context_chk, step4_txt_path_state],
        [chatbot, retrieval_info_area, side_info_panel, step4_pending_qa, step4_cand_group])

    # 截斷生成：設定全域 abort 旗標，顯示截斷訊息
    step4_abort_btn.click(request_abort, [], [])

    save_qa_btn_4.click(save_current_qa, [step4_pending_qa, step4_txt_path_state], [save_qa_msg_4])

    # 回報問題答案：寫入 ERRORLOG，不列入快取
    report_qa_btn_4.click(report_error_qa, [step4_pending_qa, step4_txt_path_state], [save_qa_msg_4])

    # ── 歷史紀錄 ───────────────────────────────────────────────
    history_dropdown.change(preview_history_folder, [history_dropdown], [history_gallery, history_msg])
    refresh_hist_btn.click(lambda: gr.update(choices=get_history_folders()), [], [history_dropdown])
    load_history_btn.click(load_history_and_go_to_step4, [history_dropdown],
                           [step1, step4, chatbot, rag_status, rag_engine_state,
                            model_selector, info_visible_state, side_info_panel, step4_txt_path_state])

    # ── 智能檢索 Tab ───────────────────────────────────────────
    # 應用程式啟動時自動建立資料夾索引
    demo.load(lambda: build_folder_metadata_index()[0], [], [folder_index_state])

    refresh_index_btn.click(
        lambda: (build_folder_metadata_index()[0], "✅ 檢索範圍已更新！"),
        [], [folder_index_state, smart_status])

    # _CONFIRM_OUTPUTS：confirm_yes_and_load_doc 的回傳值對應的元件列表
    _CONFIRM_OUTPUTS = [
        confirm_yes_btn, confirm_no_btn, smart_qa_area,
        smart_rag_engine, smart_status, smart_retrieval_info, smart_info_panel,
        smart_txt_path_state,
        smart_cand_group, smart_cand_html,
        smart_cand_radio, smart_use_cache_btn,
        smart_qa_candidates, smart_pending_q,
    ]

    # _DOC_SWITCHER_OUTPUTS：update_smart_doc_switcher 的回傳值對應的元件列表
    _DOC_SWITCHER_OUTPUTS = [
        smart_doc_switcher_group,
        smart_doc_btn_0, smart_doc_btn_1,
        smart_doc_btn_2, smart_doc_btn_3, smart_doc_btn_4,
    ]

    smart_search_btn.click(smart_search_and_confirm,
        [doc_description_input, user_question_input, smart_model_selector,
         smart_lang_input, smart_use_context_chk, folder_index_state],
        [smart_chatbot, confirm_display, confirm_yes_btn, confirm_no_btn,
         smart_qa_area, smart_rag_engine, smart_status,
         search_results_state, smart_retrieval_info, smart_info_panel],
    ).then(lambda q: q, [user_question_input], [smart_user_question]
    ).then(lambda: (gr.update(visible=True), gr.update(visible=False)),
           [], [confirm_buttons, alternative_area]
    ).then(update_smart_doc_switcher, [search_results_state], _DOC_SWITCHER_OUTPUTS)

    confirm_yes_btn.click(confirm_yes_and_load_doc,
        [search_results_state, smart_user_question, smart_use_context_chk],
        _CONFIRM_OUTPUTS
    ).then(lambda: gr.update(visible=True), [], [smart_qa_area])

    confirm_no_btn.click(confirm_no_show_alternatives, [search_results_state],
        [alternative_display, alternative_area, confirm_yes_btn, confirm_no_btn])

    select_alternative_btn.click(select_alternative_and_load_doc,
        [search_results_state, alternative_selector, smart_user_question, smart_use_context_chk],
        _CONFIRM_OUTPUTS
    ).then(lambda: gr.update(visible=True), [], [smart_qa_area])

    # 智能搜尋 Tab 的問答流程（與 Step 4 相同）
    for trigger in [smart_followup_msg.submit, smart_send_btn.click]:
        trigger(_fetch_smart, [smart_followup_msg, smart_txt_path_state],
                [smart_qa_candidates, smart_cand_group, smart_cand_html,
                 smart_cand_radio, smart_use_cache_btn, smart_pending_q]
                ).then(lambda: "", None, smart_followup_msg)

    smart_use_cache_btn.click(apply_cached_answer,
        [smart_qa_candidates, smart_cand_radio, smart_pending_q,
         smart_chatbot, smart_use_context_chk],
        [smart_chatbot, smart_retrieval_info, smart_info_panel,
         smart_pending_qa, smart_cand_group])

    smart_regen_btn.click(rag_chat_response,
        [smart_pending_q, smart_chatbot, smart_rag_engine,
         smart_model_selector, smart_lang_input, smart_use_context_chk,
         smart_txt_path_state],
        [smart_chatbot, smart_retrieval_info, smart_info_panel,
         smart_pending_qa, smart_cand_group])

    # 截斷生成
    smart_abort_btn.click(request_abort, [], [])

    save_qa_btn_smart.click(save_current_qa, [smart_pending_qa, smart_txt_path_state],
                            [save_qa_msg_smart])

    # 回報問題答案
    report_qa_btn_smart.click(report_error_qa, [smart_pending_qa, smart_txt_path_state],
                              [save_qa_msg_smart])

    smart_info_toggle.click(
        lambda v: (gr.update(visible=not v), not v),
        [smart_info_visible], [smart_info_panel, smart_info_visible])

    smart_show_img_btn.click(toggle_doc_images, [smart_txt_path_state, smart_img_visible],
                             [smart_img_group, smart_img_gallery, smart_img_visible])

    # 5 個快速切換按鈕，各自綁定對應的 search_results 索引
    for _btn, _idx in [(smart_doc_btn_0, 0), (smart_doc_btn_1, 1), (smart_doc_btn_2, 2),
                       (smart_doc_btn_3, 3), (smart_doc_btn_4, 4)]:
        _btn.click(
            lambda sr, uq, uc, idx=_idx: _load_doc_by_index(idx, sr, uq, uc),
            [search_results_state, smart_user_question, smart_use_context_chk],
            _CONFIRM_OUTPUTS,
        ).then(lambda: gr.update(visible=True), [], [smart_qa_area])

    # ── Embedding 快取 Tab ─────────────────────────────────────
    def _delete_single_emb_cache(folder_name):
        """刪除指定資料夾的 Embedding 快取，並更新快取列表和 Dropdown。"""
        if not folder_name:
            return "請先選擇資料夾", [], gr.update()
        txt_files = list((Path(TEXT_RESULT_DIR) / folder_name).glob("*.txt"))
        if not txt_files:
            return "❌ 找不到 txt 檔案", list_cached_collections(), gr.update()
        msg_r   = delete_cache_for_txt(str(txt_files[0]))
        folders = get_history_folders()
        return msg_r, list_cached_collections(), gr.update(choices=folders)

    refresh_cache_btn.click(
        lambda: (list_cached_collections(), gr.update(choices=get_history_folders())),
        [], [cache_list_display, cache_folder_dropdown])

    delete_all_cache_btn.click(
        lambda: (delete_all_cache(), list_cached_collections(), gr.update(choices=get_history_folders())),
        [], [cache_op_msg, cache_list_display, cache_folder_dropdown])

    delete_single_cache_btn.click(_delete_single_emb_cache, [cache_folder_dropdown],
                                  [cache_op_msg, cache_list_display, cache_folder_dropdown])

    # ── QA 問答快取 Tab ────────────────────────────────────────
    def _qa_refresh(folder):
        """重新整理 QA 快取列表和 Dropdown。"""
        return (gr.update(choices=get_history_folders()),
                render_qa_html(folder),
                get_qa_entry_choices(folder))

    def _qa_delete_folder(folder):
        """刪除指定資料夾的所有 QA 快取。"""
        msg_r = delete_folder_qa_cache(folder)
        return (msg_r, render_qa_html(folder),
                gr.update(choices=get_history_folders()),
                get_qa_entry_choices(folder))

    def _qa_delete_all():
        """刪除所有 QA 快取。"""
        msg_r = delete_all_qa_cache()
        return (msg_r, "", gr.update(choices=get_history_folders()),
                gr.update(choices=[], value=None))

    def _delete_single_qa_entry(folder_name, entry_id):
        """刪除 QA 快取中的單筆問答（依 UUID）。"""
        if not folder_name:
            return "❌ 請先選擇資料夾", "", gr.update()
        if not entry_id:
            return "❌ 請選擇要刪除的條目", render_qa_html(folder_name), gr.update()
        txt_files = list((Path(TEXT_RESULT_DIR) / folder_name).glob("*.txt"))
        if not txt_files:
            return "❌ 找不到 txt 檔案", render_qa_html(folder_name), gr.update()
        result = delete_qa_entry_by_id(str(txt_files[0]), entry_id)
        return result, render_qa_html(folder_name), get_qa_entry_choices(folder_name)

    qa_folder_dropdown.change(
        lambda fn: (render_qa_html(fn), get_qa_entry_choices(fn)),
        [qa_folder_dropdown], [qa_entries_html, qa_entry_dropdown])

    qa_refresh_btn.click(_qa_refresh, [qa_folder_dropdown],
                         [qa_folder_dropdown, qa_entries_html, qa_entry_dropdown])

    qa_delete_folder_btn.click(_qa_delete_folder, [qa_folder_dropdown],
                               [qa_op_msg, qa_entries_html, qa_folder_dropdown, qa_entry_dropdown])

    qa_delete_all_btn.click(_qa_delete_all, [],
                            [qa_op_msg, qa_entries_html, qa_folder_dropdown, qa_entry_dropdown])

    qa_delete_entry_btn.click(_delete_single_qa_entry,
                              [qa_folder_dropdown, qa_entry_dropdown],
                              [qa_op_msg, qa_entries_html, qa_entry_dropdown])


# ── 啟動 ───────────────────────────────────────────────────────
# server_name="0.0.0.0"：允許區域網路內其他裝置連線（如用平板操作）
# server_port=7777：預設埠號，衝突時可改為其他埠號
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
