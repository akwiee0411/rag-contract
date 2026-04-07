# ui_helpers.py
"""
UI 輔助模組

包含所有與 Gradio 介面互動相關的邏輯：
  A. 格式化輸出：將查詢結果轉成 HTML 供右側檢索面板顯示
  B. 文件圖片工具：切換文件原稿圖片的顯示
  C. QA 快取候選面板：問答前先查快取，展示相似歷史問答供選擇
  D. QA 快取管理 Tab 輔助：渲染快取列表、管理刪除
  E. 智能搜尋：依描述文字比對最相似的合約資料夾
  F. 歷史資料夾工具：掃描已處理的合約列表、預覽圖片
  G. QA 儲存 / 回答：儲存問答到快取、呼叫 RAG 生成新答案
"""

import os
import json
import uuid
import datetime
import threading
import gradio as gr
from pathlib import Path
from cache_manager import (
    load_qa_cache, save_qa_entry, delete_qa_entry_by_id,
    retrieve_similar_qa,
)

# 所有合約結果的根目錄
# ⚠️ 換機器或搬移路徑時，app.py / doc_processor.py / cache_manager.py 也要同步修改
TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"

# Error log 目錄：與 TEXT_RESULT_DIR 同層，獨立資料夾
ERRORLOG_DIR = r"C:\Users\user\Documents\rag_contract\ERRORLOG"
DAILY_LOG_DIR = r"C:\Users\user\Documents\rag_contract\DAILY_LOG"

# _abort_flag: 設為 True 時，正在執行的 rag_chat_response 會在下一個檢查點終止
# _abort_lock: 保護 flag 的執行緒鎖
_abort_flag = False
_abort_lock = threading.Lock()


def request_abort():
    """設定截斷旗標，讓正在執行的 LLM 問答在下一個檢查點中止。"""
    global _abort_flag
    with _abort_lock:
        _abort_flag = True


def clear_abort():
    """清除截斷旗標（每次新問答開始前呼叫）。"""
    global _abort_flag
    with _abort_lock:
        _abort_flag = False


def _check_abort():
    """回傳目前截斷旗標狀態（True = 已請求截斷）。"""
    with _abort_lock:
        return _abort_flag


# ==========================================
# A. 格式化輸出
# ==========================================

def format_related_clauses(clauses: list) -> str:
    """
    將向量搜尋找到的條款列表渲染成 HTML 字串，
    顯示在右側「🔍 檢索原始條款」面板的上半部。

    每個條款卡片包含：
      - 階層路徑（hierarchy_path，如「第三條 > 三、」）
      - 條款標題（clause_title）
      - 條款類型（main_clause / sub_clause / complete_section）
      - 相似度分數（向量搜尋的 cosine similarity）
      - 文字預覽（前 200 字）

    純 HTML 字串產生，不涉及 AI 邏輯。需要調整版面時修改此函式的 HTML。
    """
    if not clauses:
        return "<p>沒有找到相關條款</p>"
    html = f"<h3>🎯 相關條款 ({len(clauses)} 個)</h3>"
    for i, c in enumerate(clauses, 1):
        html += (
            f"<div style='border:1px solid #a7c4db;margin:10px 0;"
            f"padding:10px;border-radius:5px;background:#1e2a35;color:#d0e4f0;'>"
            f"<h4 style='color:#a7c4db;'>{i}. 【{c.get('hierarchy_path','N/A')}】</h4>"
            f"<p><strong>標題:</strong> {c.get('clause_title','N/A')}</p>"
            f"<p><strong>類型:</strong> {c.get('clause_type','N/A')} | "
            f"<strong>相似度:</strong> {c.get('score',0.0):.3f}</p>"
            f"<p><strong>預覽:</strong> {c.get('text_preview','N/A')}</p>"
            f"</div>"
        )
    return html


def format_complete_sections(sections: list) -> str:
    """
    將完整章節列表渲染成 HTML 字串，
    顯示在右側「🔍 檢索原始條款」面板的下半部。

    每個章節區塊包含：
      - 章節序號與標題
      - 階層路徑
      - 完整條款文字（max-height:300px 可捲動）

    純 HTML 字串產生，不涉及 AI 邏輯。
    """
    if not sections:
        return "<p>沒有完整章節內容</p>"
    html = f"<h3>📋 完整章節 ({len(sections)} 個)</h3>"
    for s in sections:
        html += (
            f"<div style='border:2px solid #a7c4db;margin:15px 0;"
            f"padding:15px;border-radius:8px;background:#1e2a35;color:#d0e4f0;'>"
            f"<h4 style='color:#a7c4db;'>章節 {s.get('section_number','')}: {s.get('section_title','')}</h4>"
            f"<p><strong>階層路徑:</strong> {s.get('hierarchy_path','')}</p>"
            f"<div style='background:#162230;padding:10px;border-radius:4px;"
            f"max-height:300px;overflow-y:auto;'>"
            f"<div style='white-space:pre-wrap;font-family:inherit;"
            f"color:#c8dded;line-height:1.5;'>{s.get('full_content','')}</div>"
            f"</div></div>"
        )
    return html


# ==========================================
# B. 文件圖片工具
# ==========================================

def get_doc_images(txt_path: str) -> list:
    """
    掃描 txt 文字檔所在資料夾，找出所有頁面圖片（page*.jpg / page*.png）。
    依頁碼數字排序（page0, page1, page2...），確保順序正確。
    txt 不存在或資料夾無圖片時回傳空列表。
    """
    if not txt_path or not os.path.exists(txt_path):
        return []
    folder = Path(txt_path).parent
    imgs   = [p for p in (list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
              if p.stem.startswith("page")]
    # 依頁碼數字排序，非數字名稱排到最後（999）
    imgs.sort(key=lambda x: int(x.stem.replace("page", ""))
              if x.stem.replace("page", "").isdigit() else 999)
    return [str(p) for p in imgs]


def toggle_doc_images(txt_path: str, visible: bool):
    """
    切換文件原稿圖片的顯示狀態。
    visible 是當前狀態（True = 正在顯示），函式取反後決定新狀態。
    顯示時同時載入圖片列表；隱藏時清空圖片列表（節省記憶體）。

    回傳：(doc_img_group 的 gr.update, doc_img_gallery 的 gr.update, 新的 visible 狀態)
    """
    new_visible = not visible
    if new_visible:
        images = get_doc_images(txt_path) if txt_path else []
        return gr.update(visible=True), gr.update(value=images), True
    return gr.update(visible=False), gr.update(value=[]), False


# ==========================================
# C. QA 候選面板（快取檢索）
# ==========================================

def fetch_qa_candidates(message: str, txt_path: str, rag_available: bool = False):
    """
    使用者送出問題後，先做語意相似度搜尋，回傳 top-3 快取問答候選。
    完全不呼叫 LLM，速度快。

    流程：
    1. retrieve_similar_qa 計算語意相似度（向量餘弦或字元重疊率）
    2. 找不到 → 顯示「找不到快取」提示，引導使用者點「重新呼叫 AI 生成」
    3. 找到 → 渲染候選卡片 HTML + Radio 選擇器

    候選卡片顯示的資訊（可修改 HTML）：
      - 相似度分數（0~1，越接近 1 越像）
      - 時間戳記（精確到分鐘）
      - 涉及的條款數 / 章節數
      - 原問題前 120 字
      - 答案前 180 字

    Parameters
    ----------
    message       : 使用者輸入的問題
    txt_path      : 合約 .txt 檔路徑（用於定位 qa_cache.json）
    rag_available : True 表示可使用向量相似度計算

    Returns
    -------
    (candidates, cand_group_update, radio_update, use_btn_update, html)
    candidates : [(entry, score), ...] 列表，供 apply_cached_answer 使用
    """
    if not txt_path:
        html = ("<div style='border:1.5px solid #f0ad4e;padding:12px;border-radius:6px;'>"
                "⚠️ 尚未載入文件，無法檢索快取</div>")
        return ([], gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False), html)

    # top_k=3：最多顯示 3 筆候選，可調整（調高顯示更多選項，調低介面更簡潔）
    candidates = retrieve_similar_qa(message, txt_path, top_k=3,
                                     rag_available=rag_available)

    if not candidates:
        html = ("<div style='border:1.5px solid #6c757d;"
                "padding:12px;border-radius:6px;color:inherit;'>"
                "📭 快取中找不到相似問答，請點擊「🔄 呼叫 AI 生成</div>")
        return ([], gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False), html)

    # 渲染候選卡片
    html = (f"<div style='padding:6px;'>"
            f"<strong>🔍 找到 {len(candidates)} 筆相似歷史問答</strong>"
            f"<p style='font-size:0.82em;color:#888;margin:4px 0 6px;'>"
            f"從下方選取後點「✅ 使用此快取答案」，或點「🔄 呼叫 AI 生成</p>")

    for i, (entry, score) in enumerate(candidates, 1):
        q   = entry.get("question", "")
        a   = entry.get("answer", "")
        ts  = entry.get("timestamp", "")
        n_c = len(entry.get("related_clauses", []))
        n_s = len(entry.get("complete_sections", []))
        html += f"""
<div style='border:1px solid #ced4da;margin:6px 0;padding:10px 12px;
border-radius:6px;background:#fff;'>
  <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
    <strong>#{i}</strong>
    <span style='font-size:0.82em;color:#6c757d;'>
      相似度 {score:.3f} &nbsp;·&nbsp; {ts[:16]}
      &nbsp;·&nbsp; {n_c} 條款 / {n_s} 章節
    </span>
  </div>
  <div style='background:#e8f4f8;padding:6px 8px;border-radius:4px;
  font-size:0.88em;margin-bottom:4px;'>
    <strong>❓</strong> {q[:120]}{"..." if len(q)>120 else ""}
  </div>
  <div style='background:#f8f9fa;padding:6px 8px;border-radius:4px;
  font-size:0.85em;color:#555;max-height:60px;overflow:hidden;'>
    {a[:180]}{"..." if len(a)>180 else ""}
  </div>
</div>"""
    html += "</div>"

    # Radio 選擇器的 label 格式：「#N 相似度 X.XXX — 問題前70字」
    choices = [
        f"#{i} 相似度 {score:.3f} — "
        f"{entry['question'][:70]}{'...' if len(entry['question'])>70 else ''}"
        for i, (entry, score) in enumerate(candidates, 1)
    ]

    return (candidates, gr.update(visible=True),
            gr.update(choices=choices, value=choices[0], visible=True),
            gr.update(visible=True), html)


def apply_cached_answer(candidates: list, selected_label: str, question: str,
                        history: list, use_complete_sections: bool):
    """
    使用者從候選面板選了某筆快取答案後呼叫。
    將答案附加到 chatbot history，並更新右側檢索詳情面板。
    此函式完全不呼叫 LLM，直接從 candidates 列表取值，速度極快。

    【selected_label 解析邏輯】
    Radio 的 label 格式為「#N 相似度 ...」，取開頭的「#N」解析出 0-based 索引。
    解析失敗（如 label 為空或格式異常）時預設使用第一筆（index=0），
    並以 max/min 確保索引在合法範圍內。

    【答案標識 badge】
    在 chatbot 的 assistant 訊息前加上：
      「💾 *（快取答案，相似度 X.XXX）*」
    讓使用者明確知道這是來自快取而非 LLM 即時生成，
    避免誤以為是最新的 AI 回答。

    【pending_qa 的建立】
    apply_cached_answer 和 rag_chat_response 都需要建立 pending_qa，
    以便使用者後續點「💾 儲存此答案」時有資料可以傳入 save_current_qa。
    來自快取的 pending_qa 使用快取記錄的 model 和 lang，保持來源一致性。

    Parameters
    ----------
    candidates           : fetch_qa_candidates 回傳的 [(entry, score), ...] 列表
    selected_label       : Radio 選取的 label 字串（格式 "#N 相似度 ..."）
    question             : 使用者目前輸入的問題（用於更新 chatbot 的 user 訊息）
    history              : 當前 chatbot 的訊息歷史列表（in-place 修改後回傳）
    use_complete_sections: 是否渲染完整章節面板（影響 full_info_html 的內容）

    Returns
    -------
    tuple：(history, full_info_html, side_panel_update, pending_qa, cand_group_update)
      history          : 已附加新問答的 chatbot 訊息列表
      full_info_html   : 右側「檢索原始條款」面板的 HTML
      side_panel_update: gr.update(visible=True)，顯示側邊面板
      pending_qa       : 供儲存或回報的問答字典（含問題、答案、條款等）
      cand_group_update: gr.update(visible=False)，隱藏候選面板
    """
    if not candidates:
        return (history,
                "<p style='color:red;'>⚠️ 沒有候選答案</p>",
                gr.update(visible=False), None, gr.update(visible=False))

    # 解析 Radio label 取得選取的索引
    idx = 0
    if selected_label:
        try:
            idx = int(selected_label.split()[0].replace("#", "")) - 1
        except Exception:
            idx = 0
    idx = max(0, min(idx, len(candidates) - 1))

    entry, score = candidates[idx]
    response     = entry.get("answer", "")
    rel_clauses  = entry.get("related_clauses", [])
    complete_secs = entry.get("complete_sections", [])
    orig_question = entry.get("question", "")

    # 快取標識 badge，顯示相似度和原問題
    cache_badge = (
        f"<div style='background:#d4edda;border:1px solid #28a745;"
        f"padding:10px;border-radius:4px;margin-bottom:12px;'>"
        f"<strong>💾 快取答案</strong>（相似度：{score:.3f}）"
        f"&nbsp;·&nbsp; 原問題：「{orig_question[:60]}"
        f"{'...' if len(orig_question)>60 else ''}」</div>"
    )

    rel_html = format_related_clauses(rel_clauses)
    sec_html = (format_complete_sections(complete_secs) if use_complete_sections
                else "<p style='color:#6c757d;'>（已關閉完整章節檢索）</p>")

    full_info_html = (
        f"<div style='padding:15px;'>{cache_badge}"
        f"<div style='background:#d1ecf1;padding:12px;margin-bottom:15px;border-radius:4px;'>"
        f"<strong>📊 快取資料：</strong> {len(rel_clauses)} 個條款"
        f"{f', {len(complete_secs)} 個章節' if use_complete_sections else ''}</div>"
        f"{rel_html}<hr>{sec_html}</div>"
    )

    history.append({"role": "user",      "content": question or orig_question})
    history.append({"role": "assistant", "content":
                    f"💾 *（快取答案，相似度 {score:.3f}）*\n\n{response}"})

    # pending_qa 供使用者點「💾 儲存此答案」時傳入 save_current_qa
    pending_qa = {
        "question":         question or orig_question,
        "answer":           response,
        "related_clauses":  rel_clauses,
        "complete_sections": complete_secs,
        "model":            entry.get("model", ""),
        "lang":             entry.get("lang", ""),
    }

    return (history, full_info_html,
            gr.update(visible=True), pending_qa, gr.update(visible=False))


# ==========================================
# D. QA 快取管理 Tab 輔助
# ==========================================

def render_qa_html(folder_name: str) -> str:
    """
    讀取指定資料夾的 qa_cache.json，渲染成 HTML 列表顯示在「💬 問答快取管理」Tab。

    【卡片顯示資訊】
    每筆問答卡片（由新到舊排列）包含：
      - 序號、時間戳記（精確到秒）、使用的模型名稱
      - UUID 縮短顯示（前 20 碼，方便對照）
      - 使用者問題（完整顯示）
      - AI 答案（限 350 字，超過顯示 "..."；可捲動）
      - 相關條款數 / 完整章節數

    【reversed 的原因】
    qa_cache.json 是依儲存時間升序（舊→新）排列，
    reversed 後最新的問答排在最上方，符合使用習慣。

    【資料夾名稱為空】
    Gradio Dropdown 初始化時 value=None，此時回傳引導文字，不報錯。

    Parameters
    ----------
    folder_name : 合約資料夾名稱（非完整路徑，只有資料夾名稱）

    Returns
    -------
    str：HTML 字串（直接設定給 gr.HTML 元件的 value）
    """
    if not folder_name:
        return "<p style='color:gray;padding:20px;'>請先選擇文件資料夾</p>"

    folder  = Path(TEXT_RESULT_DIR) / folder_name
    qa_file = folder / "qa_cache.json"

    if not qa_file.exists():
        return f"<p style='color:#888;padding:20px;'>「{folder_name}」尚無儲存的問答</p>"

    try:
        import json
        with open(qa_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        return f"<p style='color:red;'>讀取失敗：{e}</p>"

    if not entries:
        return "<p style='color:gray;padding:20px;'>此資料夾沒有儲存的問答</p>"

    html = (f"<div style='padding:10px;'>"
            f"<h3>📚 {folder_name} — 共 {len(entries)} 筆問答</h3>")

    # reversed：最新的問答顯示在最上方
    for i, e in enumerate(reversed(entries), 1):
        eid      = e.get("id", "")
        ts       = e.get("timestamp", "")
        q        = e.get("question", "")
        a        = e.get("answer", "")
        model    = e.get("model", "")
        n_clauses = len(e.get("related_clauses", []))
        n_secs    = len(e.get("complete_sections", []))
        html += f"""
<div style='border:1px solid #dee2e6;margin:12px 0;padding:15px;
border-radius:8px;background:#fff;'>
  <div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;margin-bottom:8px;'>
    <span style='font-size:0.82em;color:#6c757d;'>
      #{len(entries)-i+1} &nbsp;·&nbsp; {ts} &nbsp;·&nbsp; 模型：{model}
    </span>
    <span style='font-size:0.78em;font-family:monospace;color:#aaa;'>
      ID: {eid[:20]}...
    </span>
  </div>
  <div style='background:#e8f4f8;padding:10px;border-radius:4px;margin-bottom:8px;'>
    <strong>❓ 問：</strong>{q}
  </div>
  <div style='background:#f8f9fa;padding:10px;border-radius:4px;max-height:110px;overflow-y:auto;'>
    <strong>💬 答：</strong>{a[:350]}{"..." if len(a)>350 else ""}
  </div>
  <div style='margin-top:6px;font-size:0.82em;color:#555;'>
    📎 {n_clauses} 個條款 &nbsp;·&nbsp; {n_secs} 個章節
  </div>
</div>"""
    html += "</div>"
    return html


def get_qa_entry_choices(folder_name: str):
    """
    回傳指定資料夾的 qa_cache.json 中所有問答條目的 Dropdown 選項，
    供「刪除單筆答案」功能使用。

    【Dropdown 選項格式】
    choices 是 (label, value) 的元組列表，Gradio Dropdown 支援此格式：
      label（顯示給使用者看）= "[YYYY-MM-DD HH:MM] 問題前60字..."
      value（實際傳入處理函式）= entry_id（完整 UUID，用於精確刪除）

    【reversed 排序】
    最新的問答排在最前面，與 render_qa_html 的排列順序一致，
    方便使用者對照 HTML 列表選擇要刪除的條目。

    【資料夾無效或快取為空】
    回傳 gr.update(choices=[], value=None)，讓 Dropdown 顯示空白。
    若資料夾存在但無 .txt 檔案（OCR 未完成），也回傳空選項。

    Parameters
    ----------
    folder_name : 合約資料夾名稱

    Returns
    -------
    gr.update：包含 choices 和 value 的 Gradio 更新物件
    """
    if not folder_name:
        return gr.update(choices=[], value=None)
    folder    = Path(TEXT_RESULT_DIR) / folder_name
    txt_files = list(folder.glob("*.txt"))
    if not txt_files:
        return gr.update(choices=[], value=None)
    cache = load_qa_cache(str(txt_files[0]))
    if not cache:
        return gr.update(choices=[], value=None)
    choices = []
    for e in reversed(cache):
        ts    = e.get("timestamp", "")[:16]
        q     = e.get("question", "")
        eid   = e.get("id", "")
        label = f"[{ts}] {q[:60]}{'...' if len(q)>60 else ''}"
        choices.append((label, eid))
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


# ==========================================
# E. 智能搜尋相關
# ==========================================

def build_folder_metadata_index():
    """
    掃描 TEXT_RESULT_DIR 下所有有 .txt 的合約資料夾，
    以「資料夾名稱」作為文字建立 VectorStoreIndex，
    供智能搜尋（search_folders_by_description）使用。

    【索引的設計限制】
    索引依據「只有資料夾名稱」，不包含合約內容文字。
    這樣設計的好處是：索引建立快速（不需讀取合約內容），
    占用記憶體少，且不需每次修改合約內容時重建索引。
    代價是：搜尋效果完全取決於資料夾命名是否語意清晰。

    【命名建議】
    - ✅ 好的命名：「2024_廠商甲_採購合約」「葉月菊工程承攬合約_第三期」
    - ❌ 不好的命名：「合約001」「document」「final_v3」

    【index 的 metadata 結構】
    每個 Document 的 metadata 包含：
      folder_name : 資料夾名稱（也是合約名稱）
      txt_path    : 合約 .txt 檔案的完整路徑
      file_count  : 資料夾內的圖片數量（.jpg 和 .png 的總數）

    觸發時機：
      - 應用程式啟動時（demo.load 事件）
      - 使用者點「🔄 更新檢索範圍」按鈕

    Returns
    -------
    tuple：(index, folder_info_list)
      index            : VectorStoreIndex 物件（無任何合約時回傳 None）
      folder_info_list : [{folder_name, txt_path, file_count}, ...] 列表
    """
    from llama_index.core import Document, VectorStoreIndex
    root = Path(TEXT_RESULT_DIR)
    if not root.exists():
        return None, []

    documents, folder_info = [], []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        txt_files = list(folder.glob("*.txt"))
        if not txt_files:
            continue   # 沒有 .txt 代表尚未完成 OCR，跳過
        metadata = {
            "folder_name": folder.name,
            "txt_path":    str(txt_files[0]),
            "file_count":  len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png"))),
        }
        # 以資料夾名稱作為文字建立 Document
        documents.append(Document(text=folder.name, metadata=metadata))
        folder_info.append(metadata)

    if not documents:
        return None, []

    index = VectorStoreIndex.from_documents(documents)
    return index, folder_info


def search_folders_by_description(description: str, index, top_k: int = 5) -> list:
    """
    用使用者輸入的描述文字，在資料夾名稱索引中搜尋最相似的合約。
    此函式是「🔍 智能檢索回答」Tab 的核心搜尋邏輯。

    【搜尋機制】
    使用 VectorStoreIndex 的向量相似度搜尋：
      1. 對 description 計算 embedding 向量
      2. 與所有資料夾名稱的 embedding 做餘弦相似度比較
      3. 回傳 top_k 個最相似的結果

    【top_k=5 的設計考量】
    UI 的「搜尋結果文件」區塊有 5 個快速切換按鈕，
    設為 5 使搜尋結果與按鈕數量完全對應，不浪費也不截斷。

    Parameters
    ----------
    description : 使用者輸入的合約描述文字（如「工程承攬合約 廠商甲」）
    index       : build_folder_metadata_index 建立的 VectorStoreIndex 物件
    top_k       : 回傳前幾名候選（預設 5，與 UI 按鈕數量對應）

    Returns
    -------
    list[dict]：依相似度降冪排序的候選列表，每個字典包含：
      folder_name : 合約資料夾名稱
      txt_path    : 合約 .txt 完整路徑
      score       : 相似度分數（0.0~1.0，越接近 1 越相似）
    """
    if index is None:
        return []
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(description)
    return [
        {
            "folder_name": n.node.metadata.get("folder_name"),
            "txt_path":    n.node.metadata.get("txt_path"),
            "score":       n.score,
        }
        for n in nodes
    ]


def update_smart_doc_switcher(search_results: list):
    """
    依搜尋結果更新「搜尋結果文件」區塊的 5 個快速切換按鈕。

    【按鈕更新邏輯】
    - 無結果時：隱藏整個按鈕群組（gr.update(visible=False)）
    - 有結果時：
        - 有對應結果的按鈕：顯示，label 格式「📄 資料夾名稱　(相似度)」
        - 多餘的按鈕（搜尋結果不足 5 個）：設為 visible=False

    【固定 5 個按鈕的設計原因】
    Gradio 不支援動態增減元件數量，
    因此在 UI 初始化時固定建立 5 個按鈕，
    並在每次搜尋後透過 visible 控制顯示哪些。

    【回傳順序固定】
    (group_update, btn0, btn1, btn2, btn3, btn4) 共 6 個 gr.update，
    對應 _DOC_SWITCHER_OUTPUTS 列表的元素順序（在 app.py 中定義）。

    Parameters
    ----------
    search_results : search_folders_by_description 的回傳值

    Returns
    -------
    tuple：6 個 gr.update 物件，順序固定
    """
    if not search_results:
        return (gr.update(visible=False),
                *[gr.update(visible=False)] * 5)

    btn_updates = []
    for i in range(5):
        if i < len(search_results):
            name  = search_results[i]["folder_name"]
            score = search_results[i].get("score", 0.0)
            btn_updates.append(gr.update(value=f"📄 {name}　({score:.2f})", visible=True))
        else:
            btn_updates.append(gr.update(visible=False))

    return (gr.update(visible=True), *btn_updates)


# ==========================================
# F. 歷史資料夾工具
# ==========================================

def get_history_folders() -> list:
    """
    掃描 TEXT_RESULT_DIR 下所有子資料夾，回傳資料夾名稱列表。

    用途：
      - Tab 2「歷史紀錄」的 Dropdown choices（選擇要載入的合約）
      - Tab 3「Embedding 快取管理」的 Dropdown choices
      - Tab 4「問答快取管理」的 Dropdown choices
      - 各個 Tab 的「重新整理」按鈕點擊後更新選項

    注意事項：
      - 回傳的是資料夾名稱（str），不是完整路徑
      - 未過濾掉沒有 .txt 的資料夾（使用者可能只上傳了圖片但尚未跑 OCR）
      - 根目錄不存在時回傳空列表，不拋出例外

    Returns
    -------
    list[str]：資料夾名稱列表（可能為空）
    """
    root = Path(TEXT_RESULT_DIR)
    if not root.exists():
        return []
    return [d.name for d in root.iterdir() if d.is_dir()]


def preview_history_folder(folder_name: str):
    """
    預覽指定歷史資料夾的頁面圖片和基本資訊。
    用於「選擇歷史紀錄」Tab 的預覽功能，讓使用者確認是否為目標合約後再載入。

    【圖片排序邏輯】
    依檔名中的頁碼數字排序（page0.jpg, page1.jpg, ...）。
    若頁碼部分不是純數字（如命名錯誤的檔案），排到最後（sort key 為 stem 字串）。
    用 try/except 處理排序例外，確保至少能回傳未排序的結果。

    【狀態訊息格式】
    「📂 資料夾：<名稱>
     📸 圖片數：<N>
     📝 文字檔：✅ 存在 / ❌ 遺失」
    文字檔遺失代表 OCR 尚未完成，此時不應使用「載入此專案」。

    Parameters
    ----------
    folder_name : 合約資料夾名稱（非完整路徑）

    Returns
    -------
    tuple：(images, status_msg)
      images     : 頁面圖片路徑列表（供 gr.Gallery 顯示，可能為空）
      status_msg : 包含資料夾名稱、圖片數、.txt 狀態的文字摘要
    """

    if not folder_name:
        return [], "請選擇資料夾"

    folder_path = Path(TEXT_RESULT_DIR) / folder_name
    try:
        jpgs = list(folder_path.glob("*.jpg"))
        jpgs.sort(key=lambda x: int(x.stem.replace("page", ""))
                  if x.stem.replace("page", "").isdigit() else x.stem)
        images = [str(p) for p in jpgs]
    except Exception:
        images = [str(p) for p in folder_path.glob("*.jpg")]

    txt_files  = list(folder_path.glob("*.txt"))
    status_msg = (
        f"📂 資料夾：{folder_name}\n"
        f"📸 圖片數：{len(images)}\n"
        f"📝 文字檔：{'✅ 存在' if txt_files else '❌ 遺失'}"
    )
    return images, status_msg


# ==========================================
# G. QA 儲存 / 回答
# ==========================================

def save_current_qa(pending_qa: dict, current_txt_path: str) -> str:
    """
    將當前的問答結果儲存到 qa_cache.json。
    pending_qa 由 apply_cached_answer 或 rag_chat_response 建立，
    包含問題、答案、相關條款、完整章節、模型名稱、語言。

    pending_qa 為 None（尚未問任何問題）或 current_txt_path 為空時回傳警告訊息。
    儲存成功時回傳「✅ 已儲存（ID: ...）」，顯示 UUID 的前 12 碼。
    """
    if not pending_qa:
        return "⚠️ 沒有可儲存的答案（請先提問）"
    if not current_txt_path:
        return "⚠️ 找不到文件路徑，無法儲存"
    try:
        entry_id = save_qa_entry(
            current_txt_path,
            pending_qa["question"],
            pending_qa["answer"],
            pending_qa.get("related_clauses", []),
            pending_qa.get("complete_sections", []),
            pending_qa.get("model", ""),
            pending_qa.get("lang", ""),
        )
        return f"✅ 已儲存（ID: {entry_id[:12]}...）"
    except Exception as e:
        return f"❌ 儲存失敗：{e}"


def report_error_qa(pending_qa: dict, current_txt_path: str) -> str:
    """
    將當前問答標記為「有問題的答案」，寫入 ERRORLOG 資料夾。
    與 save_current_qa 不同：
      - 不寫入 qa_cache.json（不列入快取）
      - 改寫到 ERRORLOG_DIR/<合約名稱>/<timestamp>_error.json
      - 包含合約名稱、問題、答案、模型、時間戳記

    ERRORLOG 資料夾結構：
    ERRORLOG/
    └── <合約名稱>/
        └── 20250101_120000_error.json

    後台直接用檔案瀏覽器點開 JSON 即可查看，不需額外 Gradio 介面。
    """
    if not pending_qa:
        return "⚠️ 沒有可回報的答案（請先提問）"
    if not current_txt_path:
        return "⚠️ 找不到文件路徑，無法回報"

    try:
        folder_name = Path(current_txt_path).parent.name
        log_dir     = Path(ERRORLOG_DIR) / folder_name
        log_dir.mkdir(parents=True, exist_ok=True)

        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = log_dir / f"{ts}_error.json"

        log_entry = {
            "id":                str(uuid.uuid4()),
            "reported_at":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "contract_folder":   folder_name,
            "txt_path":          current_txt_path,
            "question":          pending_qa.get("question", ""),
            "answer":            pending_qa.get("answer", ""),
            "related_clauses":   pending_qa.get("related_clauses", []),
            "complete_sections": pending_qa.get("complete_sections", []),
            "model":             pending_qa.get("model", ""),
            "lang":              pending_qa.get("lang", ""),
            "note":              "使用者回報此答案有誤",
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        return f"⚠️ 已回報錯誤答案（{log_file.name}）"

    except Exception as e:
        return f"❌ 回報失敗：{e}"


def _auto_save_daily_rolling_log(current_txt_path: str, log_entry: dict):
    """
    修正版：自動將問答存入每日紀錄，確保 mkdir 作用於目錄而非檔案。
    """
    if not current_txt_path:
        return
    
    try:
        # 1. 取得合約資料夾名稱 (例如: "葉月菊 工程承攬合約")
        folder_name = Path(current_txt_path).parent.name
        
        # 2. 定義每日紀錄的存放目錄 (確保這裡是指向 DAILY_LOG 下的子目錄)
        daily_dir = Path(DAILY_LOG_DIR) / folder_name
        
        # ⚠️ 關鍵修正：確保 daily_dir 是一個目錄路徑
        # 如果您不小心把 daily_dir 設成了 current_txt_path，這裡就會噴 WinError 183
        daily_dir.mkdir(parents=True, exist_ok=True) 

        daily_file = daily_dir / "rolling_daily_history.json"

        # 3. 讀取與過濾邏輯... (其餘不變)
        history_data = []
        if daily_file.exists():
            try:
                with open(daily_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
            except:
                pass

        now = datetime.datetime.now()
        cutoff_time = now - datetime.timedelta(days=1)
        
        # 只保留 24 小時內的資料
        filtered_data = [
            e for e in history_data 
            if datetime.datetime.strptime(e.get("timestamp", "2000-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S") >= cutoff_time
        ]

        filtered_data.append(log_entry)

        with open(daily_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        # 這裡會印出您看到的錯誤訊息
        print(f"⚠️ 自動每日紀錄失敗: {e}")
        
def rag_chat_response(message: str, history: list, rag_engine,
                      selected_model: str, lang: str,
                      use_complete_sections: bool, current_txt_path: str = None):
    """
    直接呼叫 HierarchicalQueryEngine 產生新答案（不使用快取）。
    通常在使用者點「🔄 重新呼叫 AI 生成」，或快取無命中時觸發。

    【截斷機制】
    截斷旗標由 request_abort() 設定，parsetooltestspeedfix.py 的
    streaming loop 每個 chunk 前都會檢查，確認後呼叫 stream.close()
    真正讓 Ollama server 停止推理。

    【每次重新初始化 LLM】
    Settings.llm = Ollama(model=selected_model, ...)
    確保使用者在 UI 切換模型後，下一次問答立即生效，不需重啟程式。
    """
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama

    # 每次新問答開始前清除截斷旗標
    clear_abort()

    if rag_engine is None:
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": "⚠️ 知識庫未就緒。"})
        return (history,
                "<div style='color:orange;padding:20px;'>⚠️ 知識庫未就緒</div>",
                gr.update(visible=False), None, gr.update(visible=False))

    try:
        # 每次都重新指定模型，確保 UI 切換模型後立即生效
        Settings.llm = Ollama(model=selected_model, request_timeout=1200.0)

        combined_prompt = f"{message}\n\n(請注意：請務必使用「{lang}」回答我。)"

        rag_result    = rag_engine.query_with_complete_sections(
            combined_prompt, include_complete_sections=use_complete_sections
        )
        response      = str(rag_result["answer"])
        rel_clauses   = rag_result.get("related_clauses", [])
        complete_secs = rag_result.get("complete_sections", [])

        rel_html = format_related_clauses(rel_clauses)
        sec_html = (format_complete_sections(complete_secs) if use_complete_sections
                    else "<p style='color:#6c757d;'>（已關閉完整章節檢索）</p>")
        full_info_html = (
            f"<div style='padding:15px;'>"
            f"<div style='background:#d1ecf1;padding:12px;margin-bottom:15px;border-radius:4px;'>"
            f"<strong>📊 找到：</strong> {len(rel_clauses)} 個條款"
            f"{f', {len(complete_secs)} 個章節' if use_complete_sections else ''}</div>"
            f"{rel_html}<hr>{sec_html}</div>"
        )

        # 截斷時 pending_qa 設為 None，不讓使用者誤存截斷後的不完整答案
        was_aborted = response.startswith("⛔")
        pending_qa  = None if was_aborted else {
            "question":          message,
            "answer":            response,
            "related_clauses":   rel_clauses,
            "complete_sections": complete_secs,
            "model":             selected_model,
            "lang":              lang,
        }
        if pending_qa and current_txt_path:
            auto_log = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question":  message,
                "answer":    response,
                "model":     selected_model
            }
            _auto_save_daily_rolling_log(current_txt_path, auto_log)
    except Exception as e:
        import traceback
        traceback.print_exc()
        response       = f"❌ 錯誤: {str(e)}"
        full_info_html = f"<div style='color:red;padding:20px;'>錯誤：{str(e)}</div>"
        pending_qa     = None

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": response})

    return (history, full_info_html,
            gr.update(visible=True), pending_qa, gr.update(visible=False))
