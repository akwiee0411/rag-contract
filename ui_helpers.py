# ui_helpers.py
"""
UI 輔助模組
包含：格式化輸出、文件圖片工具、QA 候選面板、智能搜尋、文件切換按鈕
"""

import os
import gradio as gr
from pathlib import Path

from cache_manager import (
    load_qa_cache, save_qa_entry, delete_qa_entry_by_id,
    retrieve_similar_qa,
)

TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"
# ==========================================
# A. 格式化輸出
# ==========================================

def format_related_clauses(clauses: list) -> str:
    if not clauses:
        return "<p>沒有找到相關條款</p>"
    html = f"<h3>🎯 相關條款 ({len(clauses)} 個)</h3>"
    for i, c in enumerate(clauses, 1):
        html += (
            f"<div style='border:1px solid #ddd;margin:10px 0;"
            f"padding:10px;border-radius:5px;'>"
            f"<h4>{i}. 【{c.get('hierarchy_path','N/A')}】</h4>"
            f"<p><strong>標題:</strong> {c.get('clause_title','N/A')}</p>"
            f"<p><strong>類型:</strong> {c.get('clause_type','N/A')} | "
            f"<strong>相似度:</strong> {c.get('score',0.0):.3f}</p>"
            f"<p><strong>預覽:</strong> {c.get('text_preview','N/A')}</p>"
            f"</div>"
        )
    return html

def format_complete_sections(sections: list) -> str:
    if not sections:
        return "<p>沒有完整章節內容</p>"
    html = f"<h3>📋 完整章節 ({len(sections)} 個)</h3>"
    for s in sections:
        html += (
            f"<div style='border:2px solid #007bff;margin:15px 0;"
            f"padding:15px;border-radius:8px;'>"
            f"<h4>章節 {s.get('section_number','')}: {s.get('section_title','')}</h4>"
            f"<p><strong>階層路徑:</strong> {s.get('hierarchy_path','')}</p>"
            f"<div style='background:grey;padding:10px;border-radius:4px;"
            f"max-height:300px;overflow-y:auto;'>"
            f"<div style='white-space:pre-wrap;font-family:inherit;"
            f"color:#333;line-height:1.5;'>{s.get('full_content','')}</div>"
            f"</div></div>"
        )
    return html


# ==========================================
# B. 文件圖片工具
# ==========================================

def get_doc_images(txt_path: str) -> list:
    if not txt_path or not os.path.exists(txt_path):
        return []
    folder = Path(txt_path).parent
    imgs   = [p for p in (list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
              if p.stem.startswith("page")]
    imgs.sort(key=lambda x: int(x.stem.replace("page", ""))
              if x.stem.replace("page", "").isdigit() else 999)
    return [str(p) for p in imgs]

def toggle_doc_images(txt_path: str, visible: bool):
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
    送出問題後先做 embedding 檢索，回傳 top-3 快取問答候選。
    完全不呼叫 LLM。
    回傳: (candidates, cand_group_update, radio_update, use_btn_update, html)
    """
    if not txt_path:
        html = ("<div style='border:1.5px solid #f0ad4e;padding:12px;border-radius:6px;'>"
                "⚠️ 尚未載入文件，無法檢索快取</div>")
        return ([], gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False), html)

    candidates = retrieve_similar_qa(message, txt_path, top_k=3,
                                     rag_available=rag_available)

    if not candidates:
        html = ("<div style='border:1.5px solid #6c757d;"
                "padding:12px;border-radius:6px;color:inherit;'>"
                "📭 快取中找不到相似問答，請點擊「🔄 重新呼叫 AI 生成」</div>")
        return ([], gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False), html)

    html = (f"<div style='padding:6px;'>"
            f"<strong>🔍 找到 {len(candidates)} 筆相似歷史問答</strong>"
            f"<p style='font-size:0.82em;color:#888;margin:4px 0 6px;'>"
            f"從下方選取後點「✅ 使用此快取答案」，或點「🔄 重新呼叫 AI 生成」</p>")
    for i, (entry, score) in enumerate(candidates, 1):
        q   = entry.get("question", "")
        a   = entry.get("answer",   "")
        ts  = entry.get("timestamp", "")
        n_c = len(entry.get("related_clauses",   []))
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
    """使用者選了某筆快取答案 → 更新 chatbot + 詳情面板"""
    if not candidates:
        return (history,
                "<p style='color:red;'>⚠️ 沒有候選答案</p>",
                gr.update(visible=False), None, gr.update(visible=False))

    idx = 0
    if selected_label:
        try:
            idx = int(selected_label.split()[0].replace("#", "")) - 1
        except Exception:
            idx = 0
    idx = max(0, min(idx, len(candidates) - 1))

    entry, score    = candidates[idx]
    response        = entry.get("answer",            "")
    rel_clauses     = entry.get("related_clauses",   [])
    complete_secs   = entry.get("complete_sections", [])
    orig_question   = entry.get("question",          "")

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

    pending_qa = {
        "question":          question or orig_question,
        "answer":            response,
        "related_clauses":   rel_clauses,
        "complete_sections": complete_secs,
        "model":             entry.get("model", ""),
        "lang":              entry.get("lang",  ""),
    }
    return (history, full_info_html,
            gr.update(visible=True), pending_qa, gr.update(visible=False))


# ==========================================
# D. QA 快取管理 Tab 輔助
# ==========================================

def render_qa_html(folder_name: str) -> str:
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
    for i, e in enumerate(reversed(entries), 1):
        eid       = e.get("id", "")
        ts        = e.get("timestamp", "")
        q         = e.get("question", "")
        a         = e.get("answer", "")
        model     = e.get("model", "")
        n_clauses = len(e.get("related_clauses",   []))
        n_secs    = len(e.get("complete_sections", []))
        html += f"""
        <div style='border:1px solid #dee2e6;margin:12px 0;padding:15px;
                    border-radius:8px;background:#fff;'>
            <div style='display:flex;justify-content:space-between;
                        flex-wrap:wrap;gap:4px;margin-bottom:8px;'>
                <span style='font-size:0.82em;color:#6c757d;'>
                    #{len(entries)-i+1} &nbsp;·&nbsp; {ts}
                    &nbsp;·&nbsp; 模型：{model}
                </span>
                <span style='font-size:0.78em;font-family:monospace;color:#aaa;'>
                    ID: {eid[:20]}...
                </span>
            </div>
            <div style='background:#e8f4f8;padding:10px;border-radius:4px;margin-bottom:8px;'>
                <strong>❓ 問：</strong>{q}
            </div>
            <div style='background:#f8f9fa;padding:10px;border-radius:4px;
                        max-height:110px;overflow-y:auto;'>
                <strong>💬 答：</strong>{a[:350]}{"..." if len(a)>350 else ""}
            </div>
            <div style='margin-top:6px;font-size:0.82em;color:#555;'>
                📎 {n_clauses} 個條款 &nbsp;·&nbsp; {n_secs} 個章節
            </div>
        </div>"""
    html += "</div>"
    return html

def get_qa_entry_choices(folder_name: str):
    """回傳 gr.update，供 Dropdown 使用"""
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
            continue
        metadata = {
            "folder_name": folder.name,
            "txt_path":    str(txt_files[0]),
            "file_count":  len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png"))),
        }
        documents.append(Document(text=folder.name, metadata=metadata))
        folder_info.append(metadata)
    if not documents:
        return None, []
    index = VectorStoreIndex.from_documents(documents)
    return index, folder_info

def search_folders_by_description(description: str, index, top_k: int = 5) -> list:
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
    """回傳 (group_update, btn0..btn4)"""
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
    root = Path(TEXT_RESULT_DIR)
    if not root.exists():
        return []
    return [d.name for d in root.iterdir() if d.is_dir()]

def preview_history_folder(folder_name: str):
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
    if not pending_qa:
        return "⚠️ 沒有可儲存的答案（請先提問）"
    if not current_txt_path:
        return "⚠️ 找不到文件路徑，無法儲存"
    try:
        entry_id = save_qa_entry(
            current_txt_path,
            pending_qa["question"],
            pending_qa["answer"],
            pending_qa.get("related_clauses",   []),
            pending_qa.get("complete_sections", []),
            pending_qa.get("model", ""),
            pending_qa.get("lang",  ""),
        )
        return f"✅ 已儲存（ID: {entry_id[:12]}...）"
    except Exception as e:
        return f"❌ 儲存失敗：{e}"

def rag_chat_response(message: str, history: list, rag_engine,
                      selected_model: str, lang: str,
                      use_complete_sections: bool, current_txt_path: str = None):
    """直接呼叫 RAG 產生新答案"""
    from llama_index.core import Settings
    from llama_index.llms.ollama import Ollama

    if rag_engine is None:
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": "⚠️ 知識庫未就緒。"})
        return (history,
                "<div style='color:orange;padding:20px;'>⚠️ 知識庫未就緒</div>",
                gr.update(visible=False), None, gr.update(visible=False))

    try:
        Settings.llm    = Ollama(model=selected_model, request_timeout=1200.0)
        combined_prompt = f"{message}\n\n(請注意：請務必使用「{lang}」回答我。)"

        rag_result    = rag_engine.query_with_complete_sections(
            combined_prompt, include_complete_sections=use_complete_sections
        )
        response      = str(rag_result["answer"])
        rel_clauses   = rag_result.get("related_clauses",   [])
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
        pending_qa = {
            "question":          message,
            "answer":            response,
            "related_clauses":   rel_clauses,
            "complete_sections": complete_secs,
            "model":             selected_model,
            "lang":              lang,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        response       = f"❌ 錯誤: {str(e)}"
        full_info_html = f"<div style='color:red;padding:20px;'>錯誤：{str(e)}</div>"
        pending_qa     = None

    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": response})
    return (history, full_info_html,
            gr.update(visible=True), pending_qa, gr.update(visible=False))
