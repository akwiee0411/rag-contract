# cache_manager.py
"""
快取管理模組
包含：Embedding 快取 / QA 問答快取 的所有讀寫刪除操作
"""

import json
import uuid
import datetime
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────────────────────
CACHE_MARKER        = "docstore.json"
QA_CACHE_FILE       = "qa_cache.json"
_KNOWN_EMBED_FILES  = [
    "docstore.json", "index_store.json", "vector_store.json",
    "graph_store.json", "image__vector_store.json",
]
TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"

# ==========================================
# A. Embedding 快取管理
# ==========================================

def _cache_dir(txt_path: str) -> Path:
    return Path(txt_path).parent

def _cache_marker(cache: Path) -> Path:
    return cache / CACHE_MARKER

def collection_exists_and_has_data(txt_path: str) -> bool:
    return _cache_marker(_cache_dir(txt_path)).exists()

def _get_cache_json_files(folder: Path) -> list:
    return [folder / f for f in _KNOWN_EMBED_FILES if (folder / f).exists()]

def list_cached_collections() -> list:
    result = []
    root = Path(TEXT_RESULT_DIR)
    if not root.exists():
        return result
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        if _cache_marker(folder).exists():
            files    = _get_cache_json_files(folder)
            size_mb  = sum(f.stat().st_size for f in files) / 1024 / 1024
            result.append({
                "name":    folder.name,
                "path":    str(folder),
                "size_mb": round(size_mb, 2),
                "files":   [f.name for f in files],
            })
    return result

def delete_cache_for_txt(txt_path: str) -> str:
    cache = _cache_dir(txt_path)
    files = _get_cache_json_files(cache)
    if not files:
        return f"⚠️ 找不到快取：{cache.name}"
    for f in files:
        f.unlink()
    return f"✅ 已刪除快取（{cache.name}）：{', '.join(f.name for f in files)}"

def delete_all_cache() -> str:
    count = 0
    root  = Path(TEXT_RESULT_DIR)
    if root.exists():
        for folder in root.iterdir():
            if not folder.is_dir():
                continue
            for f in _get_cache_json_files(folder):
                f.unlink()
                count += 1
    return f"✅ 已清除所有 Embedding 快取（共刪除 {count} 個檔案）"


# ==========================================
# B. QA 問答快取管理
# ==========================================

def get_qa_cache_path(txt_path: str) -> Path:
    return Path(txt_path).parent / QA_CACHE_FILE

def load_qa_cache(txt_path: str) -> list:
    path = get_qa_cache_path(txt_path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_qa_entry(txt_path: str, question: str, answer: str,
                  related_clauses: list, complete_sections: list,
                  model: str, lang: str) -> str:
    """儲存一筆 QA，回傳新建立的 UUID"""
    cache = load_qa_cache(txt_path)
    entry = {
        "id":                str(uuid.uuid4()),
        "question":          question,
        "answer":            answer,
        "related_clauses":   related_clauses,
        "complete_sections": complete_sections,
        "model":             model,
        "lang":              lang,
        "timestamp":         datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    cache.append(entry)
    try:
        with open(get_qa_cache_path(txt_path), "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 儲存 QA 快取失敗：{e}")
    return entry["id"]

def delete_qa_entry_by_id(txt_path: str, entry_id: str) -> str:
    cache     = load_qa_cache(txt_path)
    new_cache = [e for e in cache if e["id"] != entry_id]
    if len(new_cache) == len(cache):
        return "⚠️ 找不到此 ID"
    try:
        with open(get_qa_cache_path(txt_path), "w", encoding="utf-8") as f:
            json.dump(new_cache, f, ensure_ascii=False, indent=2)
        return f"✅ 已刪除（ID: {entry_id[:12]}...）"
    except Exception as e:
        return f"❌ 刪除失敗：{e}"

def delete_folder_qa_cache(folder_name: str) -> str:
    qa_file = Path(TEXT_RESULT_DIR) / folder_name / QA_CACHE_FILE
    if qa_file.exists():
        qa_file.unlink()
        return f"✅ 已清除「{folder_name}」的所有問答快取"
    return f"⚠️「{folder_name}」沒有問答快取"

def delete_all_qa_cache() -> str:
    count = 0
    root  = Path(TEXT_RESULT_DIR)
    if root.exists():
        for folder in root.iterdir():
            qa_file = folder / QA_CACHE_FILE
            if qa_file.exists():
                qa_file.unlink()
                count += 1
    return f"✅ 已清除所有問答快取（共 {count} 個資料夾）"


# ==========================================
# C. QA 語意相似度檢索
# ==========================================

def retrieve_similar_qa(question: str, txt_path: str,
                        top_k: int = 3, rag_available: bool = False) -> list:
    """
    語意搜尋快取中最相似的前 top_k 筆問答。
    回傳 [(entry, score), ...] 依相似度降冪排序。
    """
    cache = load_qa_cache(txt_path)
    if not cache:
        return []

    scored = []

    if rag_available:
        try:
            from llama_index.core import Settings
            q_emb = np.array(Settings.embed_model.get_text_embedding(question))
            for entry in cache:
                c_emb = np.array(Settings.embed_model.get_text_embedding(entry["question"]))
                norm  = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
                score = float(np.dot(q_emb, c_emb) / norm) if norm > 0 else 0.0
                scored.append((entry, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception as e:
            print(f"⚠️ 向量相似度計算失敗，退化為關鍵字重疊排序：{e}")

    # Fallback：字符級重疊率
    def _overlap(a: str, b: str) -> float:
        a_set, b_set = set(a.strip()), set(b.strip())
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / max(len(a_set), len(b_set))

    for entry in cache:
        scored.append((entry, _overlap(question, entry["question"])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
