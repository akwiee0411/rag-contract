# cache_manager.py
"""
快取管理模組
============
本模組統一管理系統的兩種快取：

  A. Embedding 快取（向量索引快取）
     由 LlamaIndex 建立的向量索引，以 JSON 格式儲存在各合約資料夾內。
     快取存在時，RAG 索引建立可跳過 Embedding 計算，大幅節省時間。
     主要檔案：docstore.json、vector_store.json、index_store.json 等。

  B. QA 問答快取
     使用者儲存的問答記錄，以 JSON 陣列儲存在各合約資料夾內（qa_cache.json）。
     下次問相似問題時，可直接複用快取答案，不需重新呼叫 LLM。

目錄結構（每個合約資料夾內）：
  TEXT_RESULT_DIR/
  └── <合約名稱>/
      ├── <合約名稱>.txt         → OCR 提取的純文字
      ├── page0.jpg, page1.jpg  → 頁面圖片
      ├── docstore.json         → ← Embedding 快取（LlamaIndex 產生）
      ├── index_store.json      → ← Embedding 快取（LlamaIndex 產生）
      ├── vector_store.json     → ← Embedding 快取（LlamaIndex 產生）
      └── qa_cache.json         → ← QA 問答快取

【相依關係】
  - 被 doc_processor.py（build_or_load_index）引用：讀取 Embedding 快取狀態
  - 被 ui_helpers.py（rag_chat_response 等）引用：讀寫 QA 快取
  - 被 app.py 的 Tab 3 / Tab 4 UI 引用：提供快取管理功能

【注意事項】
  TEXT_RESULT_DIR 若需更改路徑，需同步修改以下四個檔案：
    - cache_manager.py（本檔，第 30 行）
    - doc_processor.py（第 13 行）
    - ui_helpers.py（第 29 行）
    - app.py（第 34 行）
"""

import json
import uuid
import datetime
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 常數定義
# ──────────────────────────────────────────────────────────────

# CACHE_MARKER：判斷 Embedding 快取是否存在的標記檔案名稱。
# 只要這個檔案存在，就認定整份快取有效。
# 選用 docstore.json 是因為 LlamaIndex 一定會產生此檔，不會缺少。
CACHE_MARKER = "docstore.json"

# QA_CACHE_FILE：每個合約資料夾內的問答快取 JSON 檔名
QA_CACHE_FILE = "qa_cache.json"

# _KNOWN_EMBED_FILES：LlamaIndex 持久化後會產生的所有 JSON 檔案清單。
# 刪除快取時依此列表刪除，不刪頁面圖片或文字檔。
# 若升級 LlamaIndex 版本後多出新檔，可在此補充。
_KNOWN_EMBED_FILES = [
    "docstore.json",
    "index_store.json",
    "vector_store.json",
    "graph_store.json",
    "image__vector_store.json",
]

# TEXT_RESULT_DIR：所有合約處理結果的根儲存目錄。
# ⚠️ 換機器或搬移路徑時，必須同步修改所有四個模組（見上方模組說明）。
TEXT_RESULT_DIR = r"C:\Users\user\Documents\rag_contract\text_result"


# ==========================================
# A. Embedding 快取管理
# ==========================================
# 以下函式處理 LlamaIndex 向量索引的讀取判斷與刪除。
# 「建立」索引由 doc_processor.build_or_load_index 負責；
# 本模組只負責「偵測是否存在」和「刪除」。

def _cache_dir(txt_path: str) -> Path:
    """
    根據 .txt 檔案路徑，回傳其所在資料夾的 Path 物件。

    設計邏輯：
      Embedding 快取的 JSON 檔（docstore.json 等）與 .txt 文字檔
      儲存在同一資料夾，所以直接取 .txt 的 parent 即為快取目錄。
      這樣設計的好處是：快取路徑完全由 txt_path 決定，
      不需要另外維護一個「快取路徑」的對應表。

    此函式為私有函式（名稱以 _ 開頭），
    只在 cache_manager 模組內部使用，外部應透過公開函式操作。

    範例：
      txt_path = "C:\\...\\text_result\\合約A\\合約A.txt"
      回傳     = Path("C:\\...\\text_result\\合約A")
    """
    return Path(txt_path).parent


def _cache_marker(cache: Path) -> Path:
    """
    回傳快取標記檔案（docstore.json）的完整 Path。

    判斷邏輯說明：
      此函式只回傳路徑，不做任何 IO 操作。
      外部呼叫 .exists() 才真正碰磁碟。
      此函式為私有（名稱以 _ 開頭），只在本模組內使用。

    選用 docstore.json 作為標記的原因：
      LlamaIndex 在 index.storage_context.persist() 時，
      一定會產生 docstore.json（即使沒有其他節點），
      因此以此檔案的存在作為「快取是否完整」的判斷依據最可靠。

    範例：
      cache  = Path("C:\\...\\text_result\\合約A")
      回傳   = Path("C:\\...\\text_result\\合約A\\docstore.json")
    """
    return cache / CACHE_MARKER


def collection_exists_and_has_data(txt_path: str) -> bool:
    """
    檢查指定合約是否已有 Embedding 快取。
    只要 docstore.json 存在就判定快取有效（其餘檔案可能因版本不同而缺少）。

    用途：
      build_or_load_index 在建立新索引前呼叫此函式，
      若回傳 True 就直接載入快取，跳過 Embedding 計算。

    Parameters
    ----------
    txt_path : 合約 .txt 檔案路徑

    Returns
    -------
    bool：True = 快取存在且可用；False = 需要重新建立索引
    """
    return _cache_marker(_cache_dir(txt_path)).exists()


def _get_cache_json_files(folder: Path) -> list:
    """
    掃描指定資料夾，回傳所有實際存在的 Embedding 快取 JSON 檔案路徑列表。

    實作細節：
      - 只回傳 _KNOWN_EMBED_FILES 中明確列出的已知檔案，
        不使用 folder.glob("*.json")，以避免誤刪使用者自訂的 JSON 檔案
        （例如 qa_cache.json）。
      - 若 LlamaIndex 升級後產生新的快取 JSON，
        請在 _KNOWN_EMBED_FILES 常數中補充，否則刪除快取時會漏刪。
      - 此函式為私有（以 _ 開頭），供 delete_cache_for_txt、
        list_cached_collections、doc_processor 中的快取清除邏輯呼叫。

    Parameters
    ----------
    folder : 合約資料夾的 Path

    Returns
    -------
    list[Path]：存在的快取 JSON 檔案列表（可能為空）
    """
    return [folder / f for f in _KNOWN_EMBED_FILES if (folder / f).exists()]


def list_cached_collections() -> list:
    """
    掃描 TEXT_RESULT_DIR，列出所有已有 Embedding 快取的合約資料夾。
    用於 Tab 3（Embedding 快取管理）顯示快取清單。

    每筆結果包含：
      name    : 資料夾名稱（合約名稱）
      path    : 完整路徑字串
      size_mb : 所有快取 JSON 的總大小（MB，四捨五入到小數點後 2 位）
      files   : 存在的快取 JSON 檔案名稱列表

    Returns
    -------
    list[dict]：快取資訊列表，無任何快取時回傳空列表
    """
    result = []
    root = Path(TEXT_RESULT_DIR)
    if not root.exists():
        return result
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        # 只列出有 CACHE_MARKER 的資料夾（確認快取完整性）
        if _cache_marker(folder).exists():
            files   = _get_cache_json_files(folder)
            size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
            result.append({
                "name":    folder.name,
                "path":    str(folder),
                "size_mb": round(size_mb, 2),
                "files":   [f.name for f in files],
            })
    return result


def delete_cache_for_txt(txt_path: str) -> str:
    """
    刪除指定合約的 Embedding 快取（所有已知 JSON 檔案）。
    不刪除 .txt 文字檔、頁面圖片等非快取檔案。
    刪除後下次載入該合約時，系統會自動重新計算 Embedding。

    Parameters
    ----------
    txt_path : 合約 .txt 檔案路徑（用於定位快取資料夾）

    Returns
    -------
    str：操作結果訊息（✅ 成功 / ⚠️ 找不到）
    """
    cache = _cache_dir(txt_path)
    files = _get_cache_json_files(cache)
    if not files:
        return f"⚠️ 找不到快取：{cache.name}"
    for f in files:
        f.unlink()
    return f"✅ 已刪除快取（{cache.name}）：{', '.join(f.name for f in files)}"


def delete_all_cache() -> str:
    """
    刪除 TEXT_RESULT_DIR 下所有合約資料夾的 Embedding 快取。
    適合在 Embedding 模型更換後使用，強制所有合約重新計算向量。
    不影響 .txt 文字檔、頁面圖片、qa_cache.json 等非 Embedding 快取檔案。

    Returns
    -------
    str：刪除總數的結果訊息
    """
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
# qa_cache.json 是一個 JSON 陣列，每個元素（entry）代表一筆問答記錄。
# 每筆記錄的結構：
# {
#   "id":                UUID 字串（用於精確刪除）,
#   "question":          使用者提問字串,
#   "answer":            LLM 生成的答案字串,
#   "related_clauses":   相關條款列表（來自向量搜尋結果）,
#   "complete_sections": 完整章節列表（來自 section_index 查詢）,
#   "model":             使用的 LLM 模型名稱（如 "qwen3:4b"）,
#   "lang":              回答語言設定（如 "繁體中文"）,
#   "timestamp":         建立時間（"YYYY-MM-DD HH:MM:SS" 格式）,
# }

def get_qa_cache_path(txt_path: str) -> Path:
    """
    回傳指定合約的 qa_cache.json 完整路徑。
    qa_cache.json 固定儲存在與 .txt 同一資料夾內。

    Parameters
    ----------
    txt_path : 合約 .txt 檔案路徑

    Returns
    -------
    Path：qa_cache.json 的完整路徑（檔案不一定存在）
    """
    return Path(txt_path).parent / QA_CACHE_FILE


def load_qa_cache(txt_path: str) -> list:
    """
    讀取指定合約的 qa_cache.json，回傳問答記錄列表。
    檔案不存在或 JSON 格式損壞時，均安全地回傳空列表，不拋出例外。

    Parameters
    ----------
    txt_path : 合約 .txt 檔案路徑

    Returns
    -------
    list[dict]：問答記錄列表，問題為空或讀取失敗時回傳 []
    """
    path = get_qa_cache_path(txt_path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # JSON 損壞（如檔案寫到一半中斷）時回傳空列表而非崩潰
        return []


def save_qa_entry(txt_path: str, question: str, answer: str,
                  related_clauses: list, complete_sections: list,
                  model: str, lang: str) -> str:
    """
    新增一筆問答記錄到指定合約的 qa_cache.json。

    寫入策略說明：
      - 「讀取全部 → 附加新記錄 → 整體覆寫」，而非 append 模式。
        原因：JSON 不支援行尾追加，直接 append 會破壞格式；
        整體覆寫雖然效率稍低，但確保 JSON 永遠有效且可讀。
      - 若寫入失敗（如磁碟空間不足），只印出警告，
        不拋出例外，避免影響使用者繼續問答的流程。

    【UUID 的用途】
      每筆記錄生成獨立 UUID，讓 delete_qa_entry_by_id 可以精確刪除，
      不受問題文字重複的影響（同一問題問了兩次，都有自己的 UUID）。

    Parameters
    ----------
    txt_path          : 合約 .txt 檔案路徑
    question          : 使用者提問
    answer            : LLM 或快取的答案
    related_clauses   : 向量搜尋找到的相關條款列表（可為 []）
    complete_sections : 完整章節列表（可為 []）
    model             : 使用的 LLM 模型名稱（如 "qwen3:4b"）
    lang              : 回答語言設定（如 "繁體中文"）

    Returns
    -------
    str：新建立記錄的 UUID 字串（供呼叫端顯示 ID 前綴，方便使用者對應）
    """
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
        # 儲存失敗只印出警告，不影響使用者流程（記憶體中的 entry 仍正常建立）
        print(f"⚠️ 儲存 QA 快取失敗：{e}")
    return entry["id"]


def delete_qa_entry_by_id(txt_path: str, entry_id: str) -> str:
    """
    依 UUID 刪除 qa_cache.json 中的特定問答記錄。
    採用「過濾不需要的記錄，重新整體覆寫」的方式，
    確保刪除後 JSON 格式仍然有效。

    Parameters
    ----------
    txt_path : 合約 .txt 檔案路徑
    entry_id : 要刪除的問答記錄 UUID

    Returns
    -------
    str：操作結果訊息（✅ 成功 / ⚠️ 找不到 / ❌ 失敗）
    """
    cache     = load_qa_cache(txt_path)
    new_cache = [e for e in cache if e["id"] != entry_id]
    if len(new_cache) == len(cache):
        # 找不到對應 ID，可能已被刪除或 ID 輸入有誤
        return "⚠️ 找不到此 ID"
    try:
        with open(get_qa_cache_path(txt_path), "w", encoding="utf-8") as f:
            json.dump(new_cache, f, ensure_ascii=False, indent=2)
        return f"✅ 已刪除（ID: {entry_id[:12]}...）"
    except Exception as e:
        return f"❌ 刪除失敗：{e}"


def delete_folder_qa_cache(folder_name: str) -> str:
    """
    刪除指定資料夾的整份 qa_cache.json（清除該合約所有問答記錄）。
    只刪快取 JSON，不影響 Embedding 快取或文字檔。

    Parameters
    ----------
    folder_name : 合約資料夾名稱（非完整路徑，只是資料夾名）

    Returns
    -------
    str：操作結果訊息（✅ 成功 / ⚠️ 無快取）
    """
    qa_file = Path(TEXT_RESULT_DIR) / folder_name / QA_CACHE_FILE
    if qa_file.exists():
        qa_file.unlink()
        return f"✅ 已清除「{folder_name}」的所有問答快取"
    return f"⚠️「{folder_name}」沒有問答快取"


def delete_all_qa_cache() -> str:
    """
    刪除 TEXT_RESULT_DIR 下所有合約資料夾的 qa_cache.json。
    適合在測試後清理，或需要重新收集問答時使用。
    不影響 Embedding 快取或 OCR 文字檔。

    Returns
    -------
    str：刪除的資料夾總數
    """
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
    在 qa_cache.json 中搜尋與輸入問題最相似的前 top_k 筆問答。
    供使用者送出問題後，先展示快取候選，避免重複呼叫 LLM。

    【兩種相似度計算方式】

    1. 向量餘弦相似度（rag_available=True 且 Embedding 模型已載入）：
       使用 LlamaIndex Settings.embed_model 對問題與快取問題分別計算 embedding，
       再計算餘弦相似度（cosine similarity）。
       精度高，但需要 GPU 記憶體，每次比對都會推理。

    2. 字元級重疊率（Fallback，rag_available=False 或向量計算失敗）：
       將問題拆成字元集合，計算交集比例：
         overlap = |A ∩ B| / max(|A|, |B|)
       速度快，不需 GPU，但對同義不同字的問題效果較差。
       例如「付款方式」和「支付條件」字元重疊少，可能被視為不相似。

    Parameters
    ----------
    question      : 使用者輸入的問題字串
    txt_path      : 合約 .txt 檔案路徑（用於定位 qa_cache.json）
    top_k         : 回傳最相似的前幾筆（預設 3）
    rag_available : True = 嘗試使用向量相似度；False = 直接使用字元重疊

    Returns
    -------
    list[tuple[dict, float]]：
      [(entry_dict, similarity_score), ...] 依相似度降冪排序
      entry_dict 為 qa_cache.json 中的一筆問答記錄
      similarity_score 範圍 0.0 ~ 1.0（越接近 1 越相似）
    """
    cache = load_qa_cache(txt_path)
    if not cache:
        return []

    scored = []

    if rag_available:
        try:
            from llama_index.core import Settings
            # 對輸入問題計算 embedding 向量
            q_emb = np.array(Settings.embed_model.get_text_embedding(question))
            for entry in cache:
                # 對快取中每個問題計算 embedding，再算餘弦相似度
                c_emb = np.array(Settings.embed_model.get_text_embedding(entry["question"]))
                norm  = np.linalg.norm(q_emb) * np.linalg.norm(c_emb)
                # 避免除以零（空字串的 norm 為 0）
                score = float(np.dot(q_emb, c_emb) / norm) if norm > 0 else 0.0
                scored.append((entry, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception as e:
            print(f"⚠️ 向量相似度計算失敗，退化為關鍵字重疊排序：{e}")

    # Fallback：字符級重疊率（不需 Embedding 模型）
    def _overlap(a: str, b: str) -> float:
        """計算兩個字串的字元集合交集比例（0.0 ~ 1.0）"""
        a_set, b_set = set(a.strip()), set(b.strip())
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / max(len(a_set), len(b_set))

    for entry in cache:
        scored.append((entry, _overlap(question, entry["question"])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
