# parsetooltestspeedfix.py
"""
合約階層式解析與 RAG 查詢引擎模組
====================================
本模組提供兩個核心類別：

  HierarchicalContractNodeParser（繼承自 LlamaIndex NodeParser）
    - 輸入：合約純文字（LlamaIndex Document 物件）
    - 輸出：具有階層關係的 TextNode 列表，供 LlamaIndex VectorStoreIndex 使用
    - 核心功能：
        1. 偵測合約中的條款編號（支援七種中文常見格式）
        2. 動態分析文件的層級結構（哪一種是主條款，哪一種是子條款）
        3. 切割文字為有父子關係的節點（main_clause / sub_clause / complete_section）
        4. 產生 Header 節點（條款前的合約標題與前言）

  HierarchicalQueryEngine
    - 輸入：VectorStoreIndex 物件和 TextNode 列表
    - 輸出：整合了向量檢索、完整章節取得、LLM 生成的問答結果
    - 核心功能：
        1. 向量檢索（similarity_top_k=3，找最相似的 3 個節點）
        2. 透過 section_index 直接取出完整章節（不只是片段）
        3. 組合 context，構建 prompt，呼叫 Ollama streaming API 生成答案
        4. 截斷機制：每個 streaming chunk 前檢查 ui_helpers._check_abort()
        5. 若 Ollama streaming 失敗，fallback 回 LlamaIndex 的 Settings.llm.complete

【支援的條款編號格式（clause_number_patterns，7 種）】
  0. 第X條 / 第X款 / 第X項   （例：第一條、第3條）← 最高優先级
  1. 羅馬數字、           （例：I、II. III.）
  2. 中文數字、            （例：一、二、三、）
  3. (中文數字)            （例：（一）（二））
  4. 阿拉伯數字、          （例：1、2、3.）
  5. (阿拉伯數字)          （例：(1)(2)）
  6. X.Y（小數點格式）      （例：1.1 2.3）

【節點類型說明】
  header           : 第一個條款之前的文字（合約標題、前言、甲乙方資訊）
  main_clause      : 主層級條款（如「第一條」）
  sub_clause       : 子層級條款（如「一、」隸屬於某個「第X條」下）
  complete_section : 主條款加上其所有子條款的完整文字集合
                     供問答時取出完整脈絡，不只是片段

【相依關係】
  - 被 doc_processor.py 引用：build_or_load_index 中的 import
    ⚠️ 若要更改本檔檔名，需修改以下位置（見下方「更名注意事項」）
  - 引用 ui_helpers.py 的 _check_abort：streaming 截斷檢查
    ⚠️ 若要更改 ui_helpers.py 的檔名，需修改本檔第 485 行的 from ui_helpers import _check_abort

【更名注意事項：若要將本檔從 parsetooltestspeedfix.py 改名】
  需要更改的位置共 2 處：
  1. doc_processor.py 第 15 行（build_or_load_index 函式）：
       from parsetooltestspeedfix import HierarchicalContractNodeParser, HierarchicalQueryEngine
       → 改成 from <新檔名> import ...
  2. ui_helpers.py 第 727 行（rag_chat_response 函式的 docstring）：
       提及 parsetooltestspeedfix.py 的說明文字（非程式碼，可選擇性修改）
  app.py 本身不直接 import 本模組，無需修改。
"""

import time
import re
import os
import json
from typing import List, Dict, Any, Optional, Pattern
from pydantic import Field

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, NodeWithScore, Document
from llama_index.core.node_parser import NodeParser


# ==========================================
# HierarchicalContractNodeParser
# ==========================================

class HierarchicalContractNodeParser(NodeParser):
    """
    階層式中文合約節點解析器。
    繼承自 LlamaIndex NodeParser，可直接插入 LlamaIndex 的索引建立流程。

    核心設計理念：
      合約文字具有固定的條款層級（如「第X條 > 一、 > (1)」），
      若直接用 SentenceSplitter 切割，會破壞條款的完整性。
      本解析器識別條款結構，確保每個節點對應一個完整條款（或其子節點），
      讓向量搜尋結果更有意義。

    Pydantic Field 說明：
      clause_number_patterns 使用 Field(default_factory=...) 是因為
      LlamaIndex NodeParser 繼承自 Pydantic BaseModel，
      list 類型的屬性必須用 default_factory 而非直接賦值，以避免跨實例共享。
    """

    # 七種條款編號的正則表達式，優先順序由高（索引 0）到低（索引 6）
    # 這個順序決定了「哪種格式是主條款」的判斷依據
    clause_number_patterns: List[Pattern[str]] = Field(
        default_factory=lambda: [
            # 0. 「第X條/款/項/章/節」，支援中文數字和阿拉伯數字，可選前綴「#」
            re.compile(r"^(?:#\s*)?第\s*[一二三四五六七八九十百千萬\d]+\s*[條款項章節]", re.MULTILINE),
            # 1. 羅馬數字加分隔符（I、II. III. 等）
            re.compile(r"^(?:#\s*)?[IVXLCDMivxlcdm]+\s*[、．.]", re.MULTILINE),
            # 2. 中文數字加頓號或句點（一、二、三、）
            re.compile(r"^(?:#\s*)?[一二三四五六七八九十百千萬]+\s*[、．.]", re.MULTILINE),
            # 3. 全半形括號包中文數字（（一）（二）(三)）
            re.compile(r"^(?:#\s*)?[\(（]\s*[一二三四五六七八九十]+\s*[\)）]", re.MULTILINE),
            # 4. 阿拉伯數字加頓號或句點（1、2. 3.）；(?!\d) 排除小數點格式如 1.2
            re.compile(r"^(?:#\s*)?\d+\s*(?:[、．]|\.(?!\d))", re.MULTILINE),
            # 5. 括號包阿拉伯數字（(1)(2)(3)）
            re.compile(r"^(?:#\s*)?\(\d+\)", re.MULTILINE),
            # 6. 小數點格式（1.1、2.3），通常用於技術規格或工程合約
            re.compile(r"^(?:#\s*)?\d+\.\d+", re.MULTILINE),
        ]
    )

    class Config:
        # 允許 Pattern 等非 Pydantic 原生型態作為欄位類型
        arbitrary_types_allowed = True

    def _parse_nodes(self, documents: List[Document],
                     show_progress: bool = False, **kwargs) -> List[TextNode]:
        """
        NodeParser 的核心方法，對多份文件批次解析。
        對每份 Document 呼叫 _parse_document，合併結果。

        Parameters
        ----------
        documents     : LlamaIndex Document 物件列表
        show_progress : 是否顯示進度（此處未實作，保留介面相容性）

        Returns
        -------
        List[TextNode]：所有文件的節點合併列表
        """
        nodes: List[TextNode] = []
        for doc in documents:
            nodes.extend(self._parse_document(doc))
        return nodes

    def get_nodes_from_documents(self, documents: List[Document],
                                  show_progress: bool = False, **kwargs) -> List[TextNode]:
        """
        LlamaIndex 標準介面：從 Document 列表取得節點。
        直接代理到 _parse_nodes，確保 LlamaIndex 各版本的相容性。
        doc_processor.py 的 build_or_load_index 呼叫此方法。
        """
        return self._parse_nodes(documents, show_progress=show_progress, **kwargs)

    def _parse_document(self, doc: Document) -> List[TextNode]:
        """
        解析單份合約文件，回傳帶有階層 metadata 的 TextNode 列表。

        【特殊情況：找不到任何條款】
        若文件中沒有偵測到任何條款編號（可能是目錄頁、附件等），
        將整份文件作為一個 full_document 節點回傳。

        【TextNode metadata 欄位說明】
        document_title   : 來自 doc.metadata 的 file_name
        clause_id        : 節點唯一識別（如 "Clause_3"、"Section_2"、"Header_0"）
        source           : 來自 doc.metadata 的 file_path
        clause_number_raw: 條款原始編號字串（如 "第一條"、"一、"）
        clause_title     : 解析出的條款標題（冒號前的文字）
        clause_type      : 節點類型（header / main_clause / sub_clause / complete_section）
        clause_level     : 對應的 pattern_idx（0~6，-1 表示特殊節點）
        parent_clause_id : 父節點的 clause_id（子條款用）
        section_number   : 所屬主條款的序號（1, 2, 3...）
        hierarchy_path   : 階層路徑字串（如 "第一條 > 一、"）
        is_main_clause   : 是否為主條款
        has_sub_clauses  : 是否包含子條款
        contract_id      : 來自 doc.metadata 的 contract_id
        """
        nodes: List[TextNode] = []
        text = doc.text

        # 第一步：找出所有潛在的條款起始行
        potential_clauses = self._find_all_potential_clauses(text)
        if not potential_clauses:
            # 找不到任何條款時，整份文件作為一個節點
            nodes.append(TextNode(
                text=text,
                metadata={
                    "document_title":  doc.metadata.get("file_name", "N/A"),
                    "clause_id":       "Full_Document",
                    "source":          doc.metadata.get("file_path", "N/A"),
                    "clause_type":     "full_document",
                    "clause_level":    -1,
                    "is_main_clause":  False,
                    "has_sub_clauses": False,
                }
            ))
            return nodes

        # 第二步：分析文件的層級結構（哪種格式是主條款）
        hierarchy_info = self._analyze_dynamic_hierarchy_structure(potential_clauses)

        # 第三步：依據層級結構建立章節列表
        hierarchical_sections = self._build_hierarchical_sections(
            text, potential_clauses, hierarchy_info, doc.metadata
        )

        # 第四步：將章節列表轉換為 TextNode 物件
        for section in hierarchical_sections:
            if section['text'].strip():  # 過濾空白節點
                nodes.append(TextNode(
                    text=section['text'],
                    metadata={
                        "document_title":      doc.metadata.get("file_name", "N/A"),
                        "clause_id":           section['clause_id'],
                        "source":              doc.metadata.get("file_path", "N/A"),
                        "clause_number_raw":   section['number'],
                        "clause_title":        section['title'],
                        "clause_type":         section['type'],
                        "clause_level":        section['level'],
                        "parent_clause_id":    section.get('parent_clause_id', ''),
                        "parent_clause_title": section.get('parent_clause_title', ''),
                        "section_number":      section.get('section_number', 0),
                        "hierarchy_path":      section.get('hierarchy_path', ''),
                        "is_main_clause":      section.get('is_main_clause', False),
                        "has_sub_clauses":     section.get('has_sub_clauses', False),
                        "contract_id":         doc.metadata.get("contract_id", "N/A"),
                        "orig_path":           doc.metadata.get("orig_path", "N/A"),
                        "num_pages":           doc.metadata.get("num_pages", "N/A"),
                    }
                ))
        return nodes

    # ── 階層結構分析 ─────────────────────────────────────────────

    def _analyze_dynamic_hierarchy_structure(self, potential_clauses: List[Dict]) -> Dict:
        """
        動態分析合約的條款層級結構，決定哪種條款格式是「主條款」。

        【動態判斷的必要性】
        不同合約使用不同格式作為主條款，例如：
          - 工程合約：「第一條、第二條」（pattern_idx=0）是主條款
          - 某些合約：「一、二、三、」（pattern_idx=2）是主條款
        因此採用「第一個出現的條款所屬格式」作為主層級（primary_level）。

        【返回值欄位說明】
        primary_level      : 主條款的 pattern_idx（0~6）
        levels_used        : 文件中出現過的所有 pattern_idx 列表（升序）
        level_usage        : {pattern_idx: 出現次數} 的計數字典
        structure_type     : 'single_level' / 'two_level' / 'multi_level' / 'complex'
        primary_level_name : 主層級的人類可讀名稱（如 "第X條"）

        Parameters
        ----------
        potential_clauses : _find_all_potential_clauses 回傳的條款列表

        Returns
        -------
        Dict：層級結構資訊
        """
        if not potential_clauses:
            return {'primary_level': -1, 'levels_used': [], 'structure_type': 'none'}

        # 找到第一個「真正的主條款」起始點（跳過前言等非主要條款）
        first_clause_idx = self._find_first_main_clause(potential_clauses)
        main_clauses = (potential_clauses[first_clause_idx:]
                        if first_clause_idx < len(potential_clauses)
                        else potential_clauses)

        if not main_clauses:
            return {'primary_level': -1, 'levels_used': [], 'structure_type': 'none'}

        # 第一個主條款的格式索引即為文件的主層級
        primary_level = main_clauses[0]['pattern_idx']
        print(f"動態識別主層級: 第一個條款模式索引 = {primary_level}")

        # 統計各層級出現次數
        level_usage = {}
        for clause in main_clauses:
            lv = clause['pattern_idx']
            level_usage[lv] = level_usage.get(lv, 0) + 1

        levels_used    = sorted(level_usage.keys())
        structure_type = self._determine_structure_type(levels_used, level_usage)

        print(f"層級使用統計: {level_usage}")
        print(f"確定的主層級: {primary_level} ({self._get_level_name(primary_level)})")

        return {
            'primary_level':      primary_level,
            'levels_used':        levels_used,
            'level_usage':        level_usage,
            'structure_type':     structure_type,
            'primary_level_name': self._get_level_name(primary_level),
        }

    def _determine_structure_type(self, levels_used: List[int],
                                   level_usage: Dict[int, int]) -> str:
        """
        依據使用了幾種層級，判斷文件的結構複雜程度。
        目前用於 debug 輸出，不影響節點切割邏輯。

        Returns
        -------
        str：
          'single_level' = 只有一種條款格式（扁平結構）
          'two_level'    = 主條款 + 一種子條款（最常見）
          'multi_level'  = 三種以上層級（複雜合約）
          'complex'      = 無法分類（理論上不會觸發）
        """
        n = len(levels_used)
        if n == 1:  return 'single_level'
        if n == 2:  return 'two_level'
        if n >= 3:  return 'multi_level'
        return 'complex'

    def _get_level_name(self, level: int) -> str:
        """
        將 pattern_idx 轉換成人類可讀的層級名稱，供 debug 輸出使用。
        對應 clause_number_patterns 的索引順序。

        Returns
        -------
        str：層級名稱（找不到時回傳 "Level_N"）
        """
        return {
            0: "第X條",
            1: "羅馬X、",
            2: "中文X、",
            3: "(中文X)",
            4: "數字X、",
            5: "(數字X)",
            6: "X.Y"
        }.get(level, f"Level_{level}")

    # ── 章節建構 ─────────────────────────────────────────────────

    def _build_hierarchical_sections(self, text: str, potential_clauses: List[Dict],
                                     hierarchy_info: Dict, doc_metadata: Dict) -> List[Dict]:
        """
        依據層級結構，將合約文字切割成具有父子關係的章節列表。
        這是解析器最核心的方法，決定每個節點的文字範圍和 metadata。

        【完整章節（complete_section）的產生邏輯】
        每遇到新的主條款時，將前一個主條款及其所有子條款的文字
        合併為一個 complete_section 節點，section_number 與主條款相同。
        這讓查詢引擎在找到某個子條款時，可以取回整個章節的完整文字。

        【章節編號（section_number）分配】
        只有主條款（is_main_clause=True）才遞增章節編號。
        子條款的 section_number 繼承當前所屬主條款的編號。
        Header 節點的 section_number 固定為 0。

        Parameters
        ----------
        text              : 合約完整文字
        potential_clauses : _find_all_potential_clauses 回傳的條款列表
        hierarchy_info    : _analyze_dynamic_hierarchy_structure 的回傳值
        doc_metadata      : 原始 Document 的 metadata 字典

        Returns
        -------
        List[Dict]：章節字典列表，每個字典代表一個節點的所有資訊
        """
        sections      = []
        lines         = text.split('\n')
        primary_level = hierarchy_info['primary_level']

        print(f"建立階層結構，主層級設定為: {primary_level}")
        print(f"總共找到 {len(potential_clauses)} 個潛在條款")

        # ── 處理 Header（第一個主條款之前的文字）──────────────────
        first_clause_idx = self._find_first_main_clause(potential_clauses)
        if potential_clauses and first_clause_idx < len(potential_clauses):
            first_clause_line = potential_clauses[first_clause_idx]['line_idx']
            if first_clause_line > 0:
                # 取第一個條款之前的所有行作為 Header 文字
                header_text = '\n'.join(lines[:first_clause_line]).strip()
                if header_text:
                    # 嘗試從前 5 行提取合約標題（含關鍵字的短字串）
                    contract_title = self._extract_contract_title(lines[:5])
                    sections.append({
                        'text':            header_text,
                        'number':          'Header',
                        'title':           f'合約標題與前言 - {contract_title or "合約"}',
                        'type':            'header',
                        'level':           -1,
                        'clause_id':       'Header_0',
                        'section_number':  0,
                        'is_main_clause':  False,
                        'has_sub_clauses': False,
                        'hierarchy_path':  'Header',
                    })

        # 從第一個主條款開始處理
        main_clauses = (potential_clauses[first_clause_idx:]
                        if first_clause_idx < len(potential_clauses) else [])

        # ── 預先分配主條款章節編號（line_idx → section_number）──
        # 讓後續處理子條款時可以快速查找所屬章節編號
        main_clause_sections: Dict[int, int] = {}
        counter = 1
        for clause in main_clauses:
            if clause['pattern_idx'] == primary_level:
                main_clause_sections[clause['line_idx']] = counter
                counter += 1

        print(f"識別到 {len(main_clause_sections)} 個主條款章節")

        # ── 逐一處理所有條款 ──────────────────────────────────────
        current_main_clause           = None   # 目前正在處理的主條款
        current_main_section_content  = []     # 收集主條款及其子條款的所有行（用於 complete_section）
        current_section_number        = 1      # 目前的章節編號

        for i, clause in enumerate(main_clauses):
            is_main_clause = (clause['pattern_idx'] == primary_level)

            if is_main_clause:
                current_section_number = main_clause_sections[clause['line_idx']]
                print(f"處理主條款: {clause['number']}, 分配章節編號: {current_section_number}")
            else:
                print(f"處理子條款: {clause['number']}, 歸屬章節編號: {current_section_number}")

            # 遇到新主條款 → 先將前一個主條款的完整章節（含子條款）存入 sections
            if is_main_clause and current_main_clause is not None:
                prev_sn = main_clause_sections[current_main_clause['line_idx']]
                sections.append(self._create_complete_section(
                    current_main_clause, current_main_section_content, lines, prev_sn
                ))
                print(f"創建完整章節: Section_{prev_sn}")

            if is_main_clause:
                # 重置主條款狀態
                current_main_clause          = clause
                current_main_section_content = []

            # ── 計算本條款的文字範圍 ──────────────────────────────
            # 從本條款起始行到下一個條款起始行（不含）
            start_line   = clause['line_idx']
            end_line     = (main_clauses[i + 1]['line_idx']
                            if i + 1 < len(main_clauses) else len(lines))
            clause_lines = lines[start_line:end_line]
            clause_text  = '\n'.join(clause_lines).strip()

            # 若為主條款，同時預先收集其所有子條款的行（用於 complete_section）
            if is_main_clause:
                current_main_section_content = clause_lines.copy()
                for j in range(i + 1, len(main_clauses)):
                    nc = main_clauses[j]
                    if nc['pattern_idx'] == primary_level:
                        # 遇到下一個主條款 → 停止收集
                        break
                    ns = nc['line_idx']
                    ne = (main_clauses[j + 1]['line_idx']
                          if j + 1 < len(main_clauses) else len(lines))
                    current_main_section_content.extend(lines[ns:ne])

            if clause_text:
                title_info = self._extract_smart_title(clause, clause_lines)
                # 計算 clause_id 時只計算非 complete_section 的節點數量
                clause_id  = f"Clause_{len([s for s in sections if s.get('type') != 'complete_section']) + 1}"
                sections.append({
                    'text':               clause_text,
                    'number':             clause['number'],
                    'title':              title_info['title'],
                    'type':               'main_clause' if is_main_clause else 'sub_clause',
                    'level':              clause['pattern_idx'],
                    'clause_id':          clause_id,
                    # 子條款指向所屬主條款的 Section_N ID
                    'parent_clause_id':   (f"Section_{current_section_number}"
                                           if not is_main_clause else ''),
                    'parent_clause_title': (current_main_clause['number']
                                            if (not is_main_clause and current_main_clause)
                                            else ''),
                    'section_number':     current_section_number,
                    'is_main_clause':     is_main_clause,
                    'has_sub_clauses':    False,  # 後續可擴充：偵測是否有子條款
                    'hierarchy_path':     self._build_hierarchy_path(
                        clause, current_main_clause if not is_main_clause else None
                    ),
                })
                print(f"創建條款節點: {clause_id}, 章節: {current_section_number}")

        # ── 處理最後一個主條款的完整章節 ──────────────────────────
        if current_main_clause is not None:
            final_sn = main_clause_sections[current_main_clause['line_idx']]
            sections.append(self._create_complete_section(
                current_main_clause, current_main_section_content, lines, final_sn
            ))
            print(f"創建最終完整章節: Section_{final_sn}")

        return sections

    # ── 輔助方法 ─────────────────────────────────────────────────

    def _create_complete_section(self, main_clause: Dict, section_lines: List[str],
                                  all_lines: List[str], section_num: int) -> Dict:
        """
        將主條款及其所有子條款合併，建立 complete_section 節點字典。
        complete_section 是給查詢引擎用的「完整章節」，
        包含主條款本身和所有隸屬的子條款，保留原始縮排和格式。

        【為什麼需要 complete_section？】
        向量搜尋可能只找到「第一條 一、」這個子節點（片段），
        但回答問題時需要整個「第一條」的完整內容（包含一、二、三 等）。
        complete_section 正是為了讓查詢引擎能取回完整的章節文字。

        Parameters
        ----------
        main_clause   : 主條款的條款字典
        section_lines : 主條款及其所有子條款的文字行列表
        all_lines     : 整份合約的所有行（目前未使用，保留供未來擴充）
        section_num   : 主條款的章節編號

        Returns
        -------
        Dict：complete_section 節點字典
        """
        complete_text = '\n'.join(section_lines).strip()
        title_info    = self._extract_smart_title(
            main_clause, [section_lines[0]] if section_lines else []
        )
        return {
            'text':            complete_text,
            'number':          main_clause['number'],
            'title':           f"完整章節: {title_info['title']}",
            'type':            'complete_section',
            'level':           main_clause['pattern_idx'],
            'clause_id':       f"Section_{section_num}",
            'section_number':  section_num,
            'is_main_clause':  True,
            'has_sub_clauses': True,
            'hierarchy_path':  f"Section_{section_num}_{main_clause['number']}",
        }

    def _build_hierarchy_path(self, clause: Dict, parent_clause: Optional[Dict]) -> str:
        """
        建立條款的階層路徑字串，格式為「父條款編號 > 本條款編號」。
        主條款沒有父節點，直接回傳本條款編號。

        範例：
          主條款「第一條」→ "第一條"
          子條款「一、」（隸屬第一條）→ "第一條 > 一、"

        Parameters
        ----------
        clause        : 本條款的條款字典
        parent_clause : 父條款的條款字典（主條款時為 None）

        Returns
        -------
        str：層級路徑字串，顯示在 UI 右側的「相關條款」面板
        """
        if parent_clause:
            return f"{parent_clause['number']} > {clause['number']}"
        return clause['number']

    def _extract_contract_title(self, initial_lines: List[str]) -> str:
        """
        從合約前幾行（通常是前 5 行）提取合約標題。
        標題判斷條件：含有合約相關關鍵字，且字數在 2~50 之間。

        去除 Markdown 標題前綴（「## 」等），取純文字。
        找不到時回傳空字串（呼叫端會顯示預設值「合約」）。

        Parameters
        ----------
        initial_lines : 合約文字的前幾行

        Returns
        -------
        str：提取到的合約標題，或空字串
        """
        keywords = ['合約', '契約', '協議', '同意書', '承攬', '委託', '工程',
                    '採購', '服務', '租賃', '買賣', '代理', '合作']
        for line in initial_lines:
            # 去除 Markdown 標題符號（#、##、### 等）
            clean = re.sub(r'^#+\s*', '', line.strip())
            if any(k in clean for k in keywords) and 2 <= len(clean) <= 50:
                return clean
        return ""

    def _find_all_potential_clauses(self, text: str) -> List[Dict]:
        """
        掃描合約全文，找出所有符合條款編號格式的行，並回傳其資訊列表。
        依字元位置（char_pos）升序排序，確保後續處理的順序正確。

        【每行只匹配一個 pattern】
        pattern 按優先順序（0 到 6）掃描，第一個匹配的就 break。
        確保同一行不會被多個 pattern 重複計算（如「第一條」只算 pattern 0）。

        【char_pos 的用途】
        記錄每個條款在全文中的絕對字元位置，
        用於最終排序確保條款順序與原文一致（特別是跨頁的情境）。

        Parameters
        ----------
        text : 合約完整文字字串

        Returns
        -------
        List[Dict]：每個條款的資訊字典，包含：
          line_idx       : 所在行的 0-based 索引
          char_pos       : 在全文中的字元起始位置
          number         : 條款編號（去除 # 前綴後的文字，如 "第一條"）
          full_match     : 正則匹配到的完整字串
          remaining_text : 條款編號後的剩餘文字（通常是條款標題）
          full_line      : 整行文字
          has_colon      : 剩餘文字中是否有冒號（用於標題提取）
          pattern_idx    : 使用的 pattern 索引（0~6）
          priority       : pattern_idx + 1（數字越小優先級越高）
        """
        potential_clauses = []
        lines = text.split('\n')
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue  # 跳過空行
            for pattern_idx, pattern in enumerate(self.clause_number_patterns):
                match = pattern.match(line_stripped)
                if match:
                    # 累計字元位置：加上前面所有行的長度（+1 為換行符）
                    char_pos      = sum(len(lines[i]) + 1 for i in range(line_idx))
                    full_match    = match.group(0)
                    # 去除可能的 Markdown # 前綴
                    clause_number = re.sub(r'^#\s*', '', full_match).strip()
                    remaining_text = line_stripped[len(full_match):].strip()
                    potential_clauses.append({
                        'line_idx':       line_idx,
                        'char_pos':       char_pos,
                        'number':         clause_number,
                        'full_match':     full_match,
                        'remaining_text': remaining_text,
                        'full_line':      line_stripped,
                        'has_colon':      ':' in remaining_text or '：' in remaining_text,
                        'pattern_idx':    pattern_idx,
                        'priority':       pattern_idx + 1,  # 數字越小優先級越高
                    })
                    break  # 每行只匹配一個 pattern
        return sorted(potential_clauses, key=lambda x: x['char_pos'])

    def _get_pattern_priority(self, pattern_idx: int) -> int:
        """
        取得 pattern_idx 對應的優先級數字（pattern_idx + 1）。
        數字越小優先級越高（pattern 0 的優先級為 1，最高）。
        目前此方法未被其他方法呼叫，保留供外部擴充使用。
        """
        return pattern_idx + 1

    def _find_first_main_clause(self, potential_clauses: List[Dict]) -> int:
        """
        在潛在條款列表中，找出第一個「真正的主條款」的索引。

        【為什麼不直接用列表第 0 個？】
        合約開頭可能有流水號（如文件版本號）被誤判為條款，
        需要找到真正開始的第一個主要條款。

        【判斷優先順序】
        1. priority <= 2（pattern_idx 0 或 1）：最高優先級格式，直接採用
        2. _looks_like_main_clause_start：含有合約關鍵字或是常見主條款格式

        Parameters
        ----------
        potential_clauses : _find_all_potential_clauses 回傳的條款列表

        Returns
        -------
        int：第一個主條款在 potential_clauses 中的索引（找不到時回傳 0）
        """
        if not potential_clauses:
            return 0
        for i, clause in enumerate(potential_clauses):
            if clause['priority'] <= 2:
                # pattern 0（第X條）或 1（羅馬數字）直接認定為主條款
                return i
            if self._looks_like_main_clause_start(clause):
                return i
        return 0  # 找不到明確的主條款，從頭開始

    def _looks_like_main_clause_start(self, clause: Dict) -> bool:
        """
        依據條款內容，判斷是否為合約主條款的起始點。

        【判斷邏輯】
        1. 條款文字包含常見合約語意關鍵字（如「工程」「付款」「驗收」等）
        2. 或條款格式符合「中文數字 + 頓號」（如「一、工程名稱」）
        3. 或條款格式符合「第X條/款」的中文格式

        Parameters
        ----------
        clause : 條款字典（來自 _find_all_potential_clauses）

        Returns
        -------
        bool：True = 疑似主條款起始；False = 疑似非主要條款
        """
        # 合約主條款常見語意關鍵字
        keywords = ['工程', '價款', '付款', '施工', '期間', '範圍', '名稱',
                    '責任', '義務', '保證', '驗收', '交付', '完工',
                    '合約', '契約', '條件', '規定', '約定', '詳細']
        if any(k in clause['full_line'] for k in keywords):
            return True
        n = clause['number']
        # 中文數字加頓號格式（一、二、）
        if re.match(r'^[一二三四五六七八九十]+[、．.]', n):
            return True
        # 第X條/款格式
        if re.match(r'^第\s*[一二三四五六七八九十]+\s*[條款]', n):
            return True
        return False

    def _extract_smart_title(self, clause: Dict, clause_lines: List[str]) -> Dict:
        """
        從條款的第一行提取條款標題。

        【標題提取邏輯】
        情況 1 - 有冒號（:或：）：
          取冒號前的文字作為標題
          例：「第一條：工程名稱及範圍」→ 標題 = "工程名稱及範圍"（不含冒號前的條款編號）
          ⚠️ colon_pos 取 max(find(':'), find('：')) 是為了處理全半形冒號並存的情況

        情況 2 - 無冒號：
          取條款編號後的剩餘文字作為標題
          例：「第二條 付款方式」→ 標題 = "付款方式"

        情況 3 - 無剩餘文字：
          直接用條款編號作為標題（如「三、」）

        Parameters
        ----------
        clause       : 條款字典（需含 full_match、remaining_text、number 欄位）
        clause_lines : 條款文字的行列表（只使用第一行）

        Returns
        -------
        Dict：包含 'title' 鍵的字典
        """
        if not clause_lines:
            return {'title': clause['number']}
        first_line = clause_lines[0].strip()
        full_match = clause['full_match']
        # 去除行首的條款編號部分，取後面的文字
        remaining  = first_line[len(full_match):].strip()
        if ':' in remaining or '：' in remaining:
            # 找到冒號：取冒號前的文字作為標題
            colon_pos = max(remaining.find(':'), remaining.find('：'))
            title     = remaining[:colon_pos].strip() if colon_pos >= 0 else remaining
        else:
            title = remaining if remaining else clause['number']
        return {'title': title if title else clause['number']}

    def debug_potential_clauses(self, text: str) -> List[Dict]:
        """
        除錯用公開方法：輸出所有偵測到的潛在條款的詳細資訊。
        在合約解析結果不符預期時，可呼叫此方法診斷：
          - 哪些行被偵測為條款
          - 各行使用的 pattern 索引和優先級
          - 確認主條款和子條款的分界點

        使用方式（在 Python 互動模式或 Jupyter 中）：
          parser = HierarchicalContractNodeParser()
          clauses = parser.debug_potential_clauses(合約文字)

        Parameters
        ----------
        text : 合約完整文字字串

        Returns
        -------
        List[Dict]：_find_all_potential_clauses 回傳的完整條款列表
        """
        potential_clauses = self._find_all_potential_clauses(text)
        lines = text.split('\n')
        print(f"找到 {len(potential_clauses)} 個潛在條款:")
        for i, clause in enumerate(potential_clauses):
            line_content = (lines[clause['line_idx']].strip()
                            if clause['line_idx'] < len(lines) else "")
            print(f"  {i}: 行{clause['line_idx']} - {clause['number']} "
                  f"(優先級: {clause['priority']}, pattern: {clause['pattern_idx']})")
            print(f"      內容: {line_content}")
        return potential_clauses


# ==========================================
# HierarchicalQueryEngine
# ==========================================

class HierarchicalQueryEngine:
    """
    階層式 RAG 查詢引擎。
    分離向量檢索與 LLM 生成，並在向量檢索結果的基礎上
    取回完整章節，提供更完整的問答 context。

    【與一般 LlamaIndex QueryEngine 的差異】
    一般 QueryEngine：向量搜尋 → 取片段 → LLM 生成
    本引擎：向量搜尋 → 取片段 → 查 section_index 取完整章節 → LLM 生成

    完整章節能讓 LLM 看到條款的完整脈絡，不只是搜尋片段，
    降低因 context 不完整造成的誤答機率。

    【section_index 結構】
    在 __init__ 建立時，掃描所有 complete_section 類型的節點，
    建立 {section_number: 章節詳細資訊} 的字典（_build_section_index）。
    查詢時直接用 section_number 查表，O(1) 取得完整章節文字。

    Attributes
    ----------
    index          : VectorStoreIndex 物件
    nodes          : 所有 TextNode 物件列表
    base_retriever : 向量相似度檢索器（similarity_top_k=3）
    _section_index : {section_number: 章節資訊} 的快取字典
    """

    def __init__(self, index: VectorStoreIndex, nodes: List[TextNode]):
        """
        初始化查詢引擎。

        similarity_top_k=3：每次向量搜尋回傳最相似的 3 個節點。
          調高可獲得更多 context，但 prompt 更長，LLM 推理時間增加。
          調低則速度快但可能遺漏相關條款。
          如需調整，在 app.py 中不能直接修改；需在此處修改或透過參數傳入。

        Parameters
        ----------
        index : 已建立的 VectorStoreIndex 物件
        nodes : 對應的 TextNode 列表（含 complete_section 類型）
        """
        self.index          = index
        self.nodes          = nodes
        self.base_retriever = index.as_retriever(similarity_top_k=3)
        self._section_index = self._build_section_index()

    def _build_section_index(self) -> Dict[int, Dict]:
        """
        掃描所有節點，建立 section_number → 完整章節資訊的字典。
        只收錄 clause_type == 'complete_section' 的節點。
        在 __init__ 時建立一次，後續查詢時 O(1) 查表，不需重複掃描節點。

        section_index 的每個 value 包含：
          contract_id    : 合約識別碼
          section_title  : 章節標題（來自 clause_title metadata）
          hierarchy_path : 章節的層級路徑字串
          full_content   : 完整章節文字（主條款 + 所有子條款）
          content_length : 文字字元數（供 debug 確認章節大小）
          section_number : 章節序號（與 key 相同）

        Returns
        -------
        Dict[int, Dict]：{section_number: 章節資訊} 字典
        """
        section_index = {}
        for node in self.nodes:
            md = node.metadata
            if md.get('clause_type') == 'complete_section':
                sn = md.get('section_number', 0)
                if sn > 0:
                    section_index[sn] = {
                        'contract_id':    md.get('contract_id', 'N/A'),
                        'section_title':  md.get('clause_title', 'N/A'),
                        'hierarchy_path': md.get('hierarchy_path', ''),
                        'full_content':   node.text,
                        'content_length': len(node.text),
                        'section_number': sn,
                    }
        print(f"✅ 建立章節索引緩存：{len(section_index)} 個章節")
        return section_index

    def query_with_complete_sections(self, query: str,
                                      include_complete_sections: bool = True) -> Dict:
        """
        執行完整的 RAG 問答流程，包含四個步驟。
        對應 UI 的「🔄 呼叫 AI 生成」操作。

        【步驟說明】
        Step 1 - 向量檢索：
          base_retriever.retrieve(query) 找最相似的 3 個節點
          回傳 NodeWithScore 列表，包含節點內容和相似度分數

        Step 2 - Metadata 解析：
          從每個檢索節點的 metadata 提取條款資訊（標題、類型、分數等），
          組成 related_clauses 列表供 UI 右側面板顯示
          同時收集 section_number，準備查詢完整章節

        Step 3 - 組合 Context：
          include_complete_sections=True（預設）：
            從 section_index 取出完整章節文字（主條款 + 所有子條款）
            Fallback：若 section_index 沒有對應章節，改用檢索片段
          include_complete_sections=False：
            直接使用檢索片段作為 context（context 較短，速度較快）

        Step 4 - LLM 生成（Ollama streaming）：
          使用目前 Settings.llm 的模型名稱呼叫 Ollama chat API
          支援 streaming：每個 chunk 前先檢查截斷旗標（_check_abort）
          Fallback：若 ollama streaming 失敗，退回 LlamaIndex Settings.llm.complete

        【截斷機制詳細說明】
          ui_helpers.request_abort() → 設定 _abort_flag = True
          此函式每個 chunk 前呼叫 _check_abort()
          若 True → stream.close()（關閉 HTTP 連線，Ollama server 停止生成）
          → results['answer'] = "⛔ 已截斷生成（使用者手動中止）"
          ui_helpers.rag_chat_response 偵測到截斷後，pending_qa = None（不儲存）

        Parameters
        ----------
        query                    : 使用者的完整問題字串（可能包含語言指示）
        include_complete_sections: True = 使用完整章節 context（預設）
                                   False = 只用向量搜尋片段

        Returns
        -------
        Dict：
          answer           : LLM 生成的答案字串（截斷時以 "⛔" 開頭）
          related_clauses  : 相關條款列表（供 UI 右側「🔍 檢索原始條款」顯示）
          complete_sections: 完整章節列表（供 UI 右側「📋 完整章節」顯示）
          hierarchy_info   : 查詢階層分析摘要（條款數量、層級複雜度）
        """
        results = {
            'answer': "",
            'related_clauses': [],
            'complete_sections': [],
            'hierarchy_info': {}
        }

        st = time.time()
        print(f"⏱️ [Start] 開始查詢: {query}")

        # ── 步驟 1：向量檢索 ─────────────────────────────────────
        retrieved_nodes: List[NodeWithScore] = self.base_retriever.retrieve(query)
        t1 = time.time()
        print(f"⏱️ [Step 1] Vector Retrieval 耗時: {t1 - st:.4f} 秒 | "
              f"找到 {len(retrieved_nodes)} 個片段")

        # ── 步驟 2：解析 Metadata，組成 related_clauses ──────────
        section_ids_to_fetch = set()  # 需要取完整章節的 section_number 集合
        for source_node in retrieved_nodes:
            md  = source_node.node.metadata
            sn  = md.get('section_number', 0)
            results['related_clauses'].append({
                'contract_id':    md.get('contract_id', 'N/A'),
                'clause_title':   md.get('clause_title', 'N/A'),
                'clause_type':    md.get('clause_type', 'N/A'),
                'section_number': sn,
                'hierarchy_path': md.get('hierarchy_path', ''),
                'score':          source_node.score,
                # text_preview：顯示前 200 字，讓使用者確認搜尋結果的相關性
                'text_preview':   source_node.node.text[:200] + "...",
            })
            if include_complete_sections and sn > 0:
                section_ids_to_fetch.add(sn)

        t2 = time.time()
        print(f"⏱️ [Step 2] Metadata 解析耗時: {t2 - t1:.4f} 秒")

        # ── 步驟 3：組合 Context ──────────────────────────────────
        context_text = ""
        if include_complete_sections:
            if section_ids_to_fetch:
                # 從 section_index 取出完整章節，依章節編號排序（保持原文順序）
                complete_sections_data = [
                    self._section_index[sid]
                    for sid in section_ids_to_fetch
                    if sid in self._section_index
                ]
                results['complete_sections'] = sorted(
                    complete_sections_data, key=lambda x: x['section_number']
                )
                # 組合 context：每個章節用分隔標題標記，方便 LLM 識別邊界
                context_text = "\n\n".join(
                    f"---章節 {s['section_number']} ({s['section_title']})---\n"
                    f"{s['full_content']}"
                    for s in results['complete_sections']
                )
                print(f"ℹ️ 使用模式: [完整章節] - 包含 {len(results['complete_sections'])} 個章節全文")
            else:
                # section_number = 0 的節點（如 header）沒有完整章節，降級使用片段
                context_text = "\n\n".join([n.node.text for n in retrieved_nodes])
                print("⚠️ 警告: 找不到完整章節，降級使用檢索片段")
        else:
            # 使用者關閉完整章節功能，直接用搜尋片段（節省 context 長度）
            context_text = "\n\n".join(
                f"---片段---\n{n.node.text}" for n in retrieved_nodes
            )
            print(f"ℹ️ 使用模式: [檢索片段] - 使用 {len(retrieved_nodes)} 個片段")

        t3 = time.time()

        # ── 步驟 4：LLM 生成（Ollama streaming + 截斷支援）────────
        if context_text:
            # 系統提示 + 參考資料 + 使用者問題（query 已包含語言指示）
            prompt = (
                f"你是一個專業的合約助手。請根據以下提供的參考資料回答問題。\n"
                f"如果資料中沒有答案，請直接說明。\n\n"
                f"=== 參考資料 ===\n{context_text}\n\n"
                f"=== 使用者問題 ===\n{query}"
            )

            # 從 Settings.llm 取得目前選用的模型名稱，確保與 UI 一致
            try:
                model_name = Settings.llm.model
            except AttributeError:
                # Settings.llm 未設定或不支援 .model 屬性，使用預設值
                model_name = "qwen3:4b"

            # ── 嘗試 Ollama streaming ────────────────────────────
            try:
                import ollama as _ollama
                # ⚠️ 若要更改 ui_helpers.py 的檔名，以下 import 需同步修改
                from ui_helpers import _check_abort

                stream = _ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,  # 啟用 streaming，支援逐字輸出和截斷
                )

                answer_parts = []
                aborted      = False

                for chunk in stream:
                    # 每個 chunk 前先檢查截斷旗標（request_abort 設定）
                    if _check_abort():
                        stream.close()  # 關閉 HTTP 連線 → Ollama server 立即停止生成
                        aborted = True
                        break
                    # chunk 格式：{"message": {"content": "..."}, ...}
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        answer_parts.append(content)

                if aborted:
                    results['answer'] = "⛔ 已截斷生成（使用者手動中止）"
                else:
                    results['answer'] = "".join(answer_parts)

            except Exception as e:
                # Ollama streaming 失敗（如舊版 ollama 套件不支援 stream=True）
                # 退回 LlamaIndex 標準介面（不支援截斷，但仍可完成問答）
                print(f"⚠️ ollama streaming 失敗，退回 LlamaIndex：{e}")
                response          = Settings.llm.complete(prompt)
                results['answer'] = response.text
        else:
            # context 為空（所有節點都沒有文字）
            results['answer'] = "未能檢索到相關資料。"

        # 附加查詢階層分析摘要
        results['hierarchy_info'] = self._analyze_query_hierarchy(results['related_clauses'])

        t4 = time.time()
        print(f"⏱️ [Step 4] LLM 生成耗時: {t4 - t3:.4f} 秒")
        print(f"🏁 總共耗時: {t4 - st:.4f} 秒")

        return results

    def _analyze_query_hierarchy(self, related_clauses: List[Dict]) -> Dict:
        """
        分析向量搜尋結果中涉及的條款層級分布，
        供 UI 顯示查詢的複雜度摘要（目前僅作 debug 用途，未直接顯示在前端）。

        統計項目：
          levels_involved     : 涉及的條款層級集合（如 {0, 2}）
          sections_count      : 涉及的不同章節數量
          main_clauses_count  : 主條款片段的數量
          sub_clauses_count   : 子條款片段的數量
          hierarchy_complexity: 'simple'（≤2 種層級）或 'complex'（>2 種層級）

        Parameters
        ----------
        related_clauses : query_with_complete_sections 組建的相關條款列表

        Returns
        -------
        Dict：階層分析摘要字典
        """
        if not related_clauses:
            return {}
        levels_found      = set()
        sections_involved = set()
        main_clauses      = []
        sub_clauses       = []
        for clause in related_clauses:
            levels_found.add(clause.get('level', -1))
            sn = clause.get('section_number', 0)
            if sn > 0:
                sections_involved.add(sn)
            ct = clause.get('clause_type', '')
            if ct == 'main_clause':
                main_clauses.append(clause)
            elif ct == 'sub_clause':
                sub_clauses.append(clause)
        return {
            'levels_involved':      sorted(list(levels_found)),
            'sections_count':       len(sections_involved),
            'main_clauses_count':   len(main_clauses),
            'sub_clauses_count':    len(sub_clauses),
            # 超過 2 種層級視為複雜合約（可能有工程合約的多層子條款）
            'hierarchy_complexity': 'simple' if len(levels_found) <= 2 else 'complex',
        }
