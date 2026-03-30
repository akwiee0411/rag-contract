#!/usr/bin/env python
# coding: utf-8


import time
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode, NodeWithScore
import time
import json
import os
import re
from typing import List, Pattern, Dict, Tuple, Any, Optional
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import TextNode, Document
from llama_index.core import VectorStoreIndex
from pydantic import Field

class HierarchicalContractNodeParser(NodeParser):
    """
    階層式中文合約節點解析器 - 支援動態上級/副標籤分類
    
    此解析器專門處理中文合約文件，能夠：
    1. 自動識別條款編號模式（如：第一條、一、、(一)、1、、(1)、1.1等）
    2. 動態確定主層級結構（以第一個出現的條款格式為準）
    3. 建立完整的階層關係
    4. 簡化的標題處理（第一條前的內容統一作為header）
    5. 支援多種條款格式
    """
    
    # 條款編號模式列表 - 按優先級排序，第一個匹配的作為上級標籤
    clause_number_patterns: List[Pattern[str]] = Field(
        default_factory=lambda: [
            # 層級0: 第一條、第二條（最高層級）
            re.compile(r"^(?:#\s*)?第\s*[一二三四五六七八九十百千萬\d]+\s*[條款項章節]", re.MULTILINE),  
            
            re.compile(r"^(?:#\s*)?[IVXLCDMivxlcdm]+\s*[、．.]", re.MULTILINE),
            
            # 層級1: 一、、二、（第二層級）
            re.compile(r"^(?:#\s*)?[一二三四五六七八九十百千萬]+\s*[、．.]", re.MULTILINE),         
            
            # 層級2: (一)、(二)（第三層級）
            re.compile(r"^(?:#\s*)?[\(（]\s*[一二三四五六七八九十]+\s*[\)）]", re.MULTILINE),    
            
            # 層級3: 1、、2、（第四層級）
            re.compile(r"^(?:#\s*)?\d+\s*(?:[、．]|\.(?!\d))", re.MULTILINE),                                
            
            # 層級4: (1)、(2)（第五層級）
            re.compile(r"^(?:#\s*)?\(\d+\)", re.MULTILINE),                 
            
            # 層級5: 1.1、1.2（第六層級）
            re.compile(r"^(?:#\s*)?\d+\.\d+", re.MULTILINE),                                   
        ]
    )
    
    class Config:
        # 允許任意類型（因為使用了正則表達式Pattern類型）
        arbitrary_types_allowed = True
    
    def _parse_nodes(self, documents: List[Document], show_progress: bool = False, **kwargs) -> List[TextNode]:
        """
        實現抽象方法 _parse_nodes - LlamaIndex要求的核心方法
        
        Args:
            documents: 要解析的文檔列表
            show_progress: 是否顯示進度條
            **kwargs: 其他參數
            
        Returns:
            解析後的TextNode列表
        """
        nodes: List[TextNode] = []
        
        # 逐一處理每個文檔
        for doc in documents:
            doc_nodes = self._parse_document(doc)
            nodes.extend(doc_nodes)
            
        return nodes
    
    def get_nodes_from_documents(self, documents: List[Document], show_progress: bool = False, **kwargs) -> List[TextNode]:
        """
        公開接口：從文檔列表中提取節點
        
        Args:
            documents: 文檔列表
            show_progress: 是否顯示進度
            
        Returns:
            節點列表
        """
        return self._parse_nodes(documents, show_progress=show_progress, **kwargs)
    
    def _parse_document(self, doc: Document) -> List[TextNode]:
        """
        智能解析單個文檔，建立階層結構
        
        主要步驟：
        1. 找到所有可能的條款起始位置
        2. 動態分析階層結構
        3. 建立階層式節點結構
        4. 創建最終節點
        
        Args:
            doc: 要解析的單個文檔
            
        Returns:
            該文檔解析後的節點列表
        """
        nodes: List[TextNode] = []
        text = doc.text
        
        # 步驟1: 找到所有可能的條款起始位置
        potential_clauses = self._find_all_potential_clauses(text)
        
        # 如果沒有找到任何條款，將整個文檔視為一個節點
        if not potential_clauses:
            node = TextNode(
                text=text,
                metadata={
                    "document_title": doc.metadata.get("file_name", "N/A"),
                    "clause_id": "Full_Document",
                    "source": doc.metadata.get("file_path", "N/A"),
                    "clause_type": "full_document",
                    "clause_level": -1,
                    "is_main_clause": False,
                    "has_sub_clauses": False
                }
            )
            nodes.append(node)
            return nodes
        
        # 步驟2: 動態分析階層結構 - 關鍵創新點
        hierarchy_info = self._analyze_dynamic_hierarchy_structure(potential_clauses)
        
        # 步驟3: 建立階層式節點結構
        hierarchical_sections = self._build_hierarchical_sections(
            text, potential_clauses, hierarchy_info, doc.metadata
        )
        
        # 步驟4: 創建最終的TextNode對象
        for section in hierarchical_sections:
            if section['text'].strip():  # 只處理非空內容
                new_node = TextNode(
                    text=section['text'],
                    metadata={
                        # 基本文檔信息
                        "document_title": doc.metadata.get("file_name", "N/A"),
                        "clause_id": section['clause_id'],
                        "source": doc.metadata.get("file_path", "N/A"),
                        
                        # 條款結構信息
                        "clause_number_raw": section['number'],
                        "clause_title": section['title'],
                        "clause_type": section['type'],
                        "clause_level": section['level'],
                        
                        # 階層關係信息
                        "parent_clause_id": section.get('parent_clause_id', ''),
                        "parent_clause_title": section.get('parent_clause_title', ''),
                        "section_number": section.get('section_number', 0),
                        "hierarchy_path": section.get('hierarchy_path', ''),
                        
                        # 條款特性標記
                        "is_main_clause": section.get('is_main_clause', False),
                        "has_sub_clauses": section.get('has_sub_clauses', False),
                        
                        # 合約特定元數據
                        "contract_id": doc.metadata.get("contract_id", "N/A"),
                        "orig_path": doc.metadata.get("orig_path", "N/A"),
                        "num_pages": doc.metadata.get("num_pages", "N/A"),
                    }
                )
                nodes.append(new_node)
        
        return nodes
    
    def _analyze_dynamic_hierarchy_structure(self, potential_clauses: List[Dict]) -> Dict:
        """
        動態分析合約的階層結構 - 核心創新方法
        
        關鍵創新：不預設主層級，而是根據文檔中第一個出現的條款格式
        來動態確定主層級，這樣可以適應不同的合約格式慣例。
        
        Args:
            potential_clauses: 潛在條款列表
            
        Returns:
            階層結構分析結果
        """
        if not potential_clauses:
            return {'primary_level': -1, 'levels_used': [], 'structure_type': 'none'}
        
        # 找到第一個主要條款後的所有條款模式
        first_clause_idx = self._find_first_main_clause(potential_clauses)
        main_clauses = potential_clauses[first_clause_idx:] if first_clause_idx < len(potential_clauses) else potential_clauses
        
        if not main_clauses:
            return {'primary_level': -1, 'levels_used': [], 'structure_type': 'none'}
        
        # 關鍵創新：使用第一個主要條款的pattern_idx作為主層級
        # 這樣可以自動適應不同合約的編號慣例
        primary_level = main_clauses[0]['pattern_idx']
        print(f"動態識別主層級: 第一個條款模式索引 = {primary_level}")
        
        # 統計各層級的使用情況
        level_usage = {}
        for clause in main_clauses:
            level = clause['pattern_idx']
            if level not in level_usage:
                level_usage[level] = 0
            level_usage[level] += 1
        
        # 分析層級結構類型
        levels_used = sorted(level_usage.keys())
        structure_type = self._determine_structure_type(levels_used, level_usage)
        
        # 輸出詳細分析信息供調試
        print(f"層級使用統計: {level_usage}")
        print(f"確定的主層級: {primary_level} ({self._get_level_name(primary_level)})")
        
        return {
            'primary_level': primary_level,              # 主層級索引
            'levels_used': levels_used,                  # 使用的所有層級
            'level_usage': level_usage,                  # 各層級使用次數
            'structure_type': structure_type,            # 結構類型
            'primary_level_name': self._get_level_name(primary_level)  # 主層級名稱
        }
    
    def _determine_structure_type(self, levels_used: List[int], level_usage: Dict[int, int]) -> str:
        """
        判斷階層結構類型
        
        Args:
            levels_used: 使用的層級列表
            level_usage: 各層級使用統計
            
        Returns:
            結構類型字符串
        """
        if len(levels_used) == 1:
            return 'single_level'      # 單層級結構
        elif len(levels_used) == 2:
            return 'two_level'         # 雙層級結構
        elif len(levels_used) >= 3:
            return 'multi_level'       # 多層級結構
        else:
            return 'complex'           # 複雜結構
    
    def _get_level_name(self, level: int) -> str:
        """
        獲取層級對應的名稱描述
        
        Args:
            level: 層級索引
            
        Returns:
            層級名稱
        """
        level_names = {
            0: "第X條",     # 第一條、第二條
            1: "X、",       # 一、、二、
            2: "(X)",       # (一)、(二)
            3: "X、",       # 1、、2、
            4: "(X)",       # (1)、(2)
            5: "X.X"        # 1.1、1.2
        }
        return level_names.get(level, f"Level_{level}")
    

    def _build_hierarchical_sections(self, text: str, potential_clauses: List[Dict], 
                                   hierarchy_info: Dict, doc_metadata: Dict) -> List[Dict]:
        """
        建立階層式章節結構 - 修復版章節編號邏輯
        """
        sections = []
        lines = text.split('\n')
        primary_level = hierarchy_info['primary_level']

        print(f"建立階層結構，主層級設定為: {primary_level}")
        print(f"總共找到 {len(potential_clauses)} 個潛在條款")

        # 步驟1: 處理header (保持原邏輯)
        first_clause_idx = self._find_first_main_clause(potential_clauses)
        print(f"第一個主條款索引: {first_clause_idx}")

        if potential_clauses and first_clause_idx < len(potential_clauses):
            first_clause_line = potential_clauses[first_clause_idx]['line_idx']
            if first_clause_line > 0:
                header_lines = lines[:first_clause_line]
                header_text = '\n'.join(header_lines).strip()

                if header_text:
                    contract_title = self._extract_contract_title(header_lines[:5])
                    sections.append({
                        'text': header_text,
                        'number': 'Header',
                        'title': f'合約標題與前言 - {contract_title if contract_title else "合約"}',
                        'type': 'header',
                        'level': -1,
                        'clause_id': 'Header_0',
                        'section_number': 0,
                        'is_main_clause': False,
                        'has_sub_clauses': False,
                        'hierarchy_path': 'Header'
                    })

        # 步驟2: 重新設計章節編號邏輯
        main_clauses = potential_clauses[first_clause_idx:] if first_clause_idx < len(potential_clauses) else []
        current_section_number = 1  # 從1開始計數
        current_main_clause = None
        current_main_section_content = []

        # 首先識別所有主條款並分配章節編號
        main_clause_sections = {}  # 存儲主條款對應的章節編號
        temp_section_counter = 1

        for clause in main_clauses:
            if clause['pattern_idx'] == primary_level:  # 是主條款
                main_clause_sections[clause['line_idx']] = temp_section_counter
                temp_section_counter += 1

        print(f"識別到 {len(main_clause_sections)} 個主條款章節")

        # 步驟3: 按順序處理所有條款
        for i, clause in enumerate(main_clauses):
            clause_level = clause['pattern_idx']
            is_main_clause = (clause_level == primary_level)

            # 獲取當前條款的章節編號
            if is_main_clause:
                current_section_number = main_clause_sections[clause['line_idx']]
                print(f"處理主條款: {clause['number']}, 分配章節編號: {current_section_number}")
            else:
                # 子條款使用當前主條款的章節編號
                print(f"處理子條款: {clause['number']}, 歸屬章節編號: {current_section_number}")

            # 如果遇到新的主條款，先處理前一個主條款的完整內容
            if is_main_clause and current_main_clause is not None:
                # 使用前一個主條款的章節編號
                prev_section_number = main_clause_sections[current_main_clause['line_idx']]
                complete_section = self._create_complete_section(
                    current_main_clause, 
                    current_main_section_content, 
                    lines, 
                    prev_section_number  # 修復：使用正確的章節編號
                )
                sections.append(complete_section)
                print(f"創建完整章節: Section_{prev_section_number}")

            # 更新當前主條款信息
            if is_main_clause:
                current_main_clause = clause
                current_main_section_content = []

            # 步驟4: 收集條款內容
            start_line = clause['line_idx']
            if i + 1 < len(main_clauses):
                end_line = main_clauses[i + 1]['line_idx']
            else:
                end_line = len(lines)

            clause_lines = lines[start_line:end_line]
            clause_text = '\n'.join(clause_lines).strip()

            # 如果是主條款，開始收集完整內容
            if is_main_clause:
                current_main_section_content = clause_lines.copy()

                # 預先收集後續的子條款內容
                for j in range(i + 1, len(main_clauses)):
                    next_clause = main_clauses[j]
                    if next_clause['pattern_idx'] == primary_level:  # 遇到下一個主條款停止
                        break

                    # 添加子條款內容到當前主條款的完整內容
                    next_start = next_clause['line_idx']
                    if j + 1 < len(main_clauses):
                        next_end = main_clauses[j + 1]['line_idx']
                    else:
                        next_end = len(lines)
                    current_main_section_content.extend(lines[next_start:next_end])

            # 步驟5: 創建單個條款節點
            if clause_text:
                title_info = self._extract_smart_title(clause, clause_lines)

                # 計算正確的clause_id（基於sections列表長度）
                clause_id = f"Clause_{len([s for s in sections if s.get('type') != 'complete_section']) + 1}"

                single_section = {
                    'text': clause_text,
                    'number': clause['number'],
                    'title': title_info['title'],
                    'type': 'main_clause' if is_main_clause else 'sub_clause',
                    'level': clause_level,
                    'clause_id': clause_id,
                    'parent_clause_id': f"Section_{current_section_number}" if not is_main_clause else '',
                    'parent_clause_title': current_main_clause['number'] if not is_main_clause and current_main_clause else '',
                    'section_number': current_section_number,  # 使用統一的章節編號
                    'is_main_clause': is_main_clause,
                    'has_sub_clauses': False,
                    'hierarchy_path': self._build_hierarchy_path(clause, current_main_clause if not is_main_clause else None)
                }
                sections.append(single_section)
                print(f"創建條款節點: {clause_id}, 章節: {current_section_number}")

        # 步驟6: 處理最後一個主條款的完整內容
        if current_main_clause is not None:
            final_section_number = main_clause_sections[current_main_clause['line_idx']]
            complete_section = self._create_complete_section(
                current_main_clause, 
                current_main_section_content, 
                lines, 
                final_section_number  # 修復：使用正確的最終章節編號
            )
            sections.append(complete_section)
            print(f"創建最終完整章節: Section_{final_section_number}")

        # 驗證章節編號連續性
        section_numbers = [s.get('section_number', 0) for s in sections if s.get('section_number', 0) > 0]
        expected_max = max(main_clause_sections.values()) if main_clause_sections else 0
        print(f"章節編號驗證: 期望最大值={expected_max}, 實際使用={sorted(set(section_numbers))}")

        return sections


    def debug_potential_clauses(self, text: str):
        """
        調試用：輸出所有找到的潛在條款
        """
        potential_clauses = self._find_all_potential_clauses(text)
        lines = text.split('\n')

        print(f"找到 {len(potential_clauses)} 個潛在條款:")
        for i, clause in enumerate(potential_clauses):
            line_content = lines[clause['line_idx']].strip() if clause['line_idx'] < len(lines) else ""
            print(f"  {i}: 行{clause['line_idx']} - {clause['number']} (優先級: {clause['priority']})")
            print(f"      內容: {line_content}")

        return potential_clauses
  
    def _create_complete_section(self, main_clause: Dict, section_lines: List[str], 
                               all_lines: List[str], section_num: int) -> Dict:
        """
        創建完整的主條款章節（包含所有子條款）
        
        這種完整章節對於回答涉及整個主題的問題很有用，
        因為它包含了主條款及其所有相關的子條款內容。
        
        Args:
            main_clause: 主條款信息
            section_lines: 章節所有行的內容
            all_lines: 全文所有行（預留使用）
            section_num: 章節編號
            
        Returns:
            完整章節字典
        """
        complete_text = '\n'.join(section_lines).strip()
        title_info = self._extract_smart_title(main_clause, [section_lines[0]] if section_lines else [])
        
        return {
            'text': complete_text,
            'number': main_clause['number'],
            'title': f"完整章節: {title_info['title']}",
            'type': 'complete_section',           # 標記為完整章節
            'level': main_clause['pattern_idx'],
            'clause_id': f"Section_{section_num}",
            'section_number': section_num,
            'is_main_clause': True,
            'has_sub_clauses': True,              # 完整章節通常包含子條款
            'hierarchy_path': f"Section_{section_num}_{main_clause['number']}"
        }
    
    def _build_hierarchy_path(self, clause: Dict, parent_clause: Optional[Dict]) -> str:
        """
        建立階層路徑字符串，用於顯示條款的層級關係
        
        Args:
            clause: 當前條款
            parent_clause: 上級條款（如果有）
            
        Returns:
            階層路徑字符串
        """
        if parent_clause:
            return f"{parent_clause['number']} > {clause['number']}"
        else:
            return clause['number']
    
    def _extract_contract_title(self, initial_lines: List[str]) -> str:
        """
        提取合約標題
        
        從合約開頭的幾行中識別合約標題，
        通常包含"合約"、"契約"等關鍵詞。
        
        Args:
            initial_lines: 合約開頭的行列表
            
        Returns:
            合約標題字符串
        """
        for line in initial_lines:
            line_stripped = line.strip()
            # 移除可能的 # 前綴
            clean_line = re.sub(r'^#+\s*', '', line_stripped)
            
            # 常見的合約標題關鍵詞
            title_keywords = [
                '合約', '契約', '協議', '同意書', '承攬', '委託', '工程', 
                '採購', '服務', '租賃', '買賣', '代理', '合作'
            ]
            
            # 檢查是否包含標題關鍵詞且長度合理
            if any(keyword in clean_line for keyword in title_keywords):
                if 2 <= len(clean_line) <= 50:  # 合理的標題長度
                    return clean_line
        
        return ""
    
    def _find_all_potential_clauses(self, text: str) -> List[Dict]:
        """
        找到所有潛在的條款起始位置
        
        遍歷全文的每一行，使用預定義的正則表達式模式
        來識別可能的條款編號，並記錄詳細信息。
        
        Args:
            text: 合約全文
            
        Returns:
            潛在條款信息列表
        """
        potential_clauses = []
        lines = text.split('\n')
        
        # 逐行檢查是否匹配條款模式
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:  # 跳過空行
                continue
                
            # 檢查每個條款編號模式
            for pattern_idx, pattern in enumerate(self.clause_number_patterns):
                match = pattern.match(line_stripped)
                if match:
                    # 計算該行在原文中的字符位置
                    char_pos = sum(len(lines[i]) + 1 for i in range(line_idx))
                    
                    # 提取完整匹配和純條款編號
                    full_match = match.group(0)
                    # 移除可能的 # 前綴來獲取純條款編號
                    clause_number = re.sub(r'^#\s*', '', full_match).strip()
                    
                    # 獲取條款編號後的剩餘文本
                    remaining_text = line_stripped[len(full_match):].strip()
                    
                    # 記錄條款信息
                    potential_clauses.append({
                        'line_idx': line_idx,                    # 行索引
                        'char_pos': char_pos,                    # 字符位置
                        'number': clause_number,                 # 純條款編號
                        'full_match': full_match,                # 完整匹配（包含#）
                        'remaining_text': remaining_text,        # 剩餘文本
                        'full_line': line_stripped,              # 完整行內容
                        'has_colon': ':' in remaining_text or '：' in remaining_text,  # 是否有冒號
                        'pattern_idx': pattern_idx,              # 模式索引（對應層級）
                        'priority': self._get_pattern_priority(pattern_idx)  # 優先級
                    })
                    break  # 找到匹配就停止檢查其他模式
        
        # 按字符位置排序返回
        return sorted(potential_clauses, key=lambda x: x['char_pos'])
    
    def _get_pattern_priority(self, pattern_idx: int) -> int:
        """
        獲取模式優先級，數值越小優先級越高
        
        Args:
            pattern_idx: 模式索引
            
        Returns:
            優先級數值
        """
        return pattern_idx + 1  # 直接使用索引作為優先級
    
    def _find_first_main_clause(self, potential_clauses: List[Dict]) -> int:
        """
        找到第一個主要條款的索引
        
        識別哪個條款應該被視為第一個真正的主條款，
        通常跳過序言等前置內容。
        
        Args:
            potential_clauses: 潛在條款列表
            
        Returns:
            第一個主條款的索引
        """
        if not potential_clauses:
            return 0
        
        # 尋找第一個具有高優先級的條款
        for i, clause in enumerate(potential_clauses):
            # 如果是 "一、" 或 "第一條" 等主要條款格式
            if clause['priority'] <= 2:  # 第一條(1) 或 一、(2)
                return i
            
            # 如果條款編號看起來像是主要條款的開始
            if self._looks_like_main_clause_start(clause):
                return i
        
        # 如果沒找到明顯的主條款，返回第一個
        return 0
    
    def _looks_like_main_clause_start(self, clause: Dict) -> bool:
        """
        判斷是否看起來像主條款的開始
        
        通過檢查條款內容和編號格式來判斷
        是否為主要條款的開始。
        
        Args:
            clause: 條款信息字典
            
        Returns:
            是否像主條款開始
        """
        full_line = clause['full_line']
        
        # 檢查是否包含常見的條款開始關鍵詞
        main_clause_keywords = [
            '工程', '價款', '付款', '施工', '期間', '範圍', '名稱', 
            '責任', '義務', '保證', '驗收', '交付', '完工',
            '合約', '契約', '條件', '規定', '約定', '詳細'
        ]
        
        if any(keyword in full_line for keyword in main_clause_keywords):
            return True
        
        # 檢查編號模式（移除#後的格式）
        number = clause['number']
        if re.match(r'^[一二三四五六七八九十]+[、．.]', number):
            return True
        
        if re.match(r'^第\s*[一二三四五六七八九十]+\s*[條款]', number):
            return True
            
        return False
    
    def _extract_smart_title(self, clause: Dict, clause_lines: List[str]) -> Dict:
        """
        智能提取條款標題
        
        從條款的第一行中提取有意義的標題，
        處理各種格式的條款標題。
        
        Args:
            clause: 條款信息字典
            clause_lines: 條款內容行列表
            
        Returns:
            包含標題的字典
        """
        if not clause_lines:
            return {'title': clause['number']}
        
        first_line = clause_lines[0].strip()
        full_match = clause['full_match']  # 包含可能的 # 前綴
        
        # 移除完整匹配部分，獲取剩餘部分
        remaining = first_line[len(full_match):].strip()
        
        # 處理冒號分隔的標題
        if ':' in remaining or '：' in remaining:
            colon_pos = max(remaining.find(':'), remaining.find('：'))
            if colon_pos >= 0:
                title = remaining[:colon_pos].strip()
            else:
                title = remaining
        else:
            title = remaining if remaining else clause['number']
        
        return {'title': title if title else clause['number']}


class HierarchicalQueryEngine:
    """優化版：分離檢索與生成，並增加詳細計時"""
    
    def __init__(self, index: VectorStoreIndex, nodes: List[TextNode]):
        self.index = index
        self.nodes = nodes
        
        # 優化 1: 改用 Retriever，不要在第一步就初始化 QueryEngine
        #這會大幅加快第一階段速度，因為不涉及 LLM
        self.base_retriever = index.as_retriever(similarity_top_k=3)
        
        # 建立章節索引緩存
        self._section_index = self._build_section_index()
        
    def _build_section_index(self) -> Dict[int, Dict]:
        """建立章節號到完整章節的快速查找索引 (保持原邏輯)"""
        section_index = {}
        for node in self.nodes:
            metadata = node.metadata
            # 這裡假設你的 metadata 結構是正確的
            if metadata.get('clause_type') == 'complete_section':
                section_num = metadata.get('section_number', 0)
                if section_num > 0:
                    section_index[section_num] = {
                        'contract_id': metadata.get('contract_id', 'N/A'),
                        'section_title': metadata.get('clause_title', 'N/A'),
                        'hierarchy_path': metadata.get('hierarchy_path', ''),
                        'full_content': node.text, # 完整文字
                        'content_length': len(node.text),
                        'section_number': section_num
                    }
        print(f"✅ 建立章節索引緩存：{len(section_index)} 個章節")
        return section_index

    def _analyze_query_hierarchy(self, clauses: List[Dict]) -> Dict:
        """(補全缺失的方法) 簡單分析層級，避免報錯"""
        if not clauses:
            return {"summary": "No clauses found"}
        paths = [c['hierarchy_path'] for c in clauses]
        return {"detected_paths": list(set(paths))}

    def query_with_complete_sections(self, query: str, include_complete_sections: bool = True) -> Dict:
        """
        優化後的查詢流程：
        1. Vector Retrieve (快)
        2. 抓取完整章節 (快)
        3. 組合 Context
        4. LLM Generate (慢，但只做一次)
        """
        results = {
            'answer': "",
            'related_clauses': [],
            'complete_sections': [],
            'hierarchy_info': {}
        }
        
        # --- 計時開始 ---
        st = time.time()
        print(f"⏱️ [Start] 開始查詢: {query}")

        # 步驟 1: 向量檢索 (Retrieval)
        # 這裡只找相關節點，不問 LLM，速度極快
        retrieved_nodes: List[NodeWithScore] = self.base_retriever.retrieve(query)
        
        t1 = time.time()
        print(f"⏱️ [Step 1] Vector Retrieval 耗時: {t1 - st:.4f} 秒 | 找到 {len(retrieved_nodes)} 個片段")

        # 步驟 2: 處理 Metadata 與 準備章節 ID
        section_ids_to_fetch = set()
        
        for source_node in retrieved_nodes:
            metadata = source_node.node.metadata
            section_num = metadata.get('section_number', 0)
            
            clause_info = {
                'contract_id': metadata.get('contract_id', 'N/A'),
                'clause_title': metadata.get('clause_title', 'N/A'),
                'clause_type': metadata.get('clause_type', 'N/A'),
                'section_number': section_num,
                'hierarchy_path': metadata.get('hierarchy_path', ''),
                'score': source_node.score,
                'text_preview': source_node.node.text[:200] + "..." # 預覽只存片段
            }
            results['related_clauses'].append(clause_info)
            
            # 如果需要完整章節，記錄 ID
            if include_complete_sections and section_num > 0:
                section_ids_to_fetch.add(section_num)

        t2 = time.time()
        print(f"⏱️ [Step 2] Metadata 解析耗時: {t2 - t1:.4f} 秒")

        # 步驟 3: 提取 Context 內容 (決定 LLM 看到什麼)
        context_text = ""
        
        if include_complete_sections:
            # 方案 A: 使用完整章節
            if section_ids_to_fetch:
                # 從緩存抓取完整章節
                complete_sections_data = [
                    self._section_index[sid] 
                    for sid in section_ids_to_fetch 
                    if sid in self._section_index
                ]
                # 排序並存入結果
                results['complete_sections'] = sorted(
                    complete_sections_data, 
                    key=lambda x: x['section_number']
                )
                
                # 組合 Prompt 內容：將所有相關的「完整章節」拼起來
                context_list = [f"---章節 {s['section_number']} ({s['section_title']})---\n{s['full_content']}" for s in results['complete_sections']]
                context_text = "\n\n".join(context_list)
                print(f"ℹ️ 使用模式: [完整章節] - 包含 {len(results['complete_sections'])} 個章節全文")
            else:
                # 萬一找不到對應章節，降級回使用片段
                context_text = "\n\n".join([n.node.text for n in retrieved_nodes])
                print(f"⚠️ 警告: 找不到完整章節，降級使用檢索片段")
        else:
            # 方案 B: 只使用檢索到的片段 (預設 RAG 行為)
            context_text = "\n\n".join([f"---片段---\n{n.node.text}" for n in retrieved_nodes])
            print(f"ℹ️ 使用模式: [檢索片段] - 使用 {len(retrieved_nodes)} 個片段")

        t3 = time.time()
        

        # 步驟 4: 呼叫 LLM 生成答案
        # prompt可以調整
        if context_text:
            prompt = (
                f"你是一個專業的合約助手。請根據以下提供的參考資料回答問題。\n"
                f"如果資料中沒有答案，請直接說明。\n\n"
                f"=== 參考資料 ===\n{context_text}\n\n"
                f"=== 使用者問題 ===\n{query}"
            ) 
            
            # 使用全域 Settings 中的 LLM (LlamaIndex v0.10+ 寫法)
            response = Settings.llm.complete(prompt)
            results['answer'] = response.text
        else:
            results['answer'] = "未能檢索到相關資料。"

        # 補充層級資訊
        results['hierarchy_info'] = self._analyze_query_hierarchy(results['related_clauses'])

        t4 = time.time()
        print(f"⏱️ [Step 4] LLM 生成耗時: {t4 - t3:.4f} 秒")
        print(f"🏁 總共耗時: {t4 - st:.4f} 秒")
        
        return results
    
    def _analyze_query_hierarchy(self, related_clauses: List[Dict]) -> Dict:
        """階層分析（保持不變）"""
        if not related_clauses:
            return {}
        
        levels_found = set()
        sections_involved = set()
        main_clauses = []
        sub_clauses = []
        
        for clause in related_clauses:
            level = clause.get('level', -1)
            levels_found.add(level)
            
            section_num = clause.get('section_number', 0)
            if section_num > 0:
                sections_involved.add(section_num)
            
            if clause.get('clause_type') == 'main_clause':
                main_clauses.append(clause)
            elif clause.get('clause_type') == 'sub_clause':
                sub_clauses.append(clause)
        
        return {
            'levels_involved': sorted(list(levels_found)),
            'sections_count': len(sections_involved),
            'main_clauses_count': len(main_clauses),
            'sub_clauses_count': len(sub_clauses),
            'hierarchy_complexity': 'simple' if len(levels_found) <= 2 else 'complex'
        }





