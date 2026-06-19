import re
import os
import json

class InPlaceMetallurgyRefinery:
    def __init__(self):
        self.protected_vault = {}
        self.counters = {"MATH": 0, "TABLE": 0, "IMAGE": 0}

    def _reset(self):
        self.protected_vault = {}
        self.counters = {"MATH": 0, "TABLE": 0, "IMAGE": 0}

    def _find_nearby_caption(self, text, start_pos, end_pos, block_type):
        """核心功能：在占位符附近寻找物理含义描述（如 Figure 10. Hardness profile）"""
        look_distance = 300
        search_area = text[max(0, start_pos - look_distance) : end_pos + look_distance]
        
        patterns = {
            "IMAGE": [r'(?i)Figure\s*\d+[:\.]?.*', r'(?i)Fig\.\s*\d+[:\.]?.*'],
            "TABLE": [r'(?i)Table\s*\d+[:\.]?.*'],
            "MATH": [r'(?i)Equation\s*\(\d+\).*', r'\(\d+\)']
        }
        
        for pat in patterns.get(block_type, []):
            match = re.search(pat, search_area)
            if match: return match.group(0).strip()
        return "No Caption Found"

    def _store_asset(self, match, block_type, full_text):
        """存储资产并关联上下文标题"""
        idx = self.counters[block_type]
        placeholder = f"[[BLOCK_{block_type}_{idx:03d}]]"
        caption = self._find_nearby_caption(full_text, match.start(), match.end(), block_type)
        
        self.protected_vault[placeholder] = {
            "raw_content": match.group(0).strip(),
            "caption": caption,
            "type": block_type
        }
        self.counters[block_type] += 1
        return f"\n\n{placeholder}\n\n"

    def protect_assets(self, text):
        """第一阶段：资产隔离，防止正则清洗误杀图片路径和表格"""
        text = re.sub(r'!\[.*?\]\(.*?\)', lambda m: self._store_asset(m, "IMAGE", text), text)
        table_pattern = r'\n\|.*\|.*\n\|[- |:]*\|.*\n(?:\|.*\|.*\n*)+'
        text = re.sub(table_pattern, lambda m: self._store_asset(m, "TABLE", text), text)
        text = re.sub(r'\$\$.*?\$\$', lambda m: self._store_asset(m, "MATH", text), text, flags=re.DOTALL)
        return text

    def adaptive_clean(self, text):
        """第二阶段：学术语义清洗（适配 Elsevier, MDPI, Wiley 等多种期刊排版）"""
        
        # 1. 动态截断：通过 References/参考文献 等锚点自动停止提取
        end_anchors = r'\n#+\s*(?:References?|Acknowledgements?|Appendix|参考文献|致谢).*'
        parts = re.split(end_anchors, text, flags=re.IGNORECASE | re.DOTALL)
        text = parts[0]

        # 2. 语义起点：自动跳过摘要前的作者、地址、DOI 等所有元数据噪声
        start_anchors = r'\n#+\s*(?:Abstract|摘要|1\.\s*Introduction|1\.\s*前言).*'
        start_match = re.search(start_anchors, text, flags=re.IGNORECASE)
        if start_match: text = text[start_match.start():]

        # 3. 行级清理：删除含有 DOI、邮箱、URL 和孤立页码的噪声行
        lines = text.split('\n')
        cleaned_lines = [l for l in lines if not re.search(r'doi\.org|http|[\w\.-]+@[\w\.-]+|Page\s*\d+|ISSN', l, re.I)]
        text = '\n'.join(cleaned_lines)

        # 4. 引用标记消磁：删除正文中的 [12], [5-8] 等索引
        text = re.sub(r'\[[\d\s,.\-–]+\]', '', text)
        
        # 5. 段落哈希去重：解决 PDF 分栏渲染导致的幻觉重复内容
        paragraphs = text.split('\n\n')
        final_p = []
        for i, p in enumerate(paragraphs):
            curr = p.strip()
            if not curr: continue
            if i > 0 and curr[:30] == paragraphs[i-1].strip()[:30]: continue
            final_p.append(p)
            
        return '\n\n'.join(final_p).strip()

    def process_and_save(self, file_path):
        """执行全套流程并在原位生成结果"""
        self._reset()
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 先隔离资产，再深度清洗
        cleaned = self.adaptive_clean(self.protect_assets(content))
        
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        
        # 在原始文件夹内生成 refined 文档和资产 JSON
        out_md = os.path.join(dir_name, f"refined_{base_name}")
        out_json = os.path.join(dir_name, f"assets_{base_name.replace('.md', '.json')}")
        
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(self.protected_vault, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # ================= 修改此处为你的根目录 =================
    ROOT_DIR = r"D:\test_out" 
    # =======================================================

    refinery = InPlaceMetallurgyRefinery()
    print(f"🚀 正在递归洗炼子文件夹中的语料库...")
    
    count = 0
    for root, dirs, files in os.walk(ROOT_DIR):
        for name in files:
            # 仅处理原始 Markdown 文件，避免二次清洗
            if name.endswith(".md") and not name.startswith("refined_"):
                full_path = os.path.join(root, name)
                try:
                    refinery.process_and_save(full_path)
                    count += 1
                    print(f"✅ 已洗炼并关联图片: {os.path.relpath(full_path, ROOT_DIR)}")
                except Exception as e:
                    print(f"❌ 出错 {name}: {e}")

    print(f"\n✨ 大功告成！所有文献均已在原位文件夹完成洗炼与结构化映射。")