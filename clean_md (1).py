import re
import os
import json

class MetallurgyDataCleaner:
    def __init__(self):
        # 用于存储被保护的内容，供后续步骤使用
        self.protected_vault = {}
        self.counters = {"MATH": 0, "TABLE": 0, "IMAGE": 0}

    def _store_block(self, match, block_type):
        """将匹配到的内容存入保险箱，返回占位符"""
        idx = self.counters[block_type]
        placeholder = f"[[BLOCK_{block_type}_{idx:03d}]]"
        self.protected_vault[placeholder] = match.group(0).strip()
        self.counters[block_type] += 1
        return f"\n\n{placeholder}\n\n"

    def protect_assets(self, text):
        """第一步：保护公式、表格、图片"""
        # 1. 保护块级公式 $$...$$
        text = re.sub(r'\$\$.*?\$\$', lambda m: self._store_block(m, "MATH"), text, flags=re.DOTALL)
        
        # 2. 保护 Markdown 表格 (识别连续带 | 的行)
        table_pattern = r'(\n\|.*\|.*\n\|[- |:]*\|.*\n(?:\|.*\|.*\n*)+)'
        text = re.sub(table_pattern, lambda m: self._store_block(m, "TABLE"), text)

        # 3. 保护行内公式 $...$ (排除已经处理过的占位符)
        text = re.sub(r'\$(?![^$]*?BLOCK_)[^$]+\$', lambda m: self._store_block(m, "MATH"), text)

        # 4. 保护图片 ![alt](url)
        text = re.sub(r'!\[.*?\]\(.*?\)', lambda m: self._store_block(m, "IMAGE"), text)
        
        return text

    def clean_noise(self, text):
        """第二步：精准去噪"""
        
        # 1. 截断：切除参考文献及之后的所有内容
        ref_keywords = [r'\n#+\s*References', r'\n#+\s*Bibliography', r'\n参考文献', r'\n#+\s*Notes']
        for kw in ref_keywords:
            parts = re.split(kw, text, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[0]
                break

        # 2. 截断：切除摘要/引言之前的所有作者、机构、页眉信息
        header_keywords = [r'\n#+\s*Abstract', r'\n#+\s*Introduction', r'\n摘要', r'\n引言']
        for kw in header_keywords:
            parts = re.split(kw, text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) > 1:
                text = f"\n# {kw.strip().lstrip('# ')}\n" + parts[1]
                break

        # 3. 删除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)

        # 4. 删除文中引用索引 (例如 [1], [2-5], [10, 12])
        text = re.sub(r'\[[\d\s,.\-–]+\]', '', text)
        
        # 5. 删除作者年份引用 (例如 Smith et al., 2023)
        text = re.sub(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4}\)', '', text)
        text = re.sub(r'\([\u4e00-\u9fa5]+\s+等,\s+\d{4}\)', '', text)

        # 6. 清理页眉页脚常见噪声 (单行页码、孤立 URL、版权声明)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) # 孤立页码
        text = re.sub(r'https?://\S+', '', text) # 孤立 URL
        text = re.sub(r'(?i)Downloaded from.*|Published by.*|All rights reserved.*', '', text)

        # 7. 合并由于 PDF 换行导致的断句 (可选，但推荐)
        # 如果一行不以标点结尾且下一行不是标题/列表，则尝试合并
        # text = re.sub(r'([a-zA-Z,])\n([a-z])', r'\1 \2', text)

        # 8. 格式化空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def process(self, input_file, output_dir):
        """主执行逻辑"""
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 执行流水线
        protected_text = self.protect_assets(content)
        cleaned_text = self.clean_noise(protected_text)

        # 保存清洗后的 Markdown
        base_name = os.path.basename(input_file)
        md_output = os.path.join(output_dir, f"cleaned_{base_name}")
        with open(md_output, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        # 保存对应的保险箱数据 (JSON)，供第三步描述回插使用
        vault_output = os.path.join(output_dir, f"vault_{base_name.replace('.md', '.json')}")
        with open(vault_output, 'w', encoding='utf-8') as f:
            json.dump(self.protected_vault, f, indent=4, ensure_ascii=False)

        print(f"--- 处理完成: {base_name} ---")
        print(f"成功保护资产: MATH({self.counters['MATH']}), TABLE({self.counters['TABLE']}), IMAGE({self.counters['IMAGE']})")
        print(f"输出文件: {md_output}")

# --- 使用方法 ---
if __name__ == "__main__":
    cleaner = MetallurgyDataCleaner()
    # 填入你的文件路径
    # cleaner.process("sample_paper.md", "./output")