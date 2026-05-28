import os
import json
import re
import glob
from flashtext import KeywordProcessor

# ==========================================
# 1. 实体匹配器
# ==========================================
class EntityMatcher:
    def __init__(self, entity_dict_path):
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.categories = set()
        
        print(f"[Init] 正在加载实体词典: {entity_dict_path} ...")
        
        try:
            with open(entity_dict_path, 'r', encoding='utf-8') as f:
                entity_data = json.load(f)
            
            count = 0
            for category, items in entity_data.items():
                self.categories.add(category)
                for item in items:
                    clean_item = item.strip()
                    if clean_item:
                        # FlashText 替换逻辑: "原始词" -> "原始词|类别"
                        self.keyword_processor.add_keyword(clean_item, f"{clean_item}|{category}")
                        count += 1
            
            print(f"[Init] 成功加载 {len(self.categories)} 个类别，共 {count} 个实体词条。")
            
        except Exception as e:
            print(f"[Error] 加载实体字典失败: {e}")
            print("请检查 entity.json 路径是否正确，或文件编码是否为 UTF-8。")
            raise e

    def extract(self, text):
        found = self.keyword_processor.extract_keywords(text)
        results = []
        seen = set()
        for item in found:
            if '|' in item:
                name, category = item.split('|', 1)
                if (name, category) not in seen:
                    results.append({"name": name, "category": category})
                    seen.add((name, category))
        return results

# ==========================================
# 2. 论文解析器 (增加无效图表过滤)
# ==========================================
class PaperParser:
    def __init__(self, entity_matcher):
        self.matcher = entity_matcher
        self.block_pattern = re.compile(r'\[\[BLOCK_(IMAGE|TABLE|MATH|CHART)_(\d+)\]\]')
        self.analysis_start_pattern = re.compile(r'^>\s*\*\*Expert Analysis:\*\*(.*)')

    def parse_markdown(self, file_path):
        # 尝试使用 utf-8 读取，如果报错尝试 gbk
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()

        structure = []
        current_section = "Intro/Metadata"
        global_index = 0
        
        pending_block_type = None 
        in_analysis_block = False 
        analysis_buffer = []      

        def flush_analysis_buffer():
            nonlocal global_index, in_analysis_block, pending_block_type, analysis_buffer
            if not analysis_buffer:
                return
            
            full_text = " ".join(analysis_buffer).strip()
            if full_text:
                c_type = "text"
                is_valid_chart = True

                # 智能判定内容类型
                if pending_block_type == "IMAGE": c_type = "chart_description"
                elif pending_block_type == "TABLE": c_type = "table_description"
                elif pending_block_type == "MATH": c_type = "formula_description"
                
                # 【修改点】 检测无效/空白的图表描述
                # 如果描述中包含 "completely blank" 或 "no visible content"，标记为无效
                if c_type == "chart_description":
                    lower_text = full_text.lower()
                    if "completely blank" in lower_text or "no visible content" in lower_text:
                        c_type = "invalid_chart" # 标记为无效图表
                        is_valid_chart = False
                
                # 只有有效图表才提取实体，避免提取空内容
                entities = self.matcher.extract(full_text) if is_valid_chart else []
                
                structure.append({
                    "index": global_index,
                    "section_hierarchy": current_section,
                    "content_type": c_type,
                    "text": full_text,
                    "entities": entities,
                    "entity_count": len(entities)
                })
                global_index += 1
            
            analysis_buffer = []
            in_analysis_block = False
            pending_block_type = None

        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith('#'):
                flush_analysis_buffer()
                current_section = line.lstrip('#').strip()
                continue

            block_match = self.block_pattern.search(line)
            if block_match:
                flush_analysis_buffer()
                tag_type = block_match.group(1)
                pending_block_type = tag_type
                continue

            analysis_match = self.analysis_start_pattern.match(line)
            if analysis_match:
                flush_analysis_buffer()
                in_analysis_block = True
                content = analysis_match.group(1).strip()
                if content:
                    analysis_buffer.append(content)
                continue

            if in_analysis_block and line.startswith('>'):
                clean_content = line.lstrip('>').strip()
                if clean_content:
                    analysis_buffer.append(clean_content)
                continue
            
            if in_analysis_block and not line.startswith('>'):
                flush_analysis_buffer()
            
            # 过滤元数据
            if line.startswith('type:') or line.startswith('fileName:') or line.startswith('fullContent:'):
                continue
            
            entities = self.matcher.extract(line)
            structure.append({
                "index": global_index,
                "section_hierarchy": current_section,
                "content_type": "text",
                "text": line,
                "entities": entities,
                "entity_count": len(entities)
            })
            global_index += 1

        flush_analysis_buffer()
        return structure

# ==========================================
# 3. 主执行流程 (保持路径不变)
# ==========================================
def main():
    # ------------------------------------------------------------------
    # ↓↓↓↓↓↓ 路径配置 ↓↓↓↓↓↓
    # ------------------------------------------------------------------
    
    input_dir = r"D:\课题组\中心偏析——表面裂纹markdown\end_markdown"
    output_dir = r"D:\课题组\中心偏析——表面裂纹markdown\entity_end2"
    entity_file = 'entity.json' # 假设和脚本在同一目录
    
    # ------------------------------------------------------------------
    
    if not os.path.exists(output_dir):
        print(f"[Info] 输出目录不存在，正在创建: {output_dir}")
        os.makedirs(output_dir)

    print("========================================")
    print("      冶金文献实体标注与结构化工具       ")
    print("========================================")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 1. 初始化
    try:
        matcher = EntityMatcher(entity_file)
    except FileNotFoundError:
        print(f"\n[Fatal Error] 找不到实体文件: {entity_file}")
        return

    parser = PaperParser(matcher)
    
    # 2. 获取文件列表
    md_files = glob.glob(os.path.join(input_dir, '*.md'))
    if not md_files:
        print(f"\n[Warning] 在输入目录中没有找到 .md 文件！")
        return
        
    print(f"\n找到 {len(md_files)} 个文档，开始处理...\n")
    
    # 3. 批量处理
    total_paragraphs = 0
    total_entities_found = 0
    
    for i, file_path in enumerate(md_files):
        file_name = os.path.basename(file_path)
        print(f"[{i+1}/{len(md_files)}] 处理: {file_name}", end=" ... ")
        
        try:
            structure = parser.parse_markdown(file_path)
            
            p_count = len(structure)
            e_count = sum(item['entity_count'] for item in structure)
            total_paragraphs += p_count
            total_entities_found += e_count
            
            final_json = {
                "paper_id": file_name,
                "source_file": file_path,
                "total_paragraphs": p_count,
                "total_entities": e_count,
                "structure": structure
            }
            
            output_name = file_name.replace('.md', '.json')
            output_path = os.path.join(output_dir, output_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, ensure_ascii=False, indent=2)
                
            print(f"完成 (实体: {e_count})")
            
        except Exception as e:
            print(f"\n[Error] 处理文件 {file_name} 时出错: {e}")

    print("\n========================================")
    print(f"全部完成！结果已保存在: {output_dir}")
    print("========================================")

if __name__ == "__main__":
    main()