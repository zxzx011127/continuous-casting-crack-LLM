import json
import os
import glob
import random

# ==============================================================================
#  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  【配置区域】  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ==============================================================================

# 1. 输入路径：你存放分散 QA JSON 文件的文件夹
INPUT_DIR = r"D:\课题组\中心偏析——表面裂纹markdown\QA330"

# 2. 输出路径：合并后的数据集存放位置 (建议放在上一级目录)
OUTPUT_DIR = r"D:\课题组\中心偏析——表面裂纹markdown\dataset_final2"

# 3. 数据集划分比例
TRAIN_RATIO = 0.88   # 88% 训练集
VAL_RATIO = 0.045    # 4.5% 验证集
TEST_RATIO = 0.075   # 7.5% 测试集

# ==============================================================================

def main():
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        print(f"创建输出目录: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # 2. 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    if not json_files:
        print(f"错误：在 {INPUT_DIR} 中没有找到 .json 文件！")
        return

    print(f"扫描到 {len(json_files)} 个 QA 文件，开始合并...")

    all_data = []
    total_files = 0
    empty_files = 0

    # 3. 循环读取并合并
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 检查数据格式
                if isinstance(data, list):
                    valid_entries = []
                    for entry in data:
                        # 增加对 'input' 字段的校验，保证数据三要素完整性
                        if 'instruction' in entry and 'input' in entry and 'output' in entry:
                            valid_entries.append(entry)
                    
                    if valid_entries:
                        all_data.extend(valid_entries)
                        total_files += 1
                    else:
                        empty_files += 1
                else:
                    print(f"警告：文件格式不是列表: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"读取文件出错 {file_path}: {e}")

    total_data_len = len(all_data)
    print(f"\n合并完成！")
    print(f"有效文件数: {total_files}")
    print(f"空文件数: {empty_files}")
    print(f"总数据条目: {total_data_len}")

    if total_data_len == 0:
        print("没有有效数据，程序退出。")
        return

    # 4. 打乱数据 (Shuffle)
    # 固定随机种子，保证每次运行划分结果一致，便于复现
    random.seed(42)
    random.shuffle(all_data)
    print("数据已随机打乱 (已固定随机种子: 42)。")

    # 5. 划分训练集、验证集和测试集
    train_size = int(total_data_len * TRAIN_RATIO)
    val_size = int(total_data_len * VAL_RATIO)
    # 剩下的全给测试集，防止浮点数向下取整导致最后少几条数据
    test_size = total_data_len - train_size - val_size 

    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]

    print(f"\n数据集划分情况:")
    print(f"  - 训练集 (Train): {len(train_data)} 条 ({TRAIN_RATIO*100:.1f}%)")
    print(f"  - 验证集 (Val)  : {len(val_data)} 条 ({VAL_RATIO*100:.1f}%)")
    print(f"  - 测试集 (Test) : {len(test_data)} 条 ({TEST_RATIO*100:.1f}%)")

    # 6. 保存文件
    train_path = os.path.join(OUTPUT_DIR, "train_dataset.json")
    val_path = os.path.join(OUTPUT_DIR, "val_dataset.json")
    test_path = os.path.join(OUTPUT_DIR, "test_dataset.json")
    all_path = os.path.join(OUTPUT_DIR, "all_dataset_merged.json") 

    def save_json(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    save_json(train_data, train_path)
    save_json(val_data, val_path)
    save_json(test_data, test_path)
    save_json(all_data, all_path)

    print(f"\n文件已保存:")
    print(f"  1. {train_path}")
    print(f"  2. {val_path}")
    print(f"  3. {test_path}")
    print(f"  4. {all_path}")
    print("\n数据处理彻底闭环！您可以直接使用 train_dataset.json 进行微调了。")

if __name__ == "__main__":
    main()