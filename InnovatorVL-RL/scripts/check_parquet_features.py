#!/usr/bin/env python3
"""
诊断脚本：检查 parquet 文件的特征类型问题
用于定位 AttributeError: 'Value' object has no attribute 'items' 错误
"""

import sys
from pathlib import Path
from datasets import load_dataset, Features, Image as DatasetImage, Value, Sequence

def check_dataset_features(parquet_path):
    """检查数据集的特征类型"""
    print(f"\n{'='*80}")
    print(f"检查文件: {parquet_path}")
    print(f"{'='*80}\n")
    
    try:
        # 加载数据集
        dataset = load_dataset("parquet", data_files=str(parquet_path))['train']
        
        print(f"数据集大小: {len(dataset)}")
        print(f"列名: {dataset.column_names}\n")
        
        # 检查 features 类型
        print("=" * 80)
        print("特征类型检查")
        print("=" * 80)
        print(f"features 类型: {type(dataset.features)}")
        print(f"是否为 Features 对象: {isinstance(dataset.features, Features)}")
        
        if not isinstance(dataset.features, Features):
            print(f"❌ 错误: features 不是 Features 对象！")
            print(f"   实际类型: {type(dataset.features)}")
            print(f"   features 值: {dataset.features}")
            return False
        
        print("✅ features 是 Features 对象\n")
        
        # 检查每个字段的特征类型
        print("=" * 80)
        print("各字段特征类型详情")
        print("=" * 80)
        
        for col_name in dataset.column_names:
            if col_name in dataset.features:
                feature = dataset.features[col_name]
                print(f"\n字段: {col_name}")
                print(f"  类型: {type(feature)}")
                print(f"  值: {feature}")
                
                # 特别检查 images 字段
                if col_name == "images":
                    print(f"  ⚠️  这是 images 字段，需要特别检查")
                    if isinstance(feature, Sequence):
                        print(f"    是 Sequence 类型")
                        if hasattr(feature, "feature"):
                            inner_feature = feature.feature
                            print(f"    内部特征类型: {type(inner_feature)}")
                            print(f"    内部特征值: {inner_feature}")
                            
                            if isinstance(inner_feature, DatasetImage):
                                print(f"    ✅ 内部是 Image 类型（正确）")
                            elif hasattr(inner_feature, "__contains__"):
                                print(f"    ⚠️  内部是字典类型（可能是问题所在）")
                                if hasattr(inner_feature, "keys"):
                                    try:
                                        keys = list(inner_feature.keys())
                                        print(f"    字典键: {keys}")
                                    except:
                                        pass
                            else:
                                print(f"    ⚠️  未知的内部特征类型")
                    else:
                        print(f"    ⚠️  不是 Sequence 类型")
                
                # 特别检查字典类型的字段（如 reward_model, vanilla_prompt）
                if isinstance(feature, dict):
                    print(f"  ⚠️  这是字典类型的特征（需要特别检查）")
                    print(f"  字典键: {list(feature.keys())}")
                    for k, v in feature.items():
                        print(f"    {k}: {type(v)} = {v}")
                        # 检查值是否是 Value 对象
                        from datasets import Value
                        if isinstance(v, Value):
                            print(f"      ⚠️  值是 Value 对象")
                        elif isinstance(v, dict):
                            print(f"      ⚠️  值是嵌套字典")
                            for k2, v2 in v.items():
                                print(f"        {k2}: {type(v2)} = {v2}")
                                if isinstance(v2, Value):
                                    print(f"          ⚠️  嵌套值是 Value 对象")
                
                # 检查是否是 List 包含字典（如 vanilla_prompt）
                if hasattr(feature, "feature"):
                    inner = feature.feature
                    if isinstance(inner, dict):
                        print(f"  ⚠️  List 内部是字典类型")
                        print(f"  内部字典键: {list(inner.keys())}")
                        for k, v in inner.items():
                            print(f"    {k}: {type(v)} = {v}")
                            from datasets import Value
                            if isinstance(v, Value):
                                print(f"      ⚠️  值是 Value 对象")
        
        # 检查第一个样本
        print("\n" + "=" * 80)
        print("第一个样本检查")
        print("=" * 80)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本键: {list(sample.keys())}")
            for key, value in sample.items():
                print(f"\n{key}:")
                print(f"  类型: {type(value)}")
                if isinstance(value, list):
                    print(f"  列表长度: {len(value)}")
                    if len(value) > 0:
                        print(f"  第一个元素类型: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"  第一个元素字典键: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    print(f"  字典键: {list(value.keys())}")
                    # 详细检查字典的每个值
                    for k, v in value.items():
                        print(f"    {k}: {type(v)} = {str(v)[:50]}")
                else:
                    print(f"  值预览: {str(value)[:100]}")
        
        # 特别检查嵌套字段的特征结构
        print("\n" + "=" * 80)
        print("嵌套字段特征结构详细检查")
        print("=" * 80)
        for col_name in dataset.column_names:
            if col_name in dataset.features:
                feature = dataset.features[col_name]
                print(f"\n字段: {col_name}")
                print(f"  特征类型: {type(feature)}")
                print(f"  特征值: {feature}")
                
                # 检查是否是嵌套的字典结构
                if isinstance(feature, dict):
                    print(f"  ⚠️  这是字典类型的特征（可能是问题所在）")
                    print(f"  字典键: {list(feature.keys())}")
                    for k, v in feature.items():
                        print(f"    {k}: {type(v)} = {v}")
                        # 检查值是否是 Value 对象
                        from datasets import Value
                        if isinstance(v, Value):
                            print(f"      ⚠️  值是 Value 对象，这可能导致 _align_features 出错")
                
                # 检查是否是 List 包含字典
                if hasattr(feature, "feature"):
                    inner = feature.feature
                    if isinstance(inner, dict):
                        print(f"  ⚠️  List 内部是字典类型")
                        print(f"  内部字典键: {list(inner.keys())}")
                        for k, v in inner.items():
                            print(f"    {k}: {type(v)} = {v}")
                            from datasets import Value
                            if isinstance(v, Value):
                                print(f"      ⚠️  值是 Value 对象")
        
        # 尝试转换 images 字段类型（测试改进逻辑）
        print("\n" + "=" * 80)
        print("尝试转换 images 字段类型")
        print("=" * 80)
        
        if "images" in dataset.features:
            img_feature = dataset.features["images"]
            print(f"当前 images 特征: {img_feature}")
            
            # 检查是否需要转换
            needs_conversion = False
            if hasattr(img_feature, "feature"):
                inner_feature = img_feature.feature
                if not isinstance(inner_feature, DatasetImage):
                    needs_conversion = True
                    print(f"⚠️  需要转换: 内部特征类型是 {type(inner_feature)}，不是 Image 类型")
            else:
                needs_conversion = True
                print(f"⚠️  需要转换: images 特征结构异常")
            
            if needs_conversion:
                print("\n尝试方法 1: 直接 cast 转换...")
                try:
                    standard_images_feature = Sequence(DatasetImage(decode=True))
                    new_features_dict = dict(dataset.features)
                    new_features_dict["images"] = standard_images_feature
                    new_features = Features(new_features_dict)
                    dataset_converted = dataset.cast(new_features)
                    
                    # 验证转换结果
                    if isinstance(dataset_converted.features, Features) and "images" in dataset_converted.features:
                        converted_img_feature = dataset_converted.features["images"]
                        if hasattr(converted_img_feature, "feature") and isinstance(converted_img_feature.feature, DatasetImage):
                            print("✅ 方法 1 成功: cast 转换成功！")
                            dataset = dataset_converted
                        else:
                            print(f"⚠️  方法 1 部分成功: cast 完成但类型仍不正确: {converted_img_feature}")
                            needs_conversion = True  # 继续尝试方法 2
                    else:
                        print("⚠️  方法 1 部分成功: cast 完成但 features 结构异常")
                        needs_conversion = True  # 继续尝试方法 2
                except Exception as e:
                    print(f"❌ 方法 1 失败: {type(e).__name__}: {e}")
                    needs_conversion = True  # 继续尝试方法 2
                
                if needs_conversion:
                    print("\n尝试方法 2: Workaround (移除并重新添加 images 字段)...")
                    try:
                        from datasets import Dataset
                        # 先移除 images 字段
                        temp_features_dict = {k: v for k, v in dataset.features.items() if k != "images"}
                        temp_features = Features(temp_features_dict)
                        dataset_temp = dataset.cast(temp_features)
                        
                        # 重新添加 images 字段，使用正确的特征类型
                        data_dict = {col: [dataset_temp[j][col] for j in range(len(dataset_temp))] for col in dataset_temp.column_names}
                        # 获取原始 images 数据
                        original_images = [dataset[j]["images"] for j in range(len(dataset))]
                        data_dict["images"] = original_images
                        
                        # 创建新的 features，包含正确类型的 images
                        final_features = Features({**temp_features_dict, "images": Sequence(DatasetImage(decode=True))})
                        dataset = Dataset.from_dict(data_dict, features=final_features)
                        
                        # 验证转换结果
                        if isinstance(dataset.features, Features) and "images" in dataset.features:
                            converted_img_feature = dataset.features["images"]
                            if hasattr(converted_img_feature, "feature") and isinstance(converted_img_feature.feature, DatasetImage):
                                print("✅ 方法 2 成功: Workaround 转换成功！")
                                needs_conversion = False
                            else:
                                print(f"❌ 方法 2 失败: 转换后类型仍不正确: {converted_img_feature}")
                        else:
                            print("❌ 方法 2 失败: features 结构异常")
                    except Exception as e:
                        print(f"❌ 方法 2 失败: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
        
        # 尝试与其他数据集连接（模拟实际使用场景）
        print("\n" + "=" * 80)
        print("尝试与其他数据集连接测试")
        print("=" * 80)
        
        # 创建一个简单的测试数据集
        from datasets import Dataset
        test_data = {
            "id": ["test_1"],
            "images": [[None]],  # 空的 images
            "problem": ["test problem"],
            "answer": [["test answer"]],
            "problem_type": ["test"],
            "answer_type": ["test"],
            "source": ["test"],
            "prompt_type": ["test"],
        }
        
        # 创建标准的 Features（只包含 dataset 中存在的字段）
        test_features_dict = {
            "images": Sequence(DatasetImage(decode=True)),
        }
        # 添加 dataset 中的其他字段（如果存在）
        for col in dataset.column_names:
            if col != "images" and col not in test_features_dict:
                if col in dataset.features:
                    test_features_dict[col] = dataset.features[col]
        
        # 只保留 test_data 中存在的字段
        test_data_filtered = {k: v for k, v in test_data.items() if k in test_features_dict}
        # 为缺失的字段添加默认值
        for col in test_features_dict:
            if col not in test_data_filtered:
                if col == "images":
                    test_data_filtered[col] = [[None]]
                elif isinstance(test_features_dict[col], (list, Sequence)):
                    test_data_filtered[col] = [[]]
                else:
                    test_data_filtered[col] = ["test_value"]
        
        test_features = Features(test_features_dict)
        test_dataset = Dataset.from_dict(test_data_filtered, features=test_features)
        
        print(f"测试数据集 features 类型: {type(test_dataset.features)}")
        print(f"测试数据集是否为 Features: {isinstance(test_dataset.features, Features)}")
        if "images" in test_dataset.features:
            print(f"测试数据集 images 特征: {test_dataset.features['images']}")
        
        # 尝试连接
        try:
            from datasets import concatenate_datasets
            print("\n尝试连接数据集...")
            print(f"原始数据集 images 特征: {dataset.features.get('images', 'N/A')}")
            combined = concatenate_datasets([test_dataset, dataset])
            print("✅ 连接成功！")
            print(f"合并后数据集大小: {len(combined)}")
            print(f"合并后 images 特征: {combined.features.get('images', 'N/A')}")
        except Exception as e:
            print(f"❌ 连接失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 加载数据集时出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_parquet_features.py <parquet_file_path>")
        print("\n示例:")
        print("  python check_parquet_features.py /path/to/merged_clean_full.parquet")
        sys.exit(1)
    
    parquet_path = Path(sys.argv[1])
    if not parquet_path.exists():
        print(f"❌ 文件不存在: {parquet_path}")
        sys.exit(1)
    
    success = check_dataset_features(parquet_path)
    sys.exit(0 if success else 1)

