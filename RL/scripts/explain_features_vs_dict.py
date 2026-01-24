#!/usr/bin/env python3
"""
演示普通 dict 和 Features 的区别
"""

from datasets import Features, Value, Sequence, Image as DatasetImage

print("=" * 80)
print("普通 dict vs Features 的区别")
print("=" * 80)

# 1. 普通 Python dict
print("\n1. 普通 Python dict:")
normal_dict = {
    'ground_truth': Value('string'),
    'style': Value('string')
}
print(f"   类型: {type(normal_dict)}")
print(f"   值: {normal_dict}")
print(f"   是否有 .items() 方法: {hasattr(normal_dict, 'items')}")
print(f"   是否有 .keys() 方法: {hasattr(normal_dict, 'keys')}")
print(f"   是否是 Features 对象: {isinstance(normal_dict, Features)}")

# 2. Features 对象
print("\n2. Features 对象:")
features_obj = Features({
    'ground_truth': Value('string'),
    'style': Value('string')
})
print(f"   类型: {type(features_obj)}")
print(f"   值: {features_obj}")
print(f"   是否有 .items() 方法: {hasattr(features_obj, 'items')}")
print(f"   是否有 .keys() 方法: {hasattr(features_obj, 'keys')}")
print(f"   是否是 Features 对象: {isinstance(features_obj, Features)}")

# 3. 关键区别
print("\n" + "=" * 80)
print("关键区别:")
print("=" * 80)

print("\n1. 类型检查:")
print(f"   isinstance(normal_dict, dict): {isinstance(normal_dict, dict)}")
print(f"   isinstance(normal_dict, Features): {isinstance(normal_dict, Features)}")
print(f"   isinstance(features_obj, dict): {isinstance(features_obj, dict)}")
print(f"   isinstance(features_obj, Features): {isinstance(features_obj, Features)}")

print("\n2. 方法调用:")
print("   两者都有 .items() 和 .keys() 方法")
print("   但 Features 对象有额外的数据集相关方法")

print("\n3. 在 _align_features 中的行为:")
print("   - _align_features 函数会递归处理嵌套结构")
print("   - 当它遇到 dict 时，会调用 dict.items()")
print("   - 但如果 dict 的值是 Value 对象，_align_features 会尝试递归处理")
print("   - 问题：_align_features 期望遇到 Features 对象，但遇到普通 dict 时")
print("     它会尝试对 dict 的值调用 .items()，而 Value 对象没有 .items() 方法")

# 4. 实际示例
print("\n" + "=" * 80)
print("实际示例 - 为什么会出现错误:")
print("=" * 80)

print("\n普通 dict 的情况:")
print("  reward_model = {'ground_truth': Value('string'), 'style': Value('string')}")
print("  _align_features 处理时:")
print("    - 看到 reward_model 是 dict，调用 reward_model.items()")
print("    - 得到 ('ground_truth', Value('string')) 和 ('style', Value('string'))")
print("    - 尝试递归处理 Value('string')，调用 Value('string').items()")
print("    - ❌ 错误：Value 对象没有 .items() 方法")

print("\nFeatures 对象的情况:")
print("  reward_model = Features({'ground_truth': Value('string'), 'style': Value('string')})")
print("  _align_features 处理时:")
print("    - 看到 reward_model 是 Features，调用 reward_model.items()")
print("    - 得到 ('ground_truth', Value('string')) 和 ('style', Value('string'))")
print("    - 识别 Value('string') 是叶子节点，不再递归")
print("    - ✅ 成功：正确处理")

# 5. 如何转换
print("\n" + "=" * 80)
print("如何转换:")
print("=" * 80)

print("\n从普通 dict 转换为 Features:")
print("  normal_dict = {'ground_truth': Value('string'), 'style': Value('string')}")
print("  features_obj = Features(normal_dict)")
print(f"  结果: {Features(normal_dict)}")
print(f"  类型: {type(Features(normal_dict))}")

# 6. 在数据集中的使用
print("\n" + "=" * 80)
print("在数据集中的使用:")
print("=" * 80)

print("\n正确的用法:")
print("  features = Features({")
print("      'reward_model': Features({")
print("          'ground_truth': Value('string'),")
print("          'style': Value('string')")
print("      })")
print("  })")

print("\n错误的用法（会导致 AttributeError）:")
print("  features = Features({")
print("      'reward_model': {  # 普通 dict，不是 Features")
print("          'ground_truth': Value('string'),")
print("          'style': Value('string')")
print("      }")
print("  })")

print("\n" + "=" * 80)
print("总结:")
print("=" * 80)
print("1. 普通 dict 是 Python 内置类型")
print("2. Features 是 datasets 库的特殊类型，继承自 dict 但增加了数据集相关的功能")
print("3. _align_features 函数期望嵌套结构都是 Features 对象，而不是普通 dict")
print("4. 当遇到普通 dict 时，_align_features 会尝试递归处理，导致对 Value 对象调用 .items() 而失败")
print("5. 解决方案：将所有嵌套的普通 dict 转换为 Features 对象")

