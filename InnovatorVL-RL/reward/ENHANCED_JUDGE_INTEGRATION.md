# 增强版Judge模型集成指南

本文档说明如何将新的增强版Judge模型集成到LLaVA-OneVision-1.5的训练系统中。

## 🚀 新特性

### 1. **智能多层评分**
- ✅ **格式分 (5-10%)**：鼓励标准的回答格式
- ✅ **思考分 (15-35%)**：评估推理过程的质量
- ✅ **答案分 (60-80%)**：核心正确性验证

### 2. **灵活的答案提取**
支持10+种答案格式：
```
<answer>4</answer>           # 推荐格式
\boxed{4}                    # LaTeX格式
答案是4                     # 自然语言
(A)                         # 选择题
(1,2,3,4)                   # 元组格式
```

### 3. **prompt类型识别**
自动根据system prompt类型调整评分权重：
- **thinking prompt**：重视思考过程 (思考30%, 答案60%, 格式10%)
- **normal prompt**：重视答案正确性 (思考15%, 答案80%, 格式5%)

### 4. **多层验证机制**
```
答案验证链：
├─ Level 1: 精确字符串匹配
├─ Level 2: 数学表达式等价验证
├─ Level 3: 选择题标准化匹配
├─ Level 4: 语义包含验证
└─ Level 5: 数值提取验证
```

## 📦 集成步骤

### Step 1: 更新配置文件

在训练YAML配置中添加：
```yaml
# configs/llavaov15-8b_stage2_grpo.yaml

reward:
  type: enhanced_judge
  config:
    # 基本配置
    async_pool_size: 4
    timeout: 30.0

    # 评分权重（根据prompt类型）
    scoring_weights:
      thinking_prompt:
        format: 0.10      # 格式分
        thinking: 0.30    # 思考质量
        answer: 0.60      # 答案正确性
      normal_prompt:
        format: 0.05
        thinking: 0.15
        answer: 0.80

    # 验证层级配置
    validation_layers:
      - exact_match
      - math_verify
      - semantic
      - numerical
```

### Step 2: 修改训练脚本

在`trains/grpo.py`中更新：
```python
# 替换原有的import
# from reward.reward_system import RewardSystem
from reward.enhanced_judge_adapter import create_enhanced_judge_reward

# 初始化奖励系统
reward_config = config.get('reward', {}).get('config', {})
reward_system = create_enhanced_judge_reward(
    config=reward_config,
    use_enhanced=config.get('reward', {}).get('type') == 'enhanced_judge'
)

# 启动前初始化
reward_system.start()
```

### Step 3: 确保数据传递prompt类型

修改数据加载逻辑以传递prompt_type：
```python
# 在奖励计算部分
result = reward_system.reward(
    prompt=batch.get("prompt", ""),
    completion=response,
    answer=ground_truth,
    prompt_type=batch.get("prompt_type", "normal"),  # 新增
    answer_type=batch.get("answer_type", "ANY")      # 新增
)
```

### Step 4: 环境变量配置

```bash
# 可选的环境变量
export JUDGE_ASYNC_POOL_SIZE=4
export JUDGE_TIMEOUT=30
export ENABLE_ENHANCED_JUDGE=true
```

## 🎯 使用示例

### 代码中使用
```python
from reward.enhanced_judge_adapter import create_enhanced_judge_reward

# 创建奖励系统
reward = create_enhanced_judge_reward({
    "scoring_weights": {
        "thinking": 0.30,
        "answer": 0.60,
        "format": 0.10
    }
})
reward.start()

# 评估回复
result = reward.reward(
    prompt="计算2+2",
    completion="""
<think>
让我计算2+2：
第一步：看到两个数2和2
第二步：执行加法运算
第三步：得到结果4
</think>
<answer>4</answer>
    "",
    answer="4",
    prompt_type="thinking",
    answer_type="NUMBER"
)

print(f"总分: {result['reward']:.3f}")
print(f"答案分: {result['acc_reward']:.3f}")
print(f"思考分: {result['thinking_reward']:.3f}")
print(f"格式分: {result['format_score']:.3f}")
```

### 监控训练过程
在训练日志中查看详细的评分信息：
```
[Step 100] Reward: 0.850 (格式:1.0, 思考:0.8, 答案:1.0, 方法:exact_match)
[Step 200] Reward: 0.675 (格式:1.0, 思考:0.5, 答案:0.9, 方法:math_verify)
[Step 300] Reward: 0.420 (格式:0.0, 思考:0.4, 答案:0.8, 方法:contained)
```

## 🔍 高级配置

### 1. 自定义验证层
```python
validation_layers = [
    "exact_match",      # 精确匹配
    "math_verify",      # 数学表达式验证
    "choice_normalize", # 选择题标准化
    "semantic_similar", # 语义相似度
    "numerical_approx", # 数值近似匹配
]
```

### 2. 思考质量评估参数
```python
thinking_evaluation = {
    "min_length": 20,
    "max_length": 200,
    "logic_signals": ["首先", "然后", "所以", "step 1", "finally"],
    "math_signals": ["计算", "推导", "检查", "because", "therefore"],
    "relevance_threshold": 0.3
}
```

### 3. 不同任务类型的权重
```python
# 根据任务类型调整权重
task_weights = {
    "multiple_choice": {
        "format": 0.05,
        "thinking": 0.15,
        "answer": 0.80
    },
    "math_expression": {
        "format": 0.10,
        "thinking": 0.25,
        "answer": 0.65
    },
    "spatial_reasoning": {
        "format": 0.15,      # 格式更重要（坐标等）
        "thinking": 0.35,    # 推理过程重要
        "answer": 0.50
    }
}
```

## 📊 性能优化建议

### 1. 批量处理设置
```python
config = {
    "async_pool_size": 4,      # 并行处理数量
    "batch_size": 32,          # 批处理大小
    "timeout": 30.0,           # 单个请求超时
    "max_queue_size": 1000     # 最大队列长度
}
```

### 2. 缓存配置
```python
"use_cache": True,
"cache_size": 5000,     # 最大缓存条目
"cache_ttl": 3600,      # 缓存过期时间（秒）
"cache_key_fields": ["prompt", "completion", "answer"]
```

### 3. 降级机制
```python
"fallback_to_rule": True,
"fallback_conditions": [
    "timeout",          # 超时时降级
    "circuit_breaker",  # 熔断时降级
    "high_error_rate"   # 错误率高时降级
]
```

## 🎬 最后集成到配置文件

完整配置示例：

```yaml
# configs/llavaov15-8b_stage2_enhanced.yaml

experiment_name: stage2-enhanced-judge
trial_name: trial1
seed: 42

# ... 其他配置 ...

reward:
  type: enhanced_judge  # 新增字段
  config:
    async_pool_size: 4
    timeout: 30.0
    default_prompt_type: "normal"

    # 评分权重 - 根据prompt类型自适应
    scoring_weights:
      thinking_prompt:
        format: 0.10
        thinking: 0.35
        answer: 0.55
      normal_prompt:
        format: 0.05
        thinking: 0.15
        answer: 0.80

    # 验证配置
    validation_layers:
      - exact_match
      - math_verify
      - choice_normalize
      - semantic_similar

    # 思考质量评估
    thinking_evaluation:
      min_length: 15
      logic_signals: ["首先", "然后", "所以", "step 1", "since", "because"]
      math_signals: ["计算", "推导", "check", "solve", "compute"]
      relevance_threshold: 0.25

    # 性能优化
    caching:
      enabled: true
      size: 5000
      ttl: 3600

    # 降级策略
    fallback:
      enabled: true
      conditions: ["timeout", "error_rate_high", "judge_unavailable"]

# 数据集配置
train_dataset:
  batch_size: 32
  path: /mnt/innovator/data/wenzichen/mvp-lab/RL-Data/stage2-long
  # 确保数据包含prompt_type字段
  extra_fields: ["prompt_type", "answer_type"]
```

## 🔍 监控与调试

### 检查评分分布
```bash
# 在训练日志中查看
jupyter notebook analysis/reward_analysis.ipynb
```

### 性能指标
- 平均评分：应该在0.3-0.7之间
- 思考分：思考模式应该显著高于普通模式
- 验证方法：监控使用的验证层级
- 降级率：应该低于5%

### 常见问题
1. **评分过低？** 检查权重分配和验证层级
2. **思考分太低？** 调整thinking_evaluation参数
3. **降级频繁？** 检查网络连接和judge模型响应

## ✅ 验证集成成功

运行集成测试：
```python
python -m reward.enhanced_judge_adapter --test
```

输出应该显示：
```
✓ Enhanced judge system initialized
✓ Async pool created (4 workers)
✓ Judged 10 test cases successfully
✓ Average score: 0.812
✓ Thinking vs Normal pattern recognition: OK
```

这样就成功集成了增强版Judge模型，它将提供更智能、更细粒度的奖励评估，特别适合需要推理链的多模态任务训练。""""