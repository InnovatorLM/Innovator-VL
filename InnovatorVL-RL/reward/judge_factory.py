"""
Judge Factory - 统一入口创建不同类型的Judge模型
支持：本地规则、增强Judge、vLLM部署的Judge
"""

import os
from typing import Dict, Optional
from .reward_system import RewardSystem
from .enhanced_judge_adapter import create_enhanced_judge_reward
from .vllm_judge_adapter import VLLMJudgeModelAdapter


def create_judge(config: Dict, judge_type: str = "auto") -> object:
    """
    创建Judge模型的统一工厂函数

    Args:
        config: 配置字典
        judge_type:
            - "auto": 自动检测
            - "rule": 基于规则的奖励
            - "enhanced": 增强的本地Judge
            - "vllm": vLLM部署的Judge Model

    Returns:
        Judge模型实例
    """
    if judge_type == "auto":
        # 自动检测最佳配置
        if config.get("judge", {}).get("judge_url"):
            judge_type = "vllm"
        elif config.get("reward", {}).get("type") == "enhanced_judge":
            judge_type = "enhanced"
        else:
            judge_type = "rule"

    if judge_type == "vllm":
        # vLLM部署的Judge Model
        vllm_config = {
            "judge_url": config.get("judge", {}).get("judge_url", "http://localhost:8000/v1"),
            "judge_model": config.get("judge", {}).get("model_name", "judge-model"),
            "api_key": config.get("judge", {}).get("api_key", os.getenv("JUDGE_API_KEY", "dummy")),
            "timeout": config.get("judge", {}).get("timeout", 30.0),
            "max_retries": config.get("judge", {}).get("max_retries", 3),
            "thinking_weight": 0.3,
            "answer_weight": 0.6,
            "format_weight": 0.1,
        }
        # 这里只是为了兼容性，实际应该使用完整的集成
        return VLLMJudgeModelAdapter(vllm_config)

    elif judge_type == "enhanced":
        # 增强版本地Judge
        enhanced_config = {
            "async_pool_size": config.get("reward", {}).get("config", {}).get("async_pool_size", 4),
            "timeout": config.get("reward", {}).get("config", {}).get("timeout", 30.0),
            "scoring_weights": config.get("reward", {}).get("config", {}).get("scoring_weights", {
                "thinking_prompt": {"format": 0.10, "thinking": 0.30, "answer": 0.60},
                "normal_prompt": {"format": 0.05, "thinking": 0.15, "answer": 0.80}
            }),
            "validation_layers": config.get("reward", {}).get("config", {}).get("validation_layers", [
                "exact_match", "math_verify", "choice_normalize"
            ])
        }
        return create_enhanced_judge_reward(
            config=enhanced_config,
            use_enhanced=True
        )

    elif judge_type == "rule":
        # 原始的基于规则的奖励系统
        return RewardSystem()

    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


# 配置示例
JUDGE_CONFIG_EXAMPLES = {
    # 1. vLLM部署的Judge Model
    "vllm": {
        "type": "vllm",
        "config": {
            "judge_url": "http://your-vllm-server:8000/v1",
            "model_name": "your-judge-model",  # 你的judge模型名称
            "api_key": "dummy-key",  # vLLM兼容模式
            "timeout": 30.0,
            "max_retries": 3,
            "system_prompt_version": "v1.0"
        }
    },

    # 2. 增强版本地Judge
    "enhanced": {
        "type": "enhanced",
        "config": {
            "async_pool_size": 4,
            "scoring_weights": {
                "thinking_prompt": {"format": 0.10, "thinking": 0.30, "answer": 0.60},
                "normal_prompt": {"format": 0.05, "thinking": 0.15, "answer": 0.80}
            },
            "validation_layers": [
                "exact_match",
                "math_verify",
                "choice_normalize"
            ],
            "thinking_evaluation": {
                "min_length": 15,
                "logic_threshold": 0.3,
                "relevance_threshold": 0.25
            }
        }
    },

    # 3. 传统规则-based
    "rule": {
        "type": "rule",
        "config": {}  # 使用默认配置
    }
}


# 集成到训练配置
TRAINING_CONFIG_TEMPLATE = """
# training_config.yaml

# Judge模型配置
judge:
  type: vllm  # 或 "enhanced", "rule"
  config:
    # vLLM配置
    judge_url: ${JUDGE_MODEL_URL:http://localhost:8000/v1}
    model_name: ${JUDGE_MODEL_NAME:judge-model}
    api_key: ${JUDGE_API_KEY:dummy-key}
    timeout: 30.0
    max_retries: 3
    # 评分权重
    thinking_weight: 0.30
    answer_weight: 0.60
    format_weight: 0.10

# 训练配置
reward:
  type: ${JUDGE_TYPE:enhanced_judge}  # 兼容旧配置
  config:
    # 原有配置...

# 环境变量设置：
# export JUDGE_MODEL_URL=http://your-vllm-server:8000/v1
# export JUDGE_MODEL_NAME=your-judge-model
# export JUDGE_TYPE=vllm
"""


if __name__ == "__main__":
    # 测试工厂函数
    import asyncio

    # 1. 测试vLLM模式
    vllm_config = JUDGE_CONFIG_EXAMPLES["vllm"]["config"]
    vllm_judge = create_judge({"judge": vllm_config}, "vllm")
    print("✓ vLLM judge created")

    # 2. 测试增强模式
    enhanced_config = JUDGE_CONFIG_EXAMPLES["enhanced"]["config"]
    enhanced_judge = create_judge({"reward": {"config": enhanced_config}}, "enhanced")
    print("✓ Enhanced judge created")

    # 3. 测试规则模式
    rule_judge = create_judge({}, "rule")
    print("✓ Rule judge created")

    print("\n🎯 Judge Factory is ready!")
    print("Options: vllm / enhanced / rule / auto")