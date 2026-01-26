"""
适配vLLM部署的OpenAI兼容接口的Judge Model
支持异步调用和批量处理
"""

import asyncio
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger("VLLMJudge")


@dataclass
class VLLMJudgeRequest:
    """请求给vLLM部署的judge model"""
    prompt: str
    completion: str
    answer: Optional[str] = None
    answer_type: Optional[str] = None
    question_type: Optional[str] = None  # 如 'math', 'reasoning', 'coding'
    metadata: Optional[Dict] = None


@dataclass
class VLLMJudgeResponse:
    """vLLM judge model的响应"""
    score: float
    explanation: Optional[str] = None
    detailed_scores: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


class AsyncVLLMJudgeClient:
    """异步vLLM judge model客户端"""

    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get("judge_url", "http://localhost:8000/v1")
        self.model_name = config.get("judge_model", "judge-model")
        self.api_key = config.get("api_key", "dummy-key")  # vLLM可能不需要真正的key
        self.timeout = config.get("timeout", 30.0)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        # 创建异步客户端
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def create_judge_prompt(self, request: VLLMJudgeRequest) -> str:
        """创建给judge model的系统prompt"""

        # 基础system prompt
        base_system = """You are an expert evaluator specializing in {question_type} problems.
Your task is to evaluate the quality of a model's completion based on the provided ground truth answer.

You MUST output a JSON object with the exact following structure:
{
  "score": <float between 0.0 and 1.0>,
  "explanation": "brief explanation of your evaluation",
  "detailed_scores": {
    "format": <0.0-1.0, how well the answer follows expected format>,
    "accuracy": <0.0-1.0, how accurate the answer is>,
    "reasoning": <0.0-1.0, quality of reasoning if applicable>
  },
  "confidence": <0.0-1.0, your confidence in this evaluation>
}

Evaluation criteria:
- Score 1.0: Perfect answer, fully correct and well-formatted
- Score 0.5-0.9: Partially correct or minor formatting issues
- Score 0.0-0.4: Incorrect answer or major errors"""

        # 根据问题类型调整system prompt
        if request.question_type == "math":
            system = base_system.format(question_type="mathematical") + """

Additional math evaluation rules:
1. Consider both final answer AND intermediate steps/calculations
2. Mathematical expressions can be equivalent in different forms (e.g., 1/2 = 0.5)
3. Check if reasoning steps logically lead to the conclusion
4. For computational problems, verify the calculation process"""

        elif request.question_type == "reasoning":
            system = base_system.format(question_type="logical reasoning") + """

Additional reasoning evaluation rules:
1. Evaluate the clarity and logic of the reasoning process
2. Check if conclusions logically follow from premises
3. Consider evidence and justifications provided
4. Assess the coherence and consistency of arguments"""

        elif request.question_type == "coding":
            system = base_system.format(question_type="programming") + """

Additional coding evaluation rules:
1. Check if the code solves the specified problem correctly
2. Evaluate code quality, readability, and efficiency
3. Proper syntax and error handling
4. Correct use of programming concepts"""

        else:
            system = base_system.format(question_type="general")

        return system

    def create_user_prompt(self, request: VLLMJudgeRequest) -> str:
        """创建给judge model的用户prompt"""

        prompt_parts = []

        # 问题描述
        prompt_parts.append(f"Question: {request.prompt}")

        # 标准答案（如果有）
        if request.answer:
            prompt_parts.append(f"Ground Truth Answer: {request.answer}")

        # 答案类型
        if request.answer_type:
            prompt_parts.append(f"Expected Answer Type: {request.answer_type}")

        # 模型回复
        prompt_parts.append(f"Model Completion:\n{request.completion}")

        # 评估要求
        prompt_parts.append("\nEvaluate this completion based on the criteria provided in the system message.")

        return "\n\n".join(prompt_parts)

    async def judge_single(self, request: VLLMJudgeRequest) -> VLLMJudgeResponse:
        """评判单个样本"""

        # 构建messages
        messages = [
            {
                "role": "system",
                "content": self.create_judge_prompt(request)
            },
            {
                "role": "user",
                "content": self.create_user_prompt(request)
            }
        ]

        # 添加retry机制
        for attempt in range(self.max_retries):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,  # 低温度确保一致性
                    max_tokens=500,   # 足够的空间输出JSON
                    response_format={"type": "json_object"}  # 强制JSON格式
                )

                # 解析响应
                content = completion.choices[0].message.content
                response_dict = eval(content)  # 解析JSON

                return VLLMJudgeResponse(
                    score=float(response_dict["score"]),
                    explanation=response_dict.get("explanation", ""),
                    detailed_scores=response_dict.get("detailed_scores"),
                    confidence=float(response_dict.get("confidence", 0.8))
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                else:
                    logger.error(f"All attempts failed for judge request")
                    # 返回默认值
                    return VLLMJudgeResponse(
                        score=0.0,
                        explanation=f"Judge model failed after {self.max_retries} attempts",
                        detailed_scores={"format": 0.0, "accuracy": 0.0, "reasoning": 0.0},
                        confidence=0.0
                    )

    async def judge_batch(self, requests: List[VLLMJudgeRequest]) -> List[VLLMJudgeResponse]:
        """批量评判"""
        # 并发处理所有请求
        tasks = [self.judge_single(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                final_results.append(VLLMJudgeResponse(
                    score=0.0,
                    explanation=f"Error in judge: {str(result)}",
                    detailed_scores={"format": 0.0, "accuracy": 0.0, "reasoning": 0.0},
                    confidence=0.0
                ))
            else:
                final_results.append(result)

        return final_results


# 与现有系统集成的适配器
class VLLMJudgeModelAdapter:
    """适配器，将vLLM Judge适配到现有评分系统"""

    def __init__(self, config: Dict):
        self.config = config
        self.vllm_client = AsyncVLLMJudgeClient(config)

        # 配置评分权重
        self.thinking_weight = config.get("thinking_weight", 0.3)
        self.answer_weight = config.get("answer_weight", 0.6)
        self.format_weight = config.get("format_weight", 0.1)

    async def evaluate(self, prompt: str, completion: str, answer: str,
                      prompt_type: str = "normal", answer_type: str = "ANY") -> Dict[str, float]:
        """评估单个回复"""

        # 确定问题类型
        question_type = self._determine_question_type(prompt, answer_type)

        # 创建请求
        request = VLLMJudgeRequest(
            prompt=prompt,
            completion=completion,
            answer=answer,
            answer_type=answer_type,
            question_type=question_type,
            metadata={"prompt_type": prompt_type}
        )

        # 调用vLLM judge
        response = await self.vllm_client.judge_single(request)

        # 转换结果格式
        return {
            "score": response.score,
            "thinking_score": response.detailed_scores.get("reasoning", 0.0) if response.detailed_scores else 0.0,
            "answer_score": response.detailed_scores.get("accuracy", response.score) if response.detailed_scores else response.score,
            "format_score": response.detailed_scores.get("format", 0.0) if response.detailed_scores else 0.0,
            "explanation": response.explanation,
            "confidence": response.confidence,
            "source": "vllm_judge",
            "validation_method": "vllm_model",
            "prompt_type": prompt_type
        }

    def _determine_question_type(self, prompt: str, answer_type: str) -> str:
        """根据prompt和答案类型确定问题类型"""
        if answer_type in ["NUMBER", "MATH_EXPRESSIONS"]:
            return "math"
        elif answer_type in ["HTML_CODE", "SVG_CODE", "GENERAL_CODE"]:
            return "coding"
        elif "reason" in prompt.lower() or "explain" in prompt.lower():
            return "reasoning"
        else:
            return "general"


# 使用示例
async def main():
    """使用示例"""

    # 配置
    config = {
        "judge_url": "http://localhost:8000/v1",  # 你的vLLM服务地址
        "judge_model": "your-judge-model-name",  # 你的judge模型名称
        "api_key": "dummy-key",  # vLLM兼容模式下可以是dummy key
        "thinking_weight": 0.3,
        "answer_weight": 0.6,
        "format_weight": 0.1,
        "timeout": 30.0,
        "max_retries": 3
    }

    # 创建适配器
    adapter = VLLMJudgeModelAdapter(config)

    # 测试用例
    test_inputs = [
        {
            "prompt": "计算: 156 + 234 = ?",
            "completion": """
<think>
让我计算156 + 234：
156
+234
----
390
</think>
<answer>390</answer>
            """,
            "answer": "390",
            "prompt_type": "thinking",
            "answer_type": "NUMBER"
        },
        {
            "prompt": "长方形面积计算: 长=10cm, 宽=5cm",
            "completion": "<answer>50 square cm</answer>",
            "answer": "50",
            "prompt_type": "normal",
            "answer_type": "NUMBER"
        }
    ]

    # 执行评估
    for i, test_input in enumerate(test_inputs):
        result = await adapter.evaluate(**test_input)

        print(f"\nTest {i+1} ({test_input['prompt_type']}):")
        print(f"Score: {result['score']:.3f}")
        print(f"Thinking: {result['thinking_score']:.3f}")
        print(f"Answer: {result['answer_score']:.3f}")
        print(f"Format: {result['format_score']:.3f}")
        print(f"Explanation: {result['explanation']}")
        print(f"Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())