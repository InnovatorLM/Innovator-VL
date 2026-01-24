"""
Enhanced Judge Model combining the best practices from both implementations
- Multi-layer validation like the LLM judge
- Two-part scoring for thinking + answer
- Async batch processing for performance
"""

import re
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .utils import extract_boxed_content
from math_verify import parse as math_parse, verify as math_verify, StringExtractionConfig, LatexExtractionConfig, ExprExtractionConfig

logger = logging.getLogger("EnhancedJudge")


@dataclass
class ScoringConfig:
    """Configuration for different scoring weights based on prompt type"""
    format_weight: float = 0.10      # 格式分权重
    thinking_weight: float = 0.25    # 思考质量权重
    answer_weight: float = 0.65      # 答案正确性权重
    enforces_format: bool = True     # 是否强制格式要求
    uses_think_tag: bool = True      # 是否使用思考标签


class AnswerExtractor:
    """Enhanced answer extractor with multiple fallback strategies"""

    def __init__(self):
        # 编译常用的正则表达式
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        self.boxed_pattern = re.compile(r'\boxed{(.*?)}', re.DOTALL)
        self.mcq_pattern = re.compile(r'[A-H]')
        self.number_pattern = re.compile(r'-?\d+(?:\.\d+)?')

    def extract_parts(self, completion: str) -> Tuple[Optional[str], Optional[str]]:
        """提取思考内容和答案内容"""
        thinking = None
        answer = None

        # 1. 优先提取思考部分
        think_match = self.think_pattern.search(completion)
        if think_match:
            thinking = think_match.group(1).strip()

        # 2. 提取答案（多层次提取）
        answer = self.extract_answer(completion)

        return thinking, answer

    def extract_answer(self, completion: str) -> str:
        """多层答案提取（借鉴LLM judge的设计）"""

        # Level 1: <answer>标签（新格式）
        answer_match = self.answer_pattern.search(completion)
        if answer_match:
            return answer_match.group(1).strip()

        # Level 2: oxed{}格式（数学常用）
        boxed_matches = extract_boxed_content(completion)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # Level 3: MCQ格式（选择题）
        mcq_answer = self._extract_mcq_answer(completion)
        if mcq_answer:
            return mcq_answer

        # Level 4: 回答末尾的数字/表达式
        final_answer = self._extract_final_answer(completion)
        if final_answer:
            return final_answer

        # Level 5: 整个回答的最后尝试
        lines = completion.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line:
                # 去掉常见的前缀
                prefixes = ["answer:", "the answer is:", "答案是:", "答案:"]
                for prefix in prefixes:
                    if line.lower().startswith(prefix):
                        return line[len(prefix):].strip()
                return line

        return ""

    def _extract_mcq_answer(self, text: str) -> Optional[str]:
        """从回答中提取选择题答案"""
        # 更智能的选择题答案提取
        answer_phrases = [
            "answer is", "答案是", "the answer", "正确答案是",
            "correct answer is", "选项", "choice is", "答案是"
        ]

        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            # 检查行尾的选择题格式
            for c in "ABCDEFGH":
                patterns = [
                    f" {c}.", f"({c})", f"{c})", f"{c} ",
                    f"**{c}**", f"{c}:**", f"{c}.**"
                ]
                if any(line.endswith(pattern) for pattern in patterns):
                    return c

            # 检查常见答案句式
            for phrase in answer_phrases:
                idx = line.lower().find(phrase)
                if idx != -1:
                    # 在短语后面找选择题字母
                    after_phrase = line[idx + len(phrase):].strip()
                    match = self.mcq_pattern.search(after_phrase)
                    if match:
                        return match.group(0)

        return None

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """从文本末尾提取答案（数字或简单表达式）"""
        lines = text.strip().split('\n')

        for line in reversed(lines[-3:]):  # 只看最后3行
            line = line.strip()
            if not line:
                continue

            # 检查是否是简单数字
            num_match = self.number_pattern.search(line)
            if num_match:
                # 确保这个数字是独立的答案，而不是段落的一部分
                if len(line.split()) <= 5:  # 防止在段落中
                    return num_match.group(0)

            # 检查数学表达式
            if any(op in line for op in ["+=", "-=", "*=", "/=", "=="]):
                parts = line.split("=")
                if len(parts) >= 2:
                    return parts[-1].strip()

        return None


class ThinkingEvaluator:
    """评估思考过程的质量"""

    def evaluate(self, thinking: str, prompt: str) -> Dict[str, float]:
        if not thinking or len(thinking.strip()) < 10:
            return {"overall": 0.0, "details": "no_thinking"}

        scores = {}

        # 1. 长度合理性 (避免过短或过长)
        word_count = len(thinking.split())
        if word_count < 20:
            scores["length"] = 0.3
        elif word_count < 100:
            scores["length"] = 0.8
        elif word_count < 200:
            scores["length"] = 1.0
        else:
            scores["length"] = 0.9  # 过长扣分

        # 2. 逻辑结构检查
        logic_signals = [
            "首先", "第一步", "step 1", "first",
            "然后", "接着", "其次", "next",
            "最后", "因此", "所以", "finally"
        ]
        logic_matches = sum(1 for signal in logic_signals if signal in thinking.lower())
        scores["logic"] = min(logic_matches / 3, 1.0)

        # 3. 与问题的相关性
        prompt_keywords = set(prompt.lower().split()) - set(["the", "a", "is", "what", "how"])
        relevance = sum(1 for word in prompt_keywords if word in thinking.lower()) / max(len(prompt_keywords), 1)
        scores["relevance"] = relevance

        # 4. 数学/推理内容检查
        math_signals = ["计算", "推导", "verify", "check", "因为", "since", "所以", "therefore"]
        math_matches = sum(1 for signal in math_signals if signal in thinking.lower())
        scores["reasoning"] = min(math_matches / 2, 1.0)

        # 5. 综合考虑
        overall = (scores["length"] * 0.2 +
                  scores["logic"] * 0.3 +
                  scores["relevance"] * 0.3 +
                  scores["reasoning"] * 0.2)

        return {
            "overall": overall,
            "details": scores,
            "thinking_summary": thinking[:100] + "..." if len(thinking) > 100 else thinking
        }


class AnswerValidator:
    """答案正确性验证（多层防护）"""

    def validate(self, predicted: str, ground_truth: str, answer_type: str) -> Dict[str, bool]:
        """多层级验证策略"""

        # Level 1: 严格字符串匹配（忽略大小写和空白）
        if predicted.strip().lower() == ground_truth.strip().lower():
            return {"correct": True, "method": "exact_match", "confidence": 1.0}

        # Level 2: 数学验证（对于数学表达式）
        if answer_type in ["MATH_EXPRESSIONS", "NUMBER"]:
            try:
                # 使用math_verify进行数学等价性验证
                pred_parsed = math_parse(predicted, extraction_config=[
                    StringExtractionConfig(),
                    LatexExtractionConfig(),
                    ExprExtractionConfig()
                ])
                truth_parsed = math_parse(ground_truth, extraction_config=[
                    StringExtractionConfig(),
                    LatexExtractionConfig(),
                    ExprExtractionConfig()
                ])

                if math_verify(pred_parsed, truth_parsed):
                    return {"correct": True, "method": "math_verify", "confidence": 0.95}
            except Exception:
                pass

        # Level 3: 选择题验证
        if answer_type == "MULTIPLE_CHOICE":
            # 规范化选择题答案
            pred_letter = self._normalize_choice(predicted)
            truth_letter = self._normalize_choice(ground_truth)

            if pred_letter and truth_letter and pred_letter == truth_letter:
                return {"correct": True, "method": "choice_match", "confidence": 0.9}

        # Level 4: 语义包含验证
        if ground_truth.lower() in predicted.lower() and len(ground_truth) > 1:
            return {"correct": True, "method": "contained", "confidence": 0.7}

        # Level 5: 数值验证（对于数字答案）
        if answer_type in ["NUMBER", "MATH_EXPRESSIONS"]:
            try:
                # 提取并比较数值
                pred_num = float(self._extract_number(predicted))
                truth_num = float(self._extract_number(ground_truth))
                if abs(pred_num - truth_num) < 1e-6:
                    return {"correct": True, "method": "numerical", "confidence": 0.8}
            except ValueError:
                pass

        return {"correct": False, "method": "all_failed", "confidence": 0.0}

    def _normalize_choice(self, text: str) -> Optional[str]:
        """标准化选择题答案"""
        # 清理左右空格和标点
        text = text.strip().strip(".,!?;:")

        # 检查是否是单个字母
        if len(text) == 1 and text.isalpha() and text.upper() in "ABCDEFGH":
            return text.upper()

        # 检查常见格式：(A), A., A: 等
        for c in "ABCDEFGH":
            if c in text.upper():
                # 确保这是主要答案，而不是在解释中的例子
                if text == c or text.count(c) == 1:
                    return c

        return None

    def _extract_number(self, text: str) -> Optional[str]:
        """提取数字"""
        # 使用正则提取数字
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if matches:
            return matches[-1]  # 返回最后一个数字
        return None


class EnhancedJudgeModel:
    """增强版Judge模型"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.extractor = AnswerExtractor()
        self.thinking_evaluator = ThinkingEvaluator()
        self.answer_validator = AnswerValidator()

        # 根据prompt类型配置评分权重
        self.scoring_configs = {
            "thinking": ScoringConfig(       # 需要思考链
                format_weight=0.10,
                thinking_weight=0.35,
                answer_weight=0.55,
                enforces_format=True,
                uses_think_tag=True
            ),
            "normal": ScoringConfig(         # 简单回答
                format_weight=0.05,
                thinking_weight=0.15,
                answer_weight=0.80,
                enforces_format=False,
                uses_think_tag=False
            )
        }
        self.default_config = self.scoring_configs["normal"]

    async def judge_single(
        self,
        prompt: str,
        completion: str,
        answer: str,
        prompt_type: str = "normal",
        answer_type: str = "ANY",
        **kwargs
    ) -> Dict[str, float]:
        """评判单个样本"""

        # 获取配置
        config = self.scoring_configs.get(prompt_type, self.default_config)

        # 1. 提取两部分内容
        thinking_content, answer_content = self.extractor.extract_parts(completion)

        # 2. 格式奖励
        format_score = 0.0
        if config.enforces_format:
            if config.uses_think_tag:
                format_score = 1.0 if (thinking_content is not None and answer_content is not None) else 0.0
            else:
                format_score = 1.0 if answer_content is not None else 0.0
        else:  # 宽松格式
            format_score = 0.8 if answer_content else 0.0

        # 3. 思考质量评估
        thinking_score = 0.0
        if config.uses_think_tag and thinking_content:
            thinking_eval = self.thinking_evaluator.evaluate(thinking_content, prompt)
            thinking_score = thinking_eval["overall"]
        elif not config.uses_think_tag:
            thinking_score = 0.7  # 基础分

        # 4. 答案正确性验证
        validation_result = self.answer_validator.validate(
            answer_content, answer, answer_type
        )
        answer_score = 1.0 if validation_result["correct"] else 0.0

        # 5. 综合评分（固定权重模式）
        total_score = (
            config.format_weight * format_score +
            config.thinking_weight * thinking_score +
            config.answer_weight * answer_score
        )

        return {
            "score": total_score,
            "format_score": format_score,
            "thinking_score": thinking_score,
            "answer_score": answer_score,
            "thinking_content": thinking_content,
            "predicted_answer": answer_content,
            "ground_truth": answer,
            "validation_method": validation_result.get("method", "unknown"),
            "confidence": validation_result.get("confidence", 0.0),
            "prompt_type": prompt_type
        }

    async def judge_batch(
        self,
        batch_requests: List[Dict]
    ) -> List[Dict[str, float]]:
        """批量评判"""
        tasks = [
            self.judge_single(
                prompt=req.get("prompt", ""),
                completion=req.get("completion", ""),
                answer=req.get("answer", ""),
                prompt_type=req.get("prompt_type", "normal"),
                answer_type=req.get("answer_type", "ANY")
            )
            for req in batch_requests
        ]
        return await asyncio.gather(*tasks)

    def get_config(self, prompt_type: str) -> ScoringConfig:
        """获取指定类型的配置"""
        return self.scoring_configs.get(prompt_type, self.default_config)

    def explain_scoring(self, result: Dict[str, float]) -> str:
        """解释评分结果"""
        prompt_type = result.get("prompt_type", "normal")
        config = self.get_config(prompt_type)

        return f"""
评分详情 (类型: {prompt_type}):
- 格式分 ({config.format_weight*100:.0f}%): {result['format_score']:.2f}
  * {'使用' if config.uses_think_tag else '不使用'}思考标签
- 思考分 ({config.thinking_weight*100:.0f}%): {result['thinking_score']:.2f}
- 答案分 ({config.answer_weight*100:.0f}%): {result['answer_score']:.2f}
  * 验证方法: {result.get('validation_method', 'unknown')}
- 总分: {result['score']:.3f}

预测答案: "{result['predicted_answer']}"
标准答案: "{result['ground_truth']}"
        """.strip()


# 工厂函数
async def create_enhanced_judge(config: Optional[Dict] = None) -> EnhancedJudgeModel:
    """创建增强Judge模型实例"""
    return EnhancedJudgeModel(config)


# 服务器端集成示例
class EnhancedJudgeServer:
    """FastAPI服务器集成示例"""

    def __init__(self, judge_model: EnhancedJudgeModel):
        self.judge = judge_model

    async def judge_endpoint(self, request):
        result = await self.judge.judge_single(
            prompt=request.prompt,
            completion=request.completion,
            answer=request.answer,
            prompt_type=request.prompt_type,
            answer_type=request.answer_type
        )
        return result


if __name__ == "__main__":
    import asyncio

    async def test():
        judge = await create_enhanced_judge()

        # 测试思考模式
        test_cases = [
            {
                "prompt": "计算: 2+2=?",
                "completion": """
<think>
让我来计算这个简单的数学题。
第一步：确定运算符号是加法
第二步：将两个数字相加 2 + 2
第三步：得到结果
</think>
<answer>4</answer>
                """,
                "answer": "4",
                "prompt_type": "thinking",
                "answer_type": "NUMBER"
            },
            {
                "prompt": "计算: 3*3=?",
                "completion": "<answer>9</answer>",
                "answer": "9",
                "prompt_type": "normal",
                "answer_type": "NUMBER"
            }
        ]

        for i, test_case in enumerate(test_cases):
            result = await judge.judge_single(**test_case)
            print(f"\nTest Case {i+1} ({test_case['prompt_type']} type):")
            print(f"Total Score: {result['score']:.3f}")
            print(judge.explain_scoring(result))

    asyncio.run(test())