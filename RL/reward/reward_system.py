import os
from areal.utils import logging

from dataset.const import AnswerType, ProblemType

from .bbox import bbox_reward_fn
from .bool import bool_reward_fn
from .critic import critic_reward_fn
from .format import must_have_bbox_reward_fn
from .generalcode import general_code_reward_fn
from .htmlcode import html_reward_fn
from .math import math_reward_fn
from .multiple_choice import multiplechoice_reward_fn
from .number import number_reward_fn
from .ocr import ocr_reward_fn
from .string_matching import string_matching_reward_fn
from .svgcode import svg_reward_fn
from .judge_reward import judge_reward_fn

logger = logging.getLogger("Reward System")

# Reward mode configuration
# MODE_A: TYPE_BASED_JUDGE - Use reward based on answer_type, unknown types use judge_reward
# MODE_B: ALL_JUDGE - Use judge model for all samples
# MODE_C: TYPE_BASED_RULE - Use reward based on answer_type, unknown types use rule-based (string_matching)
REWARD_MODE = os.getenv("REWARD_MODE", "TYPE_BASED_JUDGE")

REWARD_FUNCTION_MAPPING = {
    AnswerType.NUMBER: number_reward_fn,
    AnswerType.MATH_EXPRESSIONS: math_reward_fn,
    AnswerType.HTML_CODE: html_reward_fn,
    AnswerType.SVG_CODE: svg_reward_fn,
    AnswerType.BOOLEAN: bool_reward_fn,
    AnswerType.MULTIPLE_CHOICE: multiplechoice_reward_fn,
    AnswerType.OCRTEXT: ocr_reward_fn,
    AnswerType.GENERAL_CODE: general_code_reward_fn,
    AnswerType.BBOX: bbox_reward_fn,
    AnswerType.CRITIC: critic_reward_fn,
    AnswerType.ANY: string_matching_reward_fn,
    AnswerType.JUDGE: judge_reward_fn,
}



class RewardSystem:
    def __init__(self):
        pass

    def reward(
        self,
        prompt: str,
        completions: str,
        answer: str | list[str],
        answer_type: AnswerType = None,
        *args,
        **kwargs
    ):
        # Handle list answers - convert to string
        if isinstance(answer, list):
            if len(answer) == 0:
                answer = ""
            elif len(answer) == 1:
                answer = str(answer[0])
            else:
                # Multiple elements - join with comma
                answer = ", ".join(str(item) for item in answer)

        logger.debug(f"========================\nPrompt: {prompt}\n----------------------\nCompletions: {completions}\n---------------------\nAnswers: {answer}")

        # Select reward function based on REWARD_MODE
        if REWARD_MODE == "ALL_JUDGE":
            # MODE_B: All samples use judge model
            logger.debug("Reward mode: ALL_JUDGE - using judge model for all samples")
            reward_fn = judge_reward_fn
            
        elif REWARD_MODE == "TYPE_BASED_RULE":
            # MODE_C: Use mapped reward based on answer_type, unknown types use string_matching
            if answer_type not in REWARD_FUNCTION_MAPPING:
                logger.warning(f"Unknown answer type: {answer_type}. Using string matching reward as default.")
            reward_fn = REWARD_FUNCTION_MAPPING.get(answer_type, string_matching_reward_fn)
            
        else:  # REWARD_MODE == "TYPE_BASED_JUDGE" (default)
            # MODE_A: Use mapped reward based on answer_type, unknown types use judge_reward
            if answer_type not in REWARD_FUNCTION_MAPPING:
                logger.warning(f"Unknown answer type: {answer_type}. Using judge reward as default.")
            reward_fn = REWARD_FUNCTION_MAPPING.get(answer_type, judge_reward_fn)

        # Call reward function
        if reward_fn == judge_reward_fn:
            # Judge reward needs prompt and other metadata
            scores = reward_fn(completions, answer, prompt=prompt, **kwargs)
        else:
            # Other reward functions only need completions and answer
            scores = reward_fn(completions, answer)

        rewards = {
            "acc_reward": scores,
        }

        if "problem_type" in kwargs and str(kwargs["problem_type"]) == str(ProblemType.SPATIAL_REASONING):
            bbox_reward = must_have_bbox_reward_fn(completions)
            scores = scores + bbox_reward * 0.5
            rewards["bbox_reward"] = bbox_reward
        
        rewards["reward"] = scores

        logger.debug(f"Reward Scores: {rewards}")
        return rewards