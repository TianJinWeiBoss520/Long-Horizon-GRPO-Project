import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set
from urllib import error, request

from swift.rewards import ORM, orms


class ToolUseFormatReward(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step if trainer_state is not None else 0
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible

        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2

        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        for response, ans in zip(completions, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward
            rewards.append(reward)
        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    @staticmethod
    def _match_score(list1, list2):
        if list1 == list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        count1 = Counter(list1)
        count2 = Counter(list2)
        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection
        return intersection / max_possible if max_possible > 0 else 0.0

    def _compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self._match_score(gt_names, pd_names)

        local_max_possible = 1.0
        used_pd_indices = set()
        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']
            local_max_possible += 1.0 + len(gt_params)
            best_match_score = 0.0
            best_match_index = -1
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue
                pd_params = pd_tool['parameters']
                param_score = self._match_score(list(gt_params.keys()), list(pd_params.keys()))
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)
                total_score = param_score + correctness_score
                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match_index = i
            if best_match_index >= 0:
                used_pd_indices.add(best_match_index)
                score += best_match_score
        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step if trainer_state is not None else 0
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible

        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1' and global_step < 30:
            max_possible_reward = max_possible_reward / 3
            min_possible_reward = min_possible_reward / 3

        rewards = []
        for response, ans in zip(completions, solution):
            reward = 0.0
            if '<tool_call>' not in ans:
                rewards.append(reward)
                continue
            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = [json.loads(tool) for tool in gt_tool_call.split('\n') if tool.strip()]
            try:
                assert '<tool_call>' in response and '</tool_call>' in response
                pd_tool_call = response.split('<tool_call>')[1].split('</tool_call>')[0].strip()
                pd_tools = [json.loads(tool) for tool in pd_tool_call.split('\n') if tool.strip()]
                reward = self._compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward)
            except Exception:
                reward = min_possible_reward
            rewards.append(reward)
        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward


class ExternalSmokeFormatReward(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>.*$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if m else 0.0 for m in matches]


orms['external_smoke_format_reward'] = ExternalSmokeFormatReward


def _extract_first_tag_block(text: str, tag: str) -> Optional[str]:
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _has_tool_response_in_prompt(prompt: str) -> bool:
    p = prompt or ''
    return ('<tool_response>' in p or '<|im_start|>tool' in p or '"role": "tool"' in p)


def _extract_balanced_segment(text: str, start_idx: int, left: str, right: str) -> Optional[str]:
    if start_idx < 0 or start_idx >= len(text) or text[start_idx] != left:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == left:
            depth += 1
        elif ch == right:
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]
    return None


def _parse_tool_calls_from_block(block: str) -> Optional[List[Dict[str, Any]]]:
    """Parse tool calls in either JSON array or one-JSON-per-line format."""
    if not block:
        return []

    items: List[Any] = []
    block = block.strip()
    try:
        if block.startswith('['):
            parsed = json.loads(block)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = [parsed]
            else:
                return None
        else:
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    except Exception:
        return None

    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        fn = item.get('function') if isinstance(item.get('function'), dict) else {}
        name = fn.get('name') or item.get('name')
        args = fn.get('arguments', item.get('arguments', {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return None
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        if not isinstance(name, str) or not name.strip():
            return None
        normalized.append({'name': name.strip(), 'arguments': args})
    return normalized


def _extract_tool_calls(content: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Extract tool calls from either:
    1) <tool_call>...</tool_call> tagged text
    2) OpenAI-like JSON text that contains "tool_calls": [...]
    3) Direct JSON array/object of function calls

    Return:
    - list: parsed tool calls (possibly empty if no tool call signal)
    - None: tool-call signal exists but parse failed
    """
    text = _to_text(content).strip()
    if not text:
        return []

    tagged = _extract_first_tag_block(text, 'tool_call')
    if tagged is not None:
        return _parse_tool_calls_from_block(tagged)

    has_structured_signal = ('"tool_calls"' in text or ('"function"' in text and '"arguments"' in text))
    if not has_structured_signal:
        return []

    # Case A: wrapper with "tool_calls": [...]
    tc_idx = text.find('"tool_calls"')
    if tc_idx >= 0:
        arr_start = text.find('[', tc_idx)
        if arr_start >= 0:
            arr_block = _extract_balanced_segment(text, arr_start, '[', ']')
            if arr_block is not None:
                parsed = _parse_tool_calls_from_block(arr_block)
                return parsed
        # explicit signal but failed to recover a valid array
        return None

    # Case B: direct JSON-like payload containing function call objects
    if text.startswith('['):
        return _parse_tool_calls_from_block(text)
    if text.startswith('{'):
        return _parse_tool_calls_from_block(text)

    first_arr = text.find('[')
    if first_arr >= 0:
        arr_block = _extract_balanced_segment(text, first_arr, '[', ']')
        if arr_block is not None:
            return _parse_tool_calls_from_block(arr_block)

    return None


def _extract_allowed_tool_names(prompt: str) -> Set[str]:
    """Best-effort extraction of declared tool names from <tools> in prompt."""
    tools_block = _extract_first_tag_block(prompt, 'tools')
    if not tools_block:
        return set()

    names: Set[str] = set()
    for line in tools_block.splitlines():
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            obj = json.loads(line)
            fn = obj.get('function') if isinstance(obj, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get('name'), str):
                names.add(fn['name'])
        except Exception:
            continue

    if names:
        return names
    # Fallback for non-standard formatting.
    return set(re.findall(r'"name"\s*:\s*"([^"]+)"', tools_block))


class AnswerTagReward(ORM):
    """Reward completions that emit final answer in <answer>...</answer>."""

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards: List[float] = []
        for content in completions:
            answer_block = _extract_first_tag_block(content, 'answer')
            if not answer_block:
                rewards.append(0.0)
                continue
            answer_len = len(answer_block.strip())
            # Soft length shaping: avoid empty <answer>.
            if answer_len < 10:
                rewards.append(0.2)
            elif answer_len > 2500:
                rewards.append(0.6)
            else:
                rewards.append(1.0)
        return rewards


orms['external_answer_tag_reward'] = AnswerTagReward


class ToolCallSchemaReward(ORM):
    """Reward structurally valid tool calls with known tool names."""

    def __call__(self, completions, prompts=None, **kwargs) -> List[float]:
        if prompts is None:
            prompts = kwargs.get('prompts')
        if prompts is None:
            prompts = [''] * len(completions)

        rewards: List[float] = []
        for content, prompt in zip(completions, prompts):
            parsed_calls = _extract_tool_calls(content)
            if parsed_calls is None:
                rewards.append(-1.0)
                continue
            if len(parsed_calls) == 0:
                # Neutral for non-tool turns.
                rewards.append(0.0)
                continue

            allowed_tools = _extract_allowed_tool_names(prompt or '')
            valid_name_count = 0
            valid_args_count = 0
            for call in parsed_calls:
                if not allowed_tools or call['name'] in allowed_tools:
                    valid_name_count += 1
                if isinstance(call['arguments'], dict):
                    valid_args_count += 1

            name_score = valid_name_count / len(parsed_calls)
            args_score = valid_args_count / len(parsed_calls)
            # Scale to [-1, 1] to give both reward and penalty.
            score = 2 * (0.6 * name_score + 0.4 * args_score) - 1
            rewards.append(score)
        return rewards


orms['external_toolcall_schema_reward'] = ToolCallSchemaReward


class StageAwareTurnReward(ORM):
    """
    Reward turn-level policy:
    - If prompt already contains <tool_response>, prefer final <answer>.
    - Otherwise, prefer emitting <tool_call>.
    """

    def __call__(self, completions, prompts=None, **kwargs) -> List[float]:
        if prompts is None:
            prompts = kwargs.get('prompts')
        if prompts is None:
            prompts = [''] * len(completions)

        rewards: List[float] = []
        for content, prompt in zip(completions, prompts):
            has_tool_response = _has_tool_response_in_prompt(prompt or '')
            parsed_calls = _extract_tool_calls(content)
            has_tool_call = isinstance(parsed_calls, list) and len(parsed_calls) > 0
            malformed_tool_call = parsed_calls is None
            has_answer = _extract_first_tag_block(content, 'answer') is not None

            if malformed_tool_call:
                rewards.append(-0.8)
                continue

            if has_tool_response:
                if has_answer and not has_tool_call:
                    rewards.append(1.0)
                elif has_answer and has_tool_call:
                    rewards.append(0.6)
                elif has_tool_call:
                    rewards.append(0.2)
                else:
                    rewards.append(-0.5)
            else:
                if has_tool_call and not has_answer:
                    rewards.append(1.0)
                elif has_tool_call and has_answer:
                    rewards.append(0.5)
                elif has_answer:
                    rewards.append(0.2)
                else:
                    rewards.append(-0.5)
        return rewards


orms['external_stage_aware_turn_reward'] = StageAwareTurnReward


class ToolCallEfficiencyReward(ORM):
    """Penalize redundant or overlong tool-call lists in one turn."""

    @staticmethod
    def _call_signature(call: Dict[str, Any]) -> str:
        return json.dumps(
            {'name': call.get('name'), 'arguments': call.get('arguments', {})},
            sort_keys=True,
            ensure_ascii=False)

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards: List[float] = []
        for content in completions:
            parsed_calls = _extract_tool_calls(content)
            if parsed_calls is None:
                rewards.append(-0.5)
                continue
            if len(parsed_calls) == 0:
                rewards.append(0.0)
                continue

            signatures = [self._call_signature(c) for c in parsed_calls]
            unique_cnt = len(set(signatures))
            total_cnt = len(signatures)
            duplicate_ratio = 1.0 - (unique_cnt / total_cnt)

            # Mild penalty for too many calls in one response.
            # In most tool-use settings, 1-3 calls/turn is often enough.
            over_call_penalty = max(0, total_cnt - 3) * 0.15
            score = 1.0 - duplicate_ratio - over_call_penalty
            score = max(-1.0, min(1.0, score))
            rewards.append(score)
        return rewards


orms['external_toolcall_efficiency_reward'] = ToolCallEfficiencyReward


class ProcessStepReward(ORM):
    """
    Intermediate-step reward for multi-turn tool-use.

    Goal:
    - Before tool feedback is available, encourage <think> + valid <tool_call>.
    - After tool feedback appears in prompt, encourage either final <answer>
      or at least a schema-valid next tool call.
    """

    def __call__(self, completions, prompts=None, **kwargs) -> List[float]:
        if prompts is None:
            prompts = kwargs.get('prompts')
        if prompts is None:
            prompts = [''] * len(completions)

        rewards: List[float] = []
        for content, prompt in zip(completions, prompts):
            has_think = _extract_first_tag_block(content, 'think') is not None
            parsed_calls = _extract_tool_calls(content)
            has_tool_call = isinstance(parsed_calls, list) and len(parsed_calls) > 0
            malformed_tool_call = parsed_calls is None
            has_answer = _extract_first_tag_block(content, 'answer') is not None
            has_tool_response = _has_tool_response_in_prompt(prompt or '')

            score = 0.0
            if has_think:
                score += 0.2

            if malformed_tool_call:
                score -= 0.8
            elif has_tool_call:
                score += 0.4
                # small penalty on over-calling in one turn
                score -= max(0, len(parsed_calls) - 3) * 0.1
            else:
                # Without tool call, only acceptable when final answer exists
                if has_answer:
                    score += 0.3
                else:
                    score -= 0.4

            # Stage-aware nudging
            if has_tool_response:
                # Prefer closing with answer after tool feedback
                score += 0.4 if has_answer else -0.2
            else:
                # Early steps usually should call tools first
                score += 0.2 if has_tool_call else -0.2

            rewards.append(max(-1.0, min(1.0, score)))
        return rewards


orms['external_process_step_reward'] = ProcessStepReward


class AnswerLLMJudgeReward(ORM):
    """
    LLM-as-Judge reward for final <answer> quality.

    Env vars:
    - JUDGE_API_KEY: required
    - JUDGE_BASE_URL: default https://api.zhizengzeng.com/v1 
    - JUDGE_MODEL: default deepseek-v4-flash 
    - JUDGE_TIMEOUT_SEC: default 30
    """

    REWARD_SYSTEM_PROMPT = """你是一名深谙旅游行业、具有严谨逻辑与评测方法论的「旅行规划 LLM Agent 综合评审员」。现需对同一用户 Query 下，参考标准答案，对Agent 的推理路径（Path）和回答结果（Answer）分别进行分维度量化评估，请遵循客观公正的原则进行打分。请严格遵循下列指标、打分规则与输出格式。

一、评估内容格式

——————————
<USER_QUERY>
__USER_QUERY__
</USER_QUERY>

<PREDICTION_PATH>
__PREDICTION_PATH__
</PREDICTION_PATH>

<GOLD_PATH>
__GOLD_PATH__
</GOLD_PATH>

<PREDICTION_ANSWER>
__PREDICTION_ANSWER__
</PREDICTION_ANSWER>

<GOLD_ANSWER>
__GOLD_ANSWER__
</GOLD_ANSWER>
——————————

二、推理路径评测（Path Evaluation）

——————————
【评估维度说明】
1. 推理广度（Breadth）：是否从多角度（时间、空间、交通、价格、政策等）全面覆盖问题，同时无冗余或重复步骤。
2. 需求匹配度（Relevance）：各步骤与用户核心需求契合程度。
3. 细节信息丰富度（Detail）：引用的事实、数据、时间点、费用、预约规则等细节是否充分、准确且有用。

【评分规则】
• 推理路径评测时要求只关注推理路径中的实际工具调用，不用关注推理内容对信息的深入分析。
• 每个维度 0–10 分；0 表示“完全缺失”，8分以上是“优秀”，10 表示“极为出色”。
• 推理路径综合得分（Overall_P）＝三个维度均值后四舍五入取整。
——————————

三、回答结果评测（Answer Evaluation）

——————————
【评估维度说明】
1. 匹配度（Relevance）：完整响应所有子需求/限制？顺序与场景贴合？
2. 可行性（Feasibility）：安排逻辑自洽、切实可行，避免明显冲突？
3. 细节丰富度（Details）：时间表、票价、交通耗时、Tips ，天气等信息是否丰富且实用？
4. 清晰度（Clarity）：结构清晰、排版友好、可读性高？

【评分规则】
• 回答结果评测时需参考对应推理路径中的参考知识。
• 每个维度 0–10 分；0 表示“完全缺失”，8分以上是“优秀”，10 表示“极为出色”。
• 回答结果综合得分（Overall_A）＝四个维度均值后四舍五入取整。

【注意事项】
• 参考答案和Agent输出答案所使用的日期可能不同，答案里面关于天气，票价，交通费用等也可能随时间有变化，因此评测时对于具体的数字可以不做为重要评测的角度。
——————————

四、综合得分与胜负判定

——————————
综合得分 combined_scores = 0.6 * Overall_P（路径总体分） + 0.4 * Overall_A（答案总体分），四舍五入保留 1 位小数。
若 Combined 相同，则胜负判定结果为 Tie。
——————————

【输出格式（严格遵循，不要添加多余内容）】
{
  "analysis": {
    "path": "<100 字内中文评述：指出Agent推理路径亮点与不足>",
    "answer": "<100 字内中文评述：指出Agent的答案亮点与不足>"
  },
  "path_scores": {
      "breadth": <0-10>,
      "relevance": <0-10>,
      "detail": <0-10>,
      "overall_p": <0-10>
  },
  "answer_scores": {
      "relevance": <0-10>,
      "feasibility": <0-10>,
      "details": <0-10>,
      "clarity": <0-10>,
      "overall_a": <0-10>
  },
  "combined_scores": <0-10>
}
【重要要求】
• 先逐维度独立思考后再给分，确保公平客观。
• 所有评语仅基于提供的文本，不要引入外部信息。
• 所有中文评述需具体、可溯源（可引用原文片段或段落号）。
• 严格遵守 JSON 模板，以便后续程序解析。

【工具解释】
- visit工具用于访问网页并返回内容摘要。
- search工具用于执行调用google搜索接口，实现通用的、开放知识搜索。
- weather_search工具用于根据城市名称查询指定城市的天气。
- flights_search工具用于根据日期查询从某个城市出发到达某个城市的航班情况。
- train_tickets_search工具用于根据日期查询从某个城市出发到达某个城市的火车票/动车票/高铁票情。
- poi_search工具用于在一个指定的城市内搜索兴趣点（POI）的地理空间信息。
- around_search工具通过设置圆心和半径，搜索圆形区域内的地点信息。
- route_planning工具除提供多种路线规划服务。支持驾车、步行、骑行、电动车、公交路线规划。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.api_key = os.getenv('JUDGE_API_KEY') or ''
        self.base_url = os.getenv('JUDGE_BASE_URL', 'https://api.zhizengzeng.com/v1').rstrip('/')
        self.model = os.getenv('JUDGE_MODEL', 'deepseek-v4-flash')
        self.timeout_sec = int(os.getenv('JUDGE_TIMEOUT_SEC', '30'))

    @staticmethod
    def _extract_user_query_from_prompt(prompt: str) -> str:
        # Extract last user turn in chat template as judge context.
        matches = re.findall(r'<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>', prompt, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return prompt[-1200:].strip()

    @staticmethod
    def _extract_path_and_answer(text: str) -> (str, str):
        """Split full assistant output into path part and final answer part."""
        if not text:
            return '', ''
        answer = _extract_first_tag_block(text, 'answer') or ''
        if answer:
            # Keep everything before first <answer> as prediction/gold path.
            idx = text.find('<answer>')
            path = text[:idx].strip() if idx >= 0 else text.strip()
            return path, answer.strip()
        # Fallback: treat entire content as answer if tag missing.
        return text.strip(), text.strip()

    @staticmethod
    def _safe_slice(text: str, max_len: int = 12000) -> str:
        text = (text or '').strip()
        if len(text) <= max_len:
            return text
        # Keep both head and tail to preserve key context/tool calls.
        head = text[: max_len // 2]
        tail = text[-(max_len - len(head)):]
        return f'{head}\n\n...[TRUNCATED]...\n\n{tail}'

    @staticmethod
    def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
        text = (text or '').strip()
        if not text:
            return None
        # Direct JSON
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        # Markdown fenced json
        fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence:
            try:
                val = json.loads(fence.group(1))
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
        # Loose first {...}
        obj = re.search(r'(\{.*\})', text, re.DOTALL)
        if obj:
            try:
                val = json.loads(obj.group(1))
                if isinstance(val, dict):
                    return val
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_score(raw_score: Any) -> float:
        try:
            val = float(raw_score)
        except Exception:
            return 0.0
        if val > 1.0:
            # Allow 1-10 scale
            val = val / 10.0
        return max(0.0, min(1.0, val))

    def _judge_once(self, user_query: str, prediction_path: str, gold_path: str, prediction_answer: str,
                    gold_answer: str) -> float:
        if not self.api_key:
            return 0.0

        judge_prompt = self.REWARD_SYSTEM_PROMPT
        judge_prompt = judge_prompt.replace('__USER_QUERY__', self._safe_slice(user_query, 3000))
        judge_prompt = judge_prompt.replace('__PREDICTION_PATH__', self._safe_slice(prediction_path, 12000))
        judge_prompt = judge_prompt.replace('__GOLD_PATH__', self._safe_slice(gold_path, 12000))
        judge_prompt = judge_prompt.replace('__PREDICTION_ANSWER__', self._safe_slice(prediction_answer, 5000))
        judge_prompt = judge_prompt.replace('__GOLD_ANSWER__', self._safe_slice(gold_answer, 5000))

        payload = {
            'model': self.model,
            'messages': [{
                'role': 'user',
                'content': judge_prompt
            }],
            'temperature': 0,
            'response_format': {'type': 'json_object'},
        }
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = request.Request(
            url=f'{self.base_url}/chat/completions',
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST')
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode('utf-8'))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return 0.0

        try:
            content = body['choices'][0]['message']['content']
            parsed = self._extract_json_obj(content) if isinstance(content, str) else content
            if not isinstance(parsed, dict):
                return 0.0
            # Prefer new field "combined_scores" (0-10), fallback to legacy "score".
            if 'combined_scores' in parsed:
                return self._normalize_score(parsed.get('combined_scores'))
            return self._normalize_score(parsed.get('score'))
        except Exception:
            return 0.0

    def __call__(self, completions, prompts=None, solution=None, messages=None, **kwargs) -> List[float]:
        if prompts is None:
            prompts = kwargs.get('prompts')
        if prompts is None:
            prompts = [''] * len(completions)
        if solution is None:
            solution = kwargs.get('solution') or [''] * len(completions)
        if messages is None:
            messages = kwargs.get('messages')

        rewards: List[float] = []
        for i, (content, prompt) in enumerate(zip(completions, prompts)):
            # USER_QUERY: prefer prompt parsing, fallback to last user message if available.
            user_query = self._extract_user_query_from_prompt(prompt or '')
            if not user_query and isinstance(messages, list) and i < len(messages):
                msg_list = messages[i]
                if isinstance(msg_list, list):
                    users = [m.get('content', '') for m in msg_list if isinstance(m, dict) and m.get('role') == 'user']
                    user_query = users[-1] if users else ''

            pred_path, pred_answer = self._extract_path_and_answer(content or '')
            gold_raw = solution[i] if isinstance(solution, list) and i < len(solution) else ''
            gold_path, gold_answer = self._extract_path_and_answer(gold_raw or '')
            rewards.append(self._judge_once(user_query, pred_path, gold_path, pred_answer, gold_answer))
        return rewards


orms['external_answer_llm_judge_reward'] = AnswerLLMJudgeReward


def _safe_json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


class CurriculumReward(ORM):
    """
    Stage-wise curriculum reward mixer.

    This combines existing rewards and changes their weights by training progress.
    Configure via env vars (all optional):
    - CURRICULUM_TOTAL_STEPS: int, default 300
    - CURRICULUM_PHASE_RATIOS: JSON list, default [0.3, 0.3, 0.4]
    - CURRICULUM_W1: JSON list(6), default early-stage weights
    - CURRICULUM_W2: JSON list(6), default mid-stage weights
    - CURRICULUM_W3: JSON list(6), default late-stage weights
    Reward order in each weight vector:
    [process_step, answer_tag, tool_schema, stage_aware, tool_efficiency, llm_judge]
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.total_steps = max(1, int(os.getenv('CURRICULUM_TOTAL_STEPS', '300')))
        self.phase_ratios = _safe_json_loads(os.getenv('CURRICULUM_PHASE_RATIOS', '[0.3, 0.3, 0.4]'),
                                             [0.3, 0.3, 0.4])
        if not isinstance(self.phase_ratios, list) or len(self.phase_ratios) != 3:
            self.phase_ratios = [0.3, 0.3, 0.4]
        ratio_sum = sum(float(x) for x in self.phase_ratios) or 1.0
        self.phase_ratios = [float(x) / ratio_sum for x in self.phase_ratios]

        self.w1 = self._normalize_weights(
            _safe_json_loads(os.getenv('CURRICULUM_W1', '[0.35, 0.35, 0.05, 0.20, 0.05, 0.00]'),
                             [0.35, 0.35, 0.05, 0.20, 0.05, 0.00]))
        self.w2 = self._normalize_weights(
            _safe_json_loads(os.getenv('CURRICULUM_W2', '[0.20, 0.25, 0.10, 0.20, 0.10, 0.15]'),
                             [0.20, 0.25, 0.10, 0.20, 0.10, 0.15]))
        self.w3 = self._normalize_weights(
            _safe_json_loads(os.getenv('CURRICULUM_W3', '[0.10, 0.15, 0.10, 0.15, 0.10, 0.40]'),
                             [0.10, 0.15, 0.10, 0.15, 0.10, 0.40]))

        self._process = ProcessStepReward(args)
        self._answer_tag = AnswerTagReward(args)
        self._schema = ToolCallSchemaReward(args)
        self._stage = StageAwareTurnReward(args)
        self._efficiency = ToolCallEfficiencyReward(args)
        self._judge = AnswerLLMJudgeReward(args)

        # Debug trace for each sub-reward to avoid "black box" training.
        self.debug_enabled = str(os.getenv('CURRICULUM_DEBUG_SUBREWARD', '1')) == '1'
        rank = str(os.getenv('RANK', '0'))
        default_log_path = None
        if args is not None and hasattr(args, 'output_dir') and args.output_dir:
            default_log_path = os.path.join(args.output_dir, f'curriculum_subreward_rank{rank}.jsonl')
        else:
            default_log_path = os.path.join(os.getcwd(), f'curriculum_subreward_rank{rank}.jsonl')
        self.debug_log_path = os.getenv('CURRICULUM_DEBUG_LOG_PATH', default_log_path)

    @staticmethod
    def _normalize_weights(weights: Any) -> List[float]:
        if not isinstance(weights, list) or len(weights) != 6:
            weights = [1 / 6] * 6
        vals = [max(0.0, float(x)) for x in weights]
        s = sum(vals) or 1.0
        return [v / s for v in vals]

    def _current_weights(self, global_step: int) -> List[float]:
        p1 = int(self.total_steps * self.phase_ratios[0])
        p2 = int(self.total_steps * (self.phase_ratios[0] + self.phase_ratios[1]))
        if global_step < p1:
            return self.w1
        if global_step < p2:
            return self.w2
        return self.w3

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _write_debug(self, global_step: int, weights: List[float], rewards: Dict[str, List[float]],
                     mixed_rewards: List[float]) -> None:
        if not self.debug_enabled:
            return
        payload = {
            'global_step': int(global_step),
            'weights': [float(x) for x in weights],
            'sub_reward_mean': {k: self._safe_mean(v) for k, v in rewards.items()},
            'mixed_reward_mean': self._safe_mean(mixed_rewards),
        }
        try:
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
            with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception:
            # Never break training because of debug logging.
            pass

    def __call__(self, completions, prompts=None, **kwargs) -> List[float]:
        if prompts is None:
            prompts = kwargs.get('prompts')
        if prompts is None:
            prompts = [''] * len(completions)

        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step if trainer_state is not None else 0
        weights = self._current_weights(global_step)

        r_process = self._process(completions, prompts=prompts, **kwargs)
        r_answer_tag = self._answer_tag(completions, **kwargs)
        r_schema = self._schema(completions, prompts=prompts, **kwargs)
        r_stage = self._stage(completions, prompts=prompts, **kwargs)
        r_efficiency = self._efficiency(completions, **kwargs)
        r_judge = self._judge(completions, prompts=prompts, **kwargs)

        rewards: List[float] = []
        for i in range(len(completions)):
            val = (
                weights[0] * r_process[i] + weights[1] * r_answer_tag[i] + weights[2] * r_schema[i]
                + weights[3] * r_stage[i] + weights[4] * r_efficiency[i] + weights[5] * r_judge[i])
            rewards.append(max(-1.0, min(1.0, val)))
        self._write_debug(
            global_step,
            weights,
            {
                'process_step': r_process,
                'answer_tag': r_answer_tag,
                'tool_schema': r_schema,
                'stage_aware': r_stage,
                'tool_efficiency': r_efficiency,
                'llm_judge': r_judge,
            },
            rewards,
        )
        return rewards


orms['external_curriculum_reward'] = CurriculumReward
