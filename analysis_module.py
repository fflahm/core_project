# analysis_module.py
import json
from openai import OpenAI
import config

class CognitiveAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        
    def analyze_text(self, text):
        """
        Task 1: Analyzes text for keywords, tense, and abstraction.
        Returns a Python Dictionary (Structured Data).
        """
        system_prompt = """
        你是一个心理学辅助分析系统，专门用于识别用户的“反刍思维”（Rumination）特征。
        请分析用户的输入，并提取以下三个维度的特征：
        1. 关键词 (keywords): 识别是否存在“为什么”(Why)、“本应该”(Should have)、“总是”(Always) 等反刍常用词。
        2. 时态倾向 (time_orientation): 判断用户关注的是“过去”(Past)、“现在”(Present) 还是“未来”(Future)。
        3. 抽象程度 (abstraction): 判断用户是在描述“具体事件”(Concrete) 还是“抽象烦恼”(Abstract)。
        
        请务必只返回合法的 JSON 格式，不要包含Markdown标记或其他多余文本。
        格式如下：
        {
            "keywords": ["词汇1", "词汇2"],
            "time_orientation": "Past/Present/Future",
            "abstraction": "High/Medium/Low",
            "analysis_summary": "一句话简短分析"
        }
        """

        user_prompt = f"""
        请分析以下用户输入的文本：
        "{text}"
        
        ### 参考示例
        输入: "为什么这种倒霉事总是发生在我身上？我当时要是仔细一点就好了。"
        输出: {{"keywords": ["为什么", "总是", "要是...就好了"], "time_orientation": "Past", "abstraction": "High", "analysis_summary": "用户沉浸在对过去的后悔和抽象的自我归因中。"}}
        
        输入: "我刚才去食堂吃了个饭，但是排队的人有点多。"
        输出: {{"keywords": [], "time_orientation": "Past", "abstraction": "Low", "analysis_summary": "用户在描述具体的日常行为，无明显情绪困扰。"}}
        
        ### Current Input
        输入: "{text}"
        输出:
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, # Force JSON mode if available
                stream=False
            )
            
            # Parse the JSON string into a real Python Dictionary
            result_json = json.loads(response.choices[0].message.content)
            return result_json

        except Exception as e:
            print(f"Analysis Error: {e}")
            # Fallback empty structure to prevent crashes
            return {
                "keywords": [],
                "time_orientation": "Unknown",
                "abstraction": "Unknown",
                "analysis_summary": "Analysis Failed"
            }
        
    def detect_rumination(self, features, threshold=0.5):
        """
        Task 2: Binary Classification based on features.
        """
        system_prompt = """
        你是一名基于认知行为疗法（CBT）理论的心理评估专家。你的任务是根据给定的文本分析特征（JSON），判断用户当前的思维模式是否属于“反刍思维”（Rumination）。

        ### 1. 反刍思维的核心定义
        反刍思维是一种**被动、重复、抽象**地关注自身痛苦及其原因和后果，而缺乏行动解决导向的思维模式。

        ### 2. 判定逻辑（请严格按此优先级判断）
        
        **【符合反刍 (True)】**
        必须同时满足以下至少两点，且无明显行动计划：
        - **高抽象度 (High Abstraction)**：脱离具体情境，上升到性格归因（"我就是个失败者"）或普遍规律（"为什么倒霉的总是我"）。
        - **时态僵化 (Fixated Time)**：沉溺于不可改变的“过去”(Past) 或对“未来”(Future) 的灾难化想象，而非关注“当下”(Present)。
        - **消极循环 (Negative Loop)**：关键词包含绝对化词汇（总是、从未、所有）或无解的“为什么”提问。

        **【不符合反刍 (False)】**
        即使有负面情绪，符合以下任一情况即判定为 False：
        - **具体化叙述 (Concrete)**：用户在描述具体的时间、地点、人物和事件过程（如："刚才吃饭排队被人插队了，我很生气"）。这是正常的情绪宣泄。
        - **解决导向 (Solution-Oriented)**：虽然在分析过去，但目的是总结经验或制定下一步计划（如："下次我会记得提前定闹钟"）。这是建设性反思。
        - **当下状态 (Present Focus)**：描述当下的身体感觉或正在进行的动作。

        ### 3. 输出要求
        请基于输入的特征数据，综合以上判定逻辑，
        给出一个 0 到 1 之间的数值，表示“该用户当前处于反刍思维状态的可能性”。

        数值含义说明：
        - 0.0 表示几乎可以确定不属于反刍思维
        - 1.0 表示几乎可以确定属于反刍思维
        - 中间值表示边界或不确定情况

        请严格返回标准的 JSON 格式，不要包含 Markdown 标记或任何多余文本。
        格式如下：
        {
            "confidence": 0.0-1.0,
            "reasoning": "一句话简要说明判断依据（如：高抽象度 + 过去时态 + 自我攻击）"
        }

        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"特征数据: {json.dumps(features, ensure_ascii=False)}"}
                ],
                response_format={"type": "json_object"},
                stream=False
            )

            data = json.loads(response.choices[0].message.content)

            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "")

            is_ruminating = confidence >= threshold

            return is_ruminating, confidence, reasoning

        except Exception as e:
            print(f"Error in detection: {e}")
            return False, 0.0, "Error"

    def chat_response(self, history, current_text):
        """
        Generates the conversational response (The "Chat" part).
        """
        messages = [{"role": "system", "content": "你是一个温柔、善于倾听的AI伙伴。"}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_text})

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content