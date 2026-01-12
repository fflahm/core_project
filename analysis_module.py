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
        
    def detect_rumination(self, features):
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
        请基于输入的特征数据，严格返回标准的 JSON 格式，不要包含Markdown标记或其他多余文本
        格式如下：
        {
            "is_ruminating": true/false,
            "reasoning": "简短的一句话理由，指出关键的判据（如：高抽象度+过去时态+自我攻击）"
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
            return data.get("is_ruminating", False), data.get("reasoning", "")
        except Exception as e:
            print(f"Error in detection: {e}")
            return False, "Error"

    def chat_response(self, history, current_text, is_ruminating, reasoning):
        """
        Task 3: Conditional Meta-Cognitive Feedback
        Only intervenes when rumination is detected.
        """

        # --- CASE 1: NO RUMINATION → NATURAL, NON-INTRUSIVE END ---
        if not is_ruminating:
            system_prompt = """
            你是一个温和、尊重边界的 AI 伙伴。
            如果用户的表达未显示出明显的反刍或过度自我关注，
            请用一句简短、自然、不引导反思的回应结束对话，
            避免进行心理分析或干预。
            """

        # --- CASE 2: RUMINATION DETECTED → META-COGNITIVE INTERVENTION ---
        else:
            system_prompt = f"""
            你是一个“元认知引导型 AI 伙伴（Cognitive Mirror）”，
            你的目标不是解决问题，也不是评价用户的想法，
            而是**帮助用户觉察自己的思维过程本身**。

            ### 当前认知判断（仅供你参考，不要直接告诉用户）：
            - 判定为：反刍思维
            - 关键理由：{reasoning}

            ### 你的回应必须遵循以下原则：
            1. **非评判性**：不要说“这是不好的”“你不应该这样想”。
            2. **非建议性**：不要给任何解决方案或行动建议。
            3. **元认知聚焦**：关注“思维模式”，而非“事情本身”。
            4. **镜像式表达**：使用“我注意到……”“我们似乎……”。
            5. **邀请觉察**：用开放式问题邀请用户自我觉察，而不是下结论。
            6. **简短温和**：1–2 句话即可，像一面镜子，而不是一段分析报告。

            ### 推荐句式参考（不要生硬照抄）：
            - “我注意到，你的想法似乎一直在围绕着对自己的怀疑打转。”
            - “我们好像反复回到了同一个问题上，而不是某个具体的情境。”
            - “你也觉察到这种‘停不下来的思考’了吗？”

            请基于用户的原始表达，自然生成一句或两句元认知引导式回应。
            """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": current_text})

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content