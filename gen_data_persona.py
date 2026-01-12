import os
import json
import asyncio
from openai import OpenAI
import edge_tts
from openai import APIConnectionError
import random
import config

OUTPUT_FILE = "data/dataset_v3_persona.json"
AUDIO_DIR = "audio_v3_persona"

os.makedirs(AUDIO_DIR, exist_ok=True)
client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

# ================= Question Pool =================
QUESTION_POOL = {
    "study": {
        "concrete": "最近在学什么具体内容？",
        "abstract": "你对最近的学习状态整体怎么看？"
    },
    "social": {
        "concrete": "最近有没有和谁见面或聊天？",
        "abstract": "你最近对自己的人际关系感觉如何？"
    },
    "self": {
        "concrete": "这两天你的身体或精力状态怎么样？",
        "abstract": "你最近整体的状态给自己打几分？"
    },
    "present": {
        "concrete": "你现在正在做什么？",
        "abstract": "你此刻的心里状态是怎样的？"
    },
    "past": {
        "concrete": "最近有没有发生什么让你印象比较深的事情？",
        "abstract": "你最近常常会回想过去的事情吗？"
    },
    "future": {
        "concrete": "接下来几天你有什么已经确定的安排？",
        "abstract": "你对接下来一段时间最担心的是什么？"
    }
}

# ================= Personas =================
PERSONAS = {
    "rumination_prone": {
        "desc": "容易自责、反复回想、担忧未来",
        "label": 1
    },
    "anxious_but_functional": {
        "desc": "有担忧，但仍能行动",
        "label": 0
    },
    "reflective": {
        "desc": "理性反思，有行动导向",
        "label": 0
    },
    "emotionally_stable": {
        "desc": "情绪稳定，关注现实",
        "label": 0
    }
}

def build_persona_prompt(desc):
    return f"""
你是一个具有如下人格特征的人：
- {desc}

请以自然、真实、口语化的方式回答问题。
不要总结，不要建议。
长度：300–600 字。
"""

# ================= Helpers =================
async def tts(text, path):
    await edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural").save(path)

async def generate(system_prompt, user_prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
                timeout=60
            )
            return r.choices[0].message.content

        except APIConnectionError as e:
            wait = 2 + attempt * 2 + random.random()
            print(f" Connection error, retry {attempt+1}/{max_retries}, sleep {wait:.1f}s")
            await asyncio.sleep(wait)

        except Exception as e:
            print(" Other error:", e)
            break

    return None

# ================= Main =================
async def main():
    dataset = []
    idx = 0

    for persona, meta in PERSONAS.items():
        for domain, qs in QUESTION_POOL.items():
            for qtype, question in qs.items():
                sys_prompt = build_persona_prompt(meta["desc"])
                text = await generate(sys_prompt, question)
                audio = f"{AUDIO_DIR}/{persona}_{idx}.mp3"
                await tts(text, audio)

                dataset.append({
                    "id": f"{persona}_{idx}",
                    "method": "persona",
                    "persona": persona,
                    "gold_label": meta["label"],
                    "domain": domain,
                    "question_type": qtype,
                    "question": question,
                    "text": text,
                    "audio_path": audio
                })
                idx += 1

    json.dump(dataset, open(OUTPUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f" v3 persona finished: {len(dataset)} samples")

if __name__ == "__main__":
    asyncio.run(main())
