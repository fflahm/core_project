import os
import json
import asyncio
from openai import OpenAI
import edge_tts
import config

OUTPUT_FILE = "data/dataset_v1_definition.json"
AUDIO_DIR = "audio_v1"

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

# ================= System Prompts =================
SYSTEM_RUMINATION = """
你正在以【反刍型思维】进行自由表达。

特征：
- 抽象、自我归因、泛化
- 容易反复回想过去或担忧未来
- 思维打转，缺乏明确行动
- 常出现“为什么”“是不是我有问题”“总是这样”

要求：
- 口语化、自然
- 不要总结，不要建议
- 长度：300–600 字
"""

SYSTEM_NON_RUMINATION = """
你正在以【非反刍型思维】进行自由表达。

特征：
- 关注具体经历或当下状态
- 不进行反复自责或人格归因
- 有现实导向或行动感
- 即使有情绪，也不陷入循环

要求：
- 口语化、自然
- 不要总结，不要建议
- 长度：300–600 字
"""

# ================= Helpers =================
async def tts(text, path):
    await edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural").save(path)

async def generate(system_prompt, question):
    r = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        stream=False
    )
    return r.choices[0].message.content.strip()

# ================= Main =================
async def main():
    dataset = []
    idx = 0

    for domain, qs in QUESTION_POOL.items():
        for qtype, question in qs.items():

            # -------- Rumination --------
            text_r = await generate(SYSTEM_RUMINATION, question)
            audio_r = f"{AUDIO_DIR}/rumination_{idx}.mp3"
            await tts(text_r, audio_r)

            dataset.append({
                "id": f"rumination_{idx}",
                "method": "definition",
                "gold_label": 1,
                "domain": domain,
                "question_type": qtype,
                "question": question,
                "text": text_r,
                "audio_path": audio_r
            })
            idx += 1

            # -------- Non-rumination --------
            text_nr = await generate(SYSTEM_NON_RUMINATION, question)
            audio_nr = f"{AUDIO_DIR}/non_rumination_{idx}.mp3"
            await tts(text_nr, audio_nr)

            dataset.append({
                "id": f"non_rumination_{idx}",
                "method": "definition",
                "gold_label": 0,
                "domain": domain,
                "question_type": qtype,
                "question": question,
                "text": text_nr,
                "audio_path": audio_nr
            })
            idx += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"v1 generation finished: {len(dataset)} samples saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
