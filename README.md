# 基于多智能体框架的即时元认知调节系统

## 环境
``` bash
conda create -n chat python=3.10 -y
conda activate chat
conda install -c conda-forge ffmpeg -y
pip install gradio faster-whisper openai edge-tts nest_asyncio scikit-learn numpy requests
```

## 运行
### 1. 配置API
支持OpenAI兼容的API端点，请通过修改`config.py`中的内容来配置：
``` python
API_KEY = "YOUR-API-KEY"
BASE_URL = "YOUR-BASE-URL"
MODEL_NAME = "YOUR-MODEL-NAME"
```


### 2. 启动应用
运行主程序`app.py`。
``` bash
python app.py
```
启动成功后，控制台将输出本地访问地址：
```
Running on local URL: http://0.0.0.0:7860
```
然后请在浏览器中访问 http://localhost:7860 开始交互，页面中有一个`Record`按钮，点击后就可以录音或停止录音，停止录音后音频会被本地的STT模型（这里用的是Whisper）转成文字，并进入pipeline被处理。

### 3.评测模型

<for dxy>

## 项目结构说明
为支持多任务并行开发，项目采用了模块化架构：
```
root/
├── app.py                # [入口] 主程序，负责 UI 渲染与 Pipeline 调度
├── analysis_module.py    # [核心] 业务逻辑层，包含 Prompt 设计与分析算法
├── config.py             # [配置] 全局参数文件 (模型路径、API 设置)
└── README.md             # 项目文档
```

## 用户体验反馈

https://v.wjx.cn/vm/wehMlJL.aspx#