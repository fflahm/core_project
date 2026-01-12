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
Running on local URL: http://0.0.0.0:1111
```
然后请在浏览器中访问 http://localhost:1111 开始交互，页面中有一个`Record`按钮，点击后就可以录音或停止录音，停止录音后音频会被本地的STT模型转成文字，并进入pipeline被处理。

### 3.评测模型

运行程序`gen_data_defination.py`生成基于反刍思维定义的测试数据。

运行程序`gen_data_persona.py`生成基于人格的测试数据。

运行程序`evaluate.py`评估系统效果。

## 项目结构说明
为支持多任务并行开发，项目采用了模块化架构：
```
root/
├── app.py                          # [入口] 主程序，负责 UI 渲染与 Pipeline 调度
├── analysis_module.py              # [核心] 业务逻辑层，包含 Prompt 设计与分析算法
├── config.py                       # [配置] 全局参数文件 (模型路径、API 设置)
├── gen_data_defination.py          # [数据生成] 基于反刍思维定义的测试数据
├── gen_data_persona.py             # [数据生成] 基于人格的测试数据
├── evaluate.py                     # [评测] 评测模型代码
└── README.md                       # 项目文档
```

## 用户体验反馈

为了在真实的对话场景中评估系统的表现，我们设计了一份调查问卷，见https://v.wjx.cn/vm/wehMlJL.aspx#。