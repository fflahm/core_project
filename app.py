# app.py
import gradio as gr
from faster_whisper import WhisperModel
import logging
import time
import config
from analysis_module import CognitiveAnalyzer

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"--- Loading System (Model: {config.MODEL_SIZE}) ---")

# 1. Load Whisper (Hardware Layer)
try:
    stt_model = WhisperModel(
        config.MODEL_SIZE, 
        device=config.DEVICE, 
        compute_type=config.COMPUTE_TYPE
    )
    logger.info("✅ Whisper Loaded")
except Exception as e:
    logger.critical(f"❌ Whisper Failed: {e}")
    raise e

# 2. Initialize Brain (Logic Layer)
brain = CognitiveAnalyzer()

def pipeline(audio_filepath, chat_history):
    start_total = time.time()
    
    # --- PRE-CHECKS ---
    if chat_history is None: chat_history = []
    if audio_filepath is None: return chat_history, chat_history, None

    logger.info(f"🎤 Audio received: {audio_filepath}")

    # --- STEP 1: TRANSCRIBE ---
    try:
        segments, _ = stt_model.transcribe(audio_filepath, beam_size=config.BEAM_SIZE)
        user_text = " ".join([s.text for s in segments]).strip()
        logger.info(f"📝 Text: {user_text}")
        if not user_text: return chat_history, chat_history, None
    except Exception as e:
        logger.error(f"❌ Transcribe Error: {e}")
        return chat_history, chat_history, None

    # --- STEP 2: ANALYZE (TASK 1) ---
    # This is where we call the new module.
    logger.info("🔍 Analyzing Cognitive Features...")
    analysis_result = brain.analyze_text(user_text)
    is_ruminating, reasoning = brain.detect_rumination(analysis_result)
    
    # Print the structured analysis to the terminal (for debugging/demo)
    logger.info(f"📊 ANALYSIS REPORT:\n"
                f"   ----------------------------------------\n"
                f"   [Feature Extraction]\n"
                f"   - Keywords:       {analysis_result.get('keywords')}\n"
                f"   - Time Orient:    {analysis_result.get('time_orientation')}\n"
                f"   - Abstraction:    {analysis_result.get('abstraction')}\n"
                f"   - Summary:        {analysis_result.get('analysis_summary')}\n"
                f"   ----------------------------------------\n"
                f"   [Rumination Detection]\n"
                f"   - Is Ruminating:  {'🔴 YES' if is_ruminating else '🟢 NO'}\n"
                f"   - Reasoning:      {reasoning}\n"
                f"   ----------------------------------------")
    # --- STEP 3: RESPOND ---
    logger.info("🤖 Generating Response...")
    bot_reply = brain.chat_response(chat_history, user_text, is_ruminating, reasoning)

    # --- UPDATE UI ---
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": bot_reply})
    
    logger.info(f"⏱️ Total Time: {time.time() - start_total:.2f}s")
    return chat_history, chat_history, None

# --- UI LAUNCHER ---
with gr.Blocks(title="Cognitive Mirror") as app:
    gr.Markdown("## 🧠 Cognitive Mirror (Meta-Cognitive Agent)")
    
    chatbot = gr.Chatbot(height=500)
    state = gr.State([]) 
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
        clear_btn = gr.ClearButton([chatbot, state, audio_input])

    audio_input.stop_recording(
        pipeline,
        inputs=[audio_input, state],
        outputs=[chatbot, state, audio_input]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=1111)