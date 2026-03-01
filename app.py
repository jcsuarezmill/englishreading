import streamlit as st
import os
import tempfile
import asyncio
import edge_tts
import difflib
import string
import numpy as np
import eng_to_ipa as ipa
import cv2
import av
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- API SETUP ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE"

client = Groq(api_key=GROQ_API_KEY)

# --- CONSTANTS ---
TEXT_MODEL = "llama-3.1-8b-instant"
AUDIO_MODEL = "whisper-large-v3-turbo"

# --- PAGE CONFIG ---
st.set_page_config(page_title="English Ultimate V8", layout="wide", page_icon="🦁")

# --- CSS STYLING ---
st.markdown("""
<style>
    .feedback-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; }
    .correct { color: #006400; font-weight: bold; }
    .error-box { display: inline-block; background-color: #ffebee; border: 1px solid #ffcdd2; border-radius: 5px; padding: 0px 5px; color: #c0392b; font-weight: bold; }
    .ipa-sub { font-size: 0.7em; color: #7f8c8d; display: block; }
    .chat-bubble-ai { background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: left; color: black; }
    .chat-bubble-user { background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right; color: black; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'current_text' not in st.session_state: st.session_state['current_text'] = ""
if 'vocab_tips' not in st.session_state: st.session_state['vocab_tips'] = ""
if 'roleplay_history' not in st.session_state: st.session_state['roleplay_history'] = []
if 'roleplay_scenario' not in st.session_state: st.session_state['roleplay_scenario'] = None

# --- CORE FUNCTIONS ---
def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

def get_temp_file_path(suffix=".mp3"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

async def generate_emotional_audio(text, gender, emotion):
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    emotions = {
        "Neutral": {"rate": "+0%", "pitch": "+0Hz"},
        "Happy":   {"rate": "+10%", "pitch": "+4Hz"},
        "Sad":     {"rate": "-15%", "pitch": "-5Hz"},
        "Strict":  {"rate": "-5%", "pitch": "-2Hz"},
    }
    e = emotions.get(emotion, emotions["Neutral"])
    output_path = get_temp_file_path(".mp3")
    communicate = edge_tts.Communicate(text, voice, rate=e['rate'], pitch=e['pitch'])
    await communicate.save(output_path)
    return output_path

def ai_generate_coach_feedback(target, spoken):
    prompt = f"Target: '{target}'. User said: '{spoken}'. Identify pronunciation error and explain mouth shape difference (2 sentences)."
    try:
        completion = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
    except: return "Coach unavailable."

def ai_roleplay_response(history):
    try:
        completion = client.chat.completions.create(model=TEXT_MODEL, messages=history, temperature=0.7, max_tokens=150)
        return completion.choices[0].message.content
    except Exception as e: return f"Error: {str(e)}"

def transcribe_audio(audio_file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file_obj.read())
        tmp_name = tmp.name
    with open(tmp_name, "rb") as f:
        trans = client.audio.transcriptions.create(file=(tmp_name, f.read()), model=AUDIO_MODEL)
    os.remove(tmp_name)
    return trans.text

def generate_visual_diff(target, spoken):
    t_clean = clean_text(target).split()
    s_clean = clean_text(spoken).split()
    matcher = difflib.SequenceMatcher(None, t_clean, s_clean)
    html = "<div style='line-height: 2.5; font-size: 1.1em;'>"
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for word in t_clean[i1:i2]: html += f"<span class='correct'>{word}</span> "
        elif tag == 'replace':
            for k in range(i2 - i1):
                t_word = t_clean[i1+k]
                t_ipa = ipa.convert(t_word)
                html += f"<span class='error-box'>{t_word}<span class='ipa-sub'>/{t_ipa}/</span></span> "
        elif tag == 'delete':
             for word in t_clean[i1:i2]: html += f"<span style='text-decoration: line-through; color: orange;'>{word}</span> "
    html += "</div>"
    return html

# --- WEBRTC PROCESSOR ---
# We import MediaPipe INSIDE the class or logic to avoid top-level crashes if it fails
class FaceMeshProcessor(VideoTransformerBase):
    def __init__(self):
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def transform(self, frame):
        import mediapipe as mp # Redundant import but safe
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=img, landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        return img

# --- UI LAYOUT ---
st.title("🦁 English Ultimate V8")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    gender = st.selectbox("Voice", ["Male", "Female"])
    emotion = st.selectbox("Emotion", ["Neutral", "Happy", "Sad", "Strict"])

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📚 Smart Reader", "👅 Tongue Twisters", "🎭 Dynamic Roleplay", "👄 Live Mouth Lab"])

# === TAB 1: READER ===
with tab1:
    st.subheader("Reading Coach")
    mode = st.radio("Source:", ["Write a Topic", "Type My Own"], horizontal=True)
    if mode == "Write a Topic":
        topic = st.text_input("Topic", placeholder="e.g., Ordering coffee")
        if st.button("✨ Generate"):
            with st.spinner("Writing..."):
                prompt = f"Write a 3-sentence story about '{topic}'. Format: STORY ||| Vocab: Def"
                try:
                    c = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
                    content = c.choices[0].message.content
                    if "|||" in content:
                        st.session_state['current_text'], st.session_state['vocab_tips'] = content.split("|||")
                    else:
                        st.session_state['current_text'] = content
                except Exception as e: st.error(str(e))
    else:
        user_txt = st.text_area("Paste text:")
        if st.button("Set Text"): st.session_state['current_text'] = user_txt

    if st.session_state['current_text']:
        st.info(st.session_state['current_text'])
        if st.button("🔈 Listen"):
            path = asyncio.run(generate_emotional_audio(st.session_state['current_text'], gender, emotion))
            st.audio(path)
        audio_in = st.audio_input("Record")
        if audio_in:
            spoken = transcribe_audio(audio_in)
            st.markdown(generate_visual_diff(st.session_state['current_text'], spoken), unsafe_allow_html=True)
            with st.expander("🤖 Coach Feedback"):
                st.write(ai_generate_coach_feedback(st.session_state['current_text'], spoken))

# === TAB 2: TWISTERS ===
with tab2:
    st.subheader("Pronunciation Challenge")
    twister = st.selectbox("Challenge:", ["She sells seashells.", "Peter Piper picked peppers.", "Red lorry, yellow lorry."])
    st.markdown(f"## {twister}")
    if st.button("🔈 Hear Demo", key="twister_btn"):
        path = asyncio.run(generate_emotional_audio(twister, gender, "Strict"))
        st.audio(path)
    t_audio = st.audio_input("Try it")
    if t_audio:
        trans = transcribe_audio(t_audio)
        score = difflib.SequenceMatcher(None, clean_text(twister), clean_text(trans)).ratio()
        if score > 0.9: st.balloons(); st.success(f"Perfect! ({int(score*100)}%)")
        else: st.error(f"Try again. You said: {trans}")

# === TAB 3: ROLEPLAY ===
with tab3:
    st.subheader("Roleplay")
    scenarios = {"Doctor": "You are a doctor.", "Date": "You are on a date.", "Job": "You are a hiring manager."}
    scene = st.selectbox("Scenario", list(scenarios.keys()))
    
    if st.session_state['roleplay_scenario'] != scene:
        st.session_state['roleplay_scenario'] = scene
        st.session_state['roleplay_history'] = [{"role": "system", "content": scenarios[scene]}]
        init_msg = ai_roleplay_response(st.session_state['roleplay_history'])
        st.session_state['roleplay_history'].append({"role": "assistant", "content": init_msg})

    for msg in st.session_state['roleplay_history']:
        if msg['role'] == "assistant": st.markdown(f"<div class='chat-bubble-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
        elif msg['role'] == "user": st.markdown(f"<div class='chat-bubble-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)

    rp_aud = st.audio_input("Reply")
    if rp_aud:
        user_text = transcribe_audio(rp_aud)
        if user_text and (not st.session_state['roleplay_history'] or st.session_state['roleplay_history'][-1]['content'] != user_text):
            st.session_state['roleplay_history'].append({"role": "user", "content": user_text})
            ai_res = ai_roleplay_response(st.session_state['roleplay_history'])
            st.session_state['roleplay_history'].append({"role": "assistant", "content": ai_res})
            asyncio.run(generate_emotional_audio(ai_res, gender, emotion)) # Pre-generate audio
            st.rerun()

    if len(st.session_state['roleplay_history']) > 1 and st.session_state['roleplay_history'][-1]['role'] == "assistant":
        path = asyncio.run(generate_emotional_audio(st.session_state['roleplay_history'][-1]['content'], gender, emotion))
        st.audio(path, autoplay=True)

# === TAB 4: MOUTH LAB ===
with tab4:
    st.subheader("👄 Mirror")
    st.info("Start camera to see lip mesh.")
    webrtc_streamer(key="mouth", video_processor_factory=FaceMeshProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
