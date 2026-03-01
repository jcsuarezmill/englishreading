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
import librosa
import matplotlib.pyplot as plt
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="English Ultimate V10 Pro",
    layout="wide",
    page_icon="🦁",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4B0082; text-align: center; }
    .metric-card {
        background-color: #f0f2f6; border-radius: 10px; padding: 15px;
        text-align: center; border: 1px solid #dcdcdc;
    }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 0.9rem; color: #7f8c8d; }
    .correct { color: #27ae60; font-weight: bold; }
    .incorrect { color: #c0392b; font-weight: bold; text-decoration: line-through; }
    .chat-user {
        background-color: #d1ccc0; padding: 10px; border-radius: 15px 15px 0 15px;
        margin: 5px 0; text-align: right; float: right; clear: both; max-width: 70%; color: black;
    }
    .chat-ai {
        background-color: #dff9fb; padding: 10px; border-radius: 15px 15px 15px 0;
        margin: 5px 0; text-align: left; float: left; clear: both; max-width: 70%; color: black;
    }
</style>
""", unsafe_allow_html=True)

# --- API SETUP ---
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    os.environ["GROQ_API_KEY"] = "gsk_YOUR_API_KEY_HERE" # FALLBACK
    client = Groq()

# --- MEDIA PIPE SETUP (SAFE IMPORT) ---
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    st.error(f"Visual Feedback Warning: MediaPipe failed to load. The Mouth Lab will not work. (Error: {e})")

# --- CORE FUNCTIONS ---

def analyze_prosody(audio_path, transcript):
    """
    Analyzes audio for pacing, pitch, and energy using Librosa (Free).
    """
    try:
        # Load Audio
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 1. Pacing (WPM)
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60
        
        # 2. Pitch (Fundamental Frequency - F0)
        # We use PYIN (Probabilistic YIN) to estimate pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # Filter out silence/unvoiced parts
        pitch_values = f0[~np.isnan(f0)]
        
        if len(pitch_values) > 0:
            avg_pitch = np.mean(pitch_values)
            pitch_std = np.std(pitch_values) # High std = Expressive, Low std = Monotone
        else:
            avg_pitch = 0
            pitch_std = 0
            
        # 3. Energy (Root Mean Square)
        rms = librosa.feature.rms(y=y)
        avg_energy = np.mean(rms)

        return {
            "duration": duration,
            "wpm": round(wpm, 1),
            "pitch_variability": round(pitch_std, 1),
            "avg_pitch": round(avg_pitch, 1),
            "energy_score": round(avg_energy, 4)
        }
    except Exception as e:
        return None

def generate_audio_sync(text, gender, emotion):
    """Sync wrapper for TTS"""
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    emotions = {
        "Neutral": {"rate": "+0%", "pitch": "+0Hz"},
        "Happy":   {"rate": "+10%", "pitch": "+4Hz"},
        "Sad":     {"rate": "-10%", "pitch": "-5Hz"},
        "Strict":  {"rate": "-5%", "pitch": "-2Hz"},
    }
    e = emotions.get(emotion, emotions["Neutral"])
    
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    
    async def _gen():
        communicate = edge_tts.Communicate(text, voice, rate=e['rate'], pitch=e['pitch'])
        await communicate.save(path)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
             asyncio.run(_gen())
        else:
             loop.run_until_complete(_gen())
    except:
        asyncio.run(_gen())
        
    return path

def transcribe_audio(audio_file):
    if audio_file is None: return ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_name = tmp.name
    try:
        with open(tmp_name, "rb") as f:
            trans = client.audio.transcriptions.create(file=(tmp_name, f.read()), model="whisper-large-v3-turbo")
        return trans.text
    finally:
        os.remove(tmp_name)

def get_ipa(text):
    try: return ipa.convert(text)
    except: return text

def visual_diff(target, spoken):
    t_clean = target.translate(str.maketrans('', '', string.punctuation)).lower().split()
    s_clean = spoken.translate(str.maketrans('', '', string.punctuation)).lower().split()
    matcher = difflib.SequenceMatcher(None, t_clean, s_clean)
    html = ""
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            html += f"<span class='correct'>{' '.join(t_clean[i1:i2])}</span> "
        elif tag == 'replace':
            html += f"<span class='incorrect'>{' '.join(t_clean[i1:i2])}</span> "
        elif tag == 'delete':
            html += f"<span style='color:orange; text-decoration:line-through'>{' '.join(t_clean[i1:i2])}</span> "
    return html

# --- WEBRTC PROCESSOR ---
if MEDIAPIPE_AVAILABLE:
    class MouthProcessor(VideoTransformerBase):
        def __init__(self):
            self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=img, landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
            return img
else:
    class MouthProcessor(VideoTransformerBase):
        def transform(self, frame): return frame.to_ndarray(format="bgr24")

# --- UI LAYOUT ---
st.title("🦁 English Ultimate V10 Pro")

with st.sidebar:
    st.header("Settings")
    gender = st.selectbox("Voice", ["Male", "Female"])
    emotion = st.selectbox("Emotion", ["Neutral", "Happy", "Sad", "Strict"])
    st.info("New: Voice Analytics Engine Added (Librosa)")

tab1, tab2, tab3 = st.tabs(["📊 Voice Analytics Lab", "🎭 Roleplay", "👄 Mouth Camera"])

# === TAB 1: ANALYTICS LAB ===
with tab1:
    st.subheader("Deep Voice Analysis")
    st.markdown("Read the text below to analyze your Pacing, Intonation, and Pronunciation.")
    
    target_text = st.text_area("Target Text", "The quick brown fox jumps over the lazy dog. It was a bright cold day in April, and the clocks were striking thirteen.")
    
    if st.button("Listen to Example"):
        path = generate_audio_sync(target_text, gender, emotion)
        st.audio(path)
        
    audio_input = st.audio_input("Record Analysis")
    
    if audio_input:
        with st.spinner("Transcribing & Analyzing Physics of Audio..."):
            # 1. Save to temp for Librosa
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_input.getvalue())
                tmp_path = tmp.name
            
            # 2. Transcribe
            spoken_text = transcribe_audio(audio_input)
            
            # 3. Analyze Physics (Free!)
            metrics = analyze_prosody(tmp_path, spoken_text)
            os.remove(tmp_path)
            
            # --- DISPLAY RESULTS ---
            st.markdown("### 📈 Your Voice Metrics")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{metrics['wpm']}</div><div class='metric-label'>Words Per Min (WPM)</div></div>", unsafe_allow_html=True)
                if metrics['wpm'] < 100: st.warning("Too Slow")
                elif metrics['wpm'] > 160: st.warning("Too Fast")
                else: st.success("Perfect Speed")
                
            with c2:
                # Standard Deviation of Pitch indicates expressiveness
                score = metrics['pitch_variability']
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{score}</div><div class='metric-label'>Intonation Score</div></div>", unsafe_allow_html=True)
                if score < 20: st.warning("Monotone (Robot-like)")
                else: st.success("Expressive")
                
            with c3:
                 st.markdown(f"<div class='metric-card'><div class='metric-value'>{int(metrics['energy_score']*1000)}</div><div class='metric-label'>Energy Level</div></div>", unsafe_allow_html=True)
            
            st.divider()
            st.markdown("### 📝 Text Accuracy")
            st.markdown(visual_diff(target_text, spoken_text), unsafe_allow_html=True)
            st.caption(f"You said: {spoken_text}")

# === TAB 2: ROLEPLAY ===
with tab2:
    if 'rp_history' not in st.session_state: st.session_state['rp_history'] = []
    
    if st.button("Start New Scenario"):
        st.session_state['rp_history'] = [{"role": "system", "content": "You are a barista. Be friendly."}]
        intro = "Hi there! What can I get for you today?"
        st.session_state['rp_history'].append({"role": "assistant", "content": intro})
        path = generate_audio_sync(intro, gender, emotion)
        st.session_state['last_audio'] = path
        st.rerun()
        
    for msg in st.session_state['rp_history']:
        role_class = "chat-user" if msg['role'] == "user" else "chat-ai"
        if msg['role'] != "system":
            st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
            
    st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)
    
    if 'last_audio' in st.session_state:
        st.audio(st.session_state['last_audio'])
        del st.session_state['last_audio'] # Play once
        
    rp_in = st.audio_input("Reply")
    if rp_in:
        txt = transcribe_audio(rp_in)
        if txt:
            st.session_state['rp_history'].append({"role": "user", "content": txt})
            resp = client.chat.completions.create(model="llama-3.1-8b-instant", messages=st.session_state['rp_history']).choices[0].message.content
            st.session_state['rp_history'].append({"role": "assistant", "content": resp})
            path = generate_audio_sync(resp, gender, emotion)
            st.session_state['last_audio'] = path
            st.rerun()

# === TAB 3: MOUTH LAB ===
with tab3:
    if MEDIAPIPE_AVAILABLE:
        st.info("Start the camera to see the lip-sync mesh.")
        webrtc_streamer(key="mouth", mode=WebRtcMode.SENDRECV, video_processor_factory=MouthProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
    else:
        st.error("MediaPipe is unavailable in this environment. Please check packages.txt")
