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
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- PAGE CONFIG ---
st.set_page_config(page_title="English Ultimate V13 Pro", layout="wide", page_icon="🦁")

# --- CSS STYLING ---
st.markdown("""
<style>
    .big-text { font-size: 1.3rem; line-height: 1.6; font-family: sans-serif; }
    .highlight { background-color: #fff3cd; padding: 2px 5px; border-radius: 4px; }
    .metric-container {
        background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
        padding: 15px; text-align: center; margin-bottom: 10px;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #4b0082; }
    .metric-label { font-size: 0.8rem; text-transform: uppercase; color: #6c757d; letter-spacing: 1px; }
    .feedback-box {
        background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 5px;
    }
    .stress { font-weight: 900; text-decoration: underline; color: #d35400; }
    .pause { color: #c0392b; font-weight: bold; margin: 0 4px; }
</style>
""", unsafe_allow_html=True)

# --- API SETUP ---
# Falls back to environment variable or placeholder if secrets not found
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE"

client = Groq(api_key=GROQ_API_KEY)

# --- SESSION STATE ---
if 'practice_text' not in st.session_state: st.session_state['practice_text'] = ""
if 'coach_script' not in st.session_state: st.session_state['coach_script'] = ""
if 'audio_ref' not in st.session_state: st.session_state['audio_ref'] = None

# --- HELPER FUNCTIONS ---

def generate_text(topic, emotion):
    """Generates the raw practice text."""
    prompt = f"Generate a short (30-40 words) English practice text about: '{topic}'. Tone: {emotion}. OUTPUT RAW TEXT ONLY."
    try:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content.replace('"', '')
    except Exception as e:
        return f"Error: {str(e)}"

def mark_script(text, emotion):
    """Adds stress and pause markers to the text."""
    prompt = f"""
    Act as a Voice Coach. Mark this text for reading with emotion: {emotion}.
    1. Bold **stressed** words.
    2. Add || for pauses.
    Output ONLY the marked text.
    Text: {text}
    """
    try:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content
    except:
        return text

async def text_to_speech(text, gender, emotion):
    """Generates audio file."""
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    # Adjusting params for emotion simulation
    params = {
        "Neutral": {"r": "+0%", "p": "+0Hz"},
        "Happy":   {"r": "+10%", "p": "+5Hz"},
        "Sad":     {"r": "-10%", "p": "-5Hz"},
        "Strict":  {"r": "-5%", "p": "-2Hz"},
    }
    p = params.get(emotion, params["Neutral"])
    
    communicate = edge_tts.Communicate(text, voice, rate=p['r'], pitch=p['p'])
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    await communicate.save(path)
    return path

def sync_tts_gen(text, gender, emotion):
    """Wrapper to run async TTS in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): loop = asyncio.new_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(text_to_speech(text, gender, emotion))

def analyze_audio_physics(file_path, transcript):
    """Uses Librosa to extract non-linguistic features."""
    try:
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration < 0.5: return None
        
        # 1. Speaking Rate (WPM)
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60
        
        # 2. Pitch Standard Deviation (Intonation)
        # Higher std dev = more expressive. Low = monotone.
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_f0 = f0[~np.isnan(f0)]
        pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0
        
        # 3. Energy (Loudness consistency)
        rms = librosa.feature.rms(y=y)
        energy_avg = np.mean(rms)

        return {"wpm": int(wpm), "pitch_std": round(pitch_std, 1), "energy": round(energy_avg * 100, 1)}
    except:
        return None

def get_coach_feedback(target, spoken, metrics, emotion):
    """Generates the AI critique."""
    prompt = f"""
    You are a Strict English Coach.
    Context:
    - Target: "{target}"
    - Spoken: "{spoken}"
    - Tone Goal: {emotion}
    - Speed: {metrics['wpm']} WPM (Ideal: 120-150)
    - Intonation Score: {metrics['pitch_std']} (Low < 15 is Monotone)
    
    Task:
    1. Score /100 based on accuracy and emotion.
    2. Did they skip any words?
    3. Was the emotion correct? (Use the Intonation Score).
    4. Provide ONE specific correction.
    """
    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
    return res.choices[0].message.content

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    gender = st.selectbox("AI Voice", ["Male", "Female"])
    emotion = st.selectbox("Target Emotion", ["Neutral", "Happy", "Sad", "Strict"])
    st.divider()
    
    # MOUTH CAM
    with st.expander("🎥 Mouth Shape Cam"):
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            class MouthProcessor(VideoTransformerBase):
                def __init__(self): self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    res = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if res.multi_face_landmarks:
                        for fl in res.multi_face_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(image=img, landmark_list=fl, connections=mp_face_mesh.FACEMESH_LIPS)
                    return img
            webrtc_streamer(key="mouth", mode=WebRtcMode.SENDRECV, video_processor_factory=MouthProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        except:
            st.warning("Camera unavailable on this server.")

# --- MAIN LAYOUT ---
st.title("🦁 English Ultimate V13")
st.caption("Generate. Perform. Analyze.")

# 1. INPUT SECTION
col_in1, col_in2 = st.columns([3, 1])
with col_in1:
    topic = st.text_input("What do you want to practice?", placeholder="e.g. Asking for a refund, A scary story, Ordering food")
with col_in2:
    if st.button("Generate Script", use_container_width=True):
        if topic:
            with st.spinner("Creating content..."):
                # 1. Generate Text
                raw_text = generate_text(topic, emotion)
                st.session_state['practice_text'] = raw_text
                
                # 2. Generate Guide
                st.session_state['coach_script'] = mark_script(raw_text, emotion)
                
                # 3. Generate Audio
                st.session_state['audio_ref'] = sync_tts_gen(raw_text, gender, emotion)
                st.rerun()

# 2. PRACTICE SECTION
if st.session_state['practice_text']:
    st.divider()
    
    # TABS FOR VIEWING
    tab_script, tab_guide = st.tabs(["📄 Plain Text", "🎭 Acting Guide"])
    
    with tab_script:
        st.markdown(f"<div class='big-text'>{st.session_state['practice_text']}</div>", unsafe_allow_html=True)
        if st.session_state['audio_ref']:
            st.audio(st.session_state['audio_ref'])
            
    with tab_guide:
        st.info("💡 **Coach's Notes:** Emphasize bold words. Pause at || marks.")
        # HTML formatting for the guide
        formatted = st.session_state['coach_script'].replace("**", "<b>").replace("||", "<span class='pause'>||</span>")
        formatted = formatted.replace("<b>", "<span class='stress'>").replace("</b>", "</span>")
        st.markdown(f"<div class='big-text'>{formatted}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # 3. RECORDING & ANALYSIS
    st.markdown("### 🎙️ Record Your Take")
    audio_val = st.audio_input("Microphone")
    
    if audio_val:
        with st.spinner("Analyzing Physics & Linguistics..."):
            # A. Save Temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_val.getvalue())
                tmp_path = tmp.name
            
            # B. Transcribe (Linguistics)
            with open(tmp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(file=(tmp_path, f.read()), model="whisper-large-v3-turbo").text
            
            # C. Analyze (Physics)
            metrics = analyze_audio_physics(tmp_path, transcription)
            os.remove(tmp_path)
            
            if metrics:
                # DISPLAY METRICS
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"<div class='metric-container'><div class='metric-value'>{metrics['wpm']}</div><div class='metric-label'>Words Per Min</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='metric-container'><div class='metric-value'>{metrics['pitch_std']}</div><div class='metric-label'>Intonation Score</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-container'><div class='metric-value'>{metrics['energy']}</div><div class='metric-label'>Energy Level</div></div>", unsafe_allow_html=True)
                
                # DISPLAY AI CRITIQUE
                st.markdown("### 👨‍🏫 Coach Feedback")
                feedback = get_coach_feedback(st.session_state['practice_text'], transcription, metrics, emotion)
                st.markdown(f"<div class='feedback-box'>{feedback}</div>", unsafe_allow_html=True)
                
                # VISUAL DIFF
                with st.expander("Compare Words (Literal vs Spoken)"):
                    st.text(f"Target: {st.session_state['practice_text']}")
                    st.text(f"You Said: {transcription}")
                    
                    # Simple Diff
                    t_words = st.session_state['practice_text'].lower().translate(str.maketrans('', '', string.punctuation)).split()
                    s_words = transcription.lower().translate(str.maketrans('', '', string.punctuation)).split()
                    diff = difflib.SequenceMatcher(None, t_words, s_words)
                    ratio = diff.ratio()
                    st.progress(ratio, text=f"Accuracy Match: {int(ratio*100)}%")
            else:
                st.error("Audio recording was too short or silent. Please try again.")
