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
st.set_page_config(page_title="English Ultimate V11", layout="wide", page_icon="🦁")

# --- CSS STYLING ---
st.markdown("""
<style>
    .big-font { font-size: 1.2rem; font-weight: 500; line-height: 1.6; }
    .metric-box {
        background: #f8f9fa; border-radius: 8px; padding: 15px; 
        text-align: center; border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #4B0082; }
    .metric-lbl { font-size: 0.9rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }
    .correct { color: #2ecc71; font-weight: bold; }
    .incorrect { color: #e74c3c; font-weight: bold; text-decoration: line-through; }
    .correction { color: #f39c12; font-weight: bold; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# --- API SETUP ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    os.environ["GROQ_API_KEY"] = "gsk_YOUR_API_KEY_HERE" # Replace for local dev
    client = Groq()

# --- SESSION STATE ---
if 'practice_text' not in st.session_state: st.session_state['practice_text'] = ""
if 'audio_path' not in st.session_state: st.session_state['audio_path'] = None

# --- CORE LOGIC ---

def generate_content(prompt_type):
    """Uses Groq to generate ANY kind of English practice material."""
    system_prompt = """
    You are an expert English Coach. 
    Generate text based on the user's request. 
    OUTPUT ONLY THE RAW TEXT TO BE SPOKEN. 
    Do not add "Here is your text" or quotes.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_type}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating text: {e}"

def analyze_voice_quality(audio_path, transcript):
    """Uses Librosa to extract physical voice properties."""
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 1. Speed (WPM)
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60
        
        # 2. Pitch Dynamics (Monotone vs Expressive)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]
        pitch_std = np.std(f0) if len(f0) > 0 else 0
        
        # 3. Pause Ratio (Silence Detection)
        non_silent_intervals = librosa.effects.split(y, top_db=20)
        non_silent_duration = sum(end - start for start, end in non_silent_intervals) / sr
        pause_ratio = 1 - (non_silent_duration / duration)

        return {
            "wpm": int(wpm),
            "pitch_score": round(pitch_std, 1),
            "pause_ratio": round(pause_ratio * 100, 1)
        }
    except:
        return None

async def generate_tts(text, gender, emotion):
    """Async TTS Generation"""
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    # Adjust physics of voice based on emotion
    rates = {"Neutral": "+0%", "Happy": "+10%", "Sad": "-10%", "Strict": "-5%"}
    pitches = {"Neutral": "+0Hz", "Happy": "+5Hz", "Sad": "-5Hz", "Strict": "-2Hz"}
    
    communicate = edge_tts.Communicate(text, voice, rate=rates[emotion], pitch=pitches[emotion])
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    await communicate.save(path)
    return path

def get_audio_sync(text, gender, emotion):
    """Sync wrapper for TTS"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): loop = asyncio.new_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(generate_tts(text, gender, emotion))

def get_visual_diff(target, spoken):
    """Highlights errors in red and corrections in green."""
    t_words = target.translate(str.maketrans('', '', string.punctuation)).lower().split()
    s_words = spoken.translate(str.maketrans('', '', string.punctuation)).lower().split()
    
    matcher = difflib.SequenceMatcher(None, t_words, s_words)
    html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            html.append(f"<span class='correct'>{' '.join(t_words[i1:i2])}</span>")
        elif tag == 'replace':
            html.append(f"<span class='incorrect'>{' '.join(t_words[i1:i2])}</span> <span class='correction'>({' '.join(s_words[j1:j2])})</span>")
        elif tag == 'delete':
            html.append(f"<span class='incorrect'>{' '.join(t_words[i1:i2])}</span>")
        elif tag == 'insert':
            html.append(f"<span class='correction'>({' '.join(s_words[j1:j2])})</span>")
            
    return " ".join(html)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Coach Settings")
    gender = st.selectbox("AI Voice", ["Male", "Female"])
    emotion = st.selectbox("Tone", ["Neutral", "Happy", "Sad", "Strict"])
    st.divider()
    
    # MOUTH LAB (Moved to Sidebar to keep main area clean)
    with st.expander("📸 Mouth Shape Monitor"):
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            
            class MouthProcessor(VideoTransformerBase):
                def __init__(self):
                    self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    res = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if res.multi_face_landmarks:
                        for fl in res.multi_face_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                image=img, landmark_list=fl,
                                connections=mp_face_mesh.FACEMESH_LIPS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                            )
                    return img

            webrtc_streamer(key="mouth", mode=WebRtcMode.SENDRECV, video_processor_factory=MouthProcessor,
                           rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                           media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        except Exception:
            st.warning("Camera unavailable (MediaPipe missing).")

# --- MAIN LAYOUT ---
st.title("🦁 English Ultimate Generator")

# 1. THE GENERATOR (The Main Tool)
st.markdown("### 1️⃣ What do you want to practice?")
col1, col2 = st.columns([4, 1])

with col1:
    user_topic = st.text_input("Enter topic, style, or type:", 
                  placeholder="e.g., 'Hard tongue twister', 'Job interview answer about strengths', 'Story about winter'")

with col2:
    if st.button("🚀 Generate", use_container_width=True):
        with st.spinner("Writing..."):
            st.session_state['practice_text'] = generate_content(user_topic)
            # Pre-generate the audio for the new text
            st.session_state['audio_path'] = get_audio_sync(st.session_state['practice_text'], gender, emotion)

# 2. THE PRACTICE AREA
if st.session_state['practice_text']:
    st.divider()
    
    # Text Display
    st.markdown("### 2️⃣ Read Aloud")
    st.markdown(f"<div class='big-font'>{st.session_state['practice_text']}</div>", unsafe_allow_html=True)
    
    # Audio Player
    if st.session_state['audio_path']:
        if st.button("🔈 Hear Native Pronunciation"):
            st.audio(st.session_state['audio_path'])
    
    st.markdown("### 3️⃣ Analyze Your Voice")
    audio_input = st.audio_input("Record your attempt")
    
    if audio_input:
        with st.spinner("Analyzing Physics of Voice..."):
            # A. Transcribe
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_input.getvalue())
                tmp_path = tmp.name
                
            with open(tmp_path, "rb") as f:
                trans = client.audio.transcriptions.create(file=(tmp_path, f.read()), model="whisper-large-v3-turbo").text
            
            # B. Analyze (Librosa)
            metrics = analyze_voice_quality(tmp_path, trans)
            os.remove(tmp_path)
            
            # C. Display Results
            st.subheader("📊 Voice Analytics Report")
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-box'><div class='metric-val'>{metrics['wpm']}</div><div class='metric-lbl'>Words Per Min</div></div>", unsafe_allow_html=True)
                if metrics['wpm'] < 110: st.caption("🐢 Too Slow")
                elif metrics['wpm'] > 160: st.caption("🐇 Too Fast")
                else: st.caption("✅ Perfect Pace")
            
            with m2:
                # Standard deviation of pitch (higher = more expressive)
                st.markdown(f"<div class='metric-box'><div class='metric-val'>{metrics['pitch_score']}</div><div class='metric-lbl'>Intonation Score</div></div>", unsafe_allow_html=True)
                if metrics['pitch_score'] < 15: st.caption("🤖 Monotone")
                else: st.caption("🗣️ Expressive")

            with m3:
                st.markdown(f"<div class='metric-box'><div class='metric-val'>{metrics['pause_ratio']}%</div><div class='metric-lbl'>Pause Ratio</div></div>", unsafe_allow_html=True)
            
            st.divider()
            
            # Diff View
            st.markdown("#### 📝 Pronunciation Accuracy")
            st.markdown(get_visual_diff(st.session_state['practice_text'], trans), unsafe_allow_html=True)
            
            # AI Feedback
            with st.expander("🤖 Detailed Coach Feedback"):
                feedback_prompt = f"""
                Analyze this English student.
                Target: "{st.session_state['practice_text']}"
                Said: "{trans}"
                WPM: {metrics['wpm']}
                
                Give 3 specific tips to improve. Be direct.
                """
                res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": feedback_prompt}])
                st.write(res.choices[0].message.content)

else:
    st.info("👆 Type a topic above to start practicing!")
