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
st.set_page_config(page_title="English Ultimate V12: Coach Edition", layout="wide", page_icon="🦁")

# --- CSS STYLING ---
st.markdown("""
<style>
    .big-font { font-size: 1.4rem; font-weight: 500; line-height: 1.6; color: #2c3e50; }
    .coach-note { background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; }
    .metric-box {
        background: #f8f9fa; border-radius: 8px; padding: 15px; 
        text-align: center; border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-val { font-size: 2rem; font-weight: 800; color: #4B0082; }
    .metric-lbl { font-size: 0.85rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }
    .stress-word { font-weight: 900; color: #d35400; text-decoration: underline; }
    .pause-mark { color: #e74c3c; font-weight: bold; margin: 0 5px; }
</style>
""", unsafe_allow_html=True)

# --- API SETUP ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    os.environ["GROQ_API_KEY"] = "gsk_YOUR_API_KEY_HERE" # Replace for local
    client = Groq()

# --- SESSION STATE ---
if 'practice_text' not in st.session_state: st.session_state['practice_text'] = ""
if 'coach_notes' not in st.session_state: st.session_state['coach_notes'] = ""
if 'audio_path' not in st.session_state: st.session_state['audio_path'] = None

# --- LOGIC FUNCTIONS ---

def generate_content(prompt_type):
    """Generates the raw text to practice."""
    system_prompt = "You are an English content generator. Output ONLY the raw text to be spoken. No intros."
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_type}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def generate_coach_script(text, emotion):
    """Generates the 'How to Read' guide with stress and pauses."""
    system_prompt = f"""
    You are a Voice Acting Coach. 
    Rewrite the user's text to show them HOW to read it with emotion: '{emotion}'.
    Rules:
    1. Bold the words that need **STRESS**.
    2. Insert '||' where they should PAUSE.
    3. Add (bracketed notes) for tone changes like (whisper), (loudly), (slow down).
    Output only the marked-up text.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Text: {text}"}, {"role": "system", "content": system_prompt}]
        )
        return completion.choices[0].message.content
    except:
        return text

def analyze_physics(audio_path, transcript):
    """Librosa analysis for WPM, Pitch, Energy."""
    try:
        y, sr = librosa.load(audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 1: return None
        
        # 1. WPM
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60
        
        # 2. Pitch (Intonation)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]
        pitch_std = np.std(f0) if len(f0) > 0 else 0
        
        # 3. Energy (Volume/Confidence)
        rms = librosa.feature.rms(y=y)
        energy = np.mean(rms)

        return {
            "wpm": int(wpm),
            "pitch_std": round(pitch_std, 1),
            "energy": round(energy * 1000, 0) # Scaled up for readability
        }
    except:
        return None

def get_ai_critique(target, spoken, metrics, emotion):
    """The 'Real Coach' Feedback based on data."""
    prompt = f"""
    Act as a strict but encouraging Dialect Coach.
    Target Text: "{target}"
    Student Said: "{spoken}"
    Target Emotion: {emotion}
    
    Data:
    - Speed: {metrics['wpm']} WPM (Normal is 130-150)
    - Intonation Score: {metrics['pitch_std']} (Low < 15 is monotone, High > 25 is expressive)
    - Energy: {metrics['energy']} (Low < 10 is too quiet)
    
    Task:
    1. Give a score out of 100.
    2. Analyze their pacing and tone based on the data.
    3. Identify exactly one word they mispronounced or swallowed.
    4. Give 2 specific instructions for the next take.
    """
    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
    return res.choices[0].message.content

async def generate_tts(text, gender, emotion):
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    rates = {"Neutral": "+0%", "Happy": "+10%", "Sad": "-10%", "Strict": "-5%"}
    pitches = {"Neutral": "+0Hz", "Happy": "+5Hz", "Sad": "-5Hz", "Strict": "-2Hz"}
    communicate = edge_tts.Communicate(text, voice, rate=rates.get(emotion, "+0%"), pitch=pitches.get(emotion, "+0Hz"))
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    await communicate.save(path)
    return path

def sync_tts(text, gender, emotion):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): loop = asyncio.new_event_loop()
    except: loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(generate_tts(text, gender, emotion))

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Studio Settings")
    gender = st.selectbox("Coach Voice", ["Male", "Female"])
    emotion = st.selectbox("Target Emotion", ["Neutral", "Happy", "Sad", "Strict"])
    
    st.divider()
    with st.expander("📸 Mouth Lab (Camera)"):
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
        except: st.warning("Camera unavailable.")

# --- MAIN APP ---
st.title("🦁 English Ultimate: The Coach")
st.caption("Generate any text -> Get Direction -> Record -> Get Real Feedback")

# 1. INPUT
col1, col2 = st.columns([3, 1])
with col1:
    user_topic = st.text_input("Generate practice material:", placeholder="e.g. 'Opening lines of a TED Talk', 'Angry customer complaint', 'Romantic poem'")
with col2:
    if st.button("🎬 Action!", use_container_width=True):
        if user_topic:
            with st.spinner("Writing script..."):
                st.session_state['practice_text'] = generate_content(user_topic)
                st.session_state['coach_notes'] = generate_coach_script(st.session_state['practice_text'], emotion)
                st.session_state['audio_path'] = sync_tts(st.session_state['practice_text'], gender, emotion)

# 2. SCRIPT & DIRECTION
if st.session_state['practice_text']:
    st.divider()
    
    # Tabs for different views
    view_tab1, view_tab2 = st.tabs(["📄 Clean Script", "📝 Coach's Markups"])
    
    with view_tab1:
        st.markdown(f"<div class='big-font'>{st.session_state['practice_text']}</div>", unsafe_allow_html=True)
        if st.session_state['audio_path']:
             if st.button("🔈 Hear Ideal Performance"):
                st.audio(st.session_state['audio_path'])
    
    with view_tab2:
        st.info("💡 **Coach's Tip:** Emphasize the **Bold** words. Pause at the || markers.")
        # Apply simple formatting for markdown display of the notes
        formatted_notes = st.session_state['coach_notes'].replace("**", "<b>").replace("||", "<span class='pause-mark'>||</span>")
        formatted_notes = formatted_notes.replace("<b>", "<span class='stress-word'>").replace("</b>", "</span>")
        st.markdown(f"<div class='big-font'>{formatted_notes}</div>", unsafe_allow_html=True)

    # 3. RECORDING
    st.divider()
    st.subheader("🎙️ Your Take")
    audio_input = st.audio_input("Record when ready")

    if audio_input:
        with st.spinner("The Coach is listening..."):
            # Transcribe
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_input.getvalue())
                tmp_path = tmp.name
            
            with open(tmp_path, "rb") as f:
                trans = client.audio.transcriptions.create(file=(tmp_path, f.read()), model="whisper-large-v3-turbo").text
            
            # Physics Analysis
            metrics = analyze_physics(tmp_path, trans)
            os.remove(tmp_path)

            if metrics:
                # 4. RESULTS DASHBOARD
                st.markdown("### 📊 Performance Metrics")
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown(f"<div class='metric-box'><div class='metric-val'>{metrics['wpm']}</div><div class='metric-lbl'>Speed (WPM)</div></div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='metric-box'><div class='metric-val'>{metrics['pitch_std']}</div><div class='metric-lbl'>Intonation (0-50)</div></div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-box'><div class='metric-val'>{metrics['energy']}</div><div class='metric-lbl'>Energy Level</div></div>", unsafe_allow_html=True)

                st.divider()
                
                # 5. THE AI COACH CRITIQUE
                st.markdown("### 👨‍🏫 Coach's Verdict")
                critique = get_ai_critique(st.session_state['practice_text'], trans, metrics, emotion)
                
                # Parse the critique for better display
                st.markdown(f"<div class='coach-note'>{critique}</div>", unsafe_allow_html=True)
                
                # Visual Diff
                with st.expander("🔎 View Word-by-Word Errors"):
                    # Quick Diff Logic
                    target_words = st.session_state['practice_text'].lower().split()
                    spoken_words = trans.lower().split()
                    matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
                    html = []
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                        if tag == 'replace':
                            html.append(f"<span style='color:red;text-decoration:line-through'>{' '.join(target_words[i1:i2])}</span> <span style='color:green'>{' '.join(spoken_words[j1:j2])}</span>")
                        elif tag == 'delete':
                            html.append(f"<span style='color:red;text-decoration:line-through'>{' '.join(target_words[i1:i2])}</span>")
                        elif tag == 'equal':
                            html.append(f"<span style='color:black'>{' '.join(target_words[i1:i2])}</span>")
                    st.markdown(" ".join(html), unsafe_allow_html=True)
            else:
                st.error("Audio too short to analyze. Please speak longer.")
