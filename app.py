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
    # FALLBACK (Not recommended for production)
    GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE" 

client = Groq(api_key=GROQ_API_KEY)

# --- CONSTANTS ---
TEXT_MODEL = "llama-3.1-8b-instant"
AUDIO_MODEL = "whisper-large-v3-turbo"

# --- PAGE CONFIG ---
st.set_page_config(page_title="English Ultimate V8", layout="wide", page_icon="🦁")

# --- MEDIAPIPE SETUP (Global) ---
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- CSS STYLING ---
st.markdown("""
<style>
    .feedback-card {
        background-color: #f0f2f6; border-radius: 10px; padding: 15px;
        border-left: 5px solid #ff4b4b; margin-bottom: 10px;
    }
    .correct { color: #006400; font-weight: bold; }
    .error-box {
        display: inline-block; background-color: #ffebee; border: 1px solid #ffcdd2;
        border-radius: 5px; padding: 0px 5px; color: #c0392b; font-weight: bold;
    }
    .ipa-sub { font-size: 0.7em; color: #7f8c8d; display: block; }
    .chat-bubble-ai {
        background-color: #e1f5fe; padding: 10px; border-radius: 10px;
        margin-bottom: 10px; text-align: left; color: black;
    }
    .chat-bubble-user {
        background-color: #f1f8e9; padding: 10px; border-radius: 10px;
        margin-bottom: 10px; text-align: right; color: black;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'current_text' not in st.session_state: st.session_state['current_text'] = ""
if 'vocab_tips' not in st.session_state: st.session_state['vocab_tips'] = ""
if 'roleplay_history' not in st.session_state: st.session_state['roleplay_history'] = []
if 'roleplay_scenario' not in st.session_state: st.session_state['roleplay_scenario'] = None

# --- CORE FUNCTIONS ---

def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

def get_temp_file_path(suffix=".mp3"):
    """Generates a unique temp file path to prevent user collision."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd) # Close file descriptor immediately, we just need the name
    return path

async def generate_emotional_audio(text, gender, emotion):
    """
    Generates TTS audio and saves to a secure temp file.
    Returns the path to the file.
    """
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    # Emotion Map (Pitch/Rate manipulation)
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
    """Asks Llama to explain the specific pronunciation mistake."""
    prompt = f"""
    The user is learning English. 
    Target Phrase: "{target}"
    User Said: "{spoken}"
    
    Identify the main pronunciation error. Briefly explain the difference in mouth shape or tongue position (max 2 sentences).
    If they are identical, say "Perfect pronunciation."
    """
    try:
        completion = client.chat.completions.create(
            model=TEXT_MODEL, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return completion.choices[0].message.content
    except:
        return "Coach unavailable."

def ai_roleplay_response(history):
    """Generates the next line in the roleplay."""
    try:
        completion = client.chat.completions.create(
            model=TEXT_MODEL, 
            messages=history,
            temperature=0.7,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

def transcribe_audio(audio_file_obj):
    """Transcribes audio file object using Whisper."""
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
            for word in t_clean[i1:i2]:
                html += f"<span class='correct'>{word}</span> "
        elif tag == 'replace':
            for k in range(i2 - i1):
                t_word = t_clean[i1+k]
                t_ipa = ipa.convert(t_word)
                html += f"<span class='error-box'>{t_word}<span class='ipa-sub'>/{t_ipa}/</span></span> "
        elif tag == 'delete':
             for word in t_clean[i1:i2]:
                html += f"<span style='text-decoration: line-through; color: orange;'>{word}</span> "
    html += "</div>"
    return html

# --- WEBRTC PROCESSOR (MOUTH LAB) ---
class FaceMeshProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw lips specifically (using standard facemesh connections)
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
        return img

# --- UI LAYOUT ---
st.title("🦁 English Ultimate V8")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    gender = st.selectbox("Voice", ["Male", "Female"])
    emotion = st.selectbox("Emotion", ["Neutral", "Happy", "Sad", "Strict"])
    st.divider()
    st.info("Upgraded with Real-time Vision & Dynamic Roleplay")

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📚 Smart Reader", "👅 Tongue Twisters", "🎭 Dynamic Roleplay", "👄 Live Mouth Lab"])

# === TAB 1: SMART READER ===
with tab1:
    st.subheader("Reading & Pronunciation Coach")
    mode = st.radio("Source:", ["Write a Topic", "Type My Own Text"], horizontal=True)
    
    # Text Generation
    if mode == "Write a Topic":
        col1, col2 = st.columns([3, 1])
        with col1: topic = st.text_input("Topic", placeholder="e.g., Ordering coffee in Manila")
        with col2: 
            if st.button("✨ Generate", use_container_width=True):
                with st.spinner("Writing story..."):
                    prompt = f"Write a 3-sentence story about '{topic}'. Format: STORY ||| Vocab: Def"
                    try:
                        completion = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
                        content = completion.choices[0].message.content
                        if "|||" in content:
                            parts = content.split("|||")
                            st.session_state['current_text'] = parts[0].strip()
                            st.session_state['vocab_tips'] = parts[1].strip()
                        else:
                            st.session_state['current_text'] = content
                            st.session_state['vocab_tips'] = ""
                    except Exception as e:
                        st.error(str(e))
    else:
        user_txt = st.text_area("Paste text here:")
        if st.button("Set Text"):
            st.session_state['current_text'] = user_txt
            st.session_state['vocab_tips'] = "Custom text."

    # Main Interaction Area
    if st.session_state['current_text']:
        st.divider()
        st.markdown(f"#### 📖 Read this aloud:")
        st.info(st.session_state['current_text'])
        
        # Audio Playback
        if st.button("🔈 Listen to AI"):
            with st.spinner("Generating audio..."):
                audio_path = asyncio.run(generate_emotional_audio(st.session_state['current_text'], gender, emotion))
                st.audio(audio_path)
            
        # Recording & Analysis
        audio_input = st.audio_input("Record your reading")
        if audio_input:
            with st.spinner("Analyzing pronunciation..."):
                spoken_text = transcribe_audio(audio_input)
                
                # Visual Diff
                st.markdown("### 📝 Feedback")
                st.markdown(generate_visual_diff(st.session_state['current_text'], spoken_text), unsafe_allow_html=True)
                
                # AI Coach Feedback
                with st.expander("🤖 AI Pronunciation Coach", expanded=True):
                    coach_msg = ai_generate_coach_feedback(st.session_state['current_text'], spoken_text)
                    st.write(coach_msg)
                
                if st.session_state['vocab_tips']:
                    st.success(f"💡 **Vocab:** {st.session_state['vocab_tips']}")

# === TAB 2: TONGUE TWISTERS ===
with tab2:
    st.subheader("🔥 Challenge Mode")
    twister = st.selectbox("Select Challenge:", [
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
        "Red lorry, yellow lorry.",
        "The thirty-three thieves thought that they thrilled the throne."
    ])
    
    st.markdown(f"## {twister}")
    if st.button("🔈 Hear Demo"):
        path = asyncio.run(generate_emotional_audio(twister, gender, "Strict"))
        st.audio(path)

    t_audio = st.audio_input("Attempt Challenge")
    if t_audio:
        trans_text = transcribe_audio(t_audio)
        score = difflib.SequenceMatcher(None, clean_text(twister), clean_text(trans_text)).ratio()
        
        st.write(f"**You said:** {trans_text}")
        if score > 0.9: 
            st.balloons()
            st.success(f"🏆 Perfect! Score: {int(score*100)}%")
        elif score > 0.7: 
            st.warning(f"Good job! Score: {int(score*100)}%")
        else: 
            st.error(f"Keep practicing. Score: {int(score*100)}%")

# === TAB 3: DYNAMIC ROLEPLAY ===
with tab3:
    st.subheader("🎭 Interactive Roleplay")
    
    scenarios = {
        "Doctor": "You are a doctor. Ask the patient about their symptoms. Be professional.",
        "Job Interview": "You are a hiring manager. Ask the candidate why they want this job. Be strict.",
        "Date": "You are on a first date. Ask your date about their hobbies. Be friendly."
    }
    
    selected_scenario = st.selectbox("Choose Scenario", list(scenarios.keys()))
    
    # Reset history if scenario changes
    if st.session_state['roleplay_scenario'] != selected_scenario:
        st.session_state['roleplay_scenario'] = selected_scenario
        st.session_state['roleplay_history'] = [
            {"role": "system", "content": scenarios[selected_scenario]}
        ]
        # Generate opening line
        initial_msg = ai_roleplay_response(st.session_state['roleplay_history'])
        st.session_state['roleplay_history'].append({"role": "assistant", "content": initial_msg})

    # Display History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state['roleplay_history']:
            if msg['role'] == "assistant":
                st.markdown(f"<div class='chat-bubble-ai'>🤖 <b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)
            elif msg['role'] == "user":
                st.markdown(f"<div class='chat-bubble-user'>👤 <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)

    # User Input
    rp_audio = st.audio_input("Your Reply")
    
    if rp_audio:
        with st.spinner("Listening & Thinking..."):
            # 1. Transcribe
            user_text = transcribe_audio(rp_audio)
            
            # Prevent duplicate submissions of same audio buffer (basic check)
            if user_text and (len(st.session_state['roleplay_history']) == 0 or st.session_state['roleplay_history'][-1]['content'] != user_text):
                
                # 2. Append User Message
                st.session_state['roleplay_history'].append({"role": "user", "content": user_text})
                
                # 3. Get AI Response
                ai_response = ai_roleplay_response(st.session_state['roleplay_history'])
                st.session_state['roleplay_history'].append({"role": "assistant", "content": ai_response})
                
                # 4. Generate Audio for AI Response
                ai_audio_path = asyncio.run(generate_emotional_audio(ai_response, gender, emotion))
                
                st.rerun() # Refresh to show new chat

    # Play latest audio (outside loop to persist)
    if len(st.session_state['roleplay_history']) > 1 and st.session_state['roleplay_history'][-1]['role'] == "assistant":
        last_msg = st.session_state['roleplay_history'][-1]['content']
        # We regenerate audio here to ensure it plays on rerun. 
        # In a full app, we would cache the filename in session_state.
        path = asyncio.run(generate_emotional_audio(last_msg, gender, emotion))
        st.audio(path, autoplay=True)
    
    if st.button("🔄 Reset Roleplay"):
        st.session_state['roleplay_scenario'] = None
        st.rerun()

# === TAB 4: LIVE MOUTH LAB ===
with tab4:
    st.subheader("👄 Real-Time Mirror")
    st.info("Allow camera access. This works best on desktop or Android (Chrome).")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ctx = webrtc_streamer(
            key="mouth-lab",
            video_processor_factory=FaceMeshProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.markdown("""
        **Instructions:**
        1. Click START.
        2. Speak into the camera.
        3. Watch the mesh overlay on your lips.
        
        **Why?**
        English requires specific mouth shapes:
        - **TH**: Tongue between teeth.
        - **R**: Lips rounded, tongue pulled back.
        - **F/V**: Top teeth on bottom lip.
        """)
