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
import time
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="English Ultimate V9 Pro",
    layout="wide",
    page_icon="🦁",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4B0082; text-align: center; }
    .sub-header { font-size: 1.5rem; color: #333; margin-top: 20px; }
    .feedback-card {
        background-color: #f8f9fa; border-radius: 10px; padding: 20px;
        border-left: 5px solid #6c5ce7; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .correct { color: #27ae60; font-weight: bold; }
    .incorrect { color: #c0392b; font-weight: bold; text-decoration: line-through; }
    .correction { color: #e67e22; font-weight: bold; font-style: italic; }
    .ipa-text { font-family: 'Courier New', monospace; color: #7f8c8d; font-size: 0.9em; }
    .chat-user {
        background-color: #d1ccc0; padding: 15px; border-radius: 15px 15px 0 15px;
        margin: 10px 0; text-align: right; color: #2c3e50; float: right; clear: both; max-width: 70%;
    }
    .chat-ai {
        background-color: #dff9fb; padding: 15px; border-radius: 15px 15px 15px 0;
        margin: 10px 0; text-align: left; color: #2c3e50; float: left; clear: both; max-width: 70%;
    }
    .vocab-item { padding: 5px; border-bottom: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# --- API & SESSION SETUP ---
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    # Development fallback
    os.environ["GROQ_API_KEY"] = "gsk_YOUR_API_KEY_HERE" # Replace if running locally without secrets
    client = Groq()

# Session State Initialization
if 'vocab_list' not in st.session_state: st.session_state['vocab_list'] = []
if 'roleplay_history' not in st.session_state: st.session_state['roleplay_history'] = []
if 'last_audio_path' not in st.session_state: st.session_state['last_audio_path'] = None
if 'autoplay_trigger' not in st.session_state: st.session_state['autoplay_trigger'] = False
if 'quiz_data' not in st.session_state: st.session_state['quiz_data'] = None

# --- CONSTANTS ---
TEXT_MODEL = "llama-3.1-8b-instant"
AUDIO_MODEL = "whisper-large-v3-turbo"

# --- HELPER FUNCTIONS ---

def get_ipa(text):
    """Convert text to IPA phonemes."""
    try:
        return ipa.convert(text)
    except:
        return text

async def edge_tts_generate(text, voice, rate, pitch, output_file):
    """Async generator for TTS."""
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_file)

def generate_audio(text, gender, emotion):
    """
    Wrapper to handle AsyncIO loop for EdgeTTS.
    Returns path to audio file.
    """
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    
    emotions = {
        "Neutral": {"rate": "+0%", "pitch": "+0Hz"},
        "Happy":   {"rate": "+10%", "pitch": "+4Hz"},
        "Sad":     {"rate": "-10%", "pitch": "-5Hz"},
        "Strict":  {"rate": "-5%", "pitch": "-2Hz"},
    }
    e = emotions.get(emotion, emotions["Neutral"])
    
    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    
    # Handle Async Loop safely for Streamlit
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(edge_tts_generate(text, voice, e['rate'], e['pitch'], path))
    except RuntimeError:
        # Fallback for when loop is already running
        asyncio.run(edge_tts_generate(text, voice, e['rate'], e['pitch'], path))
        
    return path

def transcribe_audio(audio_file):
    """Transcribes audio using Groq Whisper."""
    if audio_file is None: return ""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_name = tmp.name
    
    try:
        with open(tmp_name, "rb") as f:
            trans = client.audio.transcriptions.create(
                file=(tmp_name, f.read()), 
                model=AUDIO_MODEL,
                language="en"
            )
        return trans.text
    finally:
        os.remove(tmp_name)

def advanced_diff(target, spoken):
    """Generates HTML diff with IPA comparison."""
    target_clean = target.translate(str.maketrans('', '', string.punctuation)).lower()
    spoken_clean = spoken.translate(str.maketrans('', '', string.punctuation)).lower()
    
    target_words = target_clean.split()
    spoken_words = spoken_clean.split()
    
    matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
    
    html = "<div style='line-height: 2.0; font-size: 1.1em;'>"
    phonetic_errors = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for word in target_words[i1:i2]:
                html += f"<span class='correct'>{word}</span> "
        elif tag == 'replace':
            for k in range(i2 - i1):
                t_word = target_words[i1+k]
                if j1+k < len(spoken_words):
                    s_word = spoken_words[j1+k]
                    t_ipa = get_ipa(t_word)
                    s_ipa = get_ipa(s_word)
                    
                    # Phonetic similarity check
                    if t_ipa == s_ipa:
                        # Sounds same, just spelled different? Mark as warning/ok
                        html += f"<span class='correct' title='Homophone match'>{t_word}</span> "
                    else:
                        html += f"<span class='incorrect' title='You said: {s_word}'>{t_word}</span> "
                        phonetic_errors.append(f"**{t_word}** (/{t_ipa}/) vs **{s_word}** (/{s_ipa}/)")
                else:
                    html += f"<span class='incorrect'>{t_word}</span> "
        elif tag == 'delete':
            for word in target_words[i1:i2]:
                html += f"<span class='incorrect' style='opacity:0.5'>[{word}]</span> "
        elif tag == 'insert':
            for word in spoken_words[j1:j2]:
                html += f"<span class='correction'>({word})</span> "
                
    html += "</div>"
    return html, phonetic_errors

# --- WEBRTC PROCESSOR ---
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MouthProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        img.flags.writeable = True
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw lips
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        return img

# --- MAIN APP UI ---

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lion.png", width=80)
    st.title("Settings")
    
    st.markdown("### 🗣️ Voice Engine")
    gender = st.selectbox("Voice Gender", ["Male", "Female"])
    emotion = st.selectbox("Tone", ["Neutral", "Happy", "Sad", "Strict"])
    
    st.divider()
    
    st.markdown("### 📓 Vocabulary Vault")
    new_word = st.text_input("Add word")
    if st.button("Save Word"):
        if new_word and new_word not in st.session_state['vocab_list']:
            st.session_state['vocab_list'].append(new_word)
            st.success(f"Saved: {new_word}")
    
    if st.session_state['vocab_list']:
        st.write("---")
        for w in st.session_state['vocab_list']:
            st.markdown(f"• **{w}** /{get_ipa(w)}/")
        if st.button("Clear Vault"):
            st.session_state['vocab_list'] = []

# Header
st.markdown("<h1 class='main-header'>🦁 English Ultimate V9 Pro</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Smart Reader", 
    "🎧 Listening Gym", 
    "🔥 Tongue Twisters", 
    "🎭 AI Roleplay", 
    "👄 Mouth Lab"
])

# === TAB 1: SMART READER ===
with tab1:
    st.markdown("### 📝 Analyze Your Pronunciation")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_text = st.text_area("Enter text to practice:", value="The quick brown fox jumps over the lazy dog.", height=100)
    with col2:
        if st.button("✨ Improve Text"):
            with st.spinner("Refining..."):
                prompt = f"Fix grammar and make this text sound more native: '{user_text}'. Return only the fixed text."
                completion = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
                user_text = completion.choices[0].message.content
                st.info("Text refined by AI!")

    # Audio Controls
    if st.button("🔈 Listen to Native Audio"):
        audio_path = generate_audio(user_text, gender, emotion)
        st.audio(audio_path)

    # Recording
    st.markdown("#### 🎙️ Your Turn")
    audio_input = st.audio_input("Record reading the text above")
    
    if audio_input:
        with st.spinner("Analyzing audio..."):
            spoken = transcribe_audio(audio_input)
            
            # Visual Diff
            diff_html, phonetics = advanced_diff(user_text, spoken)
            
            st.markdown("### Result:")
            st.markdown(diff_html, unsafe_allow_html=True)
            
            if phonetics:
                with st.expander("🧐 Phonetic Deep Dive"):
                    st.write("Words you might have mispronounced:")
                    for p in phonetics:
                        st.markdown(f"- {p}")
            
            # AI Coach
            with st.expander("🤖 Coach Feedback", expanded=True):
                prompt = f"User target: '{user_text}'. User said: '{spoken}'. Give 3 bullet points on how to improve pronunciation specifically."
                res = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
                st.write(res.choices[0].message.content)

# === TAB 2: LISTENING GYM ===
with tab2:
    st.header("🎧 Native Ear Training")
    st.write("Listen to the story WITHOUT seeing the text, then answer the question.")
    
    if st.button("🎲 Generate New Quiz"):
        with st.spinner("Creating quiz..."):
            prompt = """Generate a short 2-sentence story followed by a multiple choice question about it. 
            Format: STORY|QUESTION|OPTION_A|OPTION_B|OPTION_C|CORRECT_ANSWER_LETTER"""
            res = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
            parts = res.choices[0].message.content.split("|")
            
            if len(parts) >= 6:
                st.session_state['quiz_data'] = {
                    "story": parts[0],
                    "q": parts[1],
                    "options": [parts[2], parts[3], parts[4]],
                    "answer": parts[5].strip()
                }
                # Generate audio immediately
                st.session_state['quiz_audio'] = generate_audio(parts[0], gender, emotion)
            else:
                st.error("AI Generation failed. Try again.")

    if st.session_state['quiz_data']:
        st.audio(st.session_state['quiz_audio'])
        
        with st.form("quiz_form"):
            st.write(f"**Question:** {st.session_state['quiz_data']['q']}")
            choice = st.radio("Choose:", st.session_state['quiz_data']['options'])
            submitted = st.form_submit_button("Check Answer")
            
            if submitted:
                # Simple logic to check answer (assuming options correspond to A, B, C order)
                correct_opt = st.session_state['quiz_data']['options'][
                    ord(st.session_state['quiz_data']['answer'].upper()) - 65
                ]
                
                if choice == correct_opt:
                    st.balloons()
                    st.success("Correct!")
                else:
                    st.error(f"Wrong. The correct answer was: {correct_opt}")
                
                with st.expander("Show Transcript"):
                    st.write(st.session_state['quiz_data']['story'])

# === TAB 3: TONGUE TWISTERS ===
with tab3:
    st.header("🔥 Speed & Diction Challenge")
    
    twisters = {
        "Beginner": "Fresh fried fish, fish fresh fried, fried fish fresh, fish fried fresh.",
        "Intermediate": "The thirty-three thieves thought that they thrilled the throne throughout Thursday.",
        "Expert": "Pad kid poured curd pulled cod."
    }
    
    level = st.select_slider("Difficulty", options=["Beginner", "Intermediate", "Expert"])
    target_twister = twisters[level]
    
    st.markdown(f"### {target_twister}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Hear it (Fast)"):
            path = generate_audio(target_twister, gender, "Strict") # Strict is usually faster/crisper
            st.audio(path)
    
    tt_input = st.audio_input("Record your attempt")
    
    if tt_input:
        spoken_tt = transcribe_audio(tt_input)
        st.write(f"**You said:** {spoken_tt}")
        
        ratio = difflib.SequenceMatcher(None, target_twister.lower(), spoken_tt.lower()).ratio()
        if ratio > 0.9:
            st.success(f"🏆 Amazing! Accuracy: {int(ratio*100)}%")
        elif ratio > 0.7:
            st.warning(f"Good! Accuracy: {int(ratio*100)}%")
        else:
            st.error(f"Keep trying! Accuracy: {int(ratio*100)}%")

# === TAB 4: AI ROLEPLAY ===
with tab4:
    st.header("🎭 Real-time Conversation")
    
    # Check for Autoplay Trigger
    if st.session_state.get('autoplay_trigger') and st.session_state.get('last_audio_path'):
        st.audio(st.session_state['last_audio_path'], autoplay=True)
        st.session_state['autoplay_trigger'] = False # Reset trigger

    scenarios = ["Doctor Visit", "Job Interview", "Ordering Coffee", "Immigration Officer"]
    scenario = st.selectbox("Choose Scenario", scenarios)
    
    if st.button("🔄 Reset / Start New Chat"):
        st.session_state['roleplay_history'] = [
            {"role": "system", "content": f"Roleplay scenario: {scenario}. You are the AI character. Keep responses concise (under 30 words)."}
        ]
        # Initial greeting
        start_prompt = f"Start a roleplay about {scenario}."
        completion = client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "user", "content": start_prompt}])
        greeting = completion.choices[0].message.content
        st.session_state['roleplay_history'].append({"role": "assistant", "content": greeting})
        
        # Generate Audio
        path = generate_audio(greeting, gender, emotion)
        st.session_state['last_audio_path'] = path
        st.session_state['autoplay_trigger'] = True
        st.rerun()

    # Display Chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state['roleplay_history']:
            if msg['role'] == "user":
                st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
            elif msg['role'] == "assistant":
                st.markdown(f"<div class='chat-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("<div style='clear: both;'></div>", unsafe_allow_html=True)
    st.divider()

    # User Input
    rp_audio = st.audio_input("Reply to the AI")
    
    if rp_audio:
        # Check if we just processed this specific audio input (prevents double submit loops)
        # Note: st.audio_input doesn't have a unique ID, so we check if the last message matches
        
        with st.spinner("Listening..."):
            user_text = transcribe_audio(rp_audio)
        
        if user_text:
             # Basic debounce: Check if user_text is identical to the last user message? 
             # (Not perfect but helps. Better: check length of history)
            last_role = st.session_state['roleplay_history'][-1]['role']
            
            if last_role == "assistant":
                # 1. Add User Message
                st.session_state['roleplay_history'].append({"role": "user", "content": user_text})
                
                # 2. Get AI Response
                with st.spinner("AI thinking..."):
                    ai_res = client.chat.completions.create(
                        model=TEXT_MODEL, 
                        messages=st.session_state['roleplay_history']
                    )
                    ai_text = ai_res.choices[0].message.content
                    st.session_state['roleplay_history'].append({"role": "assistant", "content": ai_text})
                    
                    # 3. Generate Audio & Set Autoplay
                    path = generate_audio(ai_text, gender, emotion)
                    st.session_state['last_audio_path'] = path
                    st.session_state['autoplay_trigger'] = True
                    st.rerun()

# === TAB 5: MOUTH LAB ===
with tab5:
    st.header("👄 Articulation Mirror")
    st.info("Uses computer vision to overlay a face mesh on your lips. Helps with 'TH', 'R', and 'L' sounds.")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        webrtc_streamer(
            key="mouth-lab",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=MouthProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    with c2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/IPA_chart_2020.svg/800px-IPA_chart_2020.svg.png", caption="IPA Chart Reference")
