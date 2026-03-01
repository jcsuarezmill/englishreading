import streamlit as st
import os
import tempfile
import asyncio
import edge_tts
import difflib
import string
import numpy as np
import eng_to_ipa as ipa  # NEW: Phonetic Library
from groq import Groq

# --- DEPLOYMENT CONFIG (Streamlit Cloud Ready) ---
# When you deploy to Streamlit Cloud, you set this in "Secrets".
# For local use, paste your key below.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE" # <--- PASTE KEY HERE FOR LOCAL USE

client = Groq(api_key=GROQ_API_KEY)

# --- CONSTANTS ---
TEXT_MODEL = "llama-3.1-8b-instant"
AUDIO_MODEL = "whisper-large-v3-turbo"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="English Pro V5", 
    layout="centered", # Better for mobile than 'wide'
    page_icon="🦁",
    initial_sidebar_state="collapsed" # Hides sidebar on mobile for cleaner look
)

# --- MEDIAPIPE SAFETY (For Cloud Deployment) ---
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    import cv2
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError, Exception):
    pass # MediaPipe often fails on standard cloud instances without system libraries

# --- SESSION STATE ---
if 'generated_text' not in st.session_state:
    st.session_state['generated_text'] = ""
if 'vocab_list' not in st.session_state:
    st.session_state['vocab_list'] = "Generate a story to see vocabulary."

# --- ADVANCED FUNCTIONS ---

def get_phonetic_transcription(text):
    """Converts English text to IPA (International Phonetic Alphabet)"""
    # This solves the 'Naive Grading' by showing the actual sounds
    try:
        return ipa.convert(text)
    except:
        return text

def advanced_diff_view(target, spoken):
    """
    Creates a detailed comparison view with Phonetics.
    """
    target_clean = target.translate(str.maketrans('', '', string.punctuation)).lower()
    spoken_clean = spoken.translate(str.maketrans('', '', string.punctuation)).lower()
    
    t_words = target_clean.split()
    s_words = spoken_clean.split()
    
    matcher = difflib.SequenceMatcher(None, t_words, s_words)
    
    result_html = "<div style='font-family: sans-serif; line-height: 1.6;'>"
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for word in t_words[i1:i2]:
                # Green for correct
                result_html += f"<span style='color:#2ecc71; font-weight:bold; margin-right:5px;'>{word}</span>"
        elif tag == 'replace':
            for k in range(i2 - i1):
                # The word they missed
                missed_word = t_words[i1+k]
                # The word they said (if available)
                said_word = s_words[j1+k] if (j1+k) < len(s_words) else "?"
                
                # Get Phonetics
                missed_ipa = get_phonetic_transcription(missed_word)
                said_ipa = get_phonetic_transcription(said_word)

                # Advanced Red Box with IPA
                result_html += f"""
                <span style='
                    display:inline-block; 
                    background-color:#ffebee; 
                    border:1px solid #ffcdd2; 
                    border-radius:4px; 
                    padding:2px 6px; 
                    margin:2px; 
                    color:#c62828;'>
                    <strong>{missed_word}</strong><br>
                    <small style='color:#555'>/{missed_ipa}/</small>
                </span>
                <span style='color:#7f8c8d; font-size:0.8em;'> vs you: "{said_word}" /{said_ipa}/</span>
                """
        elif tag == 'delete':
            for word in t_words[i1:i2]:
                result_html += f"<span style='color:#e67e22; text-decoration:line-through; margin-right:5px;'>{word}</span>"
    
    result_html += "</div>"
    return result_html

async def text_to_speech(text, output_file, voice="en-US-ChristopherNeural", rate="+0%"):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_file)

def generate_lesson_content(topic, level):
    prompt = f"""
    Create a 3-sentence English lesson about: '{topic}'.
    Level: {level}.
    Format: Story ||| Vocab Word 1: Definition | Vocab Word 2: Definition
    """
    try:
        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = completion.choices[0].message.content
        if "|||" in content:
            parts = content.split("|||")
            return parts[0].strip(), parts[1].strip()
        return content, "No vocabulary found."
    except Exception as e:
        return f"Error: {e}. Check API Key.", ""

def analyze_mouth_shape(image_file):
    if not HAS_MEDIAPIPE: return None
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_LIPS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        return image
    return None

# --- UI START ---
st.title("🦁 English Pro V5")

# --- MOBILE FRIENDLY SETTINGS ---
with st.expander("⚙️ Settings & Controls"):
    voice_choice = st.selectbox("AI Voice", ["Male (Chris)", "Female (Aria)"])
    selected_voice = "en-US-ChristopherNeural" if "Male" in voice_choice else "en-US-AriaNeural"
    speed = st.select_slider("Speed", ["-20%", "Normal", "+20%"], value="Normal")
    speed_map = {"-20%": "-20%", "Normal": "+0%", "+20%": "+20%"}

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📚 Learn", "🎭 Act", "👄 Lab"])

# === TAB 1: SMART READER (With Phonetics) ===
with tab1:
    st.markdown("### Generate Lesson")
    topic = st.text_input("Topic", placeholder="e.g. Business Meeting, Ordering Food")
    
    if st.button("✨ Create Lesson", type="primary", use_container_width=True):
        if not topic:
            st.warning("Please enter a topic first.")
        else:
            with st.spinner("AI is writing..."):
                s, v = generate_lesson_content(topic, "Intermediate")
                st.session_state['generated_text'] = s
                st.session_state['vocab_list'] = v
                st.rerun()

    if st.session_state['generated_text']:
        st.divider()
        st.markdown("#### 📖 Read this:")
        
        # Phonetic Helper
        ipa_preview = get_phonetic_transcription(st.session_state['generated_text'])
        with st.expander("Show Phonetic Symbols (IPA)"):
            st.code(ipa_preview)
            st.caption("This shows exactly how the sounds should be pronounced.")

        st.info(st.session_state['generated_text'])
        
        # Audio Controls
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔈 Listen", use_container_width=True):
                asyncio.run(text_to_speech(st.session_state['generated_text'], "ref.mp3", selected_voice, speed_map[speed]))
                st.audio("ref.mp3")
        
        # Recording
        st.write("")
        audio_val = st.audio_input("Record Pronunciation")
        
        if audio_val:
            with st.spinner("Analyzing Phonetics..."):
                # Transcribe
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_val.read())
                    tmp_name = tmp.name
                
                with open(tmp_name, "rb") as f:
                    trans = client.audio.transcriptions.create(
                        file=(tmp_name, f.read()), 
                        model=AUDIO_MODEL,
                        language="en"
                    )
                os.remove(tmp_name)

                # DISPLAY ADVANCED FEEDBACK
                st.write("---")
                st.markdown("### 📊 Phonetic Feedback")
                feedback_html = advanced_diff_view(st.session_state['generated_text'], trans.text)
                st.markdown(feedback_html, unsafe_allow_html=True)
                
                st.info(f"💡 **Vocab Tips:** {st.session_state['vocab_list']}")

# === TAB 2: ROLEPLAY (Mobile Optimized) ===
with tab2:
    st.markdown("### 🎭 Quick Roleplay")
    scenario = st.selectbox("Scenario", ["Coffee Shop", "Doctor Visit"])
    
    scenarios = {
        "Coffee Shop": {"ai": "Next in line! What can I get you?", "user": "One iced americano, please."},
        "Doctor Visit": {"ai": "Where does it hurt exactly?", "user": "I have a sharp pain in my back."}
    }
    
    current = scenarios[scenario]
    
    with st.chat_message("assistant"):
        st.write(current["ai"])
        if st.button("▶️ Play Audio", key="rp_play", use_container_width=True):
            asyncio.run(text_to_speech(current["ai"], "rp.mp3", selected_voice, speed_map[speed]))
            st.audio("rp.mp3", autoplay=True)

    with st.chat_message("user"):
        st.write(f"**Say:** *{current['user']}*")
        rp_audio = st.audio_input("Reply", key="rp_rec")
        
        if rp_audio:
            # Quick Transcribe
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(rp_audio.read())
                tmp_name = tmp.name
            with open(tmp_name, "rb") as f:
                trans = client.audio.transcriptions.create(file=(tmp_name, f.read()), model=AUDIO_MODEL)
            os.remove(tmp_name)
            
            # Simple Feedback
            st.write(f"You said: **{trans.text}**")
            score = difflib.SequenceMatcher(None, current['user'].lower(), trans.text.lower()).ratio()
            if score > 0.8:
                st.success("✅ Perfect pronunciation!")
            else:
                st.warning("⚠️ Try again. Focus on the IPA sounds.")

# === TAB 3: MOUTH LAB ===
with tab3:
    st.markdown("### 👄 Mouth Shape")
    if HAS_MEDIAPIPE:
        st.info("Take a photo making a vowel sound.")
        img = st.camera_input("Camera")
        if img:
            res = analyze_mouth_shape(img)
            if res: st.image(res)
    else:
        st.warning("Feature disabled in this environment (Missing System Libraries).")