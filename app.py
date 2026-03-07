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
import soundfile as sf
import plotly.graph_objects as go
import noisereduce as nr
import textstat
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- PAGE CONFIG ---
st.set_page_config(page_title="English Ultimate V17 Pro", layout="wide", page_icon="🦁")

# --- CSS STYLING ---
st.markdown("""
<style>
    .big-text { font-size: 1.3rem; line-height: 1.6; font-family: sans-serif; }
    .metric-container { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; text-align: center; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);}
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #4b0082; }
    .metric-label { font-size: 0.8rem; text-transform: uppercase; color: #6c757d; letter-spacing: 1px; }
    .feedback-box { background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 5px; margin-top: 10px;}
    .stress { font-weight: 900; text-decoration: underline; color: #d35400; background-color: #fff3cd; padding: 0 4px; border-radius: 3px;}
    .pause { color: #c0392b; font-weight: bold; margin: 0 4px; font-size: 1.2rem;}
    .word-correct { color: #2e7d32; font-weight: bold; }
    .word-missing { color: #c62828; text-decoration: line-through; }
    .word-wrong { color: #ef6c00; font-style: italic; border-bottom: 2px dashed #ef6c00;}
    .level-badge { display: inline-block; padding: 5px 12px; border-radius: 15px; background: #ffe0b2; color: #e65100; font-weight: bold; font-size: 0.9rem; margin-right: 10px;}
</style>
""", unsafe_allow_html=True)

# --- API SETUP ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = "gsk_YOUR_API_KEY_HERE"

client = Groq(api_key=GROQ_API_KEY)

# --- SESSION STATE ---
if 'practice_text' not in st.session_state: st.session_state['practice_text'] = ""
if 'coach_script' not in st.session_state: st.session_state['coach_script'] = ""
if 'audio_ref_path' not in st.session_state: st.session_state['audio_ref_path'] = None
if 'reading_level' not in st.session_state: st.session_state['reading_level'] = ""

# --- HELPER FUNCTIONS ---

def generate_text(topic, emotion):
    prompt = f"Generate a highly realistic English practice text (30-50 words) about: '{topic}'. Tone: {emotion}. OUTPUT RAW TEXT ONLY."
    try:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content.replace('"', '')
    except Exception as e:
        return f"Error: {str(e)}"

def mark_script(text, emotion):
    prompt = f"""
    Act as a Master Voice Coach. Mark this script for reading with this emotion: {emotion}.
    1. Bold **stressed** words (words that carry the most emotion or importance).
    2. Add || where the reader must take a pause or breath.
    Output ONLY the marked text.
    Text: {text}
    """
    try:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content
    except:
        return text

async def text_to_speech(text, gender, emotion):
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    params = {"Neutral": {"r": "+0%", "p": "+0Hz"}, "Happy": {"r": "+5%", "p": "+5Hz"}, "Sad": {"r": "-10%", "p": "-5Hz"}, "Strict": {"r": "-5%", "p": "-2Hz"}}
    p = params.get(emotion, params["Neutral"])
    
    communicate = edge_tts.Communicate(text, voice, rate=p['r'], pitch=p['p'])
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    await communicate.save(path)
    return path

def sync_tts_gen(text, gender, emotion):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): loop = asyncio.new_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(text_to_speech(text, gender, emotion))

def get_text_metrics(text):
    grade = textstat.text_standard(text)
    ease = textstat.flesch_reading_ease(text)
    ease_desc = "Very Easy" if ease > 80 else "Conversational" if ease > 60 else "Advanced" if ease > 30 else "College Level"
    return grade, f"{ease_desc} ({ease}/100)"

def clean_audio_noise(input_path, output_path):
    data, rate = sf.read(input_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.85) # Aggressive gate for Jabra headsets
    sf.write(output_path, reduced_noise, rate)
    return output_path

def generate_visual_diff(target, spoken):
    t_words = target.lower().translate(str.maketrans('', '', string.punctuation)).split()
    s_words = spoken.lower().translate(str.maketrans('', '', string.punctuation)).split()
    matcher = difflib.SequenceMatcher(None, t_words, s_words)
    html_output =[]
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal': html_output.append(f"<span class='word-correct'>{' '.join(t_words[i1:i2])}</span>")
        elif tag == 'delete': html_output.append(f"<span class='word-missing'>{' '.join(t_words[i1:i2])}</span>")
        elif tag == 'replace': html_output.append(f"<span class='word-wrong' title='You said: {' '.join(s_words[j1:j2])}'>{' '.join(t_words[i1:i2])}</span>")
    return " ".join(html_output), matcher.ratio()

def analyze_audio_physics(file_path, transcript):
    try:
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 0.5: return None
        wpm = (len(transcript.split()) / duration) * 60
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=400)
        valid_f0 = f0[~np.isnan(f0)]
        pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0
        rms = librosa.feature.rms(y=y)
        return {"wpm": int(wpm), "pitch_std": round(pitch_std, 1), "energy": round(np.mean(rms) * 100, 1)}
    except:
        return None

def plot_intuitive_melody(ref_path, user_path):
    """The 'SingStar' Pitch Matcher - Overlays AI and User on the same 0-100% timeline!"""
    try:
        y_ref, sr_ref = librosa.load(ref_path)
        y_user, sr_user = librosa.load(user_path)

        f0_ref, _, _ = librosa.pyin(y_ref, fmin=60, fmax=400)
        f0_user, _, _ = librosa.pyin(y_user, fmin=60, fmax=400)
        
        # Clean NaNs for plotting
        f0_ref = np.nan_to_num(f0_ref, nan=0.0)
        f0_user = np.nan_to_num(f0_user, nan=0.0)
        
        # Remove absolute zeros to make the graph cleaner
        f0_ref[f0_ref == 0] = np.nan
        f0_user[f0_user == 0] = np.nan

        # Normalize time to 0% -> 100% so fast and slow readers line up perfectly!
        time_ref_perc = np.linspace(0, 100, len(f0_ref))
        time_user_perc = np.linspace(0, 100, len(f0_user))

        fig = go.Figure()

        # AI Target Line (Purple Dotted)
        fig.add_trace(go.Scatter(
            x=time_ref_perc, y=f0_ref, mode='lines', name='🎯 AI Target Melody', 
            line=dict(color='rgba(156, 39, 176, 0.7)', width=5, dash='dot')
        ))

        # User's Voice Line (Green Solid)
        fig.add_trace(go.Scatter(
            x=time_user_perc, y=f0_user, mode='lines', name='🎙️ Your Voice Melody', 
            line=dict(color='rgba(0, 150, 136, 1)', width=3)
        ))

        fig.update_layout(
            title="🎤 The Melody Matcher (Try to trace your green line over the AI's purple line!)",
            xaxis_title="Timeline of Recording (0% to 100%)",
            yaxis_title="Pitch (Highness / Lowness of Voice)",
            height=400, template="plotly_white", hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig
    except Exception as e:
        return None

def get_coach_feedback(target, spoken, metrics, emotion):
    pace_eval = "Good pace"
    if metrics['wpm'] < 110: pace_eval = "Too slow (Dragging)"
    elif metrics['wpm'] > 160: pace_eval = "Too fast (Rushing)"
    prompt = f"""
    You are an Expert English Coach. Provide direct, encouraging feedback.
    - Script: "{target}"
    - Student Said: "{spoken}"
    - Pace: {metrics['wpm']} WPM ({pace_eval})
    - Tone/Pitch Dynamism: {metrics['pitch_std']}
    Provide 3 short, actionable bullet points focusing on Missed Words, Rhythm, and Emotion.
    """
    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
    return res.choices[0].message.content

def process_script(text_to_process, gen_gender, gen_emotion):
    """Pipeline to prep the script whether it was AI generated or custom pasted"""
    with st.spinner("Processing Script, Generating AI Audio Guide & Extracting Phonetics..."):
        st.session_state['practice_text'] = text_to_process
        st.session_state['coach_script'] = mark_script(text_to_process, gen_emotion)
        st.session_state['audio_ref_path'] = sync_tts_gen(text_to_process, gen_gender, gen_emotion)
        st.session_state['reading_level'] = get_text_metrics(text_to_process)
        st.rerun()

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Coach Settings")
    gender = st.selectbox("AI Voice Coach", ["Male", "Female"])
    emotion = st.selectbox("Target Emotion", ["Neutral", "Happy", "Sad", "Strict"])
    st.divider()
    
    # 100% Retained Feature: WebRTC Camera
    with st.expander("🎥 Mouth Shape Cam", expanded=False):
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
            webrtc_streamer(key="mouth", mode=WebRtcMode.SENDRECV, video_processor_factory=MouthProcessor, rtc_configuration={"iceServers": [{"urls":["stun:stun.l.google.com:19302"]}]}, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        except:
            st.warning("Camera unavailable.")

# --- MAIN LAYOUT ---
st.title("🦁 English Ultimate V17 Pro")
st.caption("Custom Call Scripts ➔ Melody Matcher Graph ➔ Native Fluency Training")

# 1. INPUT SECTION (UPGRADED WITH CUSTOM SCRIPTING)
tab_gen, tab_custom = st.tabs(["🤖 Let AI Generate a Scenario", "✍️ Paste Your Own Script (Call Center/Monologues)"])

with tab_gen:
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        topic = st.text_input("What scenario do you want to practice?", placeholder="e.g. Handling an angry customer refund...")
    with col_in2:
        if st.button("Generate AI Script", use_container_width=True):
            if topic:
                raw_text = generate_text(topic, emotion)
                process_script(raw_text, gender, emotion)

with tab_custom:
    custom_text = st.text_area("Paste your exact script or dialogue here:", placeholder="Thank you for calling support, my name is...", height=100)
    if st.button("Process My Custom Script"):
        if custom_text:
            process_script(custom_text, gender, emotion)

# 2. PRACTICE SECTION
if st.session_state['practice_text']:
    st.divider()
    
    if st.session_state['reading_level']:
        grade, ease = st.session_state['reading_level']
        st.markdown(f"<div><span class='level-badge'>📚 Difficulty: {grade}</span><span class='level-badge'>🧠 Style: {ease}</span></div><br>", unsafe_allow_html=True)
    
    # 100% Retained Features: All Guide Tabs
    tab_guide, tab_script, tab_ipa = st.tabs(["🎭 Acting Guide", "📄 Plain Text", "🗣️ IPA Pronunciation"])
    
    with tab_guide:
        st.info("💡 **Coach's Notes:** Emphasize highlighted words. Pause and breathe at || marks.")
        formatted = st.session_state['coach_script'].replace("**", "<b>").replace("||", "<span class='pause'>||</span>")
        formatted = formatted.replace("<b>", "<span class='stress'>").replace("</b>", "</span>")
        st.markdown(f"<div class='big-text'>{formatted}</div>", unsafe_allow_html=True)
        
        st.markdown("**Listen to the Native Target:**")
        if st.session_state['audio_ref_path']:
            st.audio(st.session_state['audio_ref_path'])

    with tab_script:
        st.markdown(f"<div class='big-text'>{st.session_state['practice_text']}</div>", unsafe_allow_html=True)
        
    with tab_ipa:
        st.caption("Use the International Phonetic Alphabet (IPA) to perfect your vowel and consonant shapes.")
        ipa_text = ipa.convert(st.session_state['practice_text'])
        st.markdown(f"<div class='big-text' style='color:#555;'>{ipa_text}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # 3. RECORDING & ANALYSIS
    st.markdown("### 🎙️ The Stage: Record Your Take")
    audio_val = st.audio_input("Press Mic to Start Reading (Use your Jabra Headset!)")
    
    if audio_val:
        with st.spinner("1/3 Cleaning Jabra Audio Noise (Studio Gate)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
                tmp_raw.write(audio_val.getvalue())
                raw_path = tmp_raw.name
            clean_path = raw_path.replace(".wav", "_clean.wav")
            clean_audio_noise(raw_path, clean_path)
            
        with st.spinner("2/3 Transcribing & Scoring Linguistics..."):
            with open(clean_path, "rb") as f:
                transcription = client.audio.transcriptions.create(file=(clean_path, f.read()), model="whisper-large-v3-turbo").text
            metrics = analyze_audio_physics(clean_path, transcription)
            
        with st.spinner("3/3 Rendering 'SingStar' Melody Matcher..."):
            fig_melody = plot_intuitive_melody(st.session_state['audio_ref_path'], clean_path)
            
        if metrics:
            # --- A. NEW INTUITIVE OVERLAY GRAPH ---
            st.markdown("### 🎶 The Melody Matcher")
            if fig_melody:
                st.plotly_chart(fig_melody, use_container_width=True)

            st.markdown("**🎧 Listen to what the system heard (Noise Reduced):**")
            st.audio(clean_path)
            st.divider()

            # --- B. RETAINED: VISUAL WORD DIFF ---
            st.markdown("### 🎯 Accuracy Breakdown (Karaoke Style)")
            diff_html, acc_ratio = generate_visual_diff(st.session_state['practice_text'], transcription)
            st.markdown(f"<div style='font-size:1.2rem; background:#fff; padding:15px; border-radius:8px; border: 1px solid #ddd; line-height: 1.8;'>{diff_html}</div>", unsafe_allow_html=True)
            st.markdown(f"**Legend:** <span class='word-correct'>Green</span> = Perfect | <span class='word-missing'>Red</span> = Skipped | <span class='word-wrong'>Orange</span> = Mispronounced/Stuttered", unsafe_allow_html=True)
            st.progress(acc_ratio, text=f"Overall Pronunciation Accuracy: {int(acc_ratio*100)}%")
            st.divider()

            # --- C. RETAINED: METRICS & AI CRITIQUE ---
            st.markdown("### 📊 Scoring & AI Feedback")
            m1, m2, m3 = st.columns(3)
            pace_color = "green" if 110 <= metrics['wpm'] <= 160 else "red"
            with m1:
                st.markdown(f"<div class='metric-container'><div class='metric-value' style='color:{pace_color}'>{metrics['wpm']}</div><div class='metric-label'>Words/Min (Goal: 120-150)</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-container'><div class='metric-value'>{metrics['pitch_std']}</div><div class='metric-label'>Tone Dynamism (Emotion)</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-container'><div class='metric-value'>{metrics['energy']}</div><div class='metric-label'>Loudness / Confidence</div></div>", unsafe_allow_html=True)
            
            feedback = get_coach_feedback(st.session_state['practice_text'], transcription, metrics, emotion)
            st.markdown(f"<div class='feedback-box'>{feedback}</div>", unsafe_allow_html=True)
            
        else:
            st.error("Audio recording was too short or silent. Please try again.")
        
        try:
            os.remove(raw_path)
            os.remove(clean_path)
        except: pass
