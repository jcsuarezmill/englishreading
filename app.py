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
from plotly.subplots import make_subplots
import noisereduce as nr
import textstat
from groq import Groq
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# --- PAGE CONFIG ---
st.set_page_config(page_title="English Ultimate V16 Pro", layout="wide", page_icon="🦁")

# --- CSS STYLING ---
st.markdown("""
<style>
    .big-text { font-size: 1.3rem; line-height: 1.6; font-family: sans-serif; }
    .metric-container { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; text-align: center; margin-bottom: 10px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #4b0082; }
    .metric-label { font-size: 0.8rem; text-transform: uppercase; color: #6c757d; letter-spacing: 1px; }
    .feedback-box { background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 5px; margin-top: 10px;}
    .stress { font-weight: 900; text-decoration: underline; color: #d35400; }
    .pause { color: #c0392b; font-weight: bold; margin: 0 4px; }
    .word-correct { color: #2e7d32; font-weight: bold; }
    .word-missing { color: #c62828; text-decoration: line-through; }
    .word-wrong { color: #ef6c00; font-style: italic; border-bottom: 1px dashed #ef6c00;}
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
    prompt = f"Generate a short (30-40 words) English practice text about: '{topic}'. Tone: {emotion}. OUTPUT RAW TEXT ONLY."
    try:
        res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content.replace('"', '')
    except Exception as e:
        return f"Error: {str(e)}"

def mark_script(text, emotion):
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
    voice = "en-US-ChristopherNeural" if gender == "Male" else "en-US-AriaNeural"
    params = {"Neutral": {"r": "+0%", "p": "+0Hz"}, "Happy": {"r": "+10%", "p": "+5Hz"}, "Sad": {"r": "-10%", "p": "-5Hz"}, "Strict": {"r": "-5%", "p": "-2Hz"}}
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
    """Calculates US Grade Level and Reading Ease"""
    grade = textstat.text_standard(text)
    ease = textstat.flesch_reading_ease(text)
    ease_desc = "Very Easy" if ease > 80 else "Conversational" if ease > 60 else "Advanced" if ease > 30 else "College Level"
    return grade, f"{ease_desc} ({ease}/100)"

def clean_audio_noise(input_path, output_path):
    """Applies Studio-Grade Noise Gate to user's mic recording"""
    data, rate = sf.read(input_path)
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
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
        word_count = len(transcript.split())
        wpm = (word_count / duration) * 60
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        valid_f0 = f0[~np.isnan(f0)]
        pitch_std = np.std(valid_f0) if len(valid_f0) > 0 else 0
        rms = librosa.feature.rms(y=y)
        return {"wpm": int(wpm), "pitch_std": round(pitch_std, 1), "energy": round(np.mean(rms) * 100, 1)}
    except:
        return None

def plot_interactive_melody(ref_path, user_path):
    """Generates an Interactive Plotly Graph for Pitch and Stress/Volume"""
    try:
        y_ref, sr_ref = librosa.load(ref_path)
        y_user, sr_user = librosa.load(user_path)

        # F0 (Pitch)
        f0_ref, _, _ = librosa.pyin(y_ref, fmin=60, fmax=500)
        f0_user, _, _ = librosa.pyin(y_user, fmin=60, fmax=500)
        times_ref = librosa.times_like(f0_ref, sr=sr_ref)
        times_user = librosa.times_like(f0_user, sr=sr_user)

        # RMS (Volume / Loudness Stress)
        rms_ref = librosa.feature.rms(y=y_ref)[0]
        rms_user = librosa.feature.rms(y=y_user)[0]
        times_rms_ref = librosa.times_like(rms_ref, sr=sr_ref)
        times_rms_user = librosa.times_like(rms_user, sr=sr_user)

        # Build Interactive Subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, 
                            subplot_titles=("🎯 Target: AI Native Speaker", "🎙️ Your Voice Recording (Noise Reduced)"),
                            specs=[[{"secondary_y": True}],[{"secondary_y": True}]])

        # AI Trace: Volume (Shaded) + Pitch (Line)
        fig.add_trace(go.Scatter(x=times_rms_ref, y=rms_ref, fill='tozeroy', name='AI Volume (Stress)', line=dict(color='rgba(156, 39, 176, 0.2)')), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=times_ref, y=f0_ref, mode='lines', name='AI Pitch (Tone)', line=dict(color='#9c27b0', width=3)), row=1, col=1, secondary_y=True)

        # User Trace: Volume (Shaded) + Pitch (Line)
        fig.add_trace(go.Scatter(x=times_rms_user, y=rms_user, fill='tozeroy', name='Your Volume', line=dict(color='rgba(0, 150, 136, 0.2)')), row=2, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=times_user, y=f0_user, mode='lines', name='Your Pitch', line=dict(color='#009688', width=3)), row=2, col=1, secondary_y=True)

        # Layout adjustments
        fig.update_yaxes(title_text="Volume", secondary_y=False, showgrid=False)
        fig.update_yaxes(title_text="Pitch (Hz)", range=[60, 400], secondary_y=True, showgrid=True)
        fig.update_layout(height=650, template="plotly_white", hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20))
        
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
    - Intonation Score: {metrics['pitch_std']} (Low < 15 is Monotone)
    Provide 3 short bullet points focusing on Pronunciation, Rhythm, and Emotion.
    """
    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}])
    return res.choices[0].message.content

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    gender = st.selectbox("AI Voice", ["Male", "Female"])
    emotion = st.selectbox("Target Emotion", ["Neutral", "Happy", "Sad", "Strict"])
    st.divider()
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
            webrtc_streamer(key="mouth", mode=WebRtcMode.SENDRECV, video_processor_factory=MouthProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        except:
            st.warning("Camera unavailable on this server.")

# --- MAIN LAYOUT ---
st.title("🦁 English Ultimate V16 Pro")
st.caption("Studio-Grade Audio Cleanup ➔ Hoverable Pitch Graphs ➔ Reading Level Metrics")

# 1. INPUT SECTION
col_in1, col_in2 = st.columns([3, 1])
with col_in1:
    topic = st.text_input("What do you want to practice?", placeholder="e.g. Asking for a refund, A scary story...")
with col_in2:
    if st.button("Generate Script", use_container_width=True):
        if topic:
            with st.spinner("Creating content & AI Voice..."):
                raw_text = generate_text(topic, emotion)
                st.session_state['practice_text'] = raw_text
                st.session_state['coach_script'] = mark_script(raw_text, emotion)
                st.session_state['audio_ref_path'] = sync_tts_gen(raw_text, gender, emotion)
                st.session_state['reading_level'] = get_text_metrics(raw_text)
                st.rerun()

# 2. PRACTICE SECTION
if st.session_state['practice_text']:
    st.divider()
    
    # Textstat Reading Level Badges
    if st.session_state['reading_level']:
        grade, ease = st.session_state['reading_level']
        st.markdown(f"<div><span class='level-badge'>📚 {grade}</span><span class='level-badge'>🧠 {ease}</span></div><br>", unsafe_allow_html=True)
    
    tab_guide, tab_script, tab_ipa = st.tabs(["🎭 Acting Guide", "📄 Plain Text", "🗣️ IPA Pronunciation"])
    
    with tab_guide:
        st.info("💡 **Coach's Notes:** Emphasize bold words. Pause at || marks.")
        formatted = st.session_state['coach_script'].replace("**", "<b>").replace("||", "<span class='pause'>||</span>")
        formatted = formatted.replace("<b>", "<span class='stress'>").replace("</b>", "</span>")
        st.markdown(f"<div class='big-text'>{formatted}</div>", unsafe_allow_html=True)
        if st.session_state['audio_ref_path']:
            st.audio(st.session_state['audio_ref_path'])

    with tab_script:
        st.markdown(f"<div class='big-text'>{st.session_state['practice_text']}</div>", unsafe_allow_html=True)
        
    with tab_ipa:
        st.caption("Use the International Phonetic Alphabet to perfect your mouth shapes.")
        ipa_text = ipa.convert(st.session_state['practice_text'])
        st.markdown(f"<div class='big-text' style='color:#555;'>{ipa_text}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # 3. RECORDING & ANALYSIS
    st.markdown("### 🎙️ Your Stage: Record Your Take")
    audio_val = st.audio_input("Press Mic to Start")
    
    if audio_val:
        with st.spinner("1/3 Cleaning Audio Noise (Studio Gate)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
                tmp_raw.write(audio_val.getvalue())
                raw_path = tmp_raw.name
                
            clean_path = raw_path.replace(".wav", "_clean.wav")
            clean_audio_noise(raw_path, clean_path)
            
        with st.spinner("2/3 Transcribing & Linguistics Analysis..."):
            with open(clean_path, "rb") as f:
                transcription = client.audio.transcriptions.create(file=(clean_path, f.read()), model="whisper-large-v3-turbo").text
            metrics = analyze_audio_physics(clean_path, transcription)
            
        with st.spinner("3/3 Rendering Interactive Studio Graphs..."):
            fig_melody = plot_interactive_melody(st.session_state['audio_ref_path'], clean_path)
            
        if metrics:
            # --- A. INTERACTIVE RISE & FALL MELODY GRAPH ---
            st.markdown("### 🎶 Zoomable Melody & Pacing Matcher")
            st.markdown("""
            * **Hover your mouse** over the chart to see exact data points.
            * **Drag to zoom in** on specific words or pauses. (Double-click to zoom out).
            * Match your **Solid Line** (Tone/Pitch) and your **Shaded Area** (Loudness/Stress) to the AI!
            """)
            if fig_melody:
                st.plotly_chart(fig_melody, use_container_width=True)

            # Playback the CLEANED audio to the user so they can hear what the AI heard
            st.markdown("**🎧 Listen to your Noise-Reduced Recording:**")
            st.audio(clean_path)

            st.divider()

            # --- B. VISUAL DIFF ---
            st.markdown("### 🎯 Accuracy Breakdown")
            diff_html, acc_ratio = generate_visual_diff(st.session_state['practice_text'], transcription)
            st.markdown(f"<div style='font-size:1.2rem; background:#fff; padding:15px; border-radius:8px; border: 1px solid #ddd;'>{diff_html}</div>", unsafe_allow_html=True)
            st.markdown(f"**Legend:** <span class='word-correct'>Green</span> = Perfect | <span class='word-missing'>Red</span> = Skipped | <span class='word-wrong'>Orange</span> = Mispronounced/Replaced", unsafe_allow_html=True)
            st.progress(acc_ratio, text=f"Overall Pronunciation Accuracy: {int(acc_ratio*100)}%")
            
            st.divider()

            # --- C. METRICS & CRITIQUE ---
            st.markdown("### 📊 Scoring & AI Feedback")
            m1, m2, m3 = st.columns(3)
            pace_color = "green" if 110 <= metrics['wpm'] <= 160 else "red"
            with m1:
                st.markdown(f"<div class='metric-container'><div class='metric-value' style='color:{pace_color}'>{metrics['wpm']}</div><div class='metric-label'>Words/Min (Goal: 120-150)</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-container'><div class='metric-value'>{metrics['pitch_std']}</div><div class='metric-label'>Tone Dynamism (Low < 15)</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-container'><div class='metric-value'>{metrics['energy']}</div><div class='metric-label'>Loudness / Confidence</div></div>", unsafe_allow_html=True)
            
            feedback = get_coach_feedback(st.session_state['practice_text'], transcription, metrics, emotion)
            st.markdown(f"<div class='feedback-box'>{feedback}</div>", unsafe_allow_html=True)
            
        else:
            st.error("Audio recording was too short or silent. Please try again.")
        
        # Clean up files
        try:
            os.remove(raw_path)
            os.remove(clean_path)
        except: pass
