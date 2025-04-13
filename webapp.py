import streamlit as st
import base64

st.set_page_config(page_title="ToneClone", page_icon="🔊", layout="wide")

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import librosa.display
import plotly.graph_objects as go
import json
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which
from PIL import Image
import requests
import math

from utils.analyzer import ToneCloneAnalyzer

# Check if ffmpeg is installed
ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")

# Get the current directory of the file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
AUDIO_DIR = os.path.join(BASE_DIR, "audio_uploads")

# Create directories if they don't exist
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)

# Effect to image 
effect_to_image = {
    "plate reverb": os.path.join(IMAGES_DIR, "pedal_PLT.png"),
    "distortion": os.path.join(IMAGES_DIR, "pedal_DST.png"),
    "delay": os.path.join(IMAGES_DIR, "pedal_DLY.png"),
    "chorus": os.path.join(IMAGES_DIR, "pedal_CHR.png"),
    "flanger": os.path.join(IMAGES_DIR, "pedal_FLG.png"),
    "fuzz": os.path.join(IMAGES_DIR, "pedal_FUZ.png"),
    "auto filter": os.path.join(IMAGES_DIR, "pedal_FLT.png"),
    "overdrive": os.path.join(IMAGES_DIR, "pedal_ODV.png"),
    "octaver": os.path.join(IMAGES_DIR, "pedal_OCT.png"),
    "tremolo": os.path.join(IMAGES_DIR, "pedal_TRM.png"),  
    "phaser": os.path.join(IMAGES_DIR, "pedal_PHZ.png"),
    "hall reverb": os.path.join(IMAGES_DIR, "pedal_HLL.png"),
    "logo": os.path.join(IMAGES_DIR, "guitarlogo.png"),
    "Gary": os.path.join(IMAGES_DIR, "GaryPic.png"),
    "effects": os.path.join(IMAGES_DIR, "effects.png"),
    "spectrogram": os.path.join(IMAGES_DIR, "spectrogram.png"),
    "diagram": os.path.join(IMAGES_DIR, "data_diagram.png"),
    "architecture": os.path.join(IMAGES_DIR, "toneclone-arch.png"),
    "Kushal": os.path.join(IMAGES_DIR, "Kushal.png"),
    "Rex": os.path.join(IMAGES_DIR, "Rex.png"),
    "Jen": os.path.join(IMAGES_DIR, "Jen.jpg")
}

def downsample_waveform(y, target_len=4000):
    """Downsample waveform to target number of points for display"""
    if len(y) <= target_len:
        return y
    factor = len(y) // target_len
    return y[::factor]

def display_waveform(y, sr, start_time, end_time):
    """Create and return a Plotly waveform visualization"""

    full_duration = librosa.get_duration(y=y, sr=sr)

    y_ds = downsample_waveform(y)
    time_axis = np.linspace(0, full_duration, num=len(y_ds))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=y_ds, 
        mode="lines",
        name="",
        hovertemplate="Time: %{x:.2f} sec"
    ))
    
    # vertical lines for start and end
    fig.add_vline(
        x=start_time, 
        line_width=2, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Start", 
        annotation_position="top"
    )
    
    fig.add_vline(
        x=end_time, 
        line_width=2, 
        line_dash="dash", 
        line_color="green",
        annotation_text="End", 
        annotation_position="top"
    )
    
    # graph with only hover capability
    fig.update_layout(
        title="Waveform - Use time range selector above",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),  # Add margin to prevent cutoff
        # Fix y-axis range to prevent moving up and down
        yaxis=dict(
            fixedrange=True,
            range=[min(y_ds)*1.1, max(y_ds)*1.1]  # Set fixed range with small padding
        ),
        # Fix x-axis range to prevent horizontal panning
        xaxis=dict(
            fixedrange=True,
            range=[0, full_duration]  # Set range from 0 to full duration
        ),
        # Disable most interactive features
        dragmode=False
    )
    
    # Configure to disable most interactions but keep hover
    fig.update_layout(
        clickmode='none',
        hovermode='closest'
    )
    
    # Disable the (zoom buttons...)
    config = {
        'displayModeBar': False,
        'staticPlot': False,  # Not fully static so hover still works
        'scrollZoom': False
    }
    
    # Return both figure and config
    return fig


def display_timeline(segments, sample_length, overlap, effects, path_state):
    """
    Displays a timeline graph of detected effects with probabilities shown by intensity
    """

    def time_to_seconds(t):
        minutes, seconds = map(int, t.split(':'))
        return minutes * 60 + seconds

    # effects = dict()
    # for seg in segments:
    #     for eff in segments[seg][0]:
    #         if eff not in effects:
    #             effects[eff] = 1
    #         else:
    #             effects[eff] += 1
    # effects = sorted(effects.keys(), key=lambda x: x, reverse=False)
    if path_state == "cropped_path":
        start_time = st.session_state['start_time']
        end_time = st.session_state['end_time']
    else:
        start_time = 0.0
        end_time = st.session_state['current_audio_duration']

    time_resolution = sample_length * (100.0 - overlap) / 100.0
    time_axis = np.arange(start_time, end_time, time_resolution)

    base_colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff', '#ff8800', '#88ff00',
                   '#00ff88', '#0088ff', '#8800ff', '#ff0088']

    fig, ax = plt.subplots(figsize=(12, len(effects) * 1.2))

    for i, eff in enumerate(effects):
        for t in time_axis:
            overlapping = [
                segments[seg] for seg in segments
                if eff in segments[seg][0]
                and t - start_time <= (int(seg.split("Segment ")[1]) - 1) * time_resolution < t + time_resolution - start_time
            ]
            if overlapping:
                avg_intensity = np.mean([seg[1][eff] for seg in overlapping])
            else:
                avg_intensity = 0
            rect = patches.Rectangle(
                (t, i + 0.25), time_resolution, 0.5,
                color=base_colors[i % len(base_colors)],
                alpha=avg_intensity ** 2,
                linewidth=0
            )
            ax.add_patch(rect)
        ax.text(end_time + 1, i + 0.5, eff.replace('_', ' ').title(),
                verticalalignment='center', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlim(start_time, end_time)
    ax.set_ylim(0, len(effects))
    ax.set_yticks([])

    tick_interval = 20
    ticks = np.arange(start_time, end_time, tick_interval)
    if ticks[-1] < end_time:
        ticks = np.append(ticks, end_time)
    ax.set_xticks(ticks)

    def format_func(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f'{minutes:02d}:{seconds:02d}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    ax.set_xlabel("Time (mm:ss)")
    return fig

def classify_song(path_state):
    file_path = st.session_state[path_state]
    tcanalyzer = ToneCloneAnalyzer(str(file_path), openai_key=st.secrets["openai_key"])
    tcanalyzer.process_wav_for_model()
    tcanalyzer.submit_spectrogram_batches(
            st.secrets['aws_access_key_id'],
            st.secrets['aws_secret_access_key']
    )
    tcanalyzer.categorize_effect_confidence()
    raw_predictions, predictions_summary_for_llm, summarized_segment_results, top_3_effects = tcanalyzer.summarize_for_llm()
    st.session_state['raw_predictions'] = raw_predictions
    st.session_state['summarized_segment_results'] = summarized_segment_results
    st.session_state['predictions_summary_for_llm'] = predictions_summary_for_llm
    st.session_state['timeline_visualization'] = display_timeline(raw_predictions, tcanalyzer.sample_length,
                                                                  tcanalyzer.overlap, top_3_effects, path_state)
    tcanalyzer.chatgpt_prompt()
    user_education = tcanalyzer.parse_effects_json()

    return user_education, top_3_effects

def classify_tone():
    """Main function for tone classification interface."""
    st.markdown("<h3 style='color: #667085; font-weight: 600; letter-spacing: -0.5px; margin-top: 0;'>Upload your song sample for your ToneClone result!</h3>""",
 unsafe_allow_html=True)

    # Upload sound
    uploaded_file = st.file_uploader(' ', type='wav')
    
    # Reset session if file is removed or changed
    prev_filename = st.session_state.get("last_uploaded_filename")
    curr_filename = uploaded_file.name if uploaded_file else None
    
    if curr_filename != prev_filename:
        for key in [
            'top_3_effects', 'user_education', 'raw_predictions',
            'summarized_segment_results', 'predictions_summary_for_llm',
            'selected_effect', 'preview_path', 'cropped_path',
            'cropped_duration', 'last_classification_type',
            'current_audio_path', 'current_audio_duration',
             'timeline_visualization'
        ]:
            st.session_state.pop(key, None)
        
        st.session_state["last_uploaded_filename"] = curr_filename

    if uploaded_file is None:
        st.write('Please upload a WAV file to begin')
        return "No file uploaded"
    
    # Process the uploaded file
    file_path = Path(AUDIO_DIR) / uploaded_file.name
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Store file path in session state
    st.session_state['current_audio_path'] = str(file_path)
    
    st.success(f"✅ Uploaded: {file_path.name}")
    st.audio(str(file_path))
    
    # Convert to WAV for Processing
    audio = AudioSegment.from_file(file_path).set_channels(1)
    audio.export(file_path, format="wav")
    
    # Load audio with Librosa
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Store duration in session state
    st.session_state['current_audio_duration'] = duration
    
   
    
    # Time range selection
    # Remove dark grey background from slider
    st.markdown("""
    <style>
    /* Target the slider container */
    .stSlider {
        background-color: transparent !important;
    }

    /* Target slider's parent elements */
    .stSlider > div {
        background-color: transparent !important;
    }

    /* Keep only the slider track styled */
    .stSlider > div > div {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Time range selection
    start_time, end_time = st.slider(
        "Select Time Range (seconds)",
        min_value=0.0,
        max_value=duration,
        value=(0.0, duration),
        step=0.1
    )

    st.session_state['start_time'] = start_time
    st.session_state['end_time'] = end_time

    # Display waveform
    fig = display_waveform(y, sr, start_time, end_time)
    st.plotly_chart(
    fig, 
    use_container_width=True,
    config={
        'displayModeBar': False,
        'staticPlot': False,
        'scrollZoom': False
    }
)
    

        # Apply crop and preview
    start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
    cropped_audio = audio[start_ms:end_ms]
    
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎧 Preview Crop"):
            preview_path = Path(AUDIO_DIR) / f"preview_{uploaded_file.name}"
            cropped_path = Path(AUDIO_DIR) / f"cropped_{uploaded_file.name}"
            
            # Save both versions
            cropped_audio.export(preview_path, format="wav")
            cropped_audio.export(cropped_path, format="wav")
            
            # Store paths in session state
            st.session_state['preview_path'] = str(preview_path)
            st.session_state['cropped_path'] = str(cropped_path)
            st.session_state['cropped_duration'] = end_time - start_time
            
            # Play preview
            st.audio(str(preview_path))
            st.success("🔊 Previewing cropped audio. File is ready for classification.")
    
    with col2:
        if st.button("💾 Save Cropped File"):
            save_path = Path(AUDIO_DIR) / f"cropped_{uploaded_file.name}"
            
            # Export the cropped audio to the new file
            cropped_audio.export(save_path, format="wav")
            
            # Let the user download the saved file
            st.success(f"✅ Cropped audio saved as {save_path.name}")
            
            with open(save_path, "rb") as file:
                st.download_button(
                    label="Download Cropped Audio",
                    data=file,
                    file_name=save_path.name,
                    mime="audio/wav"
                )


    

    return "Ready for classification"

#added
def image_with_text(image_path, title, description):
    """Create HTML for image next to text section"""
    if os.path.exists(image_path):
        # Encode image to base64 for inline HTML
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode()
            
        html = f"""
        <div class="image-text-container">
            <div class="image-container">
                <img src="data:image/png;base64,{image_base64}" class="section-image" width="150">
            </div>
            <div class="text-container">
                <h3 style="color:#D3AD51; margin-top:0;">{title}</h3>
                <p>{description}</p>
            </div>
        </div>
        """
        return html
    else:
        # Fallback if image doesn't exist
        return f"""
        <h3 style="color:#D3AD51;">{title}</h3>
        <p>{description}</p>
        """

#  CSS for styling
def apply_custom_css():
    """Apply modern tech-focused styling to the ToneClone app."""
    st.markdown("""
    <style>
        /* Import modern tech fonts */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&family=Montserrat:wght@400;500;600&display=swap');
        
        /* Modern gradient background  */
        .stApp {
            background: linear-gradient(135deg, #f7f6f9 0%, #d7d5d9 50%, #c9c7cc 100%);
            background-attachment: fixed;
        }

        /* Sidebar Styling with dark tech look */
        .stSidebar {
            background-color: #ffffff !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Logo container in sidebar with tech styling */
        .sidebar-logo-container {
            padding: 20px 10px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
            background: #ffffff;
            border-radius: 0 0 10px 10px;
        }
        
        .sidebar-logo {
            width: 100px;
            height: auto;
            margin-bottom: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* ToneClone title with tech styling */
        .sidebar-title {
            font-family: 'Dancing Script', cursive;
            font-size: 28px;
            font-weight: 700;
            color: #2C7DA0;
            margin: 0;
            letter-spacing: -0.5px;
            padding: 8px 15px;
            display: inline-block;
            position: relative;
            z-index: 1;
            border-radius: 8px;
           
            
            text-shadow: 2px 2px 3px rgba(0,0,0,0.2);
        }

        /* Navigation styling - tech-focused */
        div[data-baseweb="radio"] label {
            font-family: 'Space Grotesk', sans-serif !important;
            color: #cccccc !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
        }
                
        /* Selected button state */
        div[data-baseweb="radio"] input:checked + label {
            color: #D3AD51 !important;
            font-weight: 600 !important;
            text-shadow: 0 0 5px rgba(211, 173, 81, 0.2);
        }

        /* Text styling for modern look */
        html, body, .stApp, .stText, p, label, .stTitle, .stHeader, .stSubheader {
            color: #333333 !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }
      
        /* Button styling - clean and flat */
        .stButton button {
            background-color: #0066cc !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 6px 14px !important;
            font-weight: 500 !important;
            font-family: 'Montserrat', sans-serif !important;
            font-size: 14px !important;
        }

        .stButton button:hover {
            background-color: #4d94ff !important; /* Lighter blue on hover */
        }

        /* Main page title style */
        .title-text {
            font-family: 'Dancing Script', cursive;
            font-size: 80px;
            font-weight: 700;
            text-align: left;
            color: #2C7DA0;
            margin: 0;
            text-shadow: 0 0 10px rgba(211, 173, 81, 0.3);
            letter-spacing: -0.5px;
            display: inline-block;
            padding: 15px 25px;
            text-shadow: 2px 2px 3px rgba(0,0,0,0.2);
            position: relative;
            z-index: 1;
            border-radius: 12px;
        }
                
        
        /* Subtitle text */
        .subtitle-text {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 16px;
            font-weight: 400;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            color: #667085;
            margin-top: -5px;
        }
        
        /* Add transparent frosted glass effect for columns */
        .stColumn {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px !important;
            padding: 15px !important;
            margin: 5px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
            transition: all 0.3s ease !important;
        }
        
        /* Column hover effect */
        .stColumn:hover {
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(211, 173, 81, 0.3) !important;
        }
        
        /* Container for columns to enforce horizontal layout */
        .row-container {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
            gap: 15px !important;
        }
        
        /* Fix padding inside columns */
        .stColumn > div {
            padding: 5px;
        }
        
        /* Tech-style card container */
        .tech-card {
            background: rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .tech-card:hover {
            border-color: rgba(211, 173, 81, 0.3);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        }

        /* Image-text container layout */
        .image-text-container {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .image-container {
            flex: 0 0 auto;
            margin-right: 15px;
        }
        
        .text-container {
            flex: 1 1 auto;
            padding: 15px;
        }
        
        .section-image {
            border-radius: 0;
            border: none;
            display: block;
        }
        
        /* Waveform visualization styling */
        .js-plotly-plot {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            overflow: hidden !important;
        }
        
        /* Audio player styling */
        audio {
            width: 100% !important;
            border-radius: 8px !important;
            background: rgba(32, 36, 45, 0.6) !important;
        }
        
        /* Slider styling */
        .stSlider > div > div {
            background-color: rgba(32, 36, 45, 0.6) !important;
        }
        
        .stSlider > div > div > div > div {
            background-color: #D3AD51 !important;
        }
        
        /* Code-like sections for displaying technical info */
        .code-block {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 13px !important;
            background-color: rgba(32, 36, 45, 0.7) !important;
            color: #e0e0e0 !important;
            padding: 12px !important;
            border-radius: 6px !important;
            border-left: 3px solid #D3AD51 !important;
            white-space: pre !important;
            overflow-x: auto !important;
            margin: 10px 0 !important;
        }
        
        /* Effect pedal display styling */
        .effect-pedal {
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .effect-pedal:hover {
            transform: translateY(-5px);
        }
        
        .effect-pedal img {
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        .effect-pedal:hover img {
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3), 0 0 15px rgba(211, 173, 81, 0.5);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(32, 36, 45, 0.6) !important;
            border-radius: 6px !important;
            color: white !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 500 !important;
            padding: 10px 15px !important;
        }
        
        .streamlit-expanderContent {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 0 0 6px 6px !important;
            padding: 15px !important;
        }
        
        /* Fix for file uploader */
        .stFileUploader > div {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px dashed rgba(211, 173, 81, 0.5) !important;
            border-radius: 6px !important;
            padding: 15px !important;
        }
        
        .stFileUploader > div:hover {
            border-color: #D3AD51 !important;
        }
        
        /* Success message styling */
        .success-message {
            background-color: rgba(76, 175, 80, 0.1) !important;
            border: 1px solid rgba(76, 175, 80, 0.3) !important;
            border-radius: 6px !important;
            padding: 10px 15px !important;
            color: #4CAF50 !important;
        }
        
        /* Warning message styling */
        .stWarning {
            background-color: rgba(255, 193, 7, 0.1) !important;
            border: 1px solid rgba(255, 193, 7, 0.3) !important;
            border-radius: 6px !important;
            color: #FFC107 !important;
        }
        
        /* Error message styling */
        .stError {
            background-color: rgba(244, 67, 54, 0.1) !important;
            border: 1px solid rgba(244, 67, 54, 0.3) !important;
            border-radius: 6px !important;
            color: #F44336 !important;
        }
        
        /* Special radio button styling for tech look */
        div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] div[data-baseweb="radio"] {
            background-color: rgba(32, 36, 45, 0.6) !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        div[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] div[data-baseweb="radio"]:hover {
            background-color: rgba(32, 36, 45, 0.8) !important;
            border-color: rgba(211, 173, 81, 0.3) !important;
        }
        
        /* File uploader styling */
        .uploadedFile {
            background-color: rgba(32, 36, 45, 0.6) !important;
            border-radius: 6px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        .uploadedFile:hover {
            border-color: rgba(211, 173, 81, 0.3) !important;
        }
        
        /* Progress spinner tech styling */
        .stSpinner > div > div {
            border-color: #D3AD51 transparent transparent !important;
        }
        
        /* Data table styling */
        .stDataFrame {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            overflow: hidden !important;
        }
        
        .stDataFrame th {
            background-color: rgba(32, 36, 45, 0.6) !important;
            color: #D3AD51 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            text-transform: uppercase !important;
            font-size: 12px !important;
            padding: 8px 10px !important;
        }
        
        .stDataFrame td {
            background-color: rgba(32, 36, 45, 0.2) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
            color: #333333 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-size: 13px !important;
            padding: 6px 10px !important;
        }
        
        /* Metric display tech styling */
        [data-testid="stMetricValue"] {
            color: #D3AD51 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 700 !important;
            font-size: 32px !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #333333 !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 400 !important;
            font-size: 14px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        [data-testid="stMetricDelta"] {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 500 !important;
            font-size: 14px !important;
        }
    </style>
    """, unsafe_allow_html=True)

def image_with_centered_text(image_path, heading_text, description_text, image_height="400px"):
    """Create an image with a centered text box overlay in the middle.
    
    Args:
        image_path: Path to the image file
        heading_text: Text for the heading
        description_text: Text for the description
        image_height: Height of the image container (default: "400px")
    """
    if os.path.exists(image_path):
        # Encode image to base64 for inline HTML
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode()
            
        html = f"""
        <div style="position: relative; margin-bottom: 20px; height: {image_height}; overflow: hidden; border-radius: 6px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);">
            <img src="data:image/png;base64,{image_base64}" style="width: 100%; height: 100%; object-fit: cover; object-position: center;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                       background-color: rgba(255, 255, 255, 0.85); padding: 20px; border-radius: 6px; 
                       box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); text-align: center; max-width: 80%;">
                <h3 style="color: #333333; font-weight: 700; margin-top: 0; margin-bottom: 5px;">{heading_text}</h3>
                <p style="color: #666666; margin-bottom: 0;">{description_text}</p>
            </div>
        </div>
        """
        return html
    else:
        # Fallback if image doesn't exist
        return f"""
        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 6px; margin-bottom: 20px; text-align: center; height: {image_height}; display: flex; flex-direction: column; justify-content: center;">
            <h3 style="color: #333333; margin-top: 0;">{heading_text}</h3>
            <p style="color: #666666;">{description_text}</p>
        </div>
        """

def show_home_page():
    """Home page content with modern tech styling."""
    logo_path = effect_to_image.get("logo", "")
    
    # Display logo and title with tech styling
    if os.path.exists(logo_path):
        # Encode image to base64 for inline HTML
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()
        
        # Create HTML for inline image and text with tech styling
        st.markdown(
 f"""
 <div class="title-container" style="display: flex; align-items: center; margin-bottom: 30px; background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
 <img src="data:image/jpeg;base64,{logo_base64}" style="width: 80px; height: auto; border-radius: 10px; margin-right: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); background-color: black;">
 <div>
 <div class="title-text" style="font-size: 42px; letter-spacing: -0.5px; margin-bottom: 5px;">ToneClone</div>
 <div class="subtitle-text">Guitar Pedal Effect Platform</div>
 </div>
 </div>
 """,
 unsafe_allow_html=True
)
    else:
        st.title("ToneClone")
    
    st.markdown('<div class="row-container" style="gap: 20px;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
       # Inside your show_home_page() function
        effect_img = effect_to_image.get("effects")
        if os.path.exists(effect_img):
            overlay_html = image_with_centered_text(
                effect_img,
                "Our Mission",
                "To provide future guitarists with the tools to effortlessly replicate tones through machine learning, removing barriers to creativity."
            )
            st.markdown(overlay_html, unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="tech-card">
                <h3 style="color: #667085; font-weight: 600; letter-spacing: -0.5px; margin-top: 0;">
                    How ToneClone Helps
                </h3>
                <p style="color: #333; line-height: 1.6;">
                    ToneClone addresses this challenge by analyzing guitar audio to identify the effects used and provides 
                    accessible, tailored guidance to educate guitarists about effects. This approach allows users to bridge 
                    the gap between hearing and recreating professional-quality sounds, offering a unique combination of 
                    analysis and education.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="tech-card">
                <h3 style="color: #667085; font-weight: 600; letter-spacing: -0.5px; margin-top: 0;">
                    Overcoming the Learning Curve
                </h3>
                <p style="color: #333; line-height: 1.6;">
                    Many beginner guitarists struggle with achieving professional-quality sounds using effects, 
                    especially when trying to emulate popular musicians. Distortion, delay, reverb, and modulation 
                    effects are used to create professional sounds but can be confusing to new players.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Close the container div
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="background: rgba(32, 36, 45, 0.6); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; border: 1px solid rgba(211, 173, 81, 0.3);">
            <p style="color: #D3AD51; font-weight: 600; margin: 0; font-size: 16px;">
                Customer Testimonials
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    col1 = st.columns([1])[0] 

    with col1:

    #     st.markdown("<h3 style='color: #667085; font-weight: 600; letter-spacing: -0.5px; margin-top: 0;'>But don't just take our word for it</h3>""",
    # unsafe_allow_html=True)

        # First testimonial
        st.markdown("""
        <h2 style="font-size: 25px; font-weight: bold; margin-bottom: 0px; font-style: italic;">"So easy!"</h2>
        <p style="color: #444; line-height: 1.6; margin-bottom: 10px;">
        I've always struggled to identify effects in my favorite songs, but ToneClone made it incredibly simple. Just upload, analyze, and boom! I finally know what pedals to buy. This tool has saved me countless hours of research and frustration trying to match tones by ear.
        </p>
        <p style="font-family: cursive; color: #333; font-size: 18px; margin-bottom: 0;">- Shanon, from New York</p>
        """, unsafe_allow_html=True)

        # Second testimonial
        st.markdown("""
        <h2 style="font-size: 25px; font-weight: bold; margin-bottom: 0px; font-style: italic;">"I replicated my favorite artist's sound!"</h2>
        <p style="color: #444; line-height: 1.6; margin-bottom: 10px;">
        I've been trying to nail John Mayer's tone for years without success. ToneClone identified the exact combination of effects he uses - turns out I was missing a distortion pedal! After purchasing the pedal ToneClone suggested, my friends can't believe how authentic my sound is now.
        </p>
        <p style="font-family: cursive; color: #333; font-size: 18px; margin-bottom: 0;">- Pete, from Texas</p>
        """, unsafe_allow_html=True)

        # Third testimonial
        st.markdown("""
        <h2 style="font-size: 25px; font-weight: bold; margin-bottom: 0px; font-style: italic;">"Why didn't this exist sooner?"</h2>
        <p style="color: #444; line-height: 1.6; margin-bottom: 10px;">
        As a guitar teacher, I've needed something like ToneClone for years. Now my students can upload songs they want to learn and immediately understand the effects being used. It's revolutionized my teaching approach and my students are making faster progress than ever before. Simply brilliant.
        </p>
        <p style="font-family: cursive; color: #333; font-size: 18px; margin-bottom: 0;">- Aaron, from California</p>
        """, unsafe_allow_html=True)


    # Tech-styled CTA
    st.markdown(
        """
        <div style="background: rgba(32, 36, 45, 0.6); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; border: 1px solid rgba(211, 173, 81, 0.3);">
            <p style="color: #D3AD51; font-weight: 600; margin: 0; font-size: 16px;">
                Navigate to "Upload & Crop Audio" in the sidebar to classify your guitar effect
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def how_page():
    """How it Works."""
    st.markdown("""
    <style>
    /* Force columns to display side by side with specific width */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 1rem !important;
    }
    
    /* Set explicit width for each column */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 50% !important;
        min-width: 48% !important;
        flex: 1 1 auto !important;
    }
    
    /* Make sure column content takes full width */
    [data-testid="column"] > div {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="title-text" style="font-size: 42px; letter-spacing: -0.5px; margin-bottom: 0px;">ToneClone</div>
        <div class="subtitle-text">Explanation</div>
        """,
    unsafe_allow_html=True)
    
    
    col1 = st.columns([1])[0] 

    with col1:

        st.markdown("<h3 style='color: #667085; font-weight: 600; letter-spacing: -0.5px; margin-top: 0;'>Data Source and Data Science Approach</h3>""",
    unsafe_allow_html=True)

        spectrogram = effect_to_image.get("spectrogram")
        diagram = effect_to_image.get("diagram")
        architecture = effect_to_image.get("architecture")
        Gary = effect_to_image.get("Gary")
        Rex = effect_to_image.get("Rex")
        Kushal = effect_to_image.get("Kushal")
        Jen = effect_to_image.get("Jen")

        with open(architecture, "rb") as img_file:  
            architecture_base64 = base64.b64encode(img_file.read()).decode()
        with open(spectrogram, "rb") as img_file:
            spectrogram_base64 = base64.b64encode(img_file.read()).decode()
        with open(Gary, "rb") as img_file:
            Gary_base64 = base64.b64encode(img_file.read()).decode()
        with open(Rex, "rb") as img_file:
            Rex_base64 = base64.b64encode(img_file.read()).decode()
        with open(Kushal, "rb") as img_file:
            Kushal_base64 = base64.b64encode(img_file.read()).decode()
        with open(Jen, "rb") as img_file:
            Jen_base64 = base64.b64encode(img_file.read()).decode()

        #st.subheader("Data Source and Data Science Approach")
        st.write("Our team has created a simple, user-friendly web application that beginner guitar players can use to learn about different guitar effects used in their favorite songs. Users can upload a .wav file of the guitar segment or full song they would like to learn about. If there is a certain segment they are interested in, they can use the cropping feature to only analyze that portion of the song. Once the upload process is complete, users can click on the classify button.")
        with open(architecture, "rb") as img_file:
                    architecture_base64 = base64.b64encode(img_file.read()).decode()
        st.markdown(
                    f"""
                    <div style="text-align: center; margin-bottom: 15px">
                    <img src="data:image/png;base64,{architecture_base64}" style="width: 40%;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("Under the hood, the .wav file is converted into 10 second spectrograms which are then ultimately represented by a single numpy array. This array is sent up to our custom sagemaker endpoint which hosts our fine-tuned PANN model. Once the input is fed through the model, the endpoint returns the predictions. The predictions are then processed, thresholded, and fed to ChatGPT to provide dynamic user feedback. Ultimately, the output serves the user the top three effects found in the submitted segment, a timeline of where those effects are found in the song, and further descriptive information about each effect such as famous songs using those effects and recommended effect pedals.")
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 15px;">
            <img src="data:image/png;base64,{spectrogram_base64}" style="width: 60%;">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("What makes ToneClone possible is the creation of a new, synthetic dataset of labeled guitar effects. We started with publicly available guitar arrangements and converted them to MIDI. These tracks were then processed through a high-quality virtual guitar instrument to generate realistic, clean guitar recordings. Next, we applied a wide range of digital effects and labeled them for later model training.")
        st.write("This dataset includes 100 songs, each processed with 45 different effect combinations, resulting in more than 450 hours of data.")
        
        with open(diagram, "rb") as img_file:
            diagram_base64 = base64.b64encode(img_file.read()).decode()
        # Display the diagram image
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 15px;">
            <img src="data:image/png;base64,{diagram_base64}" style="width: 60%;">
            </div>
            """,
            unsafe_allow_html=True
        )

    #st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="title-text" style="font-size: 42px; letter-spacing: -0.5px; margin-bottom: 10px;">Meet the ToneClone Team</div>
    """,
    unsafe_allow_html=True)

    # Team member 
    st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin: 5px 0;">
    <div style="width: 100px; height: 100px; border-radius: 50%; overflow: hidden; margin-right: 20px;">
    <img src="data:image/png;base64,{Gary_base64}" style="width: 100%; height: 100%; object-fit: cover;">
    </div>
    <div>
    <h4 style="margin: 0;">Gary Dionne</h4>
    <p style="margin: 0px 0 0 0; color: #666;">
    <a href="mailto:gbdionne@berkeley.edu" style="text-decoration: none; color: #0066cc;">gbdionne@berkeley.edu</a>
    </p>
    </div>
    </div>
    <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.2), rgba(0,0,0,0)); margin: 5px 0;">
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
    <div style="width: 100px; height: 100px; border-radius: 50%; overflow: hidden; margin-right: 20px;">
    <img src="data:image/png;base64,{Jen_base64}" style="width: 100%; height: 100%; object-fit: cover;">
    </div>
    <div>
    <h4 style="margin: 0;">Jennifer Tejeda</h4>
    <p style="margin: 0px 0 0 0; color: #666;">
    <a href="mailto:jtejeda@berkeley.edu" style="text-decoration: none; color: #0066cc;">jtejeda@berkeley.edu</a>
    </p>
    </div>
    </div>
    <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.2), rgba(0,0,0,0)); margin: 5px 0;">
    """,
    unsafe_allow_html=True
    )

    # Team member 3
    st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin: 5px 0;">
    <div style="width: 100px; height: 100px; border-radius: 50%; overflow: hidden; margin-right: 20px;">
    <img src="data:image/png;base64,{Rex_base64}" style="width: 100%; height: 100%; object-fit: cover;">
    </div>
    <div>
    <h4 style="margin: 0;">Rex Gao</h4>
    <p style="margin: 0px 0 0 0; color: #666;">
    <a href="mailto:rexgao@berkeley.edu" style="text-decoration: none; color: #0066cc;">rexgao@berkeley.edu</a>
    </p>
    </div>
    </div>
    <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.2), rgba(0,0,0,0)); margin: 5px 0;">
    """,
    unsafe_allow_html=True
    )

    # Team member 4
    st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin: 5px 0;">
    <div style="width: 100px; height: 100px; border-radius: 50%; overflow: hidden; margin-right: 20px;">
    <img src="data:image/png;base64,{Kushal_base64}" style="width: 100%; height: 100%; object-fit: cover;">
    </div>
    <div>
    <h4 style="margin: 0;">Kushal Gourikrishna</h4>
    <p style="margin: 0px 0 0 0; color: #666;">
    <a href="kgourikrishna@berkeley.edu" style="text-decoration: none; color: #0066cc;">kgourikrishna@berkeley.edu</a>
    </p>
    </div>
    </div>
    """,
    unsafe_allow_html=True
    )



def show_audio_page():
    # Add this custom CSS to force columns to display side by side
    st.markdown("""
    <style>
    /* Force columns to display side by side with specific width */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 1rem !important;
    }
    
    /* Set explicit width for each column */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 50% !important;
        min-width: 48% !important;
        flex: 1 1 auto !important;
    }
    
    /* Make sure column content takes full width */
    [data-testid="column"] > div {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="title-text" style="font-size: 42px; letter-spacing: -0.5px; margin-bottom: 0px;">ToneClone</div>
        <div class="subtitle-text">Classification</div>
        """,
    unsafe_allow_html=True)
    
    
    col1, col2 = st.columns([1, 1])  # Left for upload/crop, right for effects/classification

    with col1:  # LEFT COLUMN: Upload & Crop
        tone_result = classify_tone()

    with col2:  # RIGHT COLUMN: Effects & Classification
        # Add classification buttons at the top of the right column
        st.markdown("<h3 style='color: #667085; font-weight: 600; letter-spacing: -0.5px; margin-top: 0;'>Effects Classification</h3>""",
 unsafe_allow_html=True)
        classify_cols = st.columns(2)
        
        with classify_cols[0]:
            if st.button("Classify Full Song", key="col2_classify_full_button"):
                if 'current_audio_path' in st.session_state:
                    with st.spinner("Classifying full song..."):
                        try:
                            user_education, top_3_effects = classify_song('current_audio_path')
                            
                            # Store results in session state
                            st.session_state['top_3_effects'] = top_3_effects
                            st.session_state['user_education'] = user_education
                            st.session_state['last_classification_type'] = "full_song"
                            st.success("Classification Complete!")
                        except Exception as e:
                            st.error(f"Classification error: {str(e)}")
                else:
                    st.warning("Please upload an audio file first")
        
        with classify_cols[1]:
            if st.button("Classify Cropped", key="col2_classify_cropped_button"):
                if 'cropped_path' in st.session_state:
                    with st.spinner("Classifying cropped segment..."):
                        try:
                            user_education, top_3_effects = classify_song('cropped_path')
                            
                            # Store results in session state
                            st.session_state['top_3_effects'] = top_3_effects
                            st.session_state['user_education'] = user_education
                            st.session_state['last_classification_type'] = "cropped"
                            st.success("Cropped Classification Complete!")
                        except Exception as e:
                            st.error(f"Classification error: {str(e)}")
                else:
                    st.warning("Please preview or save a cropped audio first")
                    


        if 'timeline_visualization' in st.session_state and st.session_state['timeline_visualization']:
            st.markdown("<hr style='border-top: 1px solid #555; margin: 10px 0;'>", unsafe_allow_html=True)
            st.subheader("Timeline of Detected Effects")
            fig = st.session_state["timeline_visualization"]
            st.pyplot(fig, use_container_width=True)
            st.markdown("<hr style='border-top: 1px solid #555; margin: 10px 0;'>", unsafe_allow_html=True)

        if 'top_3_effects' in st.session_state and st.session_state['top_3_effects']:
            st.subheader("Top Effects Across All Segments")
        
            # Use the appropriate column
            cols = st.columns(3)

            for i, effect in enumerate(st.session_state['top_3_effects']):
                image_path = effect_to_image.get(effect, "")
                with cols[i]:
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, width=150)
    
                        # Fake center with div
                        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                        if st.button(f"Learn About {effect.title()}", key=f"{effect}_btn"):
                            st.session_state['selected_effect'] = effect
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.write(f"No image for {effect}.")

            # Show effect info if an image button was clicked
            selected = st.session_state.get('selected_effect')
            if selected:
                st.markdown(f"## {selected.capitalize()}")
    
                effect_data = st.session_state['user_education'][selected]
                st.markdown(f"**Confidence:** {effect_data.get('confidence_judgement', 'N/A')}")
                with st.expander("Description"):
                    st.markdown(f"{effect_data.get('detailed_description_and_common_use_cases', 'N/A')}")
                with st.expander("Notable Artist Usage"):
                    st.markdown(f"{effect_data.get('notable_artist_usage', 'N/A')}")
                with st.expander("Recommended Pedals"):
                    st.markdown(f"{effect_data.get('extensive_pedal_recommendations_with_brief_justifications', 'N/A')}")

def main():
    """Main application entry point."""
    # Apply custom styling
    apply_custom_css()
    
    # Start with sidebar content
    with st.sidebar:
        # Display logo at top of sidebar
        logo_path = effect_to_image.get("logo", "")
        if os.path.exists(logo_path):
            # Encode image to base64 for inline HTML
            with open(logo_path, "rb") as img_file:
                logo_base64 = base64.b64encode(img_file.read()).decode()
            
            # Create HTML for logo and title in sidebar
            st.markdown(
                f"""
            

                <div class="title-container" style="display: flex; align-items: center; margin-bottom: 30px; background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
 <img src="data:image/jpeg;base64,{logo_base64}" style="width: 80px; height: auto; border-radius: 10px; margin-right: 20px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); background-color: black;">
 <div class="sidebar-title">ToneClone</div>
 <div>
                """,
                unsafe_allow_html=True,
            )
        
        # Sidebar navigation
        pages = ["Home", "Upload & Crop Audio", "How It Works"]
        selected_page = st.sidebar.radio("Go to", pages)
    
    # Display selected page in main content area
    if selected_page == "Home":
        show_home_page()
    elif selected_page == "Upload & Crop Audio":
        show_audio_page()
    elif selected_page == "How It Works":
        how_page()

if __name__ == "__main__":
    main()