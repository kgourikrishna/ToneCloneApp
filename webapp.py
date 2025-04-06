import streamlit as st
import base64

st.set_page_config(page_title="ToneClone", page_icon="🔊", layout="wide")

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import plotly.graph_objects as go
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
    "fuzz": os.path.join(IMAGES_DIR, "pedal_fuz.png"),
    "auto filter": os.path.join(IMAGES_DIR, "pedal_FLT.png"),
    "overdrive": os.path.join(IMAGES_DIR, "pedal_odv.png"),
    "octaver": os.path.join(IMAGES_DIR, "pedal_oct.png"),
    "tremolo": os.path.join(IMAGES_DIR, "pedal_TRM.png"),  
    "phaser": os.path.join(IMAGES_DIR, "pedal_PHZ.png"),
    "hall reverb": os.path.join(IMAGES_DIR, "pedal_HLL.png"),
    "logo": os.path.join(IMAGES_DIR, "guitarlogo.png"),
    "Gary": os.path.join(IMAGES_DIR, "Gary.png"),
    "effects": os.path.join(IMAGES_DIR, "effects.png"),
    "spectrogram": os.path.join(IMAGES_DIR, "spectrogram.png"),
    "diagram": os.path.join(IMAGES_DIR, "data_diagram.png")
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

def classify_song(file_path):
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
    tcanalyzer.chatgpt_prompt()
    user_education = tcanalyzer.parse_effects_json()

    return user_education, top_3_effects

def classify_tone():
    """Main function for tone classification interface."""
    st.markdown("<h3 style='color: #fcfcfc;'>Upload your song sample for your ToneClone result!</h3>", unsafe_allow_html=True)

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
            'current_audio_path', 'current_audio_duration'
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
    start_time, end_time = st.slider(
        "Select Time Range (seconds)", 
        min_value=0.0,
        max_value=duration,
        value=(0.0, duration),  
        step=0.1
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
    
    # # Apply crop and preview
    # start_ms, end_ms = int(start_time * 1000), int(end_time * 1000)
    # cropped_audio = audio[start_ms:end_ms]
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     if st.button("🎧 Preview Crop"):
    #         preview_path = Path(AUDIO_DIR) / f"preview_{uploaded_file.name}"
    #         cropped_path = Path(AUDIO_DIR) / f"cropped_{uploaded_file.name}"
            
    #         # Save both versions
    #         cropped_audio.export(preview_path, format="wav")
    #         cropped_audio.export(cropped_path, format="wav")
            
    #         # Store paths in session state
    #         st.session_state['preview_path'] = str(preview_path)
    #         st.session_state['cropped_path'] = str(cropped_path)
    #         st.session_state['cropped_duration'] = end_time - start_time
            
    #         # Play preview
    #         st.audio(str(preview_path))
    #         st.success("🔊 Previewing cropped audio. File is ready for classification.")
    
    # with col2:
    #     if st.button("💾 Save Cropped File"):
    #         save_path = Path(AUDIO_DIR) / f"cropped_{uploaded_file.name}"
            
    #         # Export the cropped audio to the new file
    #         cropped_audio.export(save_path, format="wav")
            
    #         # Let the user download the saved file
    #         st.success(f"✅ Cropped audio saved as {save_path.name}")
            
    #         with open(save_path, "rb") as file:
    #             st.download_button(
    #                 label="Download Cropped Audio",
    #                 data=file,
    #                 file_name=save_path.name,
    #                 mime="audio/wav"
    #             )
    

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
    st.markdown("""
    <style>
        /* Import Dancing Script font and a nice sans-serif for navigation */
        @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&family=Montserrat:wght@400;500;600&display=swap');
        
        /* Black background */
        .stApp {
            background: #000000;
            background-attachment: fixed;
        }

        /* Sidebar Styling with logo space at top */
        .stSidebar {
            background-color: #121212 !important;
            border-right: 1px solid #333333;
        }
        
        /* logo container in sidebar */
        .sidebar-logo-container {
            padding: 20px 10px;
            text-align: center;
            border-bottom: 1px solid #333333;
            margin-bottom: 20px;
           
            border-radius: 0 0 10px 10px;
        }
                
        
        
        .sidebar-logo {
            width: 100px;
            height: auto;
            margin-bottom: 10px;
            border-radius: 10px;
        }
        
        /* ToneClone title with artistic background but no border */
        .sidebar-title {
            font-family: 'Dancing Script', cursive;
            font-size: 36px;
            font-weight: bold;
            color: #D3AD51;
            margin: 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            padding: 8px 15px;
            display: inline-block;
            position: relative;
            z-index: 1;
            border-radius: 8px;
        }
        
        .sidebar-title::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            border-radius: 6px;
            opacity: 0.8;
        }

        /* Navigation styling - smaller font */
        div[data-baseweb="radio"] label {
            font-family: 'Montserrat', sans-serif !important;
            color: #cccccc !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            letter-spacing: 0.3px !important;
        }
                
        /* button */
        div[data-baseweb="radio"] input:checked + label {
            color: #4d94ff !important;
            font-weight: 600 !important;
        }

        /* Text styling for dark mode */
        html, body, .stApp, .stText, p, label, .stTitle, .stHeader, .stSubheader {
            color: #e0e0e0 !important;
            font-family: 'Montserrat', sans-serif;
    
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

        /* Main page title style - gradient effect without border */
        .title-text {
            font-family: 'Dancing Script', cursive;
            font-size: 80px;
            font-weight: bold;
            text-align: left;
            color: #D3AD51;
            margin: 0;
            text-shadow: 2px 2px 3px rgba(0,0,0,0.2);
            display: inline-block;
            padding: 15px 25px;
            position: relative;
            z-index: 1;
            border-radius: 12px;
        }
        
        .title-text::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            border-radius: 10px;
            opacity: 0.8;
            filter: blur(2px);
        }
        
        /* Add grey borders around columns and ensure horizontal layout */
        .stColumn {
            border: 1px solid #CCCCCC;
            border-radius: 8px;
            padding: 10px;
            margin: 5px;
            background-color: #777777;
            display: inline-block !important;
            flex: 1 1 0 !important;
            width: 33% !important;
        }
        
        
        
                
        /* Keep borders for home page columns */
        .row-container .stColumn {
            border: 1px solid #CCCCCC;
            /* other properties */
        }

        /* Remove borders for button columns */
        div:not(.row-container) .stColumn {
            border: none !important;
        }
        
        /* Container for columns to enforce horizontal layout */
        .row-container {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
            gap: 10px;
        }
        
        /* Fix padding inside columns */
        .stColumn > div {
            padding: 5px;
        }
        
        

        /* Image-text container layout */
        .image-text-container {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .image-container {
            flex: 0 0 auto;
            margin-right: 15px;
        }
        
        .text-container {
            flex: 1 1 auto;
        }
        
        .section-image {
            border-radius: 8px;
            border: 1px solid #999999;
        }
                
        
                
    </style>
    """, unsafe_allow_html=True)

def show_home_page():
    """Home page content."""
    logo_path = effect_to_image.get("logo", "")
    
    # Display logo and title 
    if os.path.exists(logo_path):
        # Encode image to base64 for inline HTML
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()
        
        # Create HTML for inline image and text
        st.markdown(
            f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap');
                .title-container {{
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    padding: 10px;
                    justify-content: flex-start;
                }}
                .logo {{
                    width: 150px;
                    height: auto;
                    border-radius: 15px;
                }}
                .title-text {{
                    font-family: 'Dancing Script', cursive;
                    font-size: 80px;
                    font-weight: bold;
                    text-align: left;
                    color: #D3AD51;
                    margin: 0;
                }}
            </style>
            <div class="title-container">
                <img src="data:image/jpeg;base64,{logo_base64}" class="logo">
                <div class="title-text">ToneClone</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title("ToneClone")
    
    st.markdown(
    """
    <h3 style='color:#839196; text-align: center; font-weight: bold;'>Our Mission</h3>
    <p style='color:#ffffff; font-size:20px; font-style:italic; text-align: center; line-height: 1.4; max-width: 800px; margin: 0 auto;'>
    To provide future guitarists with the tools to effortlessly replicate tones through
    machine learning, removing barriers to creativity.
    </p>
    """,
    unsafe_allow_html=True,
)


    # Two-column layout with container div for horizontal layout
    st.markdown('<div class="row-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        # Content for left column 
        st.markdown("<h3 style='color:#D3AD51;'>Overcoming the Learning Curve for Guitar Effects</h3>", unsafe_allow_html=True)
        st.write("Many beginner guitarists struggle with achieving professional-quality sounds using effects, especially when trying to emulate popular musicians. Distortion, delay, reverb, and modulation effect are used to create professional sounds but can be confusing to new players.")
        # Image first, taking up full width of the column
        effect_img = effect_to_image.get("effects")
        
        if os.path.exists(effect_img):
            # Encode image to base64 for inline HTML
            with open(effect_img, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode()
                
            # Just display the image first, taking up full width
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 15px;">
                    <img src="data:image/png;base64,{image_base64}" style="width: 80%; border-radius: 8px; border: 1px solid #999999;">
                </div>
                """, 
                unsafe_allow_html=True
            )
        



    with col2:
        # Image first, taking up full width of the column
        gary_img = effect_to_image.get("Gary")
        
        if os.path.exists(gary_img):
            # Encode image to base64 for inline HTML
            with open(gary_img, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode()
                
            # Just display the image first, taking up full width
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 15px;">
                    <img src="data:image/png;base64,{image_base64}" style="width: 100%;">
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        # Then add the title and text below the image
        st.markdown("<h3 style='color:#D3AD51;'>How ToneClone Helps</h3>", unsafe_allow_html=True)
        st.write("ToneClone addresses this challenge by analyzing guitar audio to identify the effects used and provides accessible, tailored guidance to educate guitarists about effects. This approach allows users to bridge the gap between hearing and recreating professional-quality sounds, offering a unique combination of analysis and education that is not available in other products. A simple and intuitive method for new guitar players to analyze guitar tones and receive a step by step instruction to replicate.")
    
    # Close the container div
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("""
    <b style="color: #edbb24;">Navigate the sidebar to upload an audio file.</b><br>
    """, unsafe_allow_html=True)


def how_page():
    """How it Works."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap');
    </style>
    <h1 style='font-family: "Dancing Script", cursive; color: #D3AD51;'>ToneClone Explanation</h1>
    """, unsafe_allow_html=True)
    st.subheader("Data Source and Data Science Approach")
    st.write("What makes ToneClone possible is the creation of a new, synthetic dataset of labeled guitar effects. We started with publicly available guitar arrangements and converted them to MIDI. These tracks were then processed through a high-quality virtual guitar instrument to generate realistic, clean guitar recordings. Next, we applied a wide range of digital effects and labeled them for later model training.")
    st.write("This dataset includes 100 songs, each processed with 45 different effect combinations, resulting in more than 450 hours of data.")
    
    spectrogram = effect_to_image.get("spectrogram")
    diagram = effect_to_image.get("diagram")
    
    # Custom CSS to ensure black background for columns
    st.markdown("""
    <style>
    div.stColumn {
        background-color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if spectrogram is not None and os.path.exists(spectrogram):
            # Encode image to base64 for inline HTML
            with open(spectrogram, "rb") as img_file:
                spectrogram_base64 = base64.b64encode(img_file.read()).decode()
            # Display the spectrogram image
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 15px; background-color: black;">
                <img src="data:image/png;base64,{spectrogram_base64}" style="width: 100%;">
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        if diagram is not None and os.path.exists(diagram):
            # Encode image to base64 for inline HTML
            with open(diagram, "rb") as img_file:
                diagram_base64 = base64.b64encode(img_file.read()).decode()
            # Display the diagram image
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 15px; background-color: black;">
                <img src="data:image/png;base64,{diagram_base64}" style="width: 100%;">
                </div>
                """,
                unsafe_allow_html=True
            )

def show_audio_page():
    """Audio upload and analysis page."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Dancing+Script&display=swap');
        </style>
        <h1 style='font-family: "Dancing Script", cursive; color: #D3AD51;'>ToneClone Classification</h1>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])  # Left for upload/crop, right for effects/classification

    with col1:  # LEFT COLUMN: Upload & Crop
        tone_result = classify_tone()
        #st.write(f"**Result:** {tone_result}")

    with col2:  # RIGHT COLUMN: Effects & Classification
        # Add classification buttons at the top of the right column
        st.markdown("<h3 style='color: #fcfcfc;'>Effects Classification</h3>", unsafe_allow_html=True)
        classify_cols = st.columns(2)
        
        with classify_cols[0]:
            if st.button("Classify Full Song", key="col2_classify_full_button"):
                if 'current_audio_path' in st.session_state:
                    with st.spinner("Classifying full song..."):
                        try:
                            file_path = st.session_state['current_audio_path']
                            user_education, top_3_effects = classify_song(file_path)
                            
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
                            cropped_path = st.session_state['cropped_path']
                            user_education, top_3_effects = classify_song(cropped_path)
                            
                            # Store results in session state
                            st.session_state['top_3_effects'] = top_3_effects
                            st.session_state['user_education'] = user_education
                            st.session_state['last_classification_type'] = "cropped"
                            st.success("Cropped Classification Complete!")
                        except Exception as e:
                            st.error(f"Classification error: {str(e)}")
                else:
                    st.warning("Please preview or save a cropped audio first")
                    
        # Add a divider
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

                #st.write(st.session_state['raw_predictions'])
                #st.write(st.session_state['summarized_segment_results'])
                #st.write(st.session_state['predictions_summary_for_llm'])
                #st.write(st.session_state['user_education'])
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
                <div class="sidebar-logo-container">
                    <img src="data:image/jpeg;base64,{logo_base64}" class="sidebar-logo">
                    <div class="sidebar-title">ToneClone</div>
                </div>
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