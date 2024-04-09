import numpy as np
import io
from scipy.signal import spectrogram
import streamlit as st
from st_audiorec import st_audiorec
import plotly.graph_objects as go
import librosa

def plot_spectrogram(audio_data, fs):
    if audio_data.ndim > 1:  # Check if audio_data is not mono
        audio_data = np.mean(audio_data, axis=1) 
    f, t, Sxx = spectrogram(x=audio_data, fs=fs)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Convert to log scale, add a small number to avoid log(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_log,
        x=t,
        y=f,
        colorscale='Viridis'
    )).update_layout(
        title='Spectrogram',
        xaxis=dict(title='Time [sec]'),
        yaxis=dict(title='Frequency [Hz]'),
        )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_time_waveform(audio_data, fs):
    # Generate time axis
    time_axis = np.arange(len(audio_data)) / fs
    
    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=audio_data, mode='lines'))
    
    fig.update_layout(
        title='Time Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    source_selectbox = st.selectbox("Select Image Source", ["Record Audio", "Upload an Audio File"])

    if source_selectbox == "Record Audio":
        wav_audio_data = st_audiorec()
    elif source_selectbox == "Upload an Audio File":
        wav_audio_data = st.file_uploader("Upload an Audio File", ["wav", "mp3"])

    if wav_audio_data is not None:
        try:
            audio_data, fs = librosa.load(io.BytesIO(wav_audio_data))
        except:
            audio_data, fs = librosa.load(io.BytesIO(wav_audio_data.read()))
        
        col1, col2 = st.columns([2, 3])
        if audio_data is not None:
            with col1:
                plot_time_waveform(audio_data, fs)
            with col2:
                plot_spectrogram(audio_data, fs=fs)
