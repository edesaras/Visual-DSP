
import numpy as np
import streamlit as st
from sympy import symbols, lambdify
import plotly.graph_objects as go
from scipy.signal import periodogram, welch
import scipy.signal

def generate_signal(N, component_exprs, noise_amplitude):
    x = symbols('x')
    data = np.zeros(N)
    t = np.linspace(-0.5, 0.5, N, endpoint=True)
    for expr in component_exprs:
        # Parse the symbolic expression for frequency and magnitude
        freq_expr, mag_expr = map(lambda e: lambdify(x, e), expr)
        # Evaluate the expressions over the time array
        freq = freq_expr(t)
        mag = mag_expr(t)
        component_signal = mag * np.cos(2 * np.pi * freq * t)
        data += component_signal
    noise = noise_amplitude * np.random.randn(N)
    data += noise
    return t, data

def plot_periodogram(data, fs):
    freqs, Pxx = periodogram(data, fs, return_onesided=False)
    fig = go.Figure(data=go.Scatter(x=freqs, y=Pxx, mode='lines'))
    fig.update_layout(title='Periodogram', xaxis_title='Frequency (Hz)', yaxis_title='PSD (V^2/Hz)')
    st.plotly_chart(fig)

def plot_welch(data, fs, nperseg, noverlap):
    freqs, Pxx = welch(data, fs, nperseg=nperseg, noverlap=noverlap)
    fig = go.Figure(data=go.Scatter(x=freqs, y=Pxx, mode='lines'))
    fig.update_layout(title="Welch's Method", xaxis_title='Frequency (Hz)', yaxis_title='PSD (V^2/Hz)')
    st.plotly_chart(fig)

def plot_time(data, t):
    fig = go.Figure(data=go.Scatter(x=t, y=data, mode='lines'))
    fig.update_layout(title='Time Plot', xaxis_title='t', yaxis_title='Magnitude')
    st.plotly_chart(fig)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Spectrum Estimation - Visual DSP",
        layout="wide",
    )

    estimation_method = st.selectbox("Estimation Method", ["periodogram", "welch"])
    
    with st.form("freq_component_form"):
        st.write("Enter Frequency Components (Separated by comma \",\")")
        # by specifying frequency and magnitude, a modulation app can be created
        freq_input = st.text_input("Frequency Function").split(",") 
        mag_input = st.text_input("Magnitude Function").split(",")
        N = st.slider('Select N (data length)', min_value=100, max_value=1000, value=300, step=50)
        noise_amplitude = st.slider('Noise Amplitude', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        if estimation_method == "welch":
            nperseg = st.slider('Select nperseg (segment length)', min_value=10, max_value=N, value=50, step=10)
            noverlap = st.slider('Select noverlap (segment overlap)', min_value=10, max_value=nperseg - 1, value=10, step=5)
        detrend_chekbox = st.checkbox("Detrend Data")
        submit = st.form_submit_button("Plot")

    if submit:
        t, data = generate_signal(N, zip(freq_input, mag_input), noise_amplitude)
        if detrend_chekbox:
            data = scipy.signal.detrend(data)
        col1, col2 = st.columns(2)
        with col1:
            plot_time(data, t)
        with col2:
            if estimation_method == "periodogram":
                plot_periodogram(data, N)
            if estimation_method == "welch":
                plot_welch(data, N, nperseg, noverlap)

