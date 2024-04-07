import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.linalg import toeplitz
from scipy.signal import periodogram
from inspect import getsource
from scipy.signal import stft
from sympy import symbols, lambdify
import sympy

from scipy.signal import spectrogram
from st_audiorec import st_audiorec
from scipy.io import wavfile
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, VideoHTMLAttributes
import cv2
from av import VideoFrame


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

# scipy periodogram as the gold standard
def scipy_periodogram(data, N):
    f, S = periodogram(data, fs=N, return_onesided=False)
    return f, S

# scipy STFT as the gold standard
def scipy_stft(data, N, t, nperseg=100, noverlap=50):
    f, t_stft, Zxx = stft(data, fs=N, nperseg=nperseg, noverlap=noverlap, scaling="psd")
    return f, t_stft, np.abs(Zxx)  # Returning magnitude of STFT

# scipy welchs

# scipy stft

# waterfall, audio

# PERIODOGRAM ESTIMATE
def compute_periodogram(data, N, t):
    periodogram_estimate = (np.abs(np.fft.fft(data) / N) ** 2)
    periodogram_f = np.fft.fftfreq(N, d=(t[1]-t[0]))
    return periodogram_f, periodogram_estimate

# CORRELOGRAM ESTIMATE
def compute_correlogram(data, N, t):
    autocorr = np.correlate(data, data, "full") 
    autocorr = autocorr[-N:] / np.arange(N, 0, -1)
    correlogram_estimate = np.abs(np.fft.fft(autocorr) / N) 
    correlogram_f = np.fft.fftfreq(N, d=(t[1]-t[0]))
    return correlogram_f, correlogram_estimate

# AVERAGED PERIOOGRAM ESTIMATE
def compute_averaged_periodogram(data, N, t, nsegments=5):
    L = N // nsegments
    segmented_data = data.reshape((nsegments, L))
    welchs_estimate = (np.abs(np.fft.fft(segmented_data / L, axis=1)) ** 2).mean(axis=0)
    welchs_f = np.fft.fftfreq(L, d=(t[1]-t[0]))
    return welchs_f, welchs_estimate

# BLACKMAN-TUKEY ESTIMATE
def compute_blackman_tukey(data, N, t):
    windowed_correlogram = np.correlate(data, data, "full") * np.blackman(2*N - 1)
    windowed_correlogram = windowed_correlogram[-N:] / (np.blackman(2*N - 1)[-N:] ** 2).sum()
    windowed_correlogram_estimate = np.abs(np.fft.fft(windowed_correlogram)) / N
    windowed_correlogram_f = np.fft.fftfreq(N, d=(t[1]-t[0]))
    return windowed_correlogram_f, windowed_correlogram_estimate

# AR ESTIMATE
def compute_ar_estimate(data, N, t, p=5):
    r = np.correlate(data, data, mode='full')[-N:] / (np.var(data) * np.arange(N, 0, -1))
    R = toeplitz(r[:p])
    ar_coeffs, _, _, _ = np.linalg.lstsq(R, r[1:p + 1], rcond=None)
    pole_coeffs = np.concatenate(([1], -ar_coeffs))
    arf = np.linspace(-0.5, 0.5, 2048, endpoint=True)
    AR_estimate = 1 / np.abs(np.polyval(pole_coeffs, np.exp(-2j * np.pi * arf))) ** 2
    ar_f = np.linspace(-0.5 / (t[1]-t[0]), 0.5 / (t[1]-t[0]), 2048, endpoint=True) # freqs for plotting
    return ar_f, AR_estimate

# MA ESTIMATE

# ARMA ESTIMATE

# CAPON ESTIMATE

# STFT


def plot_time_domain(container, t, data, title="Time Domain Signal"):
    chart_data = pd.DataFrame({
        'Time': t,
        'Amplitude': data
    }).set_index('Time')
    with container:
        st.subheader(title)
        st.line_chart(chart_data)

def plot_estimate(container, freqs, estimate, title):
    chart_data = pd.DataFrame({
        'Frequency': freqs,
        'Estimate': estimate
    }).set_index('Frequency')
    with container:
        st.subheader(title)
        st.line_chart(chart_data)


def plot_stft(container, f, t_stft, Zxx, title="STFT"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    plt.colorbar(ax.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud'), ax=ax)
    
    with container:
        st.pyplot(fig)

def plot_waterfall_spectrogram(audio_data, fs):
    f, t, Sxx = spectrogram(audio_data, fs)
    Sxx_log = 10 * np.log10(Sxx)  # Convert to log scale

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    T, F = np.meshgrid(t, f)
    ax.plot_surface(T, F, Sxx_log, cmap='viridis')

    ax.set_title('Waterfall Spectrogram')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_zlabel('Log Power')

    st.pyplot(fig)

def plot_spectrogram(audio_data, fs, nperseg, noverlap):
    f, t, Sxx = spectrogram(audio_data, fs, nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.title('Spectrogram')
    st.pyplot(plt)

def plot_frequency_response(theta, H_circle_mag):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theta, y=H_circle_mag, mode='lines', name='|H(e^jω)|'))
    fig.update_layout(title='Frequency Response |H(e^jω)|', xaxis_title='ω (Radians)', yaxis_title='Magnitude', autosize=True)
    st.plotly_chart(fig, use_container_width=True)

def plot_z_domain(poles, zeros, cap_value=5):
    # Create a meshgrid for the z-plane
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Generate unit circle in the complex plane
    theta = np.linspace(-np.pi, np.pi, 100)
    z_circle = np.exp(1j * theta)

    # Compute H(z) along the unit circle
    H_circle = np.ones_like(z_circle, dtype=complex)
    for zero in zeros:
        H_circle *= (z_circle - zero)
    for pole in poles:
        H_circle /= (z_circle - pole)
    H_circle_mag = np.abs(H_circle)
    pos_infty_cap_value = max(H_circle_mag) + 1
    neg_infty_cap_value = min(H_circle_mag) -1

    # Initialize the H(z) surface
    H = np.ones(Z.shape, dtype=complex)
    # Apply zeros
    for zero in zeros:
        H *= (Z - zero)
    # Apply poles
    for pole in poles:
        H /= (Z - pole)
    # Compute magnitude and cap the values
    H_mag = np.abs(H)
    H_mag = np.maximum(np.minimum(H_mag, pos_infty_cap_value), neg_infty_cap_value)  # Cap values to a maximum specified by cap_value

    # Plot surface
    fig = go.Figure(data=[go.Surface(z=H_mag, x=X, y=Y, colorscale='Viridis', cmin=0, cmax=cap_value, opacity=0.6)])

    # Plot unit circle with increased thickness
    fig.add_trace(go.Scatter3d(x=np.real(z_circle), y=np.imag(z_circle), z=H_circle_mag,
                               mode='lines', line=dict(color='red', width=10), name='Unit Circle'))

    # Mark poles with 'x'
    for pole in poles:
        fig.add_trace(go.Scatter3d(x=[pole.real], y=[pole.imag], z=[pos_infty_cap_value],  # Slightly above the unit circle for visibility
                                   mode='markers', marker=dict(color='black', symbol='x', size=5), name='Pole'))

    # Mark zeros with 'o'
    for zero in zeros:
        fig.add_trace(go.Scatter3d(x=[zero.real], y=[zero.imag], z=[neg_infty_cap_value],
                                   mode='markers', marker=dict(color='blue', symbol='circle', size=10), name='Zero'))

    # Update plot layout
    fig.update_layout(title='Surface plot of H(z) with Unit Circle', autosize=True,
                    #   width=700, height=700,
                      margin=dict(l=65, r=50, b=65, t=90))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Frequency response plot
        plot_frequency_response(theta, H_circle_mag)


def video_frame_callback(frame, operations, params):
    img = frame.to_ndarray(format="bgr24")

    if operations.get("flip", False):
        img = cv2.flip(img, 1)

    if operations.get("grayscale", False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if operations.get("blur", False):
        img = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)

    if operations.get("sharpen", False):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)

    if operations.get("laplacian", False):
        img = cv2.Laplacian(img, cv2.CV_64F, ksize=params["laplacian_ksize"])
        img = cv2.convertScaleAbs(img)

    if operations.get("sobel_x", False):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=params["sobel_x_ksize"])
        img = cv2.convertScaleAbs(sobelx)

    if operations.get("sobel_y", False):
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=params["sobel_y_ksize"])
        img = cv2.convertScaleAbs(sobely)

    # Apply averaging blur if selected
    if operations.get("averaging", False):
        img = cv2.blur(img, (params["averaging_ksize"], params["averaging_ksize"]))

    # Apply median blur if selected
    if operations.get("median", False):
        img = cv2.medianBlur(img, params["median_ksize"])

    # Apply bilateral filtering if selected
    if operations.get("bilateral", False):
        img = cv2.bilateralFilter(img, params["bilateral_d"], params["bilateral_sigmaColor"], params["bilateral_sigmaSpace"])

    # Apply simple threshold if selected
    if operations.get("threshold", False):
        _, img = cv2.threshold(img, params["thresh_value"], 255, cv2.THRESH_BINARY)

    # Apply adaptive mean thresholding if selected
    if operations.get("adaptive_thresh_mean", False):
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, params["adaptive_ksize"], params["adaptive_C"])

    # Apply adaptive gaussian thresholding if selected
    if operations.get("adaptive_thresh_gaussian", False):
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, params["adaptive_ksize"], params["adaptive_C"])

    # Morphological operations
    if operations.get("erosion", False) or operations.get("dilation", False) or \
       operations.get("opening", False) or operations.get("closing", False) or \
       operations.get("morph_gradient", False) or operations.get("top_hat", False) or \
       operations.get("black_hat", False):

        # Grayscale conversion for morphological operations
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params["morph_ksize"], params["morph_ksize"]))
        
        if operations.get("erosion", False):
            img = cv2.erode(img, kernel, iterations=params["morph_iterations"])
        
        if operations.get("dilation", False):
            img = cv2.dilate(img, kernel, iterations=params["morph_iterations"])
        
        if operations.get("opening", False):
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=params["morph_iterations"])
        
        if operations.get("closing", False):
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=params["morph_iterations"])
        
        if operations.get("morph_gradient", False):
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=params["morph_iterations"])
        
        if operations.get("top_hat", False):
            img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=params["morph_iterations"])
        
        if operations.get("black_hat", False):
            img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=params["morph_iterations"])

        # Convert grayscale back to BGR for consistent output format
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert back to VideoFrame if any operation is applied
    return VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="Visual DSP", layout="wide")

    spectral_estimation, time_freq, audio, image, pole_zero = st.tabs(["spectral_estimation", "time_freq", "audio", "image", "pole_zero"])

    with image:
        st.title("Camera Effects")

        # Streamer initialization
        webrtc_streamer(key="example", video_frame_callback=lambda frame: video_frame_callback(frame, operations, params), 
                        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, style={"width": "50%"}),)
        operations = {
            "flip": st.sidebar.checkbox("Flip"),
            "grayscale": st.sidebar.checkbox("Grayscale"),
            "blur": st.sidebar.checkbox("Blur"),
            "sharpen": st.sidebar.checkbox("Sharpen"),
            "laplacian": st.sidebar.checkbox("Laplacian Edge Detection"),
            "sobel_x": st.sidebar.checkbox("Sobel Edge Detection - X Direction"),
            "sobel_y": st.sidebar.checkbox("Sobel Edge Detection - Y Direction"),
            "averaging": st.sidebar.checkbox("Averaging Blur"),
            "median": st.sidebar.checkbox("Median Blur"),
            "bilateral": st.sidebar.checkbox("Bilateral Filtering"),
            "threshold": st.sidebar.checkbox("Simple Threshold"),
            "adaptive_thresh_mean": st.sidebar.checkbox("Adaptive Threshold Mean"),
            "adaptive_thresh_gaussian": st.sidebar.checkbox("Adaptive Threshold Gaussian"),
            "erosion": st.sidebar.checkbox("Erosion"),
            "dilation": st.sidebar.checkbox("Dilation"),
            "opening": st.sidebar.checkbox("Opening"),
            "closing": st.sidebar.checkbox("Closing"),
            "morph_gradient": st.sidebar.checkbox("Morphological Gradient"),
            "top_hat": st.sidebar.checkbox("Top Hat"),
            "black_hat": st.sidebar.checkbox("Black Hat"),
        }

        # Parameters for operations
        params = {}
        with st.sidebar:
            if operations["laplacian"]:
                params["laplacian_ksize"] = st.slider("Laplacian Kernel Size", 1, 7, 3, step=2)
            if operations["sobel_x"]:
                params["sobel_x_ksize"] = st.slider("Sobel X Kernel Size", 1, 7, 3, step=2)
            if operations["sobel_y"]:
                params["sobel_y_ksize"] = st.slider("Sobel Y Kernel Size", 1, 7, 3, step=2)
            if operations["averaging"]:
                params["averaging_ksize"] = st.slider("Averaging Kernel Size", 1, 31, 5, step=2)
            if operations["median"]:
                params["median_ksize"] = st.slider("Median Blur Kernel Size", 1, 31, 5, step=2)
            if operations["bilateral"]:
                params["bilateral_d"] = st.slider("Bilateral Filter Diameter", 1, 9, 5)
                params["bilateral_sigmaColor"] = st.slider("Bilateral Filter SigmaColor", 10, 250, 75)
                params["bilateral_sigmaSpace"] = st.slider("Bilateral Filter SigmaSpace", 10, 250, 75)
            if operations["threshold"]:
                params["thresh_value"] = st.slider("Threshold Value", 0, 255, 127)
            if operations["adaptive_thresh_mean"] or operations["adaptive_thresh_gaussian"]:
                params["adaptive_ksize"] = st.slider("Adaptive Threshold Kernel Size", 3, 31, 11, step=2)
                params["adaptive_C"] = st.slider("Adaptive Threshold C Value", 0, 20, 2)
            if any(operations[op] for op in ["erosion", "dilation", "opening", "closing", "morph_gradient", "top_hat", "black_hat"]):
                params["morph_ksize"] = st.slider("Kernel Size", 1, 31, 5, step=2)
                params["morph_iterations"] = st.slider("Iterations", 1, 10, 1)

    with pole_zero:
        st.title("Z-Domain Plot with Poles, Zeros, and Unit Circle")
        # Input for poles and zeros
        poles_input = st.text_input("Enter poles separated by comma (a+bi format)", "0.5+0.5j,0.5-0.5j")
        zeros_input = st.text_input("Enter zeros separated by comma (a+bi format)", "-0.5+0.5j,-0.5-0.5j")

        # Parse inputs
        poles = [complex(p.strip()) for p in poles_input.split(',')]
        zeros = [complex(z.strip()) for z in zeros_input.split(',')]

        # Button to plot
        if st.button("Plot Z-Domain"):
            plot_z_domain(poles, zeros)

    with audio:
        st.subheader("Audio Spectrogram")

        # Display the audio recorder
        audio_data = st_audiorec()

        if audio_data is not None:
            # The audio data is a tuple with (sample_rate, audio_samples)
            fs, audio = wavfile.read(io.BytesIO(audio_data))

            if audio.ndim > 1:  # Stereo to mono conversion if needed
                audio = np.mean(audio, axis=1)

            # Plot the waterfall spectrogram
            plot_waterfall_spectrogram(audio, fs)

        audio_file = st.file_uploader("Upload an audio file", type=['wav'])
        if audio_file is not None:
            # Read audio file
            fs, audio_data = wavfile.read(audio_file)
            if audio_data.ndim > 1:  # Stereo to mono conversion if needed
                audio_data = np.mean(audio_data, axis=1)

            st.audio(audio_file, format='audio/wav')

            # Spectrogram parameters
            nperseg = st.slider('Window Length', min_value=32, max_value=512, value=256, step=16)
            noverlap = st.slider('Overlap Length', min_value=0, max_value=nperseg - 1, value=int(0.75 * nperseg), step=16)

            plot_button = st.button('Generate Spectrogram')
            if plot_button:
                plot_spectrogram(audio_data, fs, nperseg, noverlap)

    with spectral_estimation:
        # Use radio buttons to toggle between sections
        analysis_type = st.radio(
            "Choose the type of analysis:",
            ('Periodogram Methods', 'Spectrogram Methods'),
            index=0  # Default to the first option, 'Periodogram Methods'
        )

        st.subheader("Synthetic Spectrogram and Periodogram")
        # data length and noise amp. settings 
        N = st.sidebar.slider('Select N (data length)', min_value=100, max_value=1000, value=300, step=50)
        noise_amplitude = st.sidebar.slider('Noise Amplitude', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        
        if analysis_type == 'Periodogram Methods':
            # Include your Periodogram Methods code here
            st.sidebar.subheader("Periodogram Settings")
            # ... (rest of your periodogram code) ...

        elif analysis_type == 'Spectrogram Methods':
            # Include your Spectrogram Methods code here
            # Parameters for STFT
            nperseg = st.sidebar.slider('Window Length for STFT', min_value=32, max_value=N, value=100, step=2)
            noverlap = st.sidebar.slider('Overlap Length for STFT', min_value=0, max_value=nperseg-1, value=nperseg//2, step=1)

        # Checkbox for selecting charts
        selected_charts = {
            'scipy_periodogram': st.sidebar.checkbox('Scipy Periodogram', value=True),
            'periodogram_estimate': st.sidebar.checkbox('Periodogram Method', value=True),
            'correlogram_estimate': st.sidebar.checkbox('Correlogram Method', value=True),
            'averaged_periodogram': st.sidebar.checkbox('Averaged Periodogram', value=True),
            'blackman_tukey_estimate': st.sidebar.checkbox('Blackman-Tukey Method', value=True),
            'ar_estimate': st.sidebar.checkbox('AR Estimate', value=True),
            'stft_estimate': st.sidebar.checkbox('STFT Method', value=True)
        }

        # sidebar freq component states
        if 'components' not in st.session_state:
            st.session_state['components'] = []
        
        if 'delete_idx' in st.session_state:
            # Delete the marked component, then remove the 'delete_idx' flag.
            del st.session_state['components'][st.session_state['delete_idx']]
            del st.session_state['delete_idx']

        # sidebar freq component functionality
        with st.sidebar:
            st.write("Frequency Components:")
            for idx, expr in enumerate(st.session_state['components'].copy()):
                col1, col2, col3 = st.columns([3, 3, 1])
                freq_expr = col1.text_input(f'Freq {idx+1} (as expression)', value=str(expr[0]), key=f'freq{idx}')
                mag_expr = col2.text_input(f'Mag {idx+1} (as expression)', value=str(expr[1]), key=f'mag{idx}')
                # Parse the expressions using sympy
                try:
                    freq_parsed = sympy.sympify(freq_expr)
                    mag_parsed = sympy.sympify(mag_expr)
                    st.session_state['components'][idx] = (freq_parsed, mag_parsed)
                except sympy.SympifyError as e:
                    st.error(f'Error in parsing expression: {e}')

                if col3.button('🗑️', key=f'del{idx}'):
                    # Delete the component and refresh the page
                    st.session_state['components'].pop(idx)
                    st.rerun()

            if st.button('➕'):
                st.session_state['components'].append((0.0, 0.0))
                st.rerun()

        # When 'Update Plot' is clicked, compute the spectral estimates and store in session state
        if st.sidebar.button('Update Plot'):
            # Filter out empty or invalid components
            try:
                valid_components = [expr for expr in st.session_state['components']]
                # generate data and calculate spectrum
                t, data = generate_signal(N, valid_components, noise_amplitude)
                st.session_state['time_data'] = t
                st.session_state['data'] = data
                st.session_state['spectrum_data'] = {
                    'scipy_periodogram': scipy_periodogram(data, N) if selected_charts['scipy_periodogram'] else None,
                    'periodogram_estimate': compute_periodogram(data, N, t) if selected_charts['periodogram_estimate'] else None,
                    'correlogram_estimate': compute_correlogram(data, N, t) if selected_charts['correlogram_estimate'] else None,
                    'averaged_periodogram': compute_averaged_periodogram(data, N, t) if selected_charts['averaged_periodogram'] else None,
                    'blackman_tukey_estimate': compute_blackman_tukey(data, N, t) if selected_charts['blackman_tukey_estimate'] else None,
                    'ar_estimate': compute_ar_estimate(data, N, t) if selected_charts['ar_estimate'] else None,
                    'stft_estimate': scipy_stft(data, N, t, nperseg=nperseg, noverlap=noverlap) if selected_charts['stft_estimate'] else None
                }
            except Exception as e:
                st.error(f"An error occurred: {e}")
        
        # display the code for selected methods
        with st.expander("Show code for computation"):  
            if selected_charts['scipy_periodogram']:
                st.subheader("Scipy Periodogram Code")
                st.code(getsource(scipy_periodogram))

            if selected_charts['periodogram_estimate']:
                st.subheader("Periodogram Estimate Code")
                st.code(getsource(compute_periodogram))

            if selected_charts['correlogram_estimate']:
                st.subheader("Correlogram Estimate Code")
                st.code(getsource(compute_correlogram))

            if selected_charts['averaged_periodogram']:
                st.subheader("Averaged Periodogram Estimate Code")
                st.code(getsource(compute_averaged_periodogram))

            if selected_charts['blackman_tukey_estimate']:
                st.subheader("Blackman-Tukey Estimate Code")
                st.code(getsource(compute_blackman_tukey))

            if selected_charts['ar_estimate']:
                st.subheader("AR Estimate Code")
                st.code(getsource(compute_ar_estimate))

            if selected_charts['stft_estimate']:
                st.subheader("STFT Estimate Code")
                st.code(getsource(scipy_stft))
        
        # Plotting logic
        if 'time_data' in st.session_state and 'data' in st.session_state:
            # Time domain plot
            col1, _= st.columns(2)
            plot_time_domain(col1, st.session_state['time_data'], st.session_state['data'], "Time Domain Signal")

        if 'spectrum_data' in st.session_state:
            spectrum_data = st.session_state['spectrum_data']
            for title, values in [(key, spectrum_data[key]) for key in spectrum_data if spectrum_data[key]]:
                if selected_charts[title]:
                    if title == 'stft_estimate':
                        f, t_stft, Zxx = values
                        col1, _ = st.columns(2)
                        plot_stft(col1, f, t_stft, Zxx, "STFT Magnitude")
                    elif spectrum_data[title]:
                        freqs, estimate = values
                        col1, _ = st.columns(2)
                        plot_estimate(col1, freqs, estimate, title)


if __name__ == "__main__":
    main()
