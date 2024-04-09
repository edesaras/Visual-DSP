import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.graph_objects as go



def plot_freq_response(W, h, columns):
    with columns[0]:
        fig = go.Figure(data=go.Scatter(x=W, y=np.abs(h), mode="lines"))
        fig.update_layout(title="Magnitude Response Plot", xaxis_title="Magnitude", yaxis_title="Normalized Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with columns[1]:
        fig = go.Figure(data=go.Scatter(x=W, y=np.angle(h), mode="lines"))
        fig.update_layout(title="Phase Response Plot", xaxis_title="Phase", yaxis_title="Normalized Frequency")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Filter Design - Visual DSP",
        layout="wide",
    )

    design_filter, calculate_filter_params = st.tabs(["Design Filter", "Calculate Filter Parameters"])
    with calculate_filter_params:
        filter_type = st.selectbox("Select Filter Type", ["Butterworth", "Chebyshev type I", "Chebyshev type II", "Elliptical"])
        with st.form("Compute Filter Parameters"):
            st.write("Enter Filter Constraints")
            Wp = st.text_input("Passband Edge Frequency(ies)")
            Ws = st.text_input("Stopband Edge Frequency(ies)")
            gpass = st.text_input("Passband maximum loss (dB)")
            gstop = st.text_input("Stopband minimum attenuation (dB)")
            analog = False
            fs = st.text_input("Sampling Frequency")
            compute_filter_params_submit = st.form_submit_button("Compute Parameters")
        if compute_filter_params_submit:
            try:
                Wp = int(Wp)
                Ws = int(Ws)
            except:
                Wp = list(map(int, Wp.split(","))) 
                Ws = list(map(int, Ws.split(",")))
            gpass = int(gpass)
            gstop = int(gstop)
            if filter_type == "Butterworth":
                ord, Wn = scipy.signal.buttord(Wp, Ws, gpass, gstop, analog, fs)
            elif filter_type == "Chebyshev type I":
                ord, Wn = scipy.signal.cheb1ord(Wp, Ws, gpass, gstop, analog, fs)
            elif filter_type == "Chebyshev type II":
                ord, Wn = scipy.signal.cheb2ord(Wp, Ws, gpass, gstop, analog, fs)
            elif filter_type == "Elliptical":
                ord, Wn = scipy.signal.ellipord(Wp, Ws, gpass, gstop, analog, fs)
            st.write(f"ord:{ord}, Wn: {Wn}")

    with design_filter:
        filter_type = st.selectbox("Select Filter Type", ["Butterworth", "Chebyshev type I","Chebyshev type II", "Elliptical", "Bessel", "IIR Comb", "IIR Peak", "IIR Notch"])

        with st.form("Filter Design Form"):
            st.write("Enter Filter Parameters")
            if filter_type in ["IIR Notch", "IIR Peak", "IIR Comb"]:
                Wn = st.text_input("Cut-off Frequency")
                Q = st.slider('Quality Factor', min_value=1, max_value=60, value=5, step=1)
                fs = st.text_input("Sampling Frequency")
                if filter_type in ["IIR Comb"]:
                    ftype = st.selectbox("Select Filter Type", ["Notch", "Peak"])
                    pass_zero = st.checkbox("Pass Zero")
                output = "ba"
                band_type = None
            else:
                N = st.slider('Filter Order', min_value=1, max_value=51, value=3, step=1)
                if filter_type in ["Chebyshev type I", "Elliptical"]:
                    rp = st.slider('Maximum Ripple at Pass Band', min_value=.01, max_value=5., value=1., step=.01)
                if filter_type in ["Chebyshev type II", "Elliptical"]:
                    rs = st.slider('Minimum Attenuation at Stop Band', min_value=1., max_value=50., value=1., step=.01)
                Wn = st.text_input("Cut-off Frequency")
                band_type = st.selectbox("Select Band Type", ["lowpass", "highpass", "bandpass", "bandstop"])
                output = st.selectbox("Output Form", ["ba", "sos", "zpk"])
                fs = st.text_input("Sampling Frequency") 
            
            filter_design_submit = st.form_submit_button("Plot Filter")
        
        if filter_design_submit:
            if band_type in ["lowpass", "highpass"]:
                Wn = int(Wn)
            elif band_type in ["bandpass", "bandstop"]:
                Wn = list(map(int, Wn.split(","))) 
            if filter_type == "Butterworth":
                coeffs = scipy.signal.butter(int(N), Wn, band_type, False, output, int(fs))
            elif filter_type == "Chebyshev type I":
                coeffs = scipy.signal.cheby1(int(N), float(rp), Wn, band_type, False, output, int(fs))
            elif filter_type == "Chebyshev type II":
                coeffs = scipy.signal.cheby2(int(N), float(rs), Wn, band_type, False, output, int(fs))
            elif filter_type == "Elliptical":
                coeffs = scipy.signal.ellip(int(N), float(rp), float(rs), Wn, band_type, False, output, int(fs))
            elif filter_type == "Bessel":
                coeffs = scipy.signal.bessel(int(N), Wn, band_type, False, output, fs=int(fs))
            elif filter_type == "IIR Notch":
                coeffs = scipy.signal.iirnotch(int(Wn), float(Q), float(fs))
            elif filter_type == "IIR Peak":
                coeffs = scipy.signal.iirpeak(int(Wn), float(Q), float(fs))
            elif filter_type == "IIR Comb":
                coeffs = scipy.signal.iircomb(int(Wn), float(Q), ftype, float(fs), pass_zero=pass_zero)
            
            if output == "ba":
                b, a = coeffs
                st.write(f"b:{b}, a:{a}")
                W, h = scipy.signal.freqz(b, a)
            elif output == "sos":
                sos = coeffs
                st.write(f"sos:{sos}")
                W, h = scipy.signal.sosfreqz(sos)
            elif output == "zpk":
                z, p, k = coeffs
                st.write(f"z:{z}, p:{p}, k:{k}")
                W, h = scipy.signal.freqz_zpk(z, p, k)

            plot_columns = st.columns(2, gap="large")
            plot_freq_response(W, h, plot_columns)
