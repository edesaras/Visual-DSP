import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sympy import symbols, latex, Mul, simplify, expand, collect
from sympy.core.numbers import I

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

    # Plot unit circle
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
        plot_frequency_response(theta, H_circle_mag)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Spectrum Estimation - Visual DSP",
        layout="wide",
    )
    
    st.title("Z-Domain, Poles, Zeros, and Unit Circle")
    # Input for poles and zeros
    poles_input = st.text_input("Enter poles separated by comma (a+bi format)", "0.5+0.5j,0.5-0.5j")
    zeros_input = st.text_input("Enter zeros separated by comma (a+bi format)", "-0.5+0.5j,-0.5-0.5j")

    # Parse inputs
    poles = [complex(p.strip()) for p in poles_input.split(',')]
    zeros = [complex(z.strip()) for z in zeros_input.split(',')]

    # Symbolic variable
    z = symbols('z')
    numerator = Mul(*[(z - (zero.real + zero.imag*I)) for zero in zeros])
    denominator = Mul(*[(z - (pole.real + pole.imag*I)) for pole in poles])
    H_z = numerator / denominator
    H_z_simplified = collect(expand(numerator), z)  / collect(expand(denominator), z)
    latex_simplified = latex(H_z_simplified)
    latex_product = latex(H_z)
    st.latex(f"{latex_product} = {latex_simplified}")

    # Button to plot
    if st.button("Plot"):
        plot_z_domain(poles, zeros)
