import streamlit as st
import numpy as np
import scipy.signal
import plotly.graph_objects as go
from sympy import symbols, latex, Mul, Matrix, Function
from sympy.core.numbers import I

if __name__ == "__main__":
    st.set_page_config(
        page_title="System Representations - Visual DSP",
        layout="wide",
    )
    
    st.title("System Representations")
    representation_selection = st.selectbox("Select System Representation", ["Numerator/Denominator", "Zeros/Poles"])

    if representation_selection == "Zeros/Poles":
        poles_input = st.text_input("Enter poles separated by comma (a+bi format)", "0.5+0.5j,0.5-0.5j")
        zeros_input = st.text_input("Enter zeros separated by comma (a+bi format)", "-0.5+0.5j,-0.5-0.5j")
        # Parse inputs
        poles = [complex(p.strip()) for p in poles_input.split(',')]
        zeros = [complex(z.strip()) for z in zeros_input.split(',')]
        system = scipy.signal.dlti(zeros, poles, 1)
        num, den = scipy.signal.zpk2tf(zeros, poles, 1)
        a, b, c, d = scipy.signal.zpk2ss(zeros, poles, 1)

    elif representation_selection == "Numerator/Denominator":
        num_input = st.text_input("Enter numerator coeffs separated by comma (a+bi format)", "0.5+0.5j,0.5-0.5j")
        den_input = st.text_input("Enter denominator coeffs separated by comma (a+bi format)", "-0.5+0.5j,-0.5-0.5j")
        # Parse inputs
        num = [complex(c.strip()) for c in num_input.split(',')]
        den = [complex(c.strip()) for c in den_input.split(',')]
        zeros, poles, _ = scipy.signal.tf2zpk(num, den)
        a, b, c, d  = scipy.signal.tf2ss(num, den)

    repr_columns = st.columns(3, gap="large")
    z = symbols("z")
    with repr_columns[0]:
        st.subheader("Zeros and Poles")
        numerator = Mul(*[(z - (zero.real + zero.imag*I)) for zero in zeros])
        denominator = Mul(*[(z - (pole.real + pole.imag*I)) for pole in poles])
        tf = latex(numerator / denominator)
        st.latex(tf)
    with repr_columns[1]:
        st.subheader("Numerator and Denominator Coeffs")
        numerator = sum(c*z**i for i, c in enumerate(reversed(num)))
        denominator = sum(c*z**i for i, c in enumerate(reversed(den)))
        tf = latex(numerator / denominator)
        st.latex(tf)
    with repr_columns[2]:
        st.subheader("State Space Representation")
        A, B, C, D = Matrix(a), Matrix(b), Matrix(c), Matrix(d)
        st.latex(f"\\dot{{X}}(t) = {latex(A)}X(t) + {latex(B)}U(t)")
        st.latex(f"Y(t) = {latex(C)}X(t) + {latex(D)}U(t)")



    # dlti = scipy.signal.dlti(numerator, denominator, dt=0.1)
    # dlti = scipy.signal.dlti(zeros, poles, 1, dt=0.1)
    # tout, yout = scipy.signal.dstep(dlti, 0., n=100)
    # tout, yout = scipy.signal.dimpulse(dlti, 0., n=100)
    # yout = yout[0].flatten()
    # fig = go.Figure(data=go.Scatter(x=tout, y=yout, mode="lines"))
    # fig.show()
