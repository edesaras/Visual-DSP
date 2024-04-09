import streamlit as st


if __name__ == "__main__":
    st.set_page_config(
        page_title="Visual DSP",
        page_icon="⭕",
        layout="wide",
    )

    st.write("# Welcome to Visual DSP! 👋")
    st.sidebar.success("Select a DSP concept above.")

    st.markdown(
        """
        Visual DSP is a web app for learning Digital Signal Processing.        
    """
    )