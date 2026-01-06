import streamlit as st
import sys
import os
from streamlit.web import cli as stcli


def main_ui():

    st.set_page_config(page_title="Astrion LCAT Tool", layout="wide")

    st.write("# Welcome to the LCAT Mapping Tool 👋")

    st.markdown(
        """
        This application helps you map legacy Labor Categories (LCATS) to Master schedules 
        and analyze pricing trends.

        ### How to use this tool:
        1. **LCAT UI**: Use the sidebar to go to the **LCAT UI** page. Upload your Excel sheet and run the mapping logic.
        2. **Rate Trend**: Once you have matched your LCATs, go to the **Rate Trend** page to visualize price correlations.

        **👈 Select a page from the sidebar to get started!**
        """
    )


    with st.sidebar:
        col1, col2 = st.columns(2)
        with col1:
            st.page_link("pages/1_lcat_ui.py", label="Run LCAT Mapping", icon="🗺️")
        with col2:
            st.page_link("pages/2_rate_trend.py", label="View Rate Trends", icon="📈")

        st.info("You can also use the sidebar on the left to switch between pages.")

    # initializing these variables here so they can exist globally across all pages 
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'df_leg_raw' not in st.session_state:
        st.session_state['df_leg_raw'] = None
    if 'df_mas_raw' not in st.session_state:
        st.session_state['df_mas_raw'] = None


def run_app():
    """
    This is the launcher called by the 'run-lcat-tool' command.
    It finds the real path of this file and tells Streamlit to run it.
    """
    current_dir = os.path.dirname(__file__)
    app_path = os.path.join(current_dir, "app.py")
    
    sys.argv = [
        "streamlit", 
        "run", 
        app_path, 
        "--server.port=8501", 
        "--server.address=localhost"
    ]
    
    sys.exit(stcli.main())

if __name__ == "__main__":
    main_ui()