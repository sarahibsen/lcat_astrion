import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Labor Rate Trends", layout="wide")
st.title("📈 Labor Rate Trends")

if 'results' not in st.session_state or st.session_state['results'].empty:
    st.warning("Please run the Mapping Logic on Page 1 first!")
    st.stop()

results_df = st.session_state['results']
df_leg = st.session_state['df_leg_raw'] 
df_mas = st.session_state['df_mas_raw'] 

with st.sidebar:
    st.header("1. Identify Title Columns")
    st.info("Match these to what you selected on Page 1.")
    # We need to know which column in the RAW data contains the Titles 
    # so we can join the prices back to the matches.
    legacy_title_col = st.selectbox("Legacy Title Column", df_leg.columns)
    master_title_col = st.selectbox("Master Title Column", df_mas.columns)

    st.header("2. Select Price Columns")
    legacy_price_col = st.selectbox("Legacy Price/Rate Column", df_leg.columns)
    master_price_col = st.selectbox("Master Price/Rate Column", df_mas.columns)

@st.cache_data
def get_aligned_prices(matches, leg_data, mas_data, l_title, m_title, l_price, m_price):
    # Step A: Get prices for Legacy
    # We rename columns during subsetting to avoid "Unnamed: 0" collisions
    leg_subset = leg_data[[l_title, l_price]].copy()
    leg_subset.columns = ['Join_Title_L', 'Price_L']
    
    merged = matches.merge(
        leg_subset, 
        left_on='Legacy Title', 
        right_on='Join_Title_L', 
        how='left'
    )
    
    # Step B: Get prices for Master
    mas_subset = mas_data[[m_title, m_price]].copy()
    mas_subset.columns = ['Join_Title_M', 'Price_M']
    
    merged = merged.merge(
        mas_subset, 
        left_on='Master Title', 
        right_on='Join_Title_M', 
        how='left'
    )
    
    # Convert to numeric
    merged['Price_L'] = pd.to_numeric(merged['Price_L'], errors='coerce')
    merged['Price_M'] = pd.to_numeric(merged['Price_M'], errors='coerce')
    
    return merged.dropna(subset=['Price_L', 'Price_M'])

# Run the alignment
plot_df = get_aligned_prices(
    results_df, df_leg, df_mas, 
    legacy_title_col, master_title_col, 
    legacy_price_col, master_price_col
)

if plot_df.empty:
    st.error("No matching prices found. Ensure 'Title' columns match your Page 1 selections.")
else:
    x = plot_df['Price_L'].values
    y = plot_df['Price_M'].values

    # Trend Line
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6, color='#1f77b4', label='Matched LCATs')
    ax.plot(x, trend_line, color='#d62728', linestyle='--', 
            label=f'Trend: y={slope:.2f}x + {intercept:.2f}')

    ax.set_xlabel(f"Legacy Rate: {legacy_price_col}")
    ax.set_ylabel(f"Master Rate: {master_price_col}")
    ax.set_title("Price Correlation: Legacy vs Master")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Legacy Rate", f"${x.mean():.2f}")
    c2.metric("Avg Master Rate", f"${y.mean():.2f}")
    c3.metric("Avg Delta", f"${(y - x).mean():.2f}")

    st.subheader("Data Comparison Table")
    # Show clean columns for the user
    display_df = plot_df[['Legacy Title', 'Price_L', 'Master Title', 'Price_M']]
    display_df.columns = ['Legacy LCAT', 'Legacy Price', 'Master LCAT', 'Master Price']
    st.dataframe(display_df, use_container_width=True)
