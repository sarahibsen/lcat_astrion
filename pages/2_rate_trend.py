# taking the matches from page 1 and developing a trend line for the different rates 

import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np

st.title("📈 Labor Rate Trends")

if 'results' not in st.session_state or st.session_state['results'].empty:
    st.warning("Please run the Mapping Logic on Page 1 first!")
    st.stop()

results_df = st.session_state['results']
df_leg = st.session_state['df_leg_raw'] 
df_mas = st.session_state['df_mas_raw'] 

# Let user pick which year columns represent the prices
all_leg_cols = df_leg.columns.tolist()
all_mas_cols = df_mas.columns.tolist()

with st.sidebar:
    st.header("Select Price Years")
    # Heuristic: try to find columns with "Year" or numbers
    legacy_price_col = st.selectbox("Legacy Year to Compare", all_leg_cols)
    master_price_col = st.selectbox("Master Year to Compare", all_mas_cols)

# We need to pull the actual $$$ values for the matched pairs
@st.cache_data
def get_aligned_prices(matches, leg_data, mas_data, l_col, m_col):


    
    # Merge matches with Legacy prices
    merged = matches.merge(
        leg_data[['Awarded Labor Category', l_col]], 
        left_on='Legacy Title', 
        right_on='Awarded Labor Category', 
        how='left'
    )
    

    merged = merged.merge(
        mas_data[['Labor Category', m_col]], 
        left_on='Master Title', 
        right_on='Labor Category', 
        how='left'
    )
    

    merged[l_col] = pd.to_numeric(merged[l_col], errors='coerce')
    merged[m_col] = pd.to_numeric(merged[m_col], errors='coerce')
    return merged.dropna(subset=[l_col, m_col])

plot_df = get_aligned_prices(results_df, df_leg, df_mas, legacy_price_col, master_price_col)

if plot_df.empty:
    st.error("Could not find price matches for the selected years. Check column names.")
else:
    x = plot_df[legacy_price_col].values
    y = plot_df[master_price_col].values

    # Calculation for Trend Line
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6, color='blue', label='Matched LCATs')
    ax.plot(x, trend_line, color='red', linestyle='--', 
            label=f'Trend Line (y={slope:.2f}x + {intercept:.2f})')

    ax.set_xlabel(f"Legacy Rate ({legacy_price_col})")
    ax.set_ylabel(f"Master Rate ({master_price_col})")
    ax.set_title(f"Price Correlation: Legacy vs Master")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Legacy Rate", f"${x.mean():.2f}")
    col2.metric("Average Master Rate", f"${y.mean():.2f}")
    col3.metric("Avg Difference", f"${(y - x).mean():.2f}")

    st.subheader("Matched Rate Comparison Table")
    st.dataframe(plot_df[['Legacy Title', legacy_price_col, 'Master Title', master_price_col]])