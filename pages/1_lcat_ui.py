import streamlit as st
import pandas as pd
import nltk
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords

# --- CONFIGURATION ---
st.set_page_config(
    page_title = "LCAT Mapping"
)
st.title("LCAT & SIN Mapping Tool")
st.markdown("Map legacy Labor Categories to Master based on Qualifications and Semantic Description similarity.")


@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    stop_words = set(stopwords.words('english'))
    return model, stop_words

model, STOP_WORDS = load_resources()

def rate_split(df, rate_col):
    """
    Splits the 'site' and 'rate location' into their two separate dataframes so that contractor is not 
    compared to customer and vice versa
    Also make sure it is split for their respective legacy and master sheets, not just one big sheet 
    """
    # make the user to select the 'rate' / 'site location' / rate column in order to split both of the dataset sheets into their respective data frames 
    # user MUST make this distinction before we can do the mappings 
    if rate_col not in df.columns:
        return df, pd.DataFrame()
    
    # Convert to string and lowercase for comparison
    df[rate_col] = df[rate_col].astype(str).str.strip().str.lower()
   # rate_values = df[rate_col].astype(str).str.lower().str.strip()
    grouped = {site: group_df for site, group_df in df.groupby(rate_col)}
    return grouped
    

    


def split_stuck_quals(text):
    if not isinstance(text, str) or text.strip() == "":
        return []
    cleaned = re.sub(r'(\d)([A-Z])', r'\1 | \2', text)
    return [part.strip() for part in cleaned.split('|') if part.strip()]

def get_total_years_equivalent(edu_str):
    if not isinstance(edu_str, str) or edu_str.strip() == "":
        return 0
    edu_lower = edu_str.lower()
    base_years = 0
    if 'phd' in edu_lower or 'doctorate' in edu_lower: base_years = 9
    elif any(x in edu_lower for x in ['master', 'ma', 'ms', 'mba']): base_years = 6
    elif any(x in edu_lower for x in ['bachelor', 'ba', 'bs']): base_years = 4
    elif any(x in edu_lower for x in ['associate', 'aa', 'as']): base_years = 2
    elif any(x in edu_lower for x in ['high', 'hs', 'ged']): base_years = 1
    
    match = re.search(r'\+\s*(\d+)', edu_str)
    plus_years = int(match.group(1)) if match else 0
    return base_years + plus_years

def evaluate_complex_qual_string(edu_str, mode='max'):
    parts = split_stuck_quals(edu_str)
    if not parts: return 0, ""
    results = [(get_total_years_equivalent(p), p) for p in parts]
    return max(results, key=lambda x: x[0]) if mode == 'max' else min(results, key=lambda x: x[0])

def get_context_words(text):
    words = nltk.word_tokenize(re.sub(r'[^a-zA-Z\s]', '', text.lower()))
    tagged = nltk.pos_tag(words)
    return {w for w, tag in tagged if (tag.startswith('NN') or tag.startswith('JJ')) and w not in STOP_WORDS}

def draw_match_tree(matches, num_to_show=2):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    unique_legacy = list(dict.fromkeys([m['Legacy Title'] for m in matches]))[:num_to_show]
    y_pos = 85
    for l_title in unique_legacy:
        l_matches = [m for m in matches if m['Legacy Title'] == l_title][:3]
        if not l_matches: continue
        leg_x, leg_y = 10, y_pos
        rect = patches.Rectangle((leg_x, leg_y-8), 20, 16, facecolor='aliceblue', edgecolor='navy')
        ax.add_patch(rect)
        ax.text(leg_x + 10, leg_y, f"LEGACY:\n{l_title[:25]}...", ha='center', va='center', fontsize=8, fontweight='bold')
        for idx, m in enumerate(l_matches):
            m_x, m_y = 70, (y_pos + 6) - (idx * 16)
            ax.add_patch(patches.Rectangle((m_x, m_y-6), 25, 12, facecolor='#d4edda' if idx==0 else '#f8f9fa', edgecolor='black'))
            ax.text(m_x + 12.5, m_y, f"MATCH #{idx+1}\n{m['Master Title'][:25]}\nScore: {m['Similarity Score']}", ha='center', va='center', fontsize=7)
            ax.plot([leg_x + 20, m_x], [leg_y, m_y], color='gray', alpha=0.6)
        y_pos -= 50
    return fig


with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    
    if uploaded_file:
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        
        legacy_sheet = st.selectbox("Select Legacy Sheet", sheet_names)
        master_sheet = st.selectbox("Select Master Sheet", sheet_names)
        
        st.divider()
        st.header("2. Settings")
        top_k_val = st.slider("Matches per LCAT", 1, 5, 3)

if uploaded_file:
    df_leg_raw = pd.read_excel(uploaded_file, sheet_name=legacy_sheet).fillna("")
    df_mas_raw = pd.read_excel(uploaded_file, sheet_name=master_sheet).fillna("")

    df_leg_raw.columns = [str(c).strip() for c in df_leg_raw.columns]
    df_mas_raw.columns = [str(c).strip() for c in df_mas_raw.columns]

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Map Legacy Columns")
        l_sin_col = st.selectbox("SIN Column (Legacy)", df_leg_raw.columns)
        l_title_col = st.selectbox("Title Column (Legacy)", df_leg_raw.columns)
        l_desc_col = st.selectbox("Description Column (Legacy)", df_leg_raw.columns)
        l_min_col = st.selectbox("Min Quals Column (Legacy)", df_leg_raw.columns)
        l_alt_col = st.selectbox("Alt Quals Column (Legacy)", df_leg_raw.columns)
        l_rate_col = st.selectbox("Site/Rate Location (Legacy)", ["None"] + list(df_leg_raw.columns))

    with col2:
        st.subheader("Map Master Columns")
        m_sin_col = st.selectbox("SIN Column (Master)", df_mas_raw.columns)
        m_title_col = st.selectbox("Title Column (Master)", df_mas_raw.columns)
        m_desc_col = st.selectbox("Description Column (Master)", df_mas_raw.columns)
        m_min_col = st.selectbox("Min Quals Column (Master)", df_mas_raw.columns)
        m_alt_col = st.selectbox("Alt Quals Column (Master)", df_mas_raw.columns)
        m_rate_col = st.selectbox("Site/Rate Location (Master)", ["None"] + list(df_mas_raw.columns))

    if st.button("Run Mapping Logic"):
        with st.spinner("Analyzing by Site Location..."):
            leg_groups = rate_split(df_leg_raw, l_rate_col)
            mas_groups = rate_split(df_mas_raw, m_rate_col)
            
            all_matches = []
    
            for site_name, leg_subset in leg_groups.items():
                # Check if this site exists in the Master data
                if site_name not in mas_groups:
                    st.warning(f"No Master LCATs found for site: '{site_name}'. Skipping.")
                    continue
                
                mas_subset = mas_groups[site_name]
                
                # --- RUN SIMILARITY ON SUBSETS ONLY ---
                leg_descs = [str(d).strip() for d in leg_subset[l_desc_col]]
                mas_descs = [str(d).strip() for d in mas_subset[m_desc_col]]
    
                leg_embeddings = model.encode(leg_descs, convert_to_tensor=True)
                mas_embeddings = model.encode(mas_descs, convert_to_tensor=True)
                cosine_scores = util.cos_sim(leg_embeddings, mas_embeddings)
    
                # Pre-calculate master quals for this subset
                master_info = []
                for _, row in mas_subset.iterrows():
                    m_min_val, m_min_str = evaluate_complex_qual_string(str(row[m_min_col]), mode='min')
                    m_alt_val, m_alt_str = evaluate_complex_qual_string(str(row[m_alt_col]), mode='min')
                    if m_min_val > 0 and m_alt_val > 0:
                        req_val = m_min_val if m_min_val <= m_alt_val else m_alt_val
                    else:
                        req_val = max(m_min_val, m_alt_val)
                    master_info.append({"score": req_val})
    
                # Compare within this site
                for i in range(len(leg_subset)):
                    l_min_val, _ = evaluate_complex_qual_string(str(leg_subset.iloc[i][l_min_col]), mode='max')
                    l_alt_val, _ = evaluate_complex_qual_string(str(leg_subset.iloc[i][l_alt_col]), mode='max')
                    leg_best_val = max(l_min_val, l_alt_val)
    
                    valid_indices = [idx for idx, info in enumerate(master_info) if leg_best_val >= info['score']]
                    
                    if not valid_indices: 
                        continue
    
                    valid_sims = cosine_scores[i][valid_indices]
                    k = min(top_k_val, len(valid_sims))
                    top_k = torch.topk(valid_sims, k=k)
    
                    for sim_score, sub_idx in zip(top_k.values, top_k.indices):
                        m_idx = valid_indices[sub_idx.item()]
                        
                        all_matches.append({
                            "Site": site_name.upper(), # Tracks which group this came from
                            "Legacy Title": leg_subset.iloc[i][l_title_col],
                            "Master Title": mas_subset.iloc[m_idx][m_title_col],
                            "Similarity Score": round(sim_score.item(), 3),
                            "Legacy SIN": leg_subset.iloc[i][l_sin_col],
                            "Master SIN": mas_subset.iloc[m_idx][m_sin_col]
                        })

            st.session_state['results'] = pd.DataFrame(all_matches)
            st.session_state['df_leg_raw'] = df_leg_raw
            st.session_state['df_mas_raw'] = df_mas_raw

    if 'results' in st.session_state:
        res_df = st.session_state['results']
        
        st.divider()
        st.subheader("Mapping Results")
 
        c1, c2, c3 = st.columns(3)
        c1.metric("Legacy LCATs Processed", len(df_leg_raw))
        c2.metric("Matches Found", len(res_df))
        c3.metric("Avg Similarity", f"{res_df['Similarity Score'].mean():.2f}")

        st.dataframe(res_df, width='content') # might need to change this -- will be changed in newer versions

        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "lcat_mapping.csv", "text/csv")

        # Visual Tree
        st.subheader("Visualization (Top 2 Matches)")
        fig = draw_match_tree(res_df.to_dict('records'))
        st.pyplot(fig)

else:
    st.info("Please upload an Excel file in the sidebar to begin.")
