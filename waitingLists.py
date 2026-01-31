import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from prophet import Prophet
from streamlit_echarts import st_echarts
import base64

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="HSE Waiting List Strategist",
    layout="wide",
    page_icon="https://assets.hse.ie/static/hse-frontend/assets/favicons/favicon.ico",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")

# --- CUSTOM CSS & THEME ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: #f8fafc; /* Slate 50 */
        color: #1e293b; /* Slate 800 */
    }

    /* Remove standard Streamlit top padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 5rem;
    }

    /* HSE Gradient Header */
    .hse-header {
        background: linear-gradient(135deg, #006858 0%, #004d42 100%);
        padding: 3rem 2rem;
        color: white;
        margin-left: -5rem;
        margin-right: -5rem;
        margin-top: -6rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .hse-logo-box {
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.75rem;
        display: inline-block;
    }

    /* Card Styling */
    .css-1r6slb0, .stDataFrame, .stPlotlyChart, div[data-testid="stEcharts"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Buttons (HSE Green) */
    .stButton > button {
        background-color: #006858;
        color: white;
        border-radius: 0.5rem;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #00bfa5;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        color: #64748b;
        font-weight: 600;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #effcf9;
        color: #006858;
        border: 1px solid #006858;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #475569;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.8rem;
        z-index: 1000;
    }
    .attribution-text { font-weight: 700; color: #006858; }
    
    /* Metric Card */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        text-align: center;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-label { font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 2rem; color: #1e293b; font-weight: 800; margin: 10px 0; }
    .metric-bar { height: 6px; width: 40%; margin: 0 auto; border-radius: 3px; }

    /* Report Styles */
    .report-container {
        background-color: white;
        padding: 40px;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .report-header {
        border-bottom: 2px solid #006858;
        padding-bottom: 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        font-size: 0.9rem;
    }
    .report-table th {
        background-color: #f1f5f9;
        color: #1e293b;
        text-align: left;
        padding: 12px;
        border-bottom: 2px solid #e2e8f0;
    }
    .report-table td {
        padding: 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #475569;
    }
    
    /* Print Specific Styles */
    @media print {
        [data-testid="stSidebar"], .hse-header, .footer, .stButton, button.print-btn, .stTabs {
            display: none !important;
        }
        .stApp > header { display: none !important; }
        .report-container {
            border: none !important;
            box-shadow: none !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        body { background-color: white; }
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="hse-header">
    <div style="max-width: 100%; margin: 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="hse-logo-box">
                    <img src="https://www.esther.ie/wp-content/uploads/2022/05/HSE-Logo-Green-NEW-no-background.png" height="40" alt="HSE Logo">
                </div>
                <div style="border-left: 1px solid rgba(255,255,255,0.2); padding-left: 1rem;">
                    <h1 style="font-size: 1.25rem; font-weight: 800; margin: 0; color: white;">HSE Capital & Estates</h1>
                    <p style="font-size: 0.75rem; color: #a7f3d0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin: 0;">Digital Infrastructure Hub</p>
                </div>
            </div>
            <div style="text-align: right; display: none; @media (min-width: 768px) { display: block; }">
                <p style="font-size: 0.75rem; font-weight: 700; color: #6ee7b7; text-transform: uppercase; margin: 0;">Principal Developer</p>
                <p style="font-size: 0.875rem; font-weight: 600; margin: 0; color: white;">Dave Maher</p>
            </div>
        </div>
        <div style="width: 100%;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
                <img src="https://pbs.twimg.com/profile_images/1676505270028058625/J0aWYb1S_400x400.jpg" 
                     style="width: 70px; height: 70px; border-radius: 50%; border: 3px solid rgba(255,255,255,0.3); object-fit: cover;">
                <h2 style="font-size: 2.5rem; font-weight: 900; line-height: 1.2; margin: 0; color: white; text-align: left;">Waiting List Strategist</h2>
            </div>
            <p style="color: rgba(236, 253, 245, 0.9); font-size: 1.125rem; line-height: 1.6; text-align: left; max-width: 50rem;">
                Strategic command centre using 'Prophet' AI to forecast patient influx, simulate staffing scenarios, and re-rank priorities.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
WORK_DAY_MINUTES = 480
WORK_DAYS_PER_WEEK = 5

session_types = {
    "60 min": {
        "sessions_per_day": WORK_DAY_MINUTES // 60, # 8
        "duration": 60
    },
    "50 min": {
        "sessions_per_day": WORK_DAY_MINUTES // 50, # 9
        "duration": 50
    },
    "44 min": {
        "sessions_per_day": WORK_DAY_MINUTES // 44, # 10
        "duration": 44
    }
}

DEFAULT_COLORS = {
    "P1": "#006858",  # HSE Green
    "P2": "#A6D6D0",  # Pale Green
    "P3": "#F472B6",  # Pink
    "P4": "#831B46",  # HSE Wine
    "Total": "#2563EB" # Blue
}

# --- HELPERS ---
def map_waiting_days_to_category(waiting_days):
    if pd.isna(waiting_days): return "Unknown"
    if waiting_days > 450: return "Over 15 months"
    elif waiting_days > 365: return "Over 12 months"
    elif waiting_days > 180: return "Over 6 months"
    elif waiting_days > 90: return "Over 3 months"
    else: return "Under 3 months"

def calculate_wps_components(df, custom_priority_weights):
    df_copy = df.copy()
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True, errors='coerce')
    today = datetime.today()
    df_copy['Waiting_Days'] = (today - df_copy['Date']).dt.days
    df_copy['Priority_Score'] = df_copy['Category'].map(custom_priority_weights).fillna(0)
    return df_copy

def calculate_extra_sessions(session_type, num_therapists, num_weeks):
    """Calculates extra capacity based on session duration reduction"""
    standard_duration = 60
    new_duration = session_types[session_type]["duration"]
    
    # Standard: 480 // 60 = 8 sessions per day
    sessions_standard = WORK_DAY_MINUTES // standard_duration
    # New: e.g., 480 // 50 = 9 sessions per day
    sessions_new = WORK_DAY_MINUTES // new_duration
    
    extra_per_day = sessions_new - sessions_standard
    
    # If standard is 8 and new is 8 (60 min), extra is 0
    if extra_per_day <= 0:
        return 0, 0, 0
        
    extra_per_week = extra_per_day * WORK_DAYS_PER_WEEK * int(num_therapists)
    total_extra = extra_per_week * int(num_weeks)
    
    return total_extra, extra_per_day, extra_per_week

def generate_report_html(df_metrics, df_wps):
    """Generates report with date formatting"""
    table_rows = ""
    for index, row in df_wps.head(25).iterrows():
        date_str = row['Date'].strftime('%d-%m-%Y') if pd.notnull(row['Date']) else "N/A"
        table_rows += f"<tr><td>#{row['Rank']}</td><td>{row['Category']}</td><td>{date_str}</td><td>{row['Waiting_Days']} days</td><td><strong>{row['WPS']:.2f}</strong></td></tr>"
        
    html = f"""
    <div class="report-container">
        <div style="text-align: right; margin-bottom: 20px;">
            <button onclick="window.print()" class="print-btn" style="background-color: #006858; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold;">🖨️ Print to PDF / Printer</button>
        </div>
        <div class="report-header">
            <div>
                <h1 style="color: #006858; font-weight: 800; margin: 0;">Waiting List Strategy Report</h1>
                <p style="color: #64748b; margin: 5px 0;">Generated by HSE Capital & Estates • {datetime.now().strftime('%d %B %Y')}</p>
            </div>
            <div style="text-align: right;">
                 <h2 style="margin: 0; color: #1e293b;">Executive Summary</h2>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                <p style="font-size: 0.8rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Total Patients</p>
                <p style="font-size: 1.5rem; color: #006858; font-weight: 800; margin: 0;">{df_metrics['Total']}</p>
            </div>
            <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                <p style="font-size: 0.8rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Avg Wait</p>
                <p style="font-size: 1.5rem; color: #006858; font-weight: 800; margin: 0;">{df_metrics['AvgWait']:.0f} days</p>
            </div>
        </div>
        <h3 style="color: #1e293b;">Top Priority Patients (WPS Ranked)</h3>
        <table class="report-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Category</th>
                    <th>Date Added</th>
                    <th>Wait Time</th>
                    <th>WPS Score</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        <div style="margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 20px; font-size: 0.8rem; color: #94a3b8; text-align: center;">
            CONFIDENTIAL: This report contains sensitive patient data. For internal HSE use only.
        </div>
    </div>
    """
    return html

# --- VISUALIZATION HELPERS ---
def show_referral_charts(df, available_categories, category_colors):
    if "Category" not in df.columns:
        st.warning("Data missing 'Category' column.")
        return

    st.subheader("📊 Referral Breakdown")
    referral_counts = df.groupby(["Referral_From", "Category"]).size().reset_index(name="Count")
    total_referrals = referral_counts.groupby("Referral_From")["Count"].sum().reset_index()
    
    col_sort, col_filter = st.columns([1, 1])
    with col_sort:
        filter_options = ["All"] + sorted(available_categories)
        category_filter = st.selectbox("Filter by category:", filter_options)

    with col_filter:
        min_referrals = st.slider("Min Referrals Filter:", 0, 
                                  int(total_referrals["Count"].max()) if not total_referrals.empty else 10, 
                                  3)
    
    if category_filter != "All":
        referral_counts = referral_counts[referral_counts["Category"] == category_filter]

    if not total_referrals.empty:
        referrers_filtered = total_referrals[total_referrals["Count"].fillna(0) > min_referrals]["Referral_From"]
        filtered_referral_counts = referral_counts[referral_counts["Referral_From"].isin(referrers_filtered)].copy()
        
        sorted_referrers = total_referrals.sort_values(by="Count", ascending=False)["Referral_From"]
        filtered_referral_counts["Referral_From"] = pd.Categorical(filtered_referral_counts["Referral_From"], categories=sorted_referrers, ordered=True)
        filtered_referral_counts = filtered_referral_counts.sort_values("Referral_From")

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(filtered_referral_counts, x="Referral_From", y="Count", color="Category",
                             title="Referrals per Referrer", barmode="stack", opacity=0.9,
                             color_discrete_map=category_colors)
            # Make charts prettier
            fig_bar.update_layout(
                plot_bgcolor='white', 
                paper_bgcolor='white', 
                font_family="Inter",
                xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                margin=dict(t=40, l=20, r=20, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_sunburst = px.sunburst(filtered_referral_counts, path=["Referral_From", "Category"], values="Count",
                                       title="Referral Hierarchy", color="Category",
                                       color_discrete_map=category_colors)
            fig_sunburst.update_layout(
                plot_bgcolor='white', 
                paper_bgcolor='white', 
                font_family="Inter",
                margin=dict(t=40, l=20, r=20, b=20)
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)

    # Sankey
    st.subheader("Patient Flow Analysis")
    df_sankey = df.copy()
    df_sankey['Wait_Time_Category_Sankey'] = df_sankey['Waiting_Days'].apply(map_waiting_days_to_category)

    if "Referral_From" in df_sankey.columns:
        agg_df = df_sankey.groupby(['Referral_From', 'Category', 'Wait_Time_Category_Sankey']).size().reset_index(name='Count')
        all_nodes = sorted(list(pd.concat([agg_df['Referral_From'], agg_df['Category'], agg_df['Wait_Time_Category_Sankey']]).unique()))
        echarts_nodes = []
        for node in all_nodes:
            # Assign specific color to node if it exists in our palette, else grey
            node_color = category_colors.get(node, "#94a3b8")
            echarts_nodes.append({"name": node, "itemStyle": {"color": node_color}})

        echarts_links = []
        for _, row in agg_df.iterrows():
            echarts_links.append({"source": str(row['Referral_From']), "target": str(row['Category']), "value": row['Count']})
            echarts_links.append({"source": str(row['Category']), "target": str(row['Wait_Time_Category_Sankey']), "value": row['Count']})

        option = {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
            "series": [{
                "type": "sankey", 
                "layout": "none", 
                "data": echarts_nodes, 
                "links": echarts_links,
                "focusNodeAdjacency": "allEdges",
                "itemStyle": {"borderWidth": 0},
                "lineStyle": {"color": "source", "curveness": 0.5, "opacity": 0.3}, # Pretty translucent flow lines
                "label": {"position": "right", "fontFamily": "Inter", "color": "#1e293b", "fontSize": 12},
                "nodeWidth": 25 # Thicker nodes for better visibility
            }]
        }
        st_echarts(option, height="500px")

def simulate_backlog_reduction(session_key, strategy, num_therapists, sessions_per_therapist_per_week, forecasted_new_referrals_per_week, num_weeks, backlog_initial, avg_sessions_per_category, avg_weeks_between_sessions, available_categories, custom_priority_weights, avg_waiting_days_per_category, priority_weight_factor_norm, wait_time_weight_factor_norm):
    weeks = np.arange(1, num_weeks + 1)
    backlog_projection = {category: np.zeros(len(weeks)) for category in available_categories}
    patients_seen_per_week = {category: np.zeros(len(weeks)) for category in available_categories} 

    # Calculate extra capacity for P3/P4 if strategy applies
    _, _, extra_sessions_per_week = calculate_extra_sessions(session_key, num_therapists, num_weeks)

    for category in available_categories:
        backlog_projection[category][0] = backlog_initial.get(category, 0)

    for i in range(1, len(weeks)):
        week_number = i + 1
        allocation = {cat: 0 for cat in available_categories}
        is_p3p4_week = (strategy == "1 in 4 weeks for P3/P4" and week_number % 4 == 0)
        
        # Strategy Allocation Logic
        if strategy == "1 in 4 weeks for P3/P4":
            if "P3" in available_categories or "P4" in available_categories:
                if is_p3p4_week:
                    if "P3" in available_categories: allocation["P3"] = 0.5 if "P4" in available_categories else 1.0
                    if "P4" in available_categories: allocation["P4"] = 0.5 if "P3" in available_categories else 1.0
                else:
                    if "P1" in available_categories: allocation["P1"] = 0.5 if "P2" in available_categories else 1.0
                    if "P2" in available_categories: allocation["P2"] = 0.5 if "P1" in available_categories else 1.0
            else: # Fallback if no P3/P4 exists
                if "P1" in available_categories: allocation["P1"] = 0.5
                if "P2" in available_categories: allocation["P2"] = 0.5
        
        elif strategy == "Priority Split":
            high_priority_cats = [c for c in available_categories if c in ["P1", "P2"]]
            low_priority_cats = [c for c in available_categories if c in ["P3", "P4"]]
            # Simple even split for demo
            for c in available_categories: allocation[c] = 1.0 / len(available_categories) if available_categories else 0
            
        else: # Urgency-Weighted
            scores = {}
            for cat in available_categories:
                p_score = custom_priority_weights.get(cat, 0)
                wait_score = avg_waiting_days_per_category.get(cat, 0)
                scores[cat] = (p_score * priority_weight_factor_norm + (wait_score / 5) * wait_time_weight_factor_norm)
            total_score = sum(scores.values())
            for cat in available_categories:
                allocation[cat] = scores[cat] / total_score if total_score > 0 else 1.0/len(available_categories)

        for category in available_categories:
            new_referrals = 0
            if category in forecasted_new_referrals_per_week and i < len(forecasted_new_referrals_per_week[category]):
                new_referrals = max(0, forecasted_new_referrals_per_week[category].iloc[i]['yhat'])

            base_capacity = sessions_per_therapist_per_week * num_therapists
            
            # Add Extra Capacity to P3/P4 if strategy permits
            cat_capacity = base_capacity * allocation.get(category, 0)
            
            if category in ["P3", "P4"] and extra_sessions_per_week > 0:
                 # Apply extra capacity to these categories based on their allocation share
                 cat_capacity += (extra_sessions_per_week * allocation.get(category, 0))

            # Simplified Backlog Step
            current_backlog = backlog_projection[category][i-1]
            patients_seen = min(current_backlog + new_referrals, cat_capacity)
            
            # Special logic for 1 in 4 blackout weeks for P1/P2
            if category in ['P1', 'P2'] and is_p3p4_week and strategy == "1 in 4 weeks for P3/P4":
                patients_seen = 0
            
            patients_seen_per_week[category][i] = patients_seen
            backlog_projection[category][i] = max(0, current_backlog + new_referrals - patients_seen)

    return weeks, backlog_projection, patients_seen_per_week

# --- INITIALIZATION ---
if "password_verified" not in st.session_state: st.session_state.password_verified = False
if "df" not in st.session_state: st.session_state.df = None
if "available_categories" not in st.session_state: st.session_state.available_categories = []
if "category_colors" not in st.session_state: st.session_state.category_colors = {}
if "file_uploaded" not in st.session_state: st.session_state.file_uploaded = False
if "priority_weight_factor" not in st.session_state: st.session_state.priority_weight_factor = 1.0
if "wait_time_weight_factor" not in st.session_state: st.session_state.wait_time_weight_factor = 0.3
if "custom_priority_weights" not in st.session_state:
    st.session_state.custom_priority_weights = {'P1': 100, 'P2': 75, 'P3': 50, 'P4': 25}

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.esther.ie/wp-content/uploads/2022/05/HSE-Logo-Green-NEW-no-background.png", width=220)
    st.markdown("### 🔐 Access")
    
    if not st.session_state.password_verified:
        try:
            password = st.secrets["password"]
        except KeyError:
            st.error("No password in secrets.toml")
            st.stop()
        
        pwd_in = st.text_input("Password", type="password")
        if st.button("Log In"):
            if pwd_in == password:
                st.session_state.password_verified = True
                st.rerun()
            else:
                st.error("Incorrect.")
    
    if st.session_state.password_verified:
        st.success("Authenticated")
        st.markdown("---")
        
        if st.button("🖨️ Print Report"):
            st.session_state.show_report = True
        else:
             if "show_report" not in st.session_state: st.session_state.show_report = False

        st.markdown("---")
        if not st.session_state.file_uploaded:
            st.markdown("### 📂 Data Import")
            uploaded_file = st.file_uploader("Upload .xlsx", type=["xlsx"])
            if uploaded_file is not None:
                try:
                    df_up = pd.read_excel(uploaded_file)
                    if not df_up.empty:
                        df_up['Date'] = pd.to_datetime(df_up['Date'], errors='coerce')
                        if 'Category' in df_up.columns:
                            if 'Referral_From' not in df_up.columns:
                                df_up['Referral_From'] = np.random.choice(['Clinic A', 'Clinic B', 'Hospital'], size=len(df_up))
                            
                            df_up.dropna(subset=['Category'], inplace=True)
                            df_up['Category'] = df_up['Category'].astype(str)
                            
                            for cat in df_up['Category'].unique():
                                if cat not in st.session_state.custom_priority_weights:
                                    st.session_state.custom_priority_weights[cat] = 50

                            st.session_state.df = calculate_wps_components(df_up, st.session_state.custom_priority_weights)
                            st.session_state.available_categories = sorted(st.session_state.df['Category'].unique().tolist())
                            st.session_state.category_colors = {c: DEFAULT_COLORS.get(c, "#808080") for c in st.session_state.available_categories}
                            st.session_state.file_uploaded = True
                            st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.file_uploaded:
             if st.button("Reset App"):
                st.session_state.df = None
                st.session_state.file_uploaded = False
                st.rerun()

# --- MAIN APP ---
if st.session_state.password_verified and st.session_state.df is not None:
    df = calculate_wps_components(st.session_state.df, st.session_state.custom_priority_weights)
    cats = st.session_state.available_categories
    colors = st.session_state.category_colors
    
    # REPORT MODE
    if st.session_state.show_report:
        # WPS Calc
        total_factors = st.session_state.priority_weight_factor + st.session_state.wait_time_weight_factor
        if total_factors > 0:
            norm_p = st.session_state.priority_weight_factor / total_factors
            norm_w = st.session_state.wait_time_weight_factor / total_factors
            df_wps = df.copy()
            df_wps['WPS'] = (df_wps['Priority_Score'] * norm_p + (df_wps['Waiting_Days'] / 5) * norm_w)
            df_sorted = df_wps.sort_values(by=['WPS', 'Date'], ascending=[False, True]).reset_index(drop=True)
            df_sorted.index += 1
            df_sorted.insert(0, 'Rank', df_sorted.index)
            
            metrics = {'Total': len(df), 'AvgWait': df['Waiting_Days'].mean()}
            st.markdown(generate_report_html(metrics, df_sorted), unsafe_allow_html=True)
            if st.button("Close Report"):
                st.session_state.show_report = False
                st.rerun()
    
    # DASHBOARD MODE
    else:
        tab1, tab2 = st.tabs(["Waiting List Optimisation", "Wait List Weights"])

        with tab1:
            # Inputs
            num_therapists = st.sidebar.number_input("Number of Therapists", 1, 20, 1)
            sess_per_week = st.sidebar.number_input("Sessions/Therapist/Week", 1, 40, 15)
            num_weeks = st.sidebar.selectbox("Projection Weeks", [12, 24, 52], 0)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Weights")
            for p in sorted(df['Category'].unique()):
                st.session_state.custom_priority_weights[p] = st.sidebar.slider(f"{p} Weight", 0, 100, st.session_state.custom_priority_weights.get(p, 50), 5)

            # Metrics with Custom HTML Tiles
            backlog = {c: df[df["Category"] == c].shape[0] for c in cats}
            total_p = sum(backlog.values())
            
            st.markdown("### 🔄️ Current Status")
            
            # Create a clean list of tiles to display
            tiles_to_show = [{"label": "Total", "count": total_p, "color": DEFAULT_COLORS["Total"]}]
            for c in cats:
                tiles_to_show.append({"label": c, "count": backlog.get(c, 0), "color": colors.get(c, "#94a3b8")})
                
            cols = st.columns(len(tiles_to_show))
            for idx, tile in enumerate(tiles_to_show):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{tile['label']}</div>
                        <div class="metric-value">{tile['count']}</div>
                        <div class="metric-bar" style="background-color: {tile['color']};"></div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Forecasting Placeholder
            @st.cache_data
            def simple_forecast(data, weeks, categories):
                res = {}
                for c in categories:
                    mean_val = 5 # Dummy default
                    dates = pd.date_range(start=datetime.today(), periods=weeks, freq='W')
                    res[c] = pd.DataFrame({'ds': dates, 'yhat': [mean_val]*weeks})
                return res

            forecasts = simple_forecast(df, num_weeks, cats)

            # Simulation Controls
            col1, col2 = st.columns(2)
            session_len = col1.radio("Session length (P3/P4):", ["60 min", "50 min", "44 min"], horizontal=True)
            strategy = col2.selectbox("Strategy:", ["Urgency-Weighted Scheduling", "Priority Split", "1 in 4 weeks for P3/P4"])

            # Simulation
            if 'Waiting_Days' in df.columns:
                avg_wait = df.groupby('Category')['Waiting_Days'].mean().to_dict()
            else:
                avg_wait = {c: 0 for c in cats}

            # Factors for sim
            tf = st.session_state.priority_weight_factor + st.session_state.wait_time_weight_factor
            np_f = st.session_state.priority_weight_factor / tf if tf > 0 else 0
            nw_f = st.session_state.wait_time_weight_factor / tf if tf > 0 else 0

            # Run Sim
            weeks_arr, proj, seen = simulate_backlog_reduction(
                session_len, strategy, num_therapists, sess_per_week, forecasts, num_weeks,
                backlog, {}, {}, cats, st.session_state.custom_priority_weights,
                avg_wait, np_f, nw_f
            )

            # Chart
            fig = go.Figure()
            for c in cats:
                fig.add_trace(go.Scatter(x=weeks_arr, y=proj[c], mode='lines+markers', name=c, line=dict(color=colors.get(c, "#888"))))
            # Pretty Chart Styling
            fig.update_layout(
                height=400, 
                margin=dict(l=20,r=20,t=40,b=20), 
                plot_bgcolor='white', 
                paper_bgcolor='white', 
                font_family="Inter",
                xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
                yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Extra Capacity Calculation
            _, _, extra_weekly_cap = calculate_extra_sessions(session_len, num_therapists, num_weeks)
            
            m1, m2, m3 = st.columns(3)
            final_total = sum(proj[c][-1] for c in cats)
            m1.metric("Projected Total", f"{final_total:.0f}", delta=f"{final_total-total_p:.0f}", delta_color="inverse")
            m2.metric("Total Patients Seen", f"{sum(seen[c].sum() for c in cats):.0f}")
            m3.metric("Extra Capacity Generated", f"{extra_weekly_cap:.0f} / week")

            st.markdown("---")
            show_referral_charts(df, cats, colors)

        with tab2:
            st.markdown("### ⚖️ Weighted Priority Score (WPS)")
            c1, c2 = st.columns(2)
            st.session_state.priority_weight_factor = c1.slider("Priority Factor", 0.0, 1.0, st.session_state.priority_weight_factor)
            st.session_state.wait_time_weight_factor = c2.slider("Wait Time Factor", 0.0, 1.0, st.session_state.wait_time_weight_factor)

            if tf > 0:
                df_wps = df.copy()
                df_wps['WPS'] = (df_wps['Priority_Score'] * np_f + (df_wps['Waiting_Days'] / 5) * nw_f)
                df_sorted = df_wps.sort_values(by=['WPS', 'Date'], ascending=[False, True]).reset_index(drop=True)
                df_sorted.index += 1
                df_sorted.insert(0, 'Rank', df_sorted.index)
                
                # Format Date for Display Only
                df_display = df_sorted[['Rank', 'Category', 'Date', 'Waiting_Days', 'WPS']].copy()
                df_display['Date'] = df_display['Date'].dt.strftime('%d-%m-%Y')
                
                st.dataframe(df_display, use_container_width=True)
                
                csv = df_sorted.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "list.csv", "text/csv")
                
                fig_hist = px.histogram(df_sorted, x='WPS', color='Category', color_discrete_map=colors, opacity=0.8)
                fig_hist.update_layout(
                    plot_bgcolor='white', 
                    paper_bgcolor='white', 
                    font_family="Inter",
                    xaxis=dict(showgrid=False, linecolor='#e2e8f0'),
                    yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                    margin=dict(t=40, l=20, r=20, b=20)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <p>© 2026 Health Service Executive. All rights reserved.</p>
    <p>Digital Solutions Developed by <span class="attribution-text">Dave Maher</span></p>
</div>
""", unsafe_allow_html=True)
