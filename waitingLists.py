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

# --- CUSTOM CSS & THEME (React Theme Replication) ---
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
        margin-top: -6rem; /* Pull up to cover default header space */
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
        transition: box-shadow 0.2s ease-in-out;
    }
    
    .stPlotlyChart:hover, div[data-testid="stEcharts"]:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {
        color: #006858;
        font-weight: 700;
    }

    /* HSE Styled Sliders */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #006858 !important;
        box-shadow: 0 0 0 4px rgba(0, 104, 88, 0.1) !important;
    }
    div[data-baseweb="slider"] div[class*="st-"] { 
        /* Target the track line - simple approach */
        background-color: #cbd5e1; 
    }
    /* This targets the filled part of the slider track in some Streamlit versions */
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: #006858 !important;
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
    
    .attribution-text {
        font-weight: 700;
        color: #006858;
    }
    
    /* Print Report Styling */
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
</style>
""", unsafe_allow_html=True)

# --- HEADER RENDER ---
st.markdown("""
<div class="hse-header">
    <div style="max-width: 80rem; margin: 0 auto;">
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
        <div style="max-width: 42rem;">
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
                 <!-- Icon Circle -->
                <div style="background: rgba(255,255,255,0.15); width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(5px);">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                </div>
                <h2 style="font-size: 2.5rem; font-weight: 900; line-height: 1.2; margin: 0; color: white;">Waiting List Strategist</h2>
            </div>
            <p style="color: rgba(236, 253, 245, 0.9); font-size: 1.125rem; line-height: 1.6;">
                Strategic command centre using 'Prophet' AI to forecast patient influx, simulate staffing scenarios, and re-rank priorities.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- CONSTANTS AND HELPER FUNCTIONS ---

def format_time_duration(minutes):
    """Convert minutes to hours and minutes format if over 60 minutes"""
    if minutes == 0:
        return "0 minutes"
    elif minutes < 60:
        return f"{minutes} minutes"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours} hours"
        else:
            return f"{hours} hours {remaining_minutes} minutes"

WORK_DAY_MINUTES = 480
WORK_DAYS_PER_WEEK = 5

session_types = {
    "60 min": {
        "sessions_per_day": WORK_DAY_MINUTES // 60,
        "duration": 60,
        "daily_capacity": (WORK_DAY_MINUTES // 60) * 60
    },
    "50 min": {
        "sessions_per_day": WORK_DAY_MINUTES // 50,
        "duration": 50,
        "daily_capacity": (WORK_DAY_MINUTES // 50) * 50
    },
    "44 min": {
        "sessions_per_day": WORK_DAY_MINUTES // 44,
        "duration": 44,
        "daily_capacity": (WORK_DAY_MINUTES // 44) * 44
    }
}

DEFAULT_COLORS = {
    "P1": "#006858",  # HSE Green (Start)
    "P2": "#A6D6D0",  # Pale Green
    "P3": "#F472B6",  # Pink
    "P4": "#831B46",  # HSE Wine
    "Total": "#2563EB" # Blue-600
}

def map_waiting_days_to_category(waiting_days):
    if pd.isna(waiting_days):
        return "Unknown"
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

def generate_report_html(df_metrics, df_wps):
    """Generates a beautiful HTML report for printing"""
    
    # Generate Table Rows
    table_rows = ""
    for index, row in df_wps.head(15).iterrows():
        table_rows += f"""
        <tr>
            <td>#{row['Rank']}</td>
            <td>{row['Category']}</td>
            <td>{row['Date'].strftime('%d %b %Y')}</td>
            <td>{row['Waiting_Days']} days</td>
            <td><strong>{row['WPS']:.2f}</strong></td>
        </tr>
        """
        
    html = f"""
    <div class="report-container">
        <div class="report-header">
            <div>
                <h1 style="color: #006858; font-weight: 800; margin: 0;">Waiting List Strategy Report</h1>
                <p style="color: #64748b; margin: 5px 0;">Generated by HSE Capital & Estates • {datetime.now().strftime('%d %B %Y')}</p>
            </div>
            <div style="text-align: right;">
                 <h2 style="margin: 0; color: #1e293b;">Executive Summary</h2>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
            <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                <p style="font-size: 0.8rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Total Patients</p>
                <p style="font-size: 1.5rem; color: #006858; font-weight: 800; margin: 0;">{df_metrics['Total']}</p>
            </div>
            <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                <p style="font-size: 0.8rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Avg Wait</p>
                <p style="font-size: 1.5rem; color: #006858; font-weight: 800; margin: 0;">{df_metrics['AvgWait']:.0f} days</p>
            </div>
        </div>
        
        <h3 style="color: #1e293b; margin-top: 30px;">Top Priority Patients (WPS Ranked)</h3>
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

def show_referral_charts(df, available_categories, category_colors):
    if "Category" not in df.columns:
        st.warning("The uploaded file must contain 'Category' column to show referral charts.")
        return

    st.subheader("📊 Referral Breakdown")
    referral_counts = df.groupby(["Referral_From", "Category"]).size().reset_index(name="Count")
    total_referrals = referral_counts.groupby("Referral_From")["Count"].sum().reset_index()
    
    col_sort, col_filter = st.columns([1, 1])
    with col_sort:
        filter_options = ["All"] + sorted(available_categories)
        category_filter = st.selectbox("Filter by category:", filter_options, key="referral_category_filter")

    with col_filter:
        min_referrals = st.slider("Show referrers with more than X referrals:", 
                                  min_value=0, 
                                  max_value=int(total_referrals["Count"].max()) if not total_referrals.empty else 10, 
                                  value=3,
                                  key="referral_min_referrals")
    
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
                             title="Referrals per Referrer by Category",
                             labels={"Count": "Number of Referrals"},
                             barmode="stack", opacity=0.8,
                             color_discrete_map=category_colors)
            fig_bar.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_sunburst = px.sunburst(filtered_referral_counts, path=["Referral_From", "Category"], values="Count",
                                       title="Referral Breakdown by Category and Referrer",
                                       color="Category",
                                       color_discrete_map=category_colors)
            fig_sunburst.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_sunburst, use_container_width=True)

    # Sankey Diagram
    st.subheader("Flow of Patients: Referral → Category → Wait Time")
    df_sankey = df.copy()
    df_sankey['Wait_Time_Category_Sankey'] = df_sankey['Waiting_Days'].apply(map_waiting_days_to_category)

    if "Referral_From" in df_sankey.columns and "Category" in df_sankey.columns and "Wait_Time_Category_Sankey" in df_sankey.columns:
        agg_df = df_sankey.groupby(['Referral_From', 'Category', 'Wait_Time_Category_Sankey']).size().reset_index(name='Count')
        all_nodes = sorted(list(pd.concat([agg_df['Referral_From'], agg_df['Category'], agg_df['Wait_Time_Category_Sankey']]).unique()))
        echarts_nodes = [{"name": node} for node in all_nodes]

        echarts_links = []
        for _, row in agg_df.iterrows():
            echarts_links.append({"source": str(row['Referral_From']), "target": str(row['Category']), "value": row['Count']})
            echarts_links.append({"source": str(row['Category']), "target": str(row['Wait_Time_Category_Sankey']), "value": row['Count']})

        option = {
            "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
            "series": [{
                "type": "sankey", "layout": "none", "data": echarts_nodes, "links": echarts_links,
                "focusNodeAdjacency": "allEdges",
                "itemStyle": {"borderWidth": 1, "borderColor": "#aaa"},
                "lineStyle": {"color": "gradient", "curveness": 0.5},
                "label": {"position": "right"}
            }]
        }
        st_echarts(option, height="500px")
    else:
        st.warning("Required columns for Sankey diagram are missing.")

def calculate_extra_sessions(session_type, num_therapists, num_weeks):
    standard_duration = session_types["60 min"]["duration"]
    new_duration = session_types[session_type]["duration"]
    if new_duration >= standard_duration:
        return 0, 0, 0

    workday_standard_sessions = WORK_DAY_MINUTES // standard_duration
    workday_new_sessions = WORK_DAY_MINUTES // new_duration
    
    extra_sessions_per_day = workday_new_sessions - workday_standard_sessions
    extra_sessions_per_week = extra_sessions_per_day * WORK_DAYS_PER_WEEK * num_therapists
    total_extra_sessions = extra_sessions_per_week * num_weeks
    
    return total_extra_sessions, extra_sessions_per_day, extra_sessions_per_week

def simulate_backlog_reduction(session_key, strategy, num_therapists, sessions_per_therapist_per_week, forecasted_new_referrals_per_week, num_weeks, backlog_initial, avg_sessions_per_category, avg_weeks_between_sessions, available_categories, custom_priority_weights, avg_waiting_days_per_category, priority_weight_factor_norm, wait_time_weight_factor_norm):
    if num_therapists <= 0: raise ValueError("Number of therapists must be greater than 0")
    if sessions_per_therapist_per_week <= 0: raise ValueError("Sessions per therapist must be greater than 0")

    weeks = np.arange(1, num_weeks + 1)
    backlog_projection = {category: np.zeros(len(weeks)) for category in available_categories}
    patients_seen_per_week = {category: np.zeros(len(weeks)) for category in available_categories} 

    total_extra_sessions, extra_sessions_per_day, extra_sessions_per_week = calculate_extra_sessions(session_key, num_therapists, num_weeks)

    for category in available_categories:
        backlog_projection[category][0] = backlog_initial.get(category, 0)

    for i in range(1, len(weeks)):
        week_number = i + 1
        allocation = {cat: 0 for cat in available_categories}
        is_p3p4_week = (strategy == "1 in 4 weeks for P3/P4" and week_number % 4 == 0)
        
        if strategy == "1 in 4 weeks for P3/P4":
            if "P3" in available_categories or "P4" in available_categories:
                if is_p3p4_week:
                    if "P3" in available_categories: allocation["P3"] = 0.5 if "P4" in available_categories else 1.0
                    if "P4" in available_categories: allocation["P4"] = 0.5 if "P3" in available_categories else 1.0
                else:
                    if "P1" in available_categories: allocation["P1"] = 0.5 if "P2" in available_categories else 1.0
                    if "P2" in available_categories: allocation["P2"] = 0.5 if "P1" in available_categories else 1.0
            else:
                if "P1" in available_categories: allocation["P1"] = 0.5 if "P2" in available_categories else 1.0
                if "P2" in available_categories: allocation["P2"] = 0.5 if "P1" in available_categories else 1.0
        elif strategy == "Priority Split":
            high_priority_cats = [cat for cat in available_categories if cat in ["P1", "P2"]]
            low_priority_cats = [cat for cat in available_categories if cat in ["P3", "P4"]]
            for cat in available_categories: allocation[cat] = 0.0

            if not low_priority_cats and "P1" in high_priority_cats and "P2" in high_priority_cats:
                allocation["P1"], allocation["P2"] = 0.3, 0.7
            elif not low_priority_cats and "P1" in high_priority_cats: allocation["P1"] = 1.0
            elif not low_priority_cats and "P2" in high_priority_cats: allocation["P2"] = 1.0
            else:
                target_high_share = 1.0 if not low_priority_cats else (0.0 if not high_priority_cats else 0.5)
                target_low_share = 1.0 if not high_priority_cats else (0.0 if not low_priority_cats else 0.5)
                
                if high_priority_cats and target_high_share > 0:
                    for cat in high_priority_cats: allocation[cat] = target_high_share / len(high_priority_cats)
                if low_priority_cats and target_low_share > 0:
                    for cat in low_priority_cats: allocation[cat] = target_low_share / len(low_priority_cats)

        else: # Urgency-Weighted
            current_category_wps_scores = {}
            for cat in available_categories:
                priority_score = custom_priority_weights.get(cat, 0)
                avg_wait_days = avg_waiting_days_per_category.get(cat, 0)
                category_wps = (priority_score * priority_weight_factor_norm + (avg_wait_days / 5) * wait_time_weight_factor_norm)
                current_category_wps_scores[cat] = category_wps
            total_category_wps_score = sum(current_category_wps_scores.values())
            
            if total_category_wps_score > 0:
                for cat in available_categories: allocation[cat] = current_category_wps_scores[cat] / total_category_wps_score
            else:
                for cat in available_categories: allocation[cat] = 1.0 / len(available_categories) if len(available_categories) > 0 else 0

        for category in available_categories:
            if category in forecasted_new_referrals_per_week and i < len(forecasted_new_referrals_per_week[category]):
                new_referrals = max(0, forecasted_new_referrals_per_week[category].iloc[i]['yhat'])
            else:
                new_referrals = 0

            base_weekly_sessions = sessions_per_therapist_per_week * num_therapists
            base_reduction = base_weekly_sessions * allocation.get(category, 0)
            extra_reduction = 0
            if extra_sessions_per_week > 0 and category in ["P3", "P4"] and category in available_categories:
                if (is_p3p4_week and strategy == "1 in 4 weeks for P3/P4") or (strategy != "1 in 4 weeks for P3/P4"):
                    extra_reduction = extra_sessions_per_week * allocation.get(category, 0)

            total_reduction = base_reduction + extra_reduction
            avg_sessions = avg_sessions_per_category.get(category, 1)
            avg_weeks_between = avg_weeks_between_sessions.get(category, 1)

            follow_up_sessions = 0
            if avg_weeks_between > 0 and week_number % avg_weeks_between == 0:
                follow_up_sessions = new_referrals * (avg_sessions - 1)

            current_backlog = backlog_projection[category][i-1]
            
            if category in ['P1', 'P2'] and ("P3" in available_categories or "P4" in available_categories):
                if is_p3p4_week and strategy == "1 in 4 weeks for P3/P4":
                    new_backlog = current_backlog + new_referrals + follow_up_sessions
                    patients_seen_per_week[category][i] = 0
                else:
                    new_backlog = current_backlog + new_referrals - total_reduction + follow_up_sessions
                    patients_seen_per_week[category][i] = total_reduction
            else:
                new_backlog = current_backlog + new_referrals - total_reduction + follow_up_sessions
                patients_seen_per_week[category][i] = total_reduction

            backlog_projection[category][i] = max(np.floor(new_backlog), 0)

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

# --- SIDEBAR LOGIC ---
with st.sidebar:
    st.image("https://www.esther.ie/wp-content/uploads/2022/05/HSE-Logo-Green-NEW-no-background.png", width=220)
    st.markdown("### 🔐 Access")
    if not st.session_state.password_verified:
        try:
            password = st.secrets["password"]
        except KeyError:
            st.error("Missing .streamlit/secrets.toml")
            st.stop()
        user_password = st.text_input("Password", type="password", key="password_input")
        if st.button("Log In"):
            if user_password == password:
                st.session_state.password_verified = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    
    if st.session_state.password_verified:
        st.success("Authenticated")
        st.markdown("---")
        
        # PRINT BUTTON
        if st.button("🖨️ Print Report", help="Generate a printable invoice-style report"):
            st.session_state.show_report = True
        else:
             if "show_report" not in st.session_state:
                 st.session_state.show_report = False

        st.markdown("---")
        if not st.session_state.file_uploaded:
            st.markdown("### 📂 Data Import")
            uploaded_file = st.file_uploader("Upload .xlsx", type=["xlsx"], key="main_uploader")
            if uploaded_file is not None:
                try:
                    df_uploaded = pd.read_excel(uploaded_file)
                    if df_uploaded.empty:
                        st.warning("Empty file.")
                    else:
                        df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'], errors='coerce')
                        if 'Category' not in df_uploaded.columns:
                            st.error("Missing 'Category' column.")
                        else:
                            if 'Referral_From' not in df_uploaded.columns:
                                st.warning("Adding dummy 'Referral_From' data for Sankey.")
                                df_uploaded['Referral_From'] = np.random.choice(['Clinic A', 'Clinic B', 'Self-Referral', 'Hospital'], size=len(df_uploaded))

                            df_uploaded.dropna(subset=['Category'], inplace=True)
                            df_uploaded['Category'] = df_uploaded['Category'].astype(str)

                            for cat in df_uploaded['Category'].unique():
                                if cat not in st.session_state.custom_priority_weights:
                                    st.session_state.custom_priority_weights[cat] = 50

                            df_processed = calculate_wps_components(df_uploaded, st.session_state.custom_priority_weights)
                            st.session_state.df = df_processed
                            st.session_state.available_categories = sorted(df_processed['Category'].unique().tolist())
                            st.session_state.category_colors = {cat: DEFAULT_COLORS.get(cat, "#808080") for cat in st.session_state.available_categories}
                            st.session_state.file_uploaded = True
                            st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.file_uploaded:
             if st.button("Reset / Upload New File"):
                st.session_state.df = None
                st.session_state.file_uploaded = False
                st.rerun()

    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

# --- MAIN APP LOGIC ---

if st.session_state.password_verified and st.session_state.df is not None:
    df = calculate_wps_components(st.session_state.df, st.session_state.custom_priority_weights)
    available_categories = st.session_state.available_categories
    category_colors = st.session_state.category_colors
    
    # Pre-calculate WPS for Report Generation
    if st.session_state.show_report:
        # Calculate scores
        total_wps_factors = st.session_state.priority_weight_factor + st.session_state.wait_time_weight_factor
        if total_wps_factors > 0:
            norm_p = st.session_state.priority_weight_factor / total_wps_factors
            norm_w = st.session_state.wait_time_weight_factor / total_wps_factors
            df_wps = df.copy()
            df_wps['WPS'] = (df_wps['Priority_Score'] * norm_p + (df_wps['Waiting_Days'] / 5) * norm_w)
            df_sorted = df_wps.sort_values(by=['WPS', 'Date'], ascending=[False, True]).reset_index(drop=True)
            df_sorted.index += 1
            df_sorted.insert(0, 'Rank', df_sorted.index)
            
            # Metrics
            metrics = {
                'Total': len(df),
                'AvgWait': df['Waiting_Days'].mean()
            }
            
            report_html = generate_report_html(metrics, df_sorted)
            st.markdown(report_html, unsafe_allow_html=True)
            if st.button("🔙 Close Report"):
                st.session_state.show_report = False
                st.rerun()
        else:
            st.error("Please configure weights in the 'Wait List Weights' tab first.")
            st.session_state.show_report = False
            
    else:
        # NORMAL VIEW
        tab1, tab2 = st.tabs(["Waiting List Optimisation", "Wait List Weights"])

        with tab1:
            st.info("💡 **Tip:** Adjust the configuration in the sidebar (or below on mobile) to simulate different staffing scenarios.")
            
            # --- Sidebar Configurations for Tab 1 ---
            # Note: Moved visual rendering to sidebar, logic kept here
            num_therapists = st.sidebar.number_input("👩‍⚕️ Number of Therapists", 1, 20, 1, key="num_therapists_opt")
            sessions_per_therapist_per_week = st.sidebar.number_input("🗓️ Sessions/Therapist/Week", 1, 40, 15, key="sessions_per_therapist_opt")
            num_weeks = st.sidebar.selectbox("📅 Projection Weeks", [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52], 0, key="num_weeks_opt")
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("Clinical Assumptions")
            
            avg_sessions_per_category = {}
            avg_weeks_between_sessions = {}
            default_avg_sessions = {"P1": 6, "P2": 6, "P3": 4, "P4": 3}
            default_avg_weeks_between = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

            for category in available_categories:
                with st.sidebar.expander(f"{category} Configuration"):
                    avg_sessions_per_category[category] = st.number_input(f"{category} Avg Sessions", 1, value=default_avg_sessions.get(category, 1), key=f"avg_sess_{category}")
                    avg_weeks_between_sessions[category] = st.number_input(f"{category} Weeks Between", 1, value=default_avg_weeks_between.get(category, 1), key=f"avg_weeks_{category}")
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("Patient Category Weights")
            for priority in sorted(df['Category'].unique()):
                st.session_state.custom_priority_weights[priority] = st.sidebar.slider(f"{priority} Weight", 0, 100, st.session_state.custom_priority_weights.get(priority, 50), 5, key=f"weight_{priority}")

            # --- Dashboard Content ---
            backlog_initial = {cat: df[df["Category"] == cat].shape[0] for cat in available_categories}
            total_patients = sum(backlog_initial.values())
            
            st.markdown("### 🔄️ Current Status")
            cols = st.columns(len(available_categories) + 1)
            with cols[0]:
                st.metric("Total Patients", total_patients)
            for i, category in enumerate(sorted(available_categories)):
                with cols[i+1]:
                    count = backlog_initial.get(category, 0)
                    st.metric(f"{category} Patients", count, f"{round((count/total_patients)*100)}%")

            st.markdown("---")
            
            # Forecasting
            @st.cache_data
            def get_prophet_forecast(data_df, num_weeks_for_forecast, categories):
                forecasted_referrals = {}
                df_prophet_cached = data_df.copy()
                df_prophet_cached['WeekStartDate'] = df_prophet_cached['Date'].dt.to_period('W').dt.start_time
                weekly_referrals_cached = df_prophet_cached.groupby(['WeekStartDate', 'Category']).size().reset_index(name='Count')

                for category in categories:
                    category_data = weekly_referrals_cached[weekly_referrals_cached['Category'] == category].copy()
                    category_data.rename(columns={'WeekStartDate': 'ds', 'Count': 'y'}, inplace=True)
                    
                    if len(category_data) >= 2:
                        try:
                            m = Prophet(weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05) 
                            m.fit(category_data)
                            future = m.make_future_dataframe(periods=num_weeks_for_forecast, freq='W')
                            forecast = m.predict(future)
                            forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, round(x)))
                            forecasted_referrals[category] = forecast[['ds', 'yhat']]
                        except:
                            mean_val = weekly_referrals_cached[weekly_referrals_cached['Category'] == category]['Count'].mean()
                            dummy_forecast_data = {'ds': pd.to_datetime(pd.date_range(start=df_prophet_cached['Date'].max(), periods=num_weeks_for_forecast, freq='W')), 'yhat': [max(0, round(mean_val))] * num_weeks_for_forecast}
                            forecasted_referrals[category] = pd.DataFrame(dummy_forecast_data)
                    else:
                        mean_val = weekly_referrals_cached[weekly_referrals_cached['Category'] == category]['Count'].mean() if not weekly_referrals_cached[weekly_referrals_cached['Category'] == category].empty else 0
                        dummy_forecast_data = {'ds': pd.to_datetime(pd.date_range(start=df_prophet_cached['Date'].max(), periods=num_weeks_for_forecast, freq='W')), 'yhat': [max(0, round(mean_val))] * num_weeks_for_forecast}
                        forecasted_referrals[category] = pd.DataFrame(dummy_forecast_data)
                return forecasted_referrals

            with st.spinner("Running AI Forecasting..."):
                forecasted_new_referrals_per_week = get_prophet_forecast(st.session_state.df, num_weeks, available_categories)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### ⚙️ Session Logic")
                selected_session = st.radio("Session length (P3/P4):", ["60 min", "50 min", "44 min"], horizontal=True, key="selected_session_opt")
            with col2:
                st.markdown("##### 📅 Scheduling Strategy")
                strategy_options = ["Priority Split", "Urgency-Weighted Scheduling"]
                if "P3" in available_categories or "P4" in available_categories:
                    strategy_options.insert(0, "1 in 4 weeks for P3/P4")
                selected_strategy = st.selectbox("Select strategy:", strategy_options, index=strategy_options.index("Urgency-Weighted Scheduling") if "Urgency-Weighted Scheduling" in strategy_options else 0, key="selected_strategy_opt")

            # Simulation
            if 'Waiting_Days' in df.columns:
                avg_waiting_days_per_category = df.groupby('Category')['Waiting_Days'].mean().to_dict()
            else:
                avg_waiting_days_per_category = {cat: 0 for cat in available_categories}

            total_wps_factors_for_sim = st.session_state.priority_weight_factor + st.session_state.wait_time_weight_factor
            priority_weight_factor_norm_sim = st.session_state.priority_weight_factor / total_wps_factors_for_sim if total_wps_factors_for_sim > 0 else 0
            wait_time_weight_factor_norm_sim = st.session_state.wait_time_weight_factor / total_wps_factors_for_sim if total_wps_factors_for_sim > 0 else 0

            try:
                weeks, backlog_projection, patients_seen_per_week = simulate_backlog_reduction(
                    selected_session, selected_strategy, num_therapists, sessions_per_therapist_per_week,
                    forecasted_new_referrals_per_week, num_weeks, backlog_initial, avg_sessions_per_category,
                    avg_weeks_between_sessions, available_categories, st.session_state.custom_priority_weights,
                    avg_waiting_days_per_category, priority_weight_factor_norm_sim, wait_time_weight_factor_norm_sim
                )

                st.markdown("### 📉 Projected Backlog")
                fig = go.Figure()
                for category in available_categories:
                    fig.add_trace(go.Scatter(x=weeks, y=backlog_projection[category], mode='lines+markers', name=f'{category} Backlog', line=dict(color=category_colors.get(category, "#CCCCCC"))))
                fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="Weeks", yaxis_title="Patients", plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                final_total_backlog = sum(backlog_projection[cat][-1] for cat in available_categories)
                net_change = final_total_backlog - total_patients
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Net Change", f"{net_change:.0f}", delta=f"{net_change:.0f}", delta_color="inverse")
                with m_col2:
                    total_patients_seen = sum(patients_seen_per_week[cat].sum() for cat in available_categories)
                    st.metric("Total Patients Seen", f"{total_patients_seen:.0f}")
                with m_col3:
                    if "P3" in available_categories or "P4" in available_categories:
                        _, _, extra_cap = calculate_extra_sessions(selected_session, num_therapists, num_weeks)
                        st.metric("Extra Capacity Generated", f"{extra_cap} / week")

            except Exception as e:
                st.error(f"Simulation Error: {e}")

            st.markdown("---")
            show_referral_charts(df, available_categories, category_colors)

        with tab2:
            st.markdown("### ⚖️ Weighted Priority Score (WPS)")
            st.markdown("Adjust weighting factors to re-rank the patient list dynamically.")
            
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.priority_weight_factor = st.slider("Priority Weight Factor", 0.0, 1.0, st.session_state.priority_weight_factor, key="wps_p")
            with c2:
                st.session_state.wait_time_weight_factor = st.slider("Wait Time Weight Factor", 0.0, 1.0, st.session_state.wait_time_weight_factor, key="wps_w")

            total_wps_factors = st.session_state.priority_weight_factor + st.session_state.wait_time_weight_factor
            if total_wps_factors > 0:
                norm_p = st.session_state.priority_weight_factor / total_wps_factors
                norm_w = st.session_state.wait_time_weight_factor / total_wps_factors
                
                df_wps = df.copy()
                df_wps['WPS'] = (df_wps['Priority_Score'] * norm_p + (df_wps['Waiting_Days'] / 5) * norm_w)
                df_sorted = df_wps.sort_values(by=['WPS', 'Date'], ascending=[False, True]).reset_index(drop=True)
                df_sorted.index += 1
                df_sorted.insert(0, 'Rank', df_sorted.index)

                st.dataframe(df_sorted[['Rank', 'Category', 'Date', 'Waiting_Days', 'WPS']], use_container_width=True)
                
                csv = df_sorted.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Scored List", csv, "weighted_list.csv", "text/csv")
                
                st.subheader("Score Distribution")
                fig_hist = px.histogram(df_sorted, x='WPS', color='Category', color_discrete_map=category_colors, opacity=0.7)
                fig_hist.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig_hist, use_container_width=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <p>© 2026 Health Service Executive. All rights reserved.</p>
    <p>Digital Solutions Developed by <span class="attribution-text">Dave Maher</span></p>
    <p style="font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;">HSE Estates Infrastructure Intelligence</p>
</div>
""", unsafe_allow_html=True)
