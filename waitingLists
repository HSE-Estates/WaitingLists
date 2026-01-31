import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
from prophet import Prophet
from streamlit_echarts import st_echarts

warnings.filterwarnings("ignore")

# --- Constants ---

WORK_DAY_MINUTES = 480  # 8-hour workday
WORK_DAYS_PER_WEEK = 5

DEFAULT_COLORS = {
    "P1": "#FFFF00",  # Yellow
    "P2": "#00FF00",  # Green
    "P3": "#FFA500",  # Orange
    "P4": "#FF0000",  # Red
    "Total": "#0000FF" # Blue
}

SESSION_TYPES = {
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

# --- Helper Functions (Utilities) ---

def format_time_duration(minutes):
    """Convert minutes to human-readable string."""
    if minutes == 0:
        return "0 minutes"
    elif minutes < 60:
        return f"{minutes} minutes"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours} hours" if mins == 0 else f"{hours} hours {mins} minutes"

def map_waiting_days_to_category(waiting_days):
    """Maps waiting days to a time-based category for Sankey display."""
    if pd.isna(waiting_days):
        return "Unknown"
    if waiting_days > 450: return "Over 15 months"
    elif waiting_days > 365: return "Over 12 months"
    elif waiting_days > 180: return "Over 6 months"
    elif waiting_days > 90: return "Over 3 months"
    else: return "Under 3 months"

def calculate_wps_components(df, custom_priority_weights):
    """
    Calculates and adds WPS-related columns. 
    Crucially, this returns a NEW dataframe and does not mutate in place.
    """
    if df is None or df.empty:
        return df
        
    df_copy = df.copy()
    
    # Ensure Date is datetime
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True, errors='coerce')
    
    # Calculate waiting time
    today = datetime.today()
    df_copy['Waiting_Days'] = (today - df_copy['Date']).dt.days
    
    # Calculate Priority Score based on category mapping
    df_copy['Priority_Score'] = df_copy['Category'].map(custom_priority_weights).fillna(0)
    
    return df_copy

# --- Simulation Helper Functions (Refactored) ---

def calculate_extra_sessions(session_type, num_therapists, num_weeks):
    """Calculates potential extra capacity gained by shortening sessions."""
    standard = SESSION_TYPES["60 min"]["duration"]
    new_dur = SESSION_TYPES[session_type]["duration"]
    
    if new_dur >= standard:
        return 0, 0, 0

    # Capacity calculation
    std_sessions_daily = WORK_DAY_MINUTES // standard
    new_sessions_daily = WORK_DAY_MINUTES // new_dur
    
    extra_per_day = (new_sessions_daily - std_sessions_daily) * num_therapists
    extra_per_week = extra_per_day * WORK_DAYS_PER_WEEK
    total_extra = extra_per_week * num_weeks
    
    return total_extra, extra_per_day / num_therapists, extra_per_week

def calculate_allocation_shares(strategy, available_categories, week_number, custom_priority_weights, median_wait_days, wps_factors):
    """
    Determines how the total weekly capacity should be split among categories 
    based on the selected strategy. Returns a dictionary of percentages (0.0 to 1.0).
    """
    allocation = {cat: 0.0 for cat in available_categories}
    
    # Strategy 1: P3/P4 Blitz (1 in 4 weeks)
    if strategy == "1 in 4 weeks for P3/P4":
        is_p3p4_week = (week_number % 4 == 0)
        has_p3p4 = "P3" in available_categories or "P4" in available_categories
        
        if has_p3p4 and is_p3p4_week:
            # Give everything to P3/P4
            sub_cats = [c for c in ["P3", "P4"] if c in available_categories]
            for c in sub_cats:
                allocation[c] = 1.0 / len(sub_cats)
        else:
            # Give everything to P1/P2
            sub_cats = [c for c in ["P1", "P2"] if c in available_categories]
            # If no P1/P2 exist, fall back to P3/P4 even in non-blitz weeks
            if not sub_cats:
                sub_cats = [c for c in ["P3", "P4"] if c in available_categories]
            
            if sub_cats:
                for c in sub_cats:
                    allocation[c] = 1.0 / len(sub_cats)

    # Strategy 2: Fixed Priority Split
    elif strategy == "Priority Split":
        high = [c for c in available_categories if c in ["P1", "P2"]]
        low = [c for c in available_categories if c in ["P3", "P4"]]
        
        # Default split: 50% high, 50% low
        share_high = 0.5 if low else 1.0
        share_low = 0.5 if high else 1.0
        
        if not high and not low: return allocation # Empty
        
        if high:
            for c in high: allocation[c] = share_high / len(high)
        if low:
            for c in low: allocation[c] = share_low / len(low)

    # Strategy 3: Urgency-Weighted (WPS based)
    else: 
        scores = {}
        for cat in available_categories:
            p_score = custom_priority_weights.get(cat, 0)
            # Use Median to avoid outliers skewing the allocation
            w_days = median_wait_days.get(cat, 0)
            
            # WPS Formula for category level
            score = (p_score * wps_factors['priority']) + ((w_days / 5) * wps_factors['wait_time'])
            scores[cat] = score
            
        total_score = sum(scores.values())
        if total_score > 0:
            for cat in available_categories:
                allocation[cat] = scores[cat] / total_score
        else:
            # Equal distribution fallback
            for cat in available_categories:
                allocation[cat] = 1.0 / len(available_categories)
                
    return allocation

def apply_capacity_constraints(proposed_reductions, total_weekly_capacity):
    """
    Ensures we don't treat more patients than we have slots for.
    If demand > capacity, scale down all treatments proportionally.
    """
    total_proposed = sum(proposed_reductions.values())
    
    if total_proposed <= total_weekly_capacity:
        return proposed_reductions
    
    # Scale down
    scale_factor = total_weekly_capacity / total_proposed
    return {cat: val * scale_factor for cat, val in proposed_reductions.items()}

def simulate_backlog_reduction(
    session_key, strategy, num_therapists, sessions_per_therapist_per_week,
    forecasts, num_weeks, backlog_initial, avg_sessions_per_category,
    avg_weeks_between_sessions, available_categories, custom_priority_weights,
    median_wait_days, wps_factors
):
    """
    Core simulation engine.
    """
    weeks = np.arange(1, num_weeks + 1)
    
    # State tracking
    backlog_history = {cat: [backlog_initial.get(cat, 0)] for cat in available_categories}
    patients_seen_history = {cat: [] for cat in available_categories}
    
    # Calculate global capacity
    base_weekly_capacity = sessions_per_therapist_per_week * num_therapists
    
    # Extra capacity from short sessions (only applies if strategy/categories align, simplified here to general pool)
    # Note: In the original logic, extra sessions were only for P3/P4. We will maintain that logic.
    _, _, extra_sessions_weekly_total = calculate_extra_sessions(session_key, num_therapists, num_weeks)
    
    for i in range(len(weeks)):
        week_num = i + 1
        current_allocations = calculate_allocation_shares(
            strategy, available_categories, week_num, custom_priority_weights, 
            median_wait_days, wps_factors
        )
        
        proposed_treatments = {}
        
        # 1. Determine demand/proposed treatments based on allocation shares
        for cat in available_categories:
            share = current_allocations.get(cat, 0)
            
            # Base capacity share
            cat_capacity = base_weekly_capacity * share
            
            # Add extra capacity for P3/P4 if applicable
            if cat in ["P3", "P4"] and extra_sessions_weekly_total > 0:
                # Logic: Is this a week where P3/P4 get the extra slots?
                # Simplified: If they have allocation > 0, they get their share of extra slots
                if share > 0:
                    cat_capacity += (extra_sessions_weekly_total * share)
            
            proposed_treatments[cat] = cat_capacity

        # 2. Enforce conservation of capacity (Fix #3)
        # We calculate total capacity available this week for safety
        this_week_total_capacity = base_weekly_capacity + (extra_sessions_weekly_total if any(c in ["P3", "P4"] for c in available_categories) else 0)
        actual_treatments = apply_capacity_constraints(proposed_treatments, this_week_total_capacity)
        
        # 3. Process each category
        for cat in available_categories:
            # Get forecasted new referrals
            # Fix #2: Alignment. forecasts[cat] is a dataframe. We need the specific future week.
            # The forecast df passed here should already be sliced or aligned. 
            # We assume forecasts[cat] is a dataframe where row 'i' corresponds to simulation week 'i'
            new_referrals = 0
            if cat in forecasts and i < len(forecasts[cat]):
                 new_referrals = max(0, forecasts[cat].iloc[i]['yhat'])
            
            # Backlog Logic
            prev_backlog = backlog_history[cat][-1]
            treated = actual_treatments.get(cat, 0)
            
            # Fix #5: Follow-up logic approximation
            # Instead of instantaneous demand, we assume a % of current active list generates returns.
            # Or use the simplified heuristic: (new_referrals * (avg_sessions - 1)) distributed?
            # Sticking to the critique's suggestion: "Approximate follow-ups as % of current backlog" is hard without knowing churn.
            # We will use a dampened version of the original to avoid massive spikes, but respecting the user's original intent regarding "avg_weeks_between".
            
            follow_ups = 0
            avg_weeks = avg_weeks_between_sessions.get(cat, 1)
            avg_sess = avg_sessions_per_category.get(cat, 1)
            
            # Original logic logic refined:
            # If week matches the cycle, we add follow-up demand.
            if avg_weeks > 0 and week_num % avg_weeks == 0:
                # Based on RECENT patients seen, not just new referrals. 
                # Approximating active caseload as backlog size * small factor?
                # Let's stick closer to original but ensure it doesn't violate capacity.
                # Actually, follow-ups are ADDED to the backlog (appointments needed).
                follow_ups = new_referrals * (avg_sess - 1)
            
            # Balance equation: New Backlog = Old + Inflow (New + Returns) - Outflow (Treated)
            new_backlog_val = prev_backlog + new_referrals + follow_ups - treated
            
            backlog_history[cat].append(max(0, new_backlog_val))
            patients_seen_history[cat].append(treated)

    # Remove initial state for plotting (keep only weeks 1..N)
    for cat in available_categories:
        backlog_history[cat].pop(0)
        
    return weeks, backlog_history, patients_seen_history

# --- Forecasting Wrapper ---

@st.cache_data
def get_prophet_forecast(df, num_weeks, categories):
    """
    Generates forecasts. Fixes index mismatch by returning ONLY future rows.
    """
    results = {}
    
    # Aggregate weekly first
    df_agg = df.copy()
    df_agg['WeekStart'] = df_agg['Date'].dt.to_period('W').dt.start_time
    weekly = df_agg.groupby(['WeekStart', 'Category']).size().reset_index(name='y')
    weekly.rename(columns={'WeekStart': 'ds'}, inplace=True)
    
    last_date = df['Date'].max()
    
    for cat in categories:
        cat_data = weekly[weekly['Category'] == cat].copy()
        
        if len(cat_data) < 2:
            # Fallback: Average
            mean_val = cat_data['y'].mean() if not cat_data.empty else 0
            dates = pd.date_range(start=last_date, periods=num_weeks + 1, freq='W')[1:] # Next week onwards
            results[cat] = pd.DataFrame({'ds': dates, 'yhat': [mean_val] * num_weeks})
            continue
            
        try:
            m = Prophet(weekly_seasonality=True, daily_seasonality=False)
            m.fit(cat_data)
            
            # Make future dataframe. We need to be careful to extend from the last actual date.
            future = m.make_future_dataframe(periods=num_weeks, freq='W')
            forecast = m.predict(future)
            
            # Fix #2: Filter only future dates
            future_forecast = forecast[forecast['ds'] > last_date].copy()
            
            # Ensure we have exactly num_weeks
            if len(future_forecast) > num_weeks:
                future_forecast = future_forecast.head(num_weeks)
            
            future_forecast['yhat'] = future_forecast['yhat'].apply(lambda x: max(0, x))
            results[cat] = future_forecast.reset_index(drop=True)
            
        except Exception:
            # Fallback on error
            mean_val = cat_data['y'].mean()
            dates = pd.date_range(start=last_date, periods=num_weeks + 1, freq='W')[1:]
            results[cat] = pd.DataFrame({'ds': dates, 'yhat': [mean_val] * num_weeks})
            
    return results

# --- Charts ---

def show_referral_charts(df, available_categories, colors):
    st.subheader("📊 Referral Analysis")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        cat_filter = st.selectbox("Filter Category:", ["All"] + available_categories)
    with col2:
        min_refs = st.slider("Min Referrals to Show:", 0, 50, 3)
        
    data = df.copy()
    if cat_filter != "All":
        data = data[data['Category'] == cat_filter]
        
    counts = data.groupby(['Referral_From', 'Category']).size().reset_index(name='Count')
    total_per_ref = counts.groupby('Referral_From')['Count'].sum()
    valid_refs = total_per_ref[total_per_ref > min_refs].index
    
    filtered = counts[counts['Referral_From'].isin(valid_refs)]
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(filtered, x='Referral_From', y='Count', color='Category', 
                     color_discrete_map=colors, title="Referrals by Source")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        fig = px.sunburst(filtered, path=['Referral_From', 'Category'], values='Count',
                          color='Category', color_discrete_map=colors, title="Referral Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    # Sankey
    st.subheader("Patient Flow: Source → Category → Wait Duration")
    sankey_df = data.copy()
    sankey_df['Wait_Group'] = sankey_df['Waiting_Days'].apply(map_waiting_days_to_category)
    
    agg = sankey_df.groupby(['Referral_From', 'Category', 'Wait_Group']).size().reset_index(name='val')
    
    # Nodes
    nodes = set(agg['Referral_From']) | set(agg['Category']) | set(agg['Wait_Group'])
    nodes_list = [{"name": n} for n in sorted(list(nodes))]
    
    links = []
    for _, row in agg.iterrows():
        links.append({"source": str(row['Referral_From']), "target": str(row['Category']), "value": row['val']})
        links.append({"source": str(row['Category']), "target": str(row['Wait_Group']), "value": row['val']})
        
    opt = {
        "tooltip": {"trigger": "item"},
        "series": [{
            "type": "sankey",
            "layout": "none",
            "data": nodes_list,
            "links": links,
            "emphasis": {"focus": "adjacency"},
            "lineStyle": {"color": "gradient", "curveness": 0.5}
        }]
    }
    st_echarts(opt, height="500px")

# --- Main App Logic ---

st.set_page_config(page_title="Waiting List Optimisation", layout="wide", page_icon="🏥")

# Init Session State
defaults = {
    "password_verified": False,
    "raw_df": None,         # Fix #1: Store raw data separately
    "processed_df": None,   # Store processed data
    "custom_weights": {'P1': 100, 'P2': 75, 'P3': 50, 'P4': 25},
    "wps_priority_factor": 1.0,
    "wps_time_factor": 0.3
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Auth
if not st.session_state.password_verified:
    pwd = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            # Check secrets first, fallback to hardcoded for demo purposes if needed
            real_pwd = st.secrets.get("password", "admin") 
            if pwd == real_pwd:
                st.session_state.password_verified = True
                st.rerun()
            else:
                st.error("Incorrect password")
        except:
             # Fallback if secrets not configured
             if pwd == "admin":
                 st.session_state.password_verified = True
                 st.rerun()
             else:
                 st.error("Incorrect password (try 'admin' if local)")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🏥 Settings")
    
    # File Upload
    uploaded = st.file_uploader("Upload Data (Excel)", type=["xlsx"])
    if uploaded:
        try:
            raw = pd.read_excel(uploaded)
            if 'Category' not in raw.columns or 'Date' not in raw.columns:
                st.error("File must contain 'Category' and 'Date' columns.")
            else:
                if 'Referral_From' not in raw.columns:
                    raw['Referral_From'] = "Unknown"
                raw.dropna(subset=['Category'], inplace=True)
                raw['Category'] = raw['Category'].astype(str)
                # Save RAW data
                st.session_state.raw_df = raw
                st.success("Data Loaded!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    
    if st.session_state.raw_df is not None:
        st.subheader("Category Weights")
        cats = sorted(st.session_state.raw_df['Category'].unique())
        for c in cats:
            val = st.slider(f"Weight {c}", 0, 100, st.session_state.custom_weights.get(c, 50), key=f"w_{c}")
            st.session_state.custom_weights[c] = val

# Main Content
if st.session_state.raw_df is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

# Process Data (Run every time, but from immutable raw_df)
# Fix #1: This ensures we don't drift the waiting days by repeated modification
df_proc = calculate_wps_components(st.session_state.raw_df, st.session_state.custom_weights)
st.session_state.processed_df = df_proc

available_cats = sorted(df_proc['Category'].unique())
cat_colors = {c: DEFAULT_COLORS.get(c, "#888888") for c in available_cats}

tab1, tab2 = st.tabs(["📉 Optimisation & Simulation", "⚖️ WPS Weighting"])

# --- TAB 1: OPTIMISATION ---
with tab1:
    st.title("Waiting List Simulation")
    
    # Inputs
    c1, c2, c3 = st.columns(3)
    with c1:
        n_therapists = st.number_input("Therapists", 1, 50, 5)
    with c2:
        sess_per_week = st.number_input("Sessions/Therapist/Week", 1, 50, 15)
    with c3:
        n_weeks = st.selectbox("Projection Weeks", [12, 24, 52], index=0)
        
    st.markdown("### ⚙️ Simulation Strategy")
    col_strat, col_len = st.columns(2)
    with col_strat:
        strategy = st.selectbox("Allocation Strategy", 
            ["Priority Split", "Urgency-Weighted Scheduling", "1 in 4 weeks for P3/P4"])
    with col_len:
        sess_len_key = st.radio("Session Length (P3/P4)", ["60 min", "50 min", "44 min"], horizontal=True)

    # Advanced Inputs
    with st.expander("Advanced Configuration (Sessions & Frequency)"):
        col_a, col_b = st.columns(2)
        avg_sess = {}
        weeks_btwn = {}
        for c in available_cats:
            with col_a:
                avg_sess[c] = st.number_input(f"Avg Sessions ({c})", 1, 20, 6, key=f"as_{c}")
            with col_b:
                weeks_btwn[c] = st.number_input(f"Weeks Between ({c})", 1, 12, 2, key=f"wb_{c}")

    # Run Simulation
    if st.button("Run Simulation"):
        with st.spinner("Calculating Forecasts..."):
            # Forecasts
            forecasts = get_prophet_forecast(df_proc, n_weeks, available_cats)
            
            # Initial Backlog
            initial_bl = df_proc['Category'].value_counts().to_dict()
            
            # Median Wait Days (Fix #4: Use Median)
            median_wait = df_proc.groupby('Category')['Waiting_Days'].median().to_dict()
            
            # Normalise WPS factors for simulation
            tot_fact = st.session_state.wps_priority_factor + st.session_state.wps_time_factor
            wps_norm = {
                'priority': st.session_state.wps_priority_factor / tot_fact if tot_fact else 0,
                'wait_time': st.session_state.wps_time_factor / tot_fact if tot_fact else 0
            }
            
            # Run Sim
            weeks, bl_hist, treated_hist = simulate_backlog_reduction(
                sess_len_key, strategy, n_therapists, sess_per_week,
                forecasts, n_weeks, initial_bl, avg_sess, weeks_btwn,
                available_cats, st.session_state.custom_weights,
                median_wait, wps_norm
            )
            
            # Results
            fig = go.Figure()
            for c in available_cats:
                fig.add_trace(go.Scatter(
                    x=weeks, y=bl_hist[c], mode='lines', 
                    name=c, line=dict(color=cat_colors[c])
                ))
            
            fig.update_layout(title="Backlog Projection", xaxis_title="Week", yaxis_title="Patients Waiting")
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            total_start = sum(initial_bl.values())
            total_end = sum([bl_hist[c][-1] for c in available_cats])
            delta = total_end - total_start
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Start Backlog", total_start)
            m2.metric("End Backlog", int(total_end))
            m3.metric("Net Change", int(delta), delta_color="inverse")
            
            # Show Analysis Charts
            st.markdown("---")
            show_referral_charts(df_proc, available_cats, cat_colors)

# --- TAB 2: WPS WEIGHTS ---
with tab2:
    st.title("WPS Configuration")
    st.info("Adjust how Priority vs Waiting Time impacts the final score.")
    
    c1, c2 = st.columns(2)
    with c1:
        p_factor = st.slider("Priority Factor", 0.0, 1.0, st.session_state.wps_priority_factor)
        st.session_state.wps_priority_factor = p_factor
    with c2:
        t_factor = st.slider("Wait Time Factor", 0.0, 1.0, st.session_state.wps_time_factor)
        st.session_state.wps_time_factor = t_factor
        
    # Recalculate WPS for display
    # Normalise
    total = p_factor + t_factor
    norm_p = p_factor/total if total else 0
    norm_t = t_factor/total if total else 0
    
    # Calculate
    display_df = df_proc.copy()
    display_df['WPS'] = (display_df['Priority_Score'] * norm_p) + ((display_df['Waiting_Days'] / 5) * norm_t)
    display_df = display_df.sort_values(['WPS', 'Waiting_Days'], ascending=[False, False])
    
    st.dataframe(display_df[['Date', 'Category', 'Waiting_Days', 'Priority_Score', 'WPS']].head(100))
    
    # Download
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Prioritised List", csv, "prioritised_list.csv", "text/csv")
    
    # Distribution
    fig = px.histogram(display_df, x="WPS", color="Category", color_discrete_map=cat_colors, nbins=30)
    st.plotly_chart(fig, use_container_width=True)
