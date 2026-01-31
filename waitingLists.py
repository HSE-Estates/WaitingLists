import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
from prophet import Prophet # Import Prophet
from streamlit_echarts import st_echarts # Import ECharts component

warnings.filterwarnings("ignore") # Suppress pandas SettingWithCopyWarning etc.

# --- Constants and Helper Functions (Shared) ---

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

# Constants for time calculations
WORK_DAY_MINUTES = 480  # 8-hour workday
WORK_DAYS_PER_WEEK = 5

# Session configurations
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

# Define a default color map for categories
DEFAULT_COLORS = {
    "P1": "#FFFF00",  # Yellow
    "P2": "#00FF00",  # Green
    "P3": "#FFA500",  # Orange
    "P4": "#FF0000",  # Red
    "Total": "#0000FF" # Blue
}

# Function to map waiting days to time categories for Sankey display
def map_waiting_days_to_category(waiting_days):
    """Maps waiting days to a time-based category for Sankey display."""
    if pd.isna(waiting_days):
        return "Unknown"
    if waiting_days > 450: # 15 months * 30 days/month approx
        return "Over 15 months"
    elif waiting_days > 365: # 12 months
        return "Over 12 months"
    elif waiting_days > 180: # 6 months
        return "Over 6 months"
    elif waiting_days > 90: # 3 months
        return "Over 3 months"
    else:
        return "Under 3 months" # New category for those under 3 months

# --- NEW: Function to calculate WPS components ---
def calculate_wps_components(df, custom_priority_weights):
    """
    Calculates and adds WPS-related columns (Waiting_Days, Priority_Score) to the DataFrame.
    """
    df_copy = df.copy() # Work on a copy to avoid modifying original df directly in place

    # Ensure 'Date' column is datetime
    df_copy['Date'] = pd.to_datetime(df_copy['Date'], dayfirst=True, errors='coerce')

    # Calculate waiting time in days
    today = datetime.today()
    df_copy['Waiting_Days'] = (today - df_copy['Date']).dt.days

    # Calculate Priority_Score
    df_copy['Priority_Score'] = df_copy['Category'].map(custom_priority_weights).fillna(0)

    return df_copy

# --- Functions for Waiting List Optimisation Page ---

def show_referral_charts(df, available_categories, category_colors):
    """
    Displays referral breakdown charts, dynamically adjusting for available categories.
    Now includes a Sankey diagram.
    """
    if "Category" not in df.columns:
        st.warning("The uploaded file must contain 'Category' column to show referral charts.")
        return

    st.subheader("📊 Referral Breakdown")
    
    # Calculate referral counts
    referral_counts = df.groupby(["Referral_From", "Category"]).size().reset_index(name="Count")
    
    # Calculate total referrals per referrer
    total_referrals = referral_counts.groupby("Referral_From")["Count"].sum().reset_index()
    
    # Layout for sorting & filtering controls
    col_sort, col_filter = st.columns([1, 1])

    with col_sort:
        # Dynamically create category filter options
        filter_options = ["All"] + sorted(available_categories)
        category_filter = st.selectbox("Filter by category:", filter_options, key="referral_category_filter")

    with col_filter:
        min_referrals = st.slider("Show referrers with more than X referrals:", 
                                  min_value=0, 
                                  max_value=int(total_referrals["Count"].max()), 
                                  value=3,
                                  key="referral_min_referrals")
    
    # Apply category filter
    if category_filter != "All":
        referral_counts = referral_counts[referral_counts["Category"] == category_filter]

    # Apply referral count filter
    referrers_filtered = total_referrals[total_referrals["Count"].fillna(0) > min_referrals]["Referral_From"]
    filtered_referral_counts = referral_counts[referral_counts["Referral_From"].isin(referrers_filtered)].copy()

    # Sort referrers by total referrals descending
    sorted_referrers = total_referrals.sort_values(by="Count", ascending=False)["Referral_From"]
    
    filtered_referral_counts["Referral_From"] = pd.Categorical(filtered_referral_counts["Referral_From"], categories=sorted_referrers, ordered=True)
    filtered_referral_counts = filtered_referral_counts.sort_values("Referral_From")

    # Plot charts
    col1, col2 = st.columns(2)

    with col1:
        fig_bar = px.bar(filtered_referral_counts, x="Referral_From", y="Count", color="Category",
                         title="Referrals per Referrer by Category",
                         labels={"Count": "Number of Referrals"},
                         barmode="stack",
                         opacity=0.6,
                         color_discrete_map=category_colors)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_sunburst = px.sunburst(filtered_referral_counts, path=["Referral_From", "Category"], values="Count",
                                   title="Referral Breakdown by Category and Referrer",
                                   color="Category",
                                   color_discrete_map=category_colors)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    # Sankey Diagram (ECharts version)
    st.subheader("Flow of Patients: Referral → Category → Wait Time (Interactive)")
    
    df_sankey = df.copy() # Create a copy for Sankey specific processing

    # Apply the mapping for Sankey display
    df_sankey['Wait_Time_Category_Sankey'] = df_sankey['Waiting_Days'].apply(map_waiting_days_to_category)

    if "Referral_From" in df_sankey.columns and "Category" in df_sankey.columns and "Wait_Time_Category_Sankey" in df_sankey.columns:
        # Aggregate data for Sankey
        agg_df = df_sankey.groupby(['Referral_From', 'Category', 'Wait_Time_Category_Sankey']).size().reset_index(name='Count')

        # Define nodes - sorted alphabetically for consistency in ECharts
        all_nodes = sorted(list(pd.concat([agg_df['Referral_From'], agg_df['Category'], agg_df['Wait_Time_Category_Sankey']]).unique()))
        echarts_nodes = [{"name": node} for node in all_nodes]

        # Build ECharts links using node NAMES (strings)
        echarts_links = []
        for _, row in agg_df.iterrows():
            # Link from Referral_From to Category
            echarts_links.append({
                "source": str(row['Referral_From']), # Ensure string
                "target": str(row['Category']),     # Ensure string
                "value": row['Count']
            })
            # Link from Category to Wait_Time_Category_Sankey
            echarts_links.append({
                "source": str(row['Category']),     # Ensure string
                "target": str(row['Wait_Time_Category_Sankey']), # Use the new time category
                "value": row['Count']
            })

        option = {
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove"
            },
            "series": [
                {
                    "type": "sankey",
                    "layout": "none",
                    "data": echarts_nodes,
                    "links": echarts_links,
                    "focusNodeAdjacency": "allEdges",
                    "itemStyle": {
                        "borderWidth": 1,
                        "borderColor": "#aaa"
                    },
                    "lineStyle": {
                        "color": "gradient",
                        "curveness": 0.5
                    },
                    "label": {
                        "position": "right"
                    }
                }
            ]
        }
        st_echarts(option, height="600px")
    else:
        st.warning("Required columns for Sankey diagram (Referral_From, Category, and 'Date' for wait time calculation) are missing or could not be generated.")


def calculate_extra_sessions(session_type, num_therapists, num_weeks):
    """
    Calculates extra sessions based on session length reduction.
    This function is specifically for P3/P4 if they exist, otherwise it returns 0.
    """
    standard_duration = session_types["60 min"]["duration"]
    new_duration = session_types[session_type]["duration"]
    
    # Only calculate extra sessions if the session duration is actually shorter
    if new_duration >= standard_duration:
        return 0, 0, 0 # No extra sessions if duration is not reduced

    workday_standard_sessions = WORK_DAY_MINUTES // standard_duration
    workday_new_sessions = WORK_DAY_MINUTES // new_duration
    
    extra_sessions_per_day = workday_new_sessions - workday_standard_sessions
    extra_sessions_per_week = extra_sessions_per_day * WORK_DAYS_PER_WEEK * num_therapists
    total_extra_sessions = extra_sessions_per_week * num_weeks
    
    return total_extra_sessions, extra_sessions_per_day, extra_sessions_per_week

def simulate_backlog_reduction(session_key, strategy, num_therapists, sessions_per_therapist_per_week, forecasted_new_referrals_per_week, num_weeks, backlog_initial, avg_sessions_per_category, avg_weeks_between_sessions, available_categories, custom_priority_weights, avg_waiting_days_per_category, priority_weight_factor_norm, wait_time_weight_factor_norm):
    """
    Simulate backlog reduction based on session configurations and strategies with follow-up sessions,
    dynamically adjusting for available categories.
    `forecasted_new_referrals_per_week` is now a dictionary of DataFrames,
    where each DataFrame contains 'ds' and 'yhat' for a category's forecast.
    `custom_priority_weights` is a dictionary of weights for P1, P2, P3, P4.
    New parameters added for Urgency-Weighted Scheduling to incorporate all WPS factors.
    """
    if num_therapists <= 0:
        raise ValueError("Number of therapists must be greater than 0")
    if sessions_per_therapist_per_week <= 0:
        raise ValueError("Sessions per therapist must be greater than 0")

    weeks = np.arange(1, num_weeks + 1)
    backlog_projection = {category: np.zeros(len(weeks)) for category in available_categories}
    patients_seen_per_week = {category: np.zeros(len(weeks)) for category in available_categories} 

    # Calculate extra sessions, only relevant if P3/P4 exist and strategy uses them
    total_extra_sessions, extra_sessions_per_day, extra_sessions_per_week = calculate_extra_sessions(session_key, num_therapists, num_weeks)

    # Set initial backlog
    for category in available_categories:
        backlog_projection[category][0] = backlog_initial.get(category, 0) # Use .get with default 0 for safety

    for i in range(1, len(weeks)):
        week_number = i + 1
        # Initialise allocation for all available categories to 0
        allocation = {cat: 0 for cat in available_categories}

        is_p3p4_week = (strategy == "1 in 4 weeks for P3/P4" and week_number % 4 == 0)
        
        # Determine current week's allocation based on strategy and available categories
        if strategy == "1 in 4 weeks for P3/P4":
            if "P3" in available_categories or "P4" in available_categories:
                if is_p3p4_week:
                    if "P3" in available_categories: allocation["P3"] = 0.5 if "P4" in available_categories else 1.0
                    if "P4" in available_categories: allocation["P4"] = 0.5 if "P3" in available_categories else 1.0
                else:
                    if "P1" in available_categories: allocation["P1"] = 0.5 if "P2" in available_categories else 1.0
                    if "P2" in available_categories: allocation["P2"] = 0.5 if "P1" in available_categories else 1.0
            else: # If no P3/P4, distribute among P1/P2
                if "P1" in available_categories: allocation["P1"] = 0.5 if "P2" in available_categories else 1.0
                if "P2" in available_categories: allocation["P2"] = 0.5 if "P1" in available_categories else 1.0
        elif strategy == "Priority Split":
            high_priority_cats = [cat for cat in available_categories if cat in ["P1", "P2"]]
            low_priority_cats = [cat for cat in available_categories if cat in ["P3", "P4"]]

            # Initialise all allocations to 0.0
            for cat in available_categories:
                allocation[cat] = 0.0

            # Special case: Only P1 and P2 are present
            if not low_priority_cats and "P1" in high_priority_cats and "P2" in high_priority_cats:
                allocation["P1"] = 0.3
                allocation["P2"] = 0.7
            elif not low_priority_cats and "P1" in high_priority_cats: # Only P1
                allocation["P1"] = 1.0
            elif not low_priority_cats and "P2" in high_priority_cats: # Only P2
                allocation["P2"] = 1.0
            else:
                # General case: split 50% for high priority, 50% for low priority
                target_high_share = 0.5
                target_low_share = 0.5

                # Adjust shares if only one group of categories is present
                if not low_priority_cats and high_priority_cats:
                    target_high_share = 1.0
                    target_low_share = 0.0
                elif not high_priority_cats and low_priority_cats:
                    target_high_share = 0.0
                    target_low_share = 1.0

                # Distribute shares among categories within their groups
                if high_priority_cats and target_high_share > 0:
                    split_per_high_cat = target_high_share / len(high_priority_cats)
                    for cat in high_priority_cats:
                        allocation[cat] = split_per_high_cat
                
                if low_priority_cats and target_low_share > 0:
                    split_per_low_cat = target_low_share / len(low_priority_cats)
                    for cat in low_priority_cats:
                        allocation[cat] = split_per_low_cat

        else:  # Urgency-Weighted - now incorporates all WPS components for category allocation
            current_category_wps_scores = {}
            for cat in available_categories:
                # Get the base priority weight for the category
                priority_score = custom_priority_weights.get(cat, 0)
                # Get the average waiting days for the category
                avg_wait_days = avg_waiting_days_per_category.get(cat, 0)

                # Calculate a "category WPS" for allocation, using normalised weight factors
                category_wps = (
                    priority_score * priority_weight_factor_norm
                    + (avg_wait_days / 5) * wait_time_weight_factor_norm # Scaled by 5
                )
                current_category_wps_scores[cat] = category_wps

            total_category_wps_score = sum(current_category_wps_scores.values())
            
            if total_category_wps_score > 0:
                for cat in available_categories:
                    allocation[cat] = current_category_wps_scores[cat] / total_category_wps_score
            else:
                # Fallback if no weights are defined or all are zero
                for cat in available_categories:
                    allocation[cat] = 1.0 / len(available_categories) if len(available_categories) > 0 else 0


        for category in available_categories:
            # Get the forecasted new referrals for the current week
            # We need to map the simulation week (1-indexed) to the forecast index (0-indexed)
            # Ensure we don't go out of bounds for the forecast
            if category in forecasted_new_referrals_per_week and i < len(forecasted_new_referrals_per_week[category]):
                new_referrals = max(0, forecasted_new_referrals_per_week[category].iloc[i]['yhat']) # Use yhat from Prophet forecast
            else:
                new_referrals = 0 # Default to 0 if no forecast available for this week/category

            base_weekly_sessions = sessions_per_therapist_per_week * num_therapists
            base_reduction = base_weekly_sessions * allocation.get(category, 0) # Use .get for safety

            extra_reduction = 0
            # Apply extra reduction only if P3/P4 exist and conditions are met
            if extra_sessions_per_week > 0 and category in ["P3", "P4"] and category in available_categories:
                if (is_p3p4_week and strategy == "1 in 4 weeks for P3/P4") or (strategy != "1 in 4 weeks for P3/P4"):
                    extra_reduction = extra_sessions_per_week * allocation.get(category, 0)

            total_reduction = base_reduction + extra_reduction

            # Ensure avg_sessions and avg_weeks_between are available for the category
            avg_sessions = avg_sessions_per_category.get(category, 1) # Default to 1 session
            avg_weeks_between = avg_weeks_between_sessions.get(category, 1) # Default to 1 week

            follow_up_sessions = 0
            if avg_weeks_between > 0 and week_number % avg_weeks_between == 0:
                follow_up_sessions = new_referrals * (avg_sessions - 1)

            current_backlog = backlog_projection[category][i-1]
            
            # Special handling for "1 in 4 weeks for P3/P4" strategy for P1/P2 categories
            if category in ['P1', 'P2'] and ("P3" in available_categories or "P4" in available_categories):
                if is_p3p4_week and strategy == "1 in 4 weeks for P3/P4":
                    # P1/P2 don't get sessions in P3/P4 weeks under this strategy
                    new_backlog = current_backlog + new_referrals + follow_up_sessions
                    patients_seen_per_week[category][i] = 0 # No patients seen for P1/P2 in this week
                else:
                    new_backlog = current_backlog + new_referrals - total_reduction + follow_up_sessions
                    patients_seen_per_week[category][i] = total_reduction
            else:
                new_backlog = current_backlog + new_referrals - total_reduction + follow_up_sessions
                patients_seen_per_week[category][i] = total_reduction

            backlog_projection[category][i] = max(np.floor(new_backlog), 0)

    return weeks, backlog_projection, patients_seen_per_week

# --- Streamlit Application Layout ---

st.set_page_config(
    page_title="Waiting List",
    layout="wide",
    page_icon="https://www.ehealthireland.ie/media/k1app1wt/hse-logo-black-png.png"
)

# Initialize all session state variables at the very beginning
if "password_verified" not in st.session_state:
    st.session_state.password_verified = False
if "df" not in st.session_state:
    st.session_state.df = None
if "available_categories" not in st.session_state:
    st.session_state.available_categories = []
if "category_colors" not in st.session_state:
    st.session_state.category_colors = {}
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "priority_weight_factor" not in st.session_state:
    st.session_state.priority_weight_factor = 1.0
if "wait_time_weight_factor" not in st.session_state:
    st.session_state.wait_time_weight_factor = 0.3
if "custom_priority_weights" not in st.session_state:
    st.session_state.custom_priority_weights = {
        'P1': 100,
        'P2': 75,
        'P3': 50,
        'P4': 25
    }

if not st.session_state.password_verified:
    try:
        password = st.secrets["password"]
    except KeyError:
        st.error("Password not found in Streamlit secrets. Please set it up in .streamlit/secrets.toml.")
        st.stop()

    user_password = st.text_input("Enter password to access the app:", type="password", key="password_input")
    submit_button = st.button("Submit")

    if submit_button:
        if user_password == password:
            st.session_state.password_verified = True
            st.rerun()
        else:
            st.warning("Incorrect password. Please try again.")
else:
    st.sidebar.image("https://www.ehealthireland.ie/media/k1app1wt/hse-logo-black-png.png", width=200)
    st.sidebar.title("📃 Waiting List App")
    if st.sidebar.button("Created by Dave Maher"):
        st.sidebar.write("This application intellectual property belongs to Dave Maher.")

    st.sidebar.markdown("---")
    
    # Central file uploader and data processing
    # Display file uploader only if no file has been uploaded yet
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader("📂 Upload an Excel file with waiting list data", type=["xlsx"], key="main_uploader")

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_excel(uploaded_file)
                
                # Check if DataFrame is empty after reading
                if df_uploaded.empty:
                    st.warning("The uploaded Excel file is empty. Please upload a file with data.")
                    st.session_state.df = None
                    st.session_state.file_uploaded = False # Keep uploader visible
                else:
                    # Initial data cleaning
                    df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'], errors='coerce')
                    if 'Category' not in df_uploaded.columns:
                        st.error("The uploaded file must contain a 'Category' column.")
                        st.session_state.df = None
                        st.session_state.file_uploaded = False # Keep uploader visible
                    
                    if 'Referral_From' not in df_uploaded.columns:
                        st.warning("⚠️ The 'Referral_From' column was not found in your uploaded data. A dummy 'Referral_From' column has been generated for the Sankey diagram to ensure functionality. Please ensure your input file contains a 'Referral_From' column for accurate results.")
                        dummy_referrals = np.random.choice(['Clinic A', 'Clinic B', 'Self-Referral', 'Hospital'], size=len(df_uploaded))
                        df_uploaded['Referral_From'] = dummy_referrals

                    df_uploaded.dropna(subset=['Category'], inplace=True)
                    df_uploaded['Category'] = df_uploaded['Category'].astype(str)

                    # Initialise custom_priority_weights for new categories
                    for cat in df_uploaded['Category'].unique():
                        if cat not in st.session_state.custom_priority_weights:
                            st.session_state.custom_priority_weights[cat] = 50 # A reasonable default

                    # Call the new function to calculate WPS components
                    df_processed = calculate_wps_components(
                        df_uploaded,
                        st.session_state.custom_priority_weights
                    )

                    st.session_state.df = df_processed
                    st.session_state.available_categories = sorted(df_processed['Category'].unique().tolist())
                    st.session_state.category_colors = {cat: DEFAULT_COLORS.get(cat, "#808080") for cat in st.session_state.available_categories}
                    
                    st.session_state.file_uploaded = True # Set flag to hide uploader
                    st.success("File uploaded and processed successfully! You can now navigate the app.")
                    st.rerun() # Rerun to hide the uploader and display the content
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")
                st.session_state.df = None # Clear data on error
                st.session_state.file_uploaded = False # Keep uploader visible
    elif st.session_state.file_uploaded:
        st.info("File already uploaded. Use the tabs below to navigate.")

    # Check if data is loaded before rendering pages
    if st.session_state.df is None:
        pass # The initial st.info("Please upload an Excel file...") handles this
    else:
        # IMPORTANT: Recalculate WPS components every time the app runs
        # to ensure they reflect the latest slider values for weights/multipliers.
        df = calculate_wps_components(
            st.session_state.df,
            st.session_state.custom_priority_weights
        )
        available_categories = st.session_state.available_categories
        category_colors = st.session_state.category_colors

        # Use st.tabs for navigation
        tab1, tab2 = st.tabs(["Waiting List Optimisation", "Wait List Weights"])

        with tab1:
            st.title("📃 Waiting List Optimisation")

            st.markdown("""
            This application helps optimise therapy waitlist management by:
            - Analysing current waitlist data
            - Projecting future waitlist trends
            - Simulating different scheduling strategies
            - Providing ML-based predictions
            """)

            st.subheader("📊 Current Data Overview")
            st.write(df.head())

            # Moved Configuration to sidebar for Tab 1
            st.sidebar.header("⚙️ Configuration")
            
            num_therapists = st.sidebar.number_input(
                "👩‍⚕️ Number of Therapists",
                min_value=1,
                max_value=20,
                value=1,
                help="Enter the number of available therapists (1-20)",
                key="num_therapists_opt"
            )
            
            sessions_per_therapist_per_week = st.sidebar.number_input(
                "🗓️ Sessions per Therapist per Week",
                min_value=1,
                max_value=40,
                value=15,
                help="Enter the number of sessions per therapist per week (1-40)",
                key="sessions_per_therapist_opt"
            )
            
            num_weeks = st.sidebar.selectbox(
                "📅 Number of Weeks for Projection",
                [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52],
                index=0,
                key="num_weeks_opt"
            )
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Average Sessions and Weeks Between")
            
            avg_sessions_per_category = {}
            avg_weeks_between_sessions = {}
            
            default_avg_sessions = {"P1": 6, "P2": 6, "P3": 4, "P4": 3}
            default_avg_weeks_between = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

            for category in available_categories:
                avg_sessions_per_category[category] = st.sidebar.number_input(
                    f"Average Sessions for {category}", 
                    min_value=1, 
                    value=default_avg_sessions.get(category, 1),
                    key=f"avg_sessions_opt_{category}"
                )
                
                avg_weeks_between_sessions[category] = st.sidebar.number_input(
                    f"Average Weeks Between Sessions for {category}", 
                    min_value=1, 
                    value=default_avg_weeks_between.get(category, 1),
                    key=f"avg_weeks_between_opt_{category}"
                )

            st.sidebar.markdown("---")
            
            # Consolidated Patient Category Weights section with expander
            st.sidebar.subheader("⚖️ Patient Category Weights")
            with st.sidebar.expander("More Info on Patient Category Weights"):
                st.markdown("""
                These weights define the relative importance of each patient category (P1 to P4)
                for both the 'Urgency-Weighted Scheduling' strategy (on this tab) and as the
                'Priority Score' component in the Weighted Priority Score (WPS) calculation
                (on the 'Wait List Weights' tab). Higher weights mean more sessions will be
                allocated to that category in simulations, and a higher contribution to WPS.
                """)

            unique_priorities = sorted(df['Category'].unique().tolist())
            for priority in unique_priorities:
                current_value = st.sidebar.slider(
                    f"Weight for {priority}", 
                    min_value=0,
                    max_value=100,
                    value=st.session_state.custom_priority_weights.get(priority, 50), 
                    step=5,
                    key=f"custom_priority_weight_{priority}"
                )
                st.session_state.custom_priority_weights[priority] = current_value


            st.markdown("---")
            st.subheader("🔄️ Initial Backlog Counts")
            
            backlog_initial = {
                category: df[df["Category"] == category].shape[0] 
                for category in available_categories
            }
            total_patients = sum(backlog_initial.values())
            
            backlog_percentage = {}
            if total_patients > 0:
                backlog_percentage = {
                    category: round((count / total_patients) * 100, 0)
                    for category, count in backlog_initial.items()
                }
            else:
                st.info("No patients found in the uploaded data.")

            display_categories = sorted(available_categories) + ["Total"]
            columns = st.columns(len(display_categories))
            
            for i, category in enumerate(display_categories):
                with columns[i]:
                    if category == "Total":
                        st.metric(f"{category} Patients", total_patients)
                        st.markdown(f"""
                            <div style="background-color: {DEFAULT_COLORS['Total']}; width: 100%; height: 20px; opacity: 0.6; border-radius: 10px;"></div>
                        """, unsafe_allow_html=True)
                    else:
                        count = backlog_initial.get(category, 0)
                        percentage = backlog_percentage.get(category, 0)
                        st.metric(f"{category} Patients", count)
                        progress_color = category_colors.get(category, "#CCCCCC")
                        st.markdown(f"""
                            <div style="background-color: {progress_color}; width: {percentage}%; height: 20px; opacity: 0.6; border-radius: 10px;"></div>
                        """, unsafe_allow_html=True)
                        st.write(f"{percentage:.0f}% of total")

            st.markdown("---")
            st.subheader("📉📈 Forecasting New Referrals with Prophet")
            
            # Cached function for Prophet forecasting
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
                        except Exception as e:
                            st.warning(f"Could not train Prophet for {category} (Error: {e}). Falling back to historical average.")
                            mean_val = weekly_referrals_cached[weekly_referrals_cached['Category'] == category]['Count'].mean()
                            dummy_forecast_data = {'ds': pd.to_datetime(pd.date_range(start=df_prophet_cached['Date'].max(), periods=num_weeks_for_forecast, freq='W')),
                                                   'yhat': [max(0, round(mean_val))] * num_weeks_for_forecast}
                            forecasted_referrals[category] = pd.DataFrame(dummy_forecast_data)
                    else:
                        st.warning(f"Insufficient data for Prophet forecasting for {category}. Falling back to historical average.")
                        mean_val = weekly_referrals_cached[weekly_referrals_cached['Category'] == category]['Count'].mean() if not weekly_referrals_cached[weekly_referrals_cached['Category'] == category].empty else 0
                        dummy_forecast_data = {'ds': pd.to_datetime(pd.date_range(start=df_prophet_cached['Date'].max(), periods=num_weeks_for_forecast, freq='W')),
                                               'yhat': [max(0, round(mean_val))] * num_weeks_for_forecast}
                        forecasted_referrals[category] = pd.DataFrame(dummy_forecast_data)
                return forecasted_referrals

            with st.spinner("Training Prophet models and forecasting new referrals..."):
                forecasted_new_referrals_per_week = get_prophet_forecast(st.session_state.df, num_weeks, available_categories)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⚙️ Session Configuration")
                selected_session = st.radio(
                    "🕒 Session length (P3/P4):",
                    ["60 min", "50 min", "44 min"],
                    key="selected_session_opt"
                )
            with col2:
                st.subheader("📅 Scheduling Strategy")
                strategy_options = [
                    "Priority Split",
                    "Urgency-Weighted Scheduling"
                ]
                if "P3" in available_categories or "P4" in available_categories:
                    strategy_options.insert(0, "1 in 4 weeks for P3/P4")

                selected_strategy = st.selectbox(
                    "📌 Select strategy:",
                    strategy_options,
                    index=strategy_options.index("Urgency-Weighted Scheduling") if "Urgency-Weighted Scheduling" in strategy_options else 0,
                    key="selected_strategy_opt"
                )

            # --- Calculate average WPS components for simulation ---
            if 'Waiting_Days' in df.columns:
                avg_waiting_days_per_category = df.groupby('Category')['Waiting_Days'].mean().to_dict()
            else:
                avg_waiting_days_per_category = {cat: 0 for cat in available_categories}


            # Normalise factors to sum to 1 for simulation (consistent with WPS calculation)
            total_wps_factors_for_sim = st.session_state.priority_weight_factor + st.session_state.wait_time_weight_factor
            if total_wps_factors_for_sim == 0:
                priority_weight_factor_norm_sim = 0
                wait_time_weight_factor_norm_sim = 0
            else:
                priority_weight_factor_norm_sim = st.session_state.priority_weight_factor / total_wps_factors_for_sim
                wait_time_weight_factor_norm_sim = st.session_state.wait_time_weight_factor / total_wps_factors_for_sim
            
            risk_weight_factor_norm_sim = 0


            try:
                weeks, backlog_projection, patients_seen_per_week = simulate_backlog_reduction(
                    selected_session,
                    selected_strategy,
                    num_therapists,
                    sessions_per_therapist_per_week,
                    forecasted_new_referrals_per_week,
                    num_weeks,
                    backlog_initial,
                    avg_sessions_per_category,
                    avg_weeks_between_sessions,
                    available_categories,
                    st.session_state.custom_priority_weights,
                    avg_waiting_days_per_category,
                    priority_weight_factor_norm_sim,
                    wait_time_weight_factor_norm_sim
                )
                
                st.subheader("📉 Simulation Results")
                st.info("""
                This simulation projects backlog changes based on your inputs and Prophet's forecasted new referrals.
                A decreasing backlog indicates effective management.
                The '1 in 4 weeks for P3/P4' strategy may show temporary growth for P1/P2
                as their sessions are paused to prioritise P3/P4.
                To observe consistent backlog growth with other strategies, consider reducing the number of therapists
                or sessions per therapist per week, or increasing average sessions/decreasing weeks between sessions.
                """)

                fig = go.Figure()
                for category in available_categories:
                    projection = backlog_projection[category]
                    fig.add_trace(go.Scatter(
                        x=weeks,
                        y=projection,
                        mode='lines+markers', # Keep lines and markers
                        name=f'{category} Backlog',
                        line=dict(color=category_colors.get(category, "#CCCCCC")), # Use category colors
                        # Removed fill='tozeroy'
                        hovertemplate="Week: %{x}<br>" + category + ": %{y} patients"
                    ))
                fig.update_layout(
                    title=f"📉 Backlog Reduction Over {num_weeks} Weeks",
                    xaxis_title="Weeks",
                    xaxis_range=[0, num_weeks],
                    yaxis_title="Number of Patients",
                    yaxis_range=[0, max(max(p) for p in backlog_projection.values()) * 1.1 if backlog_projection else 10]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🎯 Efficiency Metrics")
                col1, col2 = st.columns(2)
                
                final_total_backlog = sum(backlog_projection[cat][-1] for cat in available_categories)
                initial_total_backlog = sum(backlog_initial.values())
                net_backlog_change = final_total_backlog - initial_total_backlog

                with col1:
                    st.metric(
                        "Net Change in Total Backlog",
                        f"{net_backlog_change:.0f} patients",
                        delta=f"{net_backlog_change:.0f} patients",
                        delta_color="inverse"
                    )

                    if "P3" in available_categories or "P4" in available_categories:
                        total_extra_sessions, extra_sessions_per_day, extra_sessions_per_week = calculate_extra_sessions(
                            selected_session,
                            num_therapists,
                            num_weeks
                        )
                        
                        st.metric(
                            "Extra Sessions from Shorter Session Length",
                            f"{extra_sessions_per_day} sessions per day per therapist"
                        )
                        st.metric(
                            "Extra Weekly Capacity (All Therapists)",
                            f"{extra_sessions_per_week} sessions per week"
                        )
                    else:
                        st.info("P3/P4 categories are not present in the data, so specific P3/P4 efficiency metrics are not displayed.")
                
                with col2:
                    if "P3" in available_categories or "P4" in available_categories:
                        p3p4_weeks_multiplier = num_weeks // 4 if selected_strategy == "1 in 4 weeks for P3/P4" else num_weeks
                        total_p3p4_sessions_capacity = 0
                        if selected_strategy == "1 in 4 weeks for P3/P4":
                            total_p3p4_sessions_capacity = (extra_sessions_per_week * p3p4_weeks_multiplier)
                        else:
                            total_p3p4_sessions_capacity = sum(patients_seen_per_week[cat].sum() for cat in ["P3", "P4"] if cat in available_categories)

                        st.metric(
                            "Total P3/P4 Sessions Capacity (Simulated)",
                            f"{total_p3p4_sessions_capacity:.0f} sessions"
                        )
                    
                    total_patients_seen = sum(patients_seen_per_week[cat].sum() for cat in available_categories)
                    st.metric(
                        "Total Patients Seen (Simulated)",
                        f"{total_patients_seen:.0f} patients"
                    )

            except ValueError as e:
                st.error(f"Configuration Error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred during simulation: {e}")

            st.markdown("---")
            show_referral_charts(df, available_categories, category_colors)

            st.markdown("---")
            st.markdown("This implementation has been tested using test data. Adjustments may be required to ensure optimal performance with real-world waiting list data.")
            st.markdown("Created by Dave Maher")


        with tab2:
            st.title("⚖️ Dynamic Weighted Scheduling for Waiting Lists")

            st.markdown("""
            This page allows you to dynamically adjust weighting factors for patient priority and wait time
            to generate a Weighted Priority Score (WPS) for each patient.
            """)

            with st.expander("💡 What the software does:", expanded=False):
                st.markdown("""
                **💻 What the software does:**

                **📥 You give it a spreadsheet with a list of patients:**
                * When they joined the waiting list (date).
                * How urgent they are (P1 = super urgent, P4 = least urgent).

                **🧠 It calculates a score for each patient, called the Weighted Priority Score (WPS):**
                * It’s like giving each person a “fairness” score for how soon they should get help.
                * The score depends on 2 adjustable factors:
                    * **Priority Score:** Based on the patient's category (P1, P2, etc.) and the "Patient Category Weights" set on the 'Waiting List Optimisation' tab.
                    * **Waiting Days:** The actual number of days the patient has been waiting, divided by a scaling factor of `5` to balance its impact.

                **The formula for WPS is:**
                `WPS = (Priority_Score * Normalised_Priority_Weight_Factor) +`
                `      (Waiting_Days / 5 * Normalised_Wait_Time_Weight_Factor)`

                Where:
                * `Normalised_..._Weight_Factor` are the normalised values of the "Priority Weight Factor" and "Wait Time Weight Factor" sliders below, ensuring they sum to 1.
                * `Priority_Score` is directly from the "Patient Category Weights" (e.g., P1=100, P2=75, etc.).

                **Higher WPS = more urgent + waited longer**

                **🕹️ On the screen, you can move sliders to adjust how much each of those things matters.**
                * Do you want urgency (Priority Score) to matter more? Adjust the "Priority Weight Factor" slider.
                * Should waiting time count for more? Adjust the "Wait Time Weight Factor" slider.
                * Or maybe you want to balance both equally.

                **📊 The software recalculates everyone’s score instantly and shows a list of patients, sorted from most to least urgent based on the new settings.**
                **📥 You can download the updated list to use it in the hospital.**
                """)


            # Get unique priorities in the dataset
            # Ensure df is not None before accessing its columns
            if st.session_state.df is not None:
                unique_priorities = df['Category'].unique()
            else:
                unique_priorities = [] # Or handle as appropriate if no data is loaded

            # Consolidated WPS Component Weights section with expander
            st.sidebar.subheader("Adjust WPS Component Weights")
            with st.sidebar.expander("More Info on WPS Component Weights"):
                st.markdown("""
                These factors determine the overall influence of Priority and Wait Time
                in the Weighted Priority Score (WPS) calculation. Adjust them to prioritise
                different aspects when ranking individual patients.
                """)

            # Main weight factors, using session state
            # Ensure session state variables are initialized before being used as default values
            st.session_state.priority_weight_factor = st.sidebar.slider("Priority Weight Factor", 0.0, 1.0, st.session_state.priority_weight_factor, key="wps_priority_factor_tab2")
            st.session_state.wait_time_weight_factor = st.sidebar.slider("Wait Time Weight Factor", 0.0, 1.0, st.session_state.wait_time_weight_factor, key="wps_wait_time_factor_tab2")

            # Normalise factors to sum to 1 (only priority and wait time)
            total_wps_factors = st.session_state.priority_weight_factor + st.session_state.wait_time_weight_factor
            if total_wps_factors == 0: # Avoid division by zero if all factors are 0
                st.sidebar.warning("All weighting factors are zero. Please adjust them to calculate WPS.")
                priority_weight_factor_norm = 0
                wait_time_weight_factor_norm = 0
            else:
                priority_weight_factor_norm = st.session_state.priority_weight_factor / total_wps_factors
                wait_time_weight_factor_norm = st.session_state.wait_time_weight_factor / total_wps_factors

            st.sidebar.markdown(f"""
            <small>Normalised Factors:</small><br>
            <small>Priority: {priority_weight_factor_norm:.2f}</small><br>
            <small>Wait Time: {wait_time_weight_factor_norm:.2f}</small><br>
            """, unsafe_allow_html=True)

            st.sidebar.markdown("---")
            
            # Create a copy of the DataFrame for this page's calculations (already has WPS components)
            if st.session_state.df is not None:
                df_wps = df.copy() # df here already has the latest WPS components from the main app logic

                # Recalculate WPS with updated Normalised factors.
                df_wps['WPS'] = (
                    df_wps['Priority_Score'] * priority_weight_factor_norm
                    + (df_wps['Waiting_Days'] / 5) * wait_time_weight_factor_norm
                )

                # Sort by WPS descending, then by Date ascending (oldest to newest)
                df_sorted = df_wps.sort_values(by=['WPS', 'Date'], ascending=[False, True])

                # Add 'Rank' column
                df_sorted = df_sorted.reset_index(drop=True)
                df_sorted.index = df_sorted.index + 1
                df_sorted.insert(0, 'Rank', df_sorted.index)

                st.write("### Patients Sorted by Weighted Priority Score", df_sorted)
                st.info("""
                The 'Priority Score' used in this table is derived from the 'Patient Category Weights'
                sliders found in the sidebar of the 'Waiting List Optimisation' tab.
                """)


                # Option to download the scored dataset
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')
                
                csv = convert_df_to_csv(df_sorted)
                st.download_button("📥 Download Scored Data as CSV", csv, "weighted_waiting_list.csv", "text/csv", key="download_wps_csv")

                # Add a simple visualisation for WPS distribution
                st.subheader("Weighted Priority Score Distribution")
                fig_hist_wps = px.histogram(df_sorted, x='WPS', color='Category',
                                            title='Distribution of Weighted Priority Scores by Category',
                                            color_discrete_map=category_colors,
                                            opacity=0.7)
                st.plotly_chart(fig_hist_wps, use_container_width=True)
            else:
                st.info("Please upload an Excel file in the 'Waiting List Optimisation' tab to view and adjust weights.")
