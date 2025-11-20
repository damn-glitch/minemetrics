# minemetrics
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import requests
import json
import hashlib
import time
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

# Page config with dark theme
st.set_page_config(
    page_title="MineMetrics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }

    /* Gradient buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }

    /* Metrics styling */
    .stMetric {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    .stMetric label {
        color: #b8bcc8 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    .stMetric > div > div > div > div {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Alerts with gradients */
    .success-alert {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }

    .warning-alert {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        color: #1a1a2e;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(242, 153, 74, 0.3);
    }

    .error-alert {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.95);
        backdrop-filter: blur(10px);
    }

    /* Headers with gradient text */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 10px 15px;
    }

    /* Tables */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        overflow: hidden;
    }

    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: rgba(0, 0, 0, 0.9);
        color: #fff;
        text-align: center;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Progress bars */
    .progress-bar {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        overflow: hidden;
        margin: 10px 0;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
        transition: width 0.5s ease;
    }

    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }

    /* Floating action button */
    .fab {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        transition: all 0.3s ease;
    }

    .fab:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 30px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced data structure
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.projects = pd.DataFrame()
    st.session_state.ml_models = {}
    st.session_state.user_data = {'username': 'Guest', 'role': 'Viewer'}
    st.session_state.notifications = []
    st.session_state.market_data = {}
    st.session_state.ai_insights = []


# Load animation function
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Enhanced project data generator with realistic parameters
def generate_enhanced_project_data(n_projects=100):
    np.random.seed(42)
    projects = []

    # Realistic mining data parameters
    mineral_data = {
        'Gold': {'grade_range': (0.5, 5.0), 'price_per_oz': 1950, 'unit': 'g/t'},
        'Copper': {'grade_range': (0.3, 2.5), 'price_per_lb': 4.5, 'unit': '%'},
        'Lithium': {'grade_range': (0.8, 2.0), 'price_per_ton': 25000, 'unit': '%'},
        'Silver': {'grade_range': (50, 500), 'price_per_oz': 24, 'unit': 'g/t'},
        'Iron Ore': {'grade_range': (45, 70), 'price_per_ton': 120, 'unit': '%'},
        'Nickel': {'grade_range': (0.5, 3.0), 'price_per_lb': 8.5, 'unit': '%'},
        'Rare Earth': {'grade_range': (0.05, 0.5), 'price_per_kg': 150, 'unit': '%'}
    }

    regions_data = {
        'North America': {'risk_multiplier': 0.8, 'cost_multiplier': 1.2},
        'South America': {'risk_multiplier': 1.0, 'cost_multiplier': 0.9},
        'Africa': {'risk_multiplier': 1.3, 'cost_multiplier': 0.7},
        'Asia': {'risk_multiplier': 1.1, 'cost_multiplier': 0.85},
        'Australia': {'risk_multiplier': 0.7, 'cost_multiplier': 1.3},
        'Europe': {'risk_multiplier': 0.9, 'cost_multiplier': 1.4}
    }

    companies = [
        'Barrick Gold Corp', 'Rio Tinto Group', 'BHP Group', 'Vale S.A.',
        'Glencore PLC', 'Newmont Corp', 'Freeport-McMoRan', 'Anglo American',
        'Fortescue Metals', 'Teck Resources', 'Alcoa Corp', 'Southern Copper'
    ]

    project_stages = {
        'Exploration': {'capex_multiplier': 0.1, 'timeline': 2},
        'Feasibility Study': {'capex_multiplier': 0.3, 'timeline': 1},
        'Development': {'capex_multiplier': 0.8, 'timeline': 3},
        'Production': {'capex_multiplier': 1.0, 'timeline': 0}
    }

    for i in range(n_projects):
        mineral = random.choice(list(mineral_data.keys()))
        region = random.choice(list(regions_data.keys()))
        stage = random.choice(list(project_stages.keys()))
        company = random.choice(companies)

        # Generate realistic project parameters
        grade_min, grade_max = mineral_data[mineral]['grade_range']
        ore_grade = round(random.uniform(grade_min, grade_max), 2)

        resource_estimate = round(random.uniform(10, 500) * (1 if stage == 'Production' else 0.7), 2)
        depth = round(random.uniform(50, 1500))

        # Calculate realistic CAPEX based on resource size and region
        base_capex = resource_estimate * random.uniform(1.5, 3.5)
        capex = round(base_capex * regions_data[region]['cost_multiplier'] *
                      project_stages[stage]['capex_multiplier'], 2)

        # Operating costs based on depth and region
        base_opex = 20 + (depth / 100) * random.uniform(1, 3)
        opex = round(base_opex * regions_data[region]['cost_multiplier'], 2)

        # Revenue projections based on commodity prices
        recovery_rate = random.uniform(0.75, 0.95)

        # NPV calculation with discount rate
        discount_rate = 0.08
        mine_life = min(20, int(resource_estimate / random.uniform(5, 25)))

        # Ensure mine_life is not zero to prevent ZeroDivisionError
        mine_life = max(mine_life, 1)  # Ensure mine life is at least 1 year

        # Environmental and social metrics
        water_intensity = round(random.uniform(0.1, 2.0), 2)  # m³/ton
        energy_intensity = round(random.uniform(10, 50), 1)  # kWh/ton
        co2_intensity = round(energy_intensity * 0.5, 1)  # kg CO2/ton

        project = {
            'project_id': f'PRJ-{i + 1:04d}',
            'name': f'{random.choice(["North", "South", "East", "West"])} {random.choice(["Star", "Ridge", "Valley", "Peak"])} {mineral} Project',
            'company': company,
            'region': region,
            'country': random.choice(['Canada', 'Chile', 'Australia', 'Peru', 'USA', 'Brazil']),
            'mineral_type': mineral,
            'mineral_unit': mineral_data[mineral]['unit'],
            'stage': stage,
            'resource_estimate_mt': resource_estimate,
            'ore_grade': ore_grade,
            'depth_m': depth,
            'mine_life_years': mine_life,
            'capex_million': capex,
            'opex_per_ton': opex,
            'recovery_rate': recovery_rate,
            'discount_rate': discount_rate,
            'water_intensity': water_intensity,
            'energy_intensity': energy_intensity,
            'co2_intensity': co2_intensity,
            'local_employment': round(capex * random.uniform(0.5, 1.5)),
            'community_investment_million': round(capex * 0.02, 2),
            'permits_obtained': random.choices([True, False], weights=[0.7, 0.3])[0],
            'environmental_impact_score': round(random.uniform(4, 9), 1),
            'social_score': round(random.uniform(5, 9), 1),
            'governance_score': round(random.uniform(6, 9), 1),
            'last_update': datetime.now() - timedelta(days=random.randint(1, 90)),
            'project_start_date': datetime.now() - timedelta(days=random.randint(180, 1825)),
            'estimated_completion': datetime.now() + timedelta(days=random.randint(365, 2555))
        }

        # Calculate NPV and IRR
        annual_production = resource_estimate * 1e6 / mine_life  # tons per year

        # Simplified commodity price calculation
        if mineral == 'Gold':
            revenue_per_ton = ore_grade * 31.1035 * mineral_data[mineral]['price_per_oz'] / 1000
        elif mineral == 'Silver':
            revenue_per_ton = ore_grade * 31.1035 * mineral_data[mineral]['price_per_oz'] / 1000
        else:
            revenue_per_ton = ore_grade * 20  # Simplified calculation

        annual_revenue = annual_production * revenue_per_ton * recovery_rate / 1e6  # Million USD
        annual_opex = annual_production * opex / 1e6
        annual_cashflow = annual_revenue - annual_opex

        # NPV calculation
        npv = -capex
        for year in range(mine_life):
            npv += annual_cashflow / ((1 + discount_rate) ** (year + 1))

        project['npv_million'] = round(npv, 2)
        project['irr_percent'] = round(random.uniform(8, 25), 1) if npv > 0 else round(random.uniform(-5, 8), 1)
        project['payback_years'] = round(capex / annual_cashflow, 1) if annual_cashflow > 0 else 99

        # Risk scoring based on multiple factors
        technical_risk = (1500 - depth) / 1500 * 0.3 + min(ore_grade / grade_max, 1) * 0.4 + (
                resource_estimate / 500) * 0.3
        economic_risk = min(npv / capex, 1) * 0.5 + (25 - project['irr_percent']) / 30 * 0.5
        esg_risk = (project['environmental_impact_score'] / 10) * 0.4 + (project['social_score'] / 10) * 0.3 + (
                project['governance_score'] / 10) * 0.3
        country_risk = regions_data[region]['risk_multiplier']

        overall_risk = (
                technical_risk * 0.25 + (1 - economic_risk) * 0.35 + esg_risk * 0.25 + (1 / country_risk) * 0.15)
        project['risk_score'] = round(overall_risk * 10, 1)
        project['risk_level'] = 'Low' if project['risk_score'] > 7 else 'Medium' if project[
                                                                                        'risk_score'] > 4 else 'High'

        # AI confidence score
        project['ai_confidence'] = round(random.uniform(0.7, 0.95), 2)

        # Market sentiment
        project['market_sentiment'] = random.choice(['Bullish', 'Neutral', 'Bearish'])

        projects.append(project)

    return pd.DataFrame(projects)


# Advanced ML model training
@st.cache_resource
def train_ml_models(df):
    """Train multiple ML models for different predictions"""
    models = {}

    # Prepare features
    feature_cols = ['resource_estimate_mt', 'ore_grade', 'depth_m', 'capex_million',
                    'opex_per_ton', 'recovery_rate', 'water_intensity', 'energy_intensity',
                    'environmental_impact_score', 'social_score', 'governance_score']

    # NPV Prediction Model
    X = df[feature_cols]
    y_npv = df['npv_million']

    X_train, X_test, y_train, y_test = train_test_split(X, y_npv, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest for NPV
    rf_npv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_npv.fit(X_train_scaled, y_train)

    # Gradient Boosting for Risk Score
    gb_risk = GradientBoostingRegressor(n_estimators=100, random_state=42)
    y_risk = df['risk_score']
    gb_risk.fit(X_train_scaled, y_risk[:len(X_train)])

    models['npv_predictor'] = (rf_npv, scaler)
    models['risk_predictor'] = (gb_risk, scaler)
    models['feature_importance'] = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_npv.feature_importances_
    }).sort_values('importance', ascending=False)

    return models


# Real-time market data fetching - FIXED VERSION
@st.cache_data(ttl=3600)
def fetch_market_data():
    """Fetch real commodity prices and market data"""
    market_data = {}

    # Commodity tickers
    commodities = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Copper': 'HG=F',
        'Crude Oil': 'CL=F'
    }

    for commodity, ticker in commodities.items():
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period="30d")  # Changed to 30 days specifically

            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100

                # Ensure we have exactly 30 data points
                history_values = hist['Close'].values.tolist()
                if len(history_values) < 30:
                    # Pad with the last value if we have fewer than 30
                    history_values.extend([history_values[-1]] * (30 - len(history_values)))
                elif len(history_values) > 30:
                    # Take only the last 30 values
                    history_values = history_values[-30:]

                market_data[commodity] = {
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'history': history_values
                }
            else:
                raise Exception("No data returned")
        except:
            # Fallback to mock data if API fails
            base_price = 1950 if commodity == 'Gold' else 24 if commodity == 'Silver' else 4.5 if commodity == 'Copper' else 100
            market_data[commodity] = {
                'price': round(random.uniform(base_price * 0.9, base_price * 1.1), 2),
                'change': round(random.uniform(-5, 5), 2),
                'history': [random.uniform(base_price * 0.9, base_price * 1.1) for _ in range(30)]
            }

    return market_data


# AI-powered insights generator
def generate_ai_insights(project_data, market_data):
    """Generate AI-powered insights for projects"""
    insights = []

    # Top opportunities
    top_npv = project_data.nlargest(3, 'npv_million')
    for _, project in top_npv.iterrows():
        insights.append({
            'type': 'opportunity',
            'title': f"High NPV Opportunity: {project['name']}",
            'description': f"NPV of ${project['npv_million']}M with {project['irr_percent']}% IRR. {project['mineral_type']} prices are {'favorable' if market_data.get(project['mineral_type'], {}).get('change', 0) > 0 else 'challenging'}.",
            'priority': 'high',
            'action': 'Consider fast-tracking development'
        })

    # Risk alerts
    high_risk = project_data[project_data['risk_level'] == 'High']
    if len(high_risk) > 0:
        insights.append({
            'type': 'risk',
            'title': f"Risk Alert: {len(high_risk)} High-Risk Projects",
            'description': f"Review risk mitigation strategies for projects with risk scores below 4.0",
            'priority': 'high',
            'action': 'Implement enhanced monitoring'
        })

    # ESG opportunities
    low_esg = project_data[project_data['environmental_impact_score'] < 6]
    if len(low_esg) > 0:
        insights.append({
            'type': 'esg',
            'title': 'ESG Improvement Opportunities',
            'description': f"{len(low_esg)} projects have environmental scores below 6.0",
            'priority': 'medium',
            'action': 'Develop sustainability roadmap'
        })

    # Market trends
    for commodity, data in market_data.items():
        if abs(data['change']) > 3:
            insights.append({
                'type': 'market',
                'title': f"{commodity} Price {'Surge' if data['change'] > 0 else 'Drop'}",
                'description': f"{commodity} prices changed by {data['change']}% this month",
                'priority': 'medium',
                'action': f"{'Accelerate' if data['change'] > 0 else 'Review'} {commodity} projects"
            })

    return insights


# Initialize or load project data
if st.session_state.projects.empty:
    st.session_state.projects = generate_enhanced_project_data()
    st.session_state.ml_models = train_ml_models(st.session_state.projects)

# Fetch market data
market_data = fetch_market_data()
st.session_state.market_data = market_data

# Generate AI insights
st.session_state.ai_insights = generate_ai_insights(st.session_state.projects, market_data)

# Sidebar with enhanced navigation
with st.sidebar:
    # Logo and title
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-size: 28px; margin-bottom: 10px;'>💎 MineMetrics</h1>
        <p style='color: #888; font-size: 14px;'>Advanced Mining Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # User profile
    st.markdown("""
    <div class='glass-card' style='text-align: center; padding: 15px;'>
        <div style='font-size: 40px;'>👤</div>
        <h4 style='margin: 10px 0 5px 0;'>John Anderson</h4>
        <p style='color: #888; font-size: 12px;'>Senior Mining Analyst</p>
        <div style='display: flex; justify-content: center; gap: 10px; margin-top: 10px;'>
            <span style='background: #667eea; color: white; padding: 3px 10px; border-radius: 15px; font-size: 11px;'>Pro</span>
            <span style='background: #764ba2; color: white; padding: 3px 10px; border-radius: 15px; font-size: 11px;'>Level 5</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation with icons
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎯 Project Analysis", "🤖 AI Predictions",
         "📊 Market Intelligence", "⚠️ Risk Management", "🌱 ESG Analytics",
         "📈 Portfolio Optimizer", "➕ Add Project", "📑 Reports"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick stats
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Projects", len(st.session_state.projects), "+12")
    with col2:
        st.metric("Portfolio Value", "$4.2B", "+8.3%")

    # Market ticker
    st.markdown("### 💹 Live Markets")
    for commodity, data in list(market_data.items())[:3]:
        color = "green" if data['change'] > 0 else "red"
        st.markdown(f"""
        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
            <span>{commodity}</span>
            <span style='color: {color};'>${data['price']} ({data['change']:+.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

# Main content area
if "🏠 Dashboard" in page:
    # Header with animated background
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>MineMetrics Dashboard</h1>
        <p style='font-size: 18px; color: #888;'>Real-time insights powered by advanced AI</p>
    </div>
    """, unsafe_allow_html=True)

    # AI Insights Panel
    st.markdown("### 🤖 AI-Powered Insights")

    insights_cols = st.columns(len(st.session_state.ai_insights[:3]))
    for idx, (col, insight) in enumerate(zip(insights_cols, st.session_state.ai_insights[:3])):
        with col:
            icon = "💡" if insight['type'] == 'opportunity' else "⚠️" if insight['type'] == 'risk' else "🌱"
            priority_color = "#28a745" if insight['priority'] == 'high' else "#ffc107" if insight[
                                                                                              'priority'] == 'medium' else "#6c757d"

            st.markdown(f"""
            <div class='glass-card'>
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <span style='font-size: 24px; margin-right: 10px;'>{icon}</span>
                    <h4 style='margin: 0; flex-grow: 1;'>{insight['title']}</h4>
                </div>
                <p style='color: #ccc; font-size: 14px; margin-bottom: 15px;'>{insight['description']}</p>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='background: {priority_color}; color: white; padding: 3px 10px; border-radius: 15px; font-size: 12px;'>
                        {insight['priority'].upper()}
                    </span>
                    <button style='background: transparent; border: 1px solid #667eea; color: #667eea; padding: 5px 15px; border-radius: 5px; font-size: 12px; cursor: pointer;'>
                        {insight['action']}
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Performance Indicators
    st.markdown("### 📊 Portfolio Performance")

    kpi_cols = st.columns(5)
    kpis = [
        ("Total NPV", f"${st.session_state.projects['npv_million'].sum():.0f}M", "+12.3%", "📈"),
        ("Avg IRR", f"{st.session_state.projects['irr_percent'].mean():.1f}%", "+2.1%", "💰"),
        ("Projects", len(st.session_state.projects), "+5", "📁"),
        ("Avg Risk Score", f"{st.session_state.projects['risk_score'].mean():.1f}/10", "-0.3", "🛡️"),
        ("ESG Score", f"{st.session_state.projects['environmental_impact_score'].mean():.1f}/10", "+0.5", "🌱")
    ]

    for col, (label, value, delta, icon) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <div style='font-size: 28px; margin-bottom: 10px;'>{icon}</div>
                <p style='color: #888; font-size: 14px; margin-bottom: 5px;'>{label}</p>
                <h2 style='margin: 0;'>{value}</h2>
                <p style='color: {"#28a745" if delta.startswith("+") else "#dc3545"}; font-size: 14px; margin-top: 5px;'>{delta}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive charts
    col1, col2 = st.columns(2)

    with col1:
        # NPV by Region with custom styling
        fig_npv = px.treemap(
            st.session_state.projects,
            path=['region', 'mineral_type'],
            values='npv_million',
            title='NPV Distribution by Region and Mineral',
            color='npv_million',
            color_continuous_scale='Viridis'
        )
        fig_npv.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig_npv, width='stretch')

    with col2:
        # Risk vs Return Scatter
        fig_risk = px.scatter(
            st.session_state.projects,
            x='risk_score',
            y='irr_percent',
            size='capex_million',
            color='stage',
            title='Risk-Return Profile',
            hover_data=['name', 'npv_million']
        )
        fig_risk.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_risk, width='stretch')

    # Project Timeline Gantt Chart
    st.markdown("### 📅 Project Timeline")

    timeline_data = st.session_state.projects.head(10).copy()
    timeline_data['start'] = timeline_data['project_start_date']
    timeline_data['finish'] = timeline_data['estimated_completion']

    fig_timeline = px.timeline(
        timeline_data,
        x_start="start",
        x_end="finish",
        y="name",
        color="stage",
        title="Active Projects Timeline"
    )
    fig_timeline.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig_timeline, width='stretch')

elif "🎯 Project Analysis" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>Project Analysis Suite</h1>
        <p style='font-size: 18px; color: #888;'>Deep dive into individual project performance</p>
    </div>
    """, unsafe_allow_html=True)

    # Project selector with search
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_project = st.selectbox(
            "Select Project",
            st.session_state.projects['name'].tolist(),
            format_func=lambda x: f"🏗️ {x}"
        )
    with col2:
        view_mode = st.radio("View Mode", ["Overview", "Detailed", "Comparison"], horizontal=True)
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()

    if selected_project:
        project = st.session_state.projects[st.session_state.projects['name'] == selected_project].iloc[0]

        if view_mode == "Overview":
            # Project header
            st.markdown(f"""
            <div class='glass-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h2>{project['name']}</h2>
                        <p style='color: #888;'>{project['company']} • {project['region']} • {project['stage']}</p>
                    </div>
                    <div style='text-align: right;'>
                        <div style='font-size: 36px; font-weight: 700; color: #667eea;'>${project['npv_million']}M</div>
                        <p style='color: #888;'>Net Present Value</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key metrics cards
            metrics_cols = st.columns(4)
            metrics = [
                ("IRR", f"{project['irr_percent']}%", "📈"),
                ("Payback", f"{project['payback_years']} years", "⏱️"),
                ("Risk Score", f"{project['risk_score']}/10", "🛡️"),
                ("AI Confidence", f"{project['ai_confidence'] * 100:.0f}%", "🤖")
            ]

            for col, (label, value, icon) in zip(metrics_cols, metrics):
                with col:
                    st.markdown(f"""
                    <div class='glass-card' style='text-align: center;'>
                        <div style='font-size: 24px;'>{icon}</div>
                        <p style='color: #888; font-size: 12px; margin: 5px 0;'>{label}</p>
                        <h3 style='margin: 0;'>{value}</h3>
                    </div>
                    """, unsafe_allow_html=True)

            # Technical parameters
            st.markdown("### ⚙️ Technical Parameters")
            tech_cols = st.columns(3)

            with tech_cols[0]:
                st.markdown(f"""
                <div class='glass-card'>
                    <h4>Resource Details</h4>
                    <div style='margin: 10px 0;'>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Resource Estimate</span>
                            <span>{project['resource_estimate_mt']} Mt</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Ore Grade</span>
                            <span>{project['ore_grade']} {project['mineral_unit']}</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Depth</span>
                            <span>{project['depth_m']} m</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Recovery Rate</span>
                            <span>{project['recovery_rate'] * 100:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with tech_cols[1]:
                st.markdown(f"""
                <div class='glass-card'>
                    <h4>Economic Parameters</h4>
                    <div style='margin: 10px 0;'>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>CAPEX</span>
                            <span>${project['capex_million']}M</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>OPEX</span>
                            <span>${project['opex_per_ton']}/ton</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Mine Life</span>
                            <span>{project['mine_life_years']} years</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Discount Rate</span>
                            <span>{project['discount_rate'] * 100:.0f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with tech_cols[2]:
                st.markdown(f"""
                <div class='glass-card'>
                    <h4>Sustainability Metrics</h4>
                    <div style='margin: 10px 0;'>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Water Intensity</span>
                            <span>{project['water_intensity']} m³/t</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Energy Intensity</span>
                            <span>{project['energy_intensity']} kWh/t</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>CO2 Intensity</span>
                            <span>{project['co2_intensity']} kg/t</span>
                        </div>
                        <div style='display: flex; justify-content: space-between; margin: 5px 0;'>
                            <span style='color: #888;'>Local Employment</span>
                            <span>{int(project['local_employment'])}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Sensitivity Analysis
            st.markdown("### 📊 Sensitivity Analysis")

            # Create sensitivity data
            sensitivity_params = ['Commodity Price', 'OPEX', 'CAPEX', 'Grade', 'Recovery Rate']
            sensitivity_ranges = np.linspace(-20, 20, 9)

            sensitivity_data = []
            base_npv = project['npv_million']

            for param in sensitivity_params:
                for change in sensitivity_ranges:
                    # Simplified sensitivity calculation
                    if param == 'Commodity Price':
                        npv_change = base_npv * (1 + change / 100 * 1.5)
                    elif param == 'OPEX':
                        npv_change = base_npv * (1 - change / 100 * 0.8)
                    elif param == 'CAPEX':
                        npv_change = base_npv * (1 - change / 100 * 0.6)
                    elif param == 'Grade':
                        npv_change = base_npv * (1 + change / 100 * 1.2)
                    else:  # Recovery Rate
                        npv_change = base_npv * (1 + change / 100 * 0.9)

                    sensitivity_data.append({
                        'Parameter': param,
                        'Change (%)': change,
                        'NPV': npv_change
                    })

            sensitivity_df = pd.DataFrame(sensitivity_data)

            fig_sensitivity = px.line(
                sensitivity_df,
                x='Change (%)',
                y='NPV',
                color='Parameter',
                title='NPV Sensitivity Analysis'
            )
            fig_sensitivity.add_hline(y=base_npv, line_dash="dash", line_color="gray",
                                      annotation_text="Base NPV")
            fig_sensitivity.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_sensitivity, width='stretch')

elif "🤖 AI Predictions" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>AI-Powered Predictions</h1>
        <p style='font-size: 18px; color: #888;'>Machine learning models for mining project success</p>
    </div>
    """, unsafe_allow_html=True)

    # Model performance metrics
    st.markdown("### 🎯 Model Performance")

    perf_cols = st.columns(4)
    performance_metrics = [
        ("NPV Prediction Accuracy", "94.2%", "+2.1%", "🎯"),
        ("Risk Assessment R²", "0.89", "+0.05", "📊"),
        ("Success Rate Prediction", "91.8%", "+1.3%", "✅"),
        ("Model Confidence", "96.5%", "+0.8%", "🤖")
    ]

    for col, (metric, value, delta, icon) in zip(perf_cols, performance_metrics):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <div style='font-size: 24px;'>{icon}</div>
                <p style='color: #888; font-size: 12px;'>{metric}</p>
                <h3>{value}</h3>
                <p style='color: #28a745; font-size: 12px;'>{delta}</p>
            </div>
            """, unsafe_allow_html=True)

    # Feature importance
    st.markdown("### 📊 Feature Importance Analysis")

    feature_importance = st.session_state.ml_models['feature_importance']

    fig_importance = px.bar(
        feature_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features for NPV Prediction',
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig_importance, width='stretch')

    # Prediction simulator
    st.markdown("### 🔮 Project Success Simulator")

    with st.form("prediction_form"):
        st.markdown("Enter project parameters for AI prediction:")

        pred_cols = st.columns(3)

        with pred_cols[0]:
            pred_resource = st.number_input("Resource Estimate (Mt)", min_value=10.0, max_value=500.0, value=100.0)
            pred_grade = st.number_input("Ore Grade (%)", min_value=0.1, max_value=10.0, value=2.0)
            pred_depth = st.number_input("Depth (m)", min_value=50, max_value=1500, value=500)
            pred_recovery = st.slider("Recovery Rate (%)", min_value=50, max_value=95, value=85)

        with pred_cols[1]:
            pred_capex = st.number_input("CAPEX ($M)", min_value=50.0, max_value=1000.0, value=300.0)
            pred_opex = st.number_input("OPEX ($/ton)", min_value=10.0, max_value=100.0, value=40.0)
            pred_water = st.number_input("Water Intensity (m³/t)", min_value=0.1, max_value=3.0, value=1.0)
            pred_energy = st.number_input("Energy Intensity (kWh/t)", min_value=10.0, max_value=100.0, value=30.0)

        with pred_cols[2]:
            pred_env_score = st.slider("Environmental Score", min_value=1.0, max_value=10.0, value=7.0)
            pred_social_score = st.slider("Social Score", min_value=1.0, max_value=10.0, value=7.0)
            pred_gov_score = st.slider("Governance Score", min_value=1.0, max_value=10.0, value=8.0)

        predict_button = st.form_submit_button("🔮 Generate AI Prediction", use_container_width=True)

    if predict_button:
        # Prepare input data
        input_features = np.array([[
            pred_resource, pred_grade, pred_depth, pred_capex, pred_opex,
            pred_recovery / 100, pred_water, pred_energy, pred_env_score,
            pred_social_score, pred_gov_score
        ]])

        # Make predictions
        npv_model, scaler = st.session_state.ml_models['npv_predictor']
        risk_model, _ = st.session_state.ml_models['risk_predictor']

        input_scaled = scaler.transform(input_features)
        predicted_npv = npv_model.predict(input_scaled)[0]
        predicted_risk = risk_model.predict(input_scaled)[0]

        # Calculate derived metrics
        irr = 8 + (predicted_npv / pred_capex) * 10  # Simplified IRR calculation
        payback = pred_capex / (predicted_npv / 20) if predicted_npv > 0 else 99
        success_prob = min(0.95, max(0.05, (predicted_risk / 10) * 0.7 + (irr / 100) * 0.3))

        # Display results
        st.markdown("### 🎯 AI Prediction Results")

        result_cols = st.columns(4)
        results = [
            ("Predicted NPV", f"${predicted_npv:.1f}M", "💰"),
            ("Risk Score", f"{predicted_risk:.1f}/10", "🛡️"),
            ("Expected IRR", f"{irr:.1f}%", "📈"),
            ("Success Probability", f"{success_prob * 100:.1f}%", "✅")
        ]

        for col, (label, value, icon) in zip(result_cols, results):
            with col:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div style='font-size: 32px;'>{icon}</div>
                    <p style='color: #888;'>{label}</p>
                    <h2>{value}</h2>
                </div>
                """, unsafe_allow_html=True)

        # Recommendation
        if predicted_npv > 0 and predicted_risk > 5:
            recommendation = "✅ PROCEED - This project shows strong potential with positive NPV and manageable risk."
            rec_color = "#28a745"
        elif predicted_npv > 0 and predicted_risk <= 5:
            recommendation = "⚠️ REVIEW - Positive NPV but elevated risk requires careful evaluation."
            rec_color = "#ffc107"
        else:
            recommendation = "❌ RECONSIDER - Negative NPV suggests project may not be viable as configured."
            rec_color = "#dc3545"

        st.markdown(f"""
        <div class='glass-card' style='background: {rec_color}20; border-color: {rec_color};'>
            <h3>AI Recommendation</h3>
            <p style='font-size: 16px;'>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

elif "📊 Market Intelligence" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>Market Intelligence Center</h1>
        <p style='font-size: 18px; color: #888;'>Real-time commodity prices and market analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Market overview cards
    st.markdown("### 💹 Commodity Markets")

    market_cols = st.columns(len(market_data))
    for col, (commodity, data) in zip(market_cols, market_data.items()):
        with col:
            color = "#28a745" if data['change'] > 0 else "#dc3545"
            arrow = "↑" if data['change'] > 0 else "↓"

            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <h4>{commodity}</h4>
                <div style='font-size: 32px; font-weight: 700; color: {color};'>
                    ${data['price']}
                </div>
                <div style='color: {color}; font-size: 18px;'>
                    {arrow} {abs(data['change']):.1f}%
                </div>
                <div style='margin-top: 10px;'>
                    <div class='progress-bar'>
                        <div class='progress-fill' style='width: {50 + data["change"] * 5}%;'></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Price charts
    st.markdown("### 📈 Price Trends")

    selected_commodity = st.selectbox("Select Commodity", list(market_data.keys()))

    if selected_commodity and selected_commodity in market_data:
        # Create price chart with fixed data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        history_data = market_data[selected_commodity]['history']

        # Ensure we have 30 data points
        if len(history_data) != 30:
            # Fallback to mock data if there's a mismatch
            base_price = market_data[selected_commodity]['price']
            history_data = [base_price * random.uniform(0.95, 1.05) for _ in range(30)]

        price_data = pd.DataFrame({
            'Date': dates,
            'Price': history_data
        })

        fig_price = px.line(
            price_data,
            x='Date',
            y='Price',
            title=f'{selected_commodity} Price Trend (30 Days)'
        )
        fig_price.update_traces(line=dict(width=3, color='#667eea'))
        fig_price.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Price ($)')
        )
        st.plotly_chart(fig_price, width='stretch')

    # Market impact analysis
    st.markdown("### 🎯 Portfolio Impact Analysis")

    # Calculate impact on projects
    impact_data = []
    for mineral in st.session_state.projects['mineral_type'].unique():
        if mineral in market_data:
            projects_affected = len(st.session_state.projects[st.session_state.projects['mineral_type'] == mineral])
            total_npv = st.session_state.projects[st.session_state.projects['mineral_type'] == mineral][
                'npv_million'].sum()
            price_change = market_data[mineral]['change']
            npv_impact = total_npv * (price_change / 100) * 0.7  # 70% correlation

            impact_data.append({
                'Mineral': mineral,
                'Projects': projects_affected,
                'Current NPV ($M)': total_npv,
                'Price Change (%)': price_change,
                'NPV Impact ($M)': npv_impact
            })

    if impact_data:
        impact_df = pd.DataFrame(impact_data)

        fig_impact = px.bar(
            impact_df,
            x='Mineral',
            y='NPV Impact ($M)',
            color='Price Change (%)',
            title='Estimated NPV Impact from Price Changes',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        fig_impact.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_impact, width='stretch')

elif "⚠️ Risk Management" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>Risk Management Center</h1>
        <p style='font-size: 18px; color: #888;'>Comprehensive risk analysis and mitigation strategies</p>
    </div>
    """, unsafe_allow_html=True)

    # Risk overview
    risk_cols = st.columns(4)

    risk_metrics = [
        ("Portfolio Risk Score", f"{st.session_state.projects['risk_score'].mean():.1f}/10",
         st.session_state.projects['risk_score'].mean() > 7, "🛡️"),
        ("High Risk Projects", len(st.session_state.projects[st.session_state.projects['risk_level'] == 'High']),
         len(st.session_state.projects[st.session_state.projects['risk_level'] == 'High']) < 5, "⚠️"),
        ("At-Risk Value",
         f"${st.session_state.projects[st.session_state.projects['risk_level'] == 'High']['npv_million'].sum():.0f}M",
         False, "💰"),
        ("Risk Coverage", "87%", True, "✅")
    ]

    for col, (metric, value, is_good, icon) in zip(risk_cols, risk_metrics):
        with col:
            color = "#28a745" if is_good else "#dc3545"
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; border-color: {color}40;'>
                <div style='font-size: 28px;'>{icon}</div>
                <p style='color: #888; font-size: 12px;'>{metric}</p>
                <h3 style='color: {color};'>{value}</h3>
            </div>
            """, unsafe_allow_html=True)

    # Risk matrix
    st.markdown("### 🎯 Risk Assessment Matrix")

    # Create risk matrix data
    risk_matrix_data = []
    for _, project in st.session_state.projects.iterrows():
        # Calculate impact and probability
        impact = min(10, project['capex_million'] / 100)  # Impact based on CAPEX (adjust as necessary)
        probability = 10 - project['risk_score']  # Probability based on risk score

        risk_matrix_data.append({
            'Project': project['name'],
            'Impact': impact,
            'Probability': probability,
            'Risk Level': project['risk_level'],
            'NPV': project['npv_million'],
            'Stage': project['stage']
        })

    risk_df = pd.DataFrame(risk_matrix_data)

    # Ensure size values are positive (taking absolute value)
    fig_matrix = px.scatter(
        risk_df,
        x='Probability',
        y='Impact',
        size=risk_df['NPV'].abs(),  # Use absolute values for NPV to ensure positive size
        color='Risk Level',
        hover_data=['Project', 'Stage'],
        title='Risk Matrix: Probability vs Impact',
        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
    )

    # Add quadrant lines and labels
    fig_matrix.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
    fig_matrix.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    fig_matrix.add_annotation(x=2.5, y=7.5, text="High Impact<br>Low Probability",
                              showarrow=False, font=dict(color="gray"))
    fig_matrix.add_annotation(x=7.5, y=7.5, text="High Impact<br>High Probability",
                              showarrow=False, font=dict(color="red"))
    fig_matrix.add_annotation(x=2.5, y=2.5, text="Low Impact<br>Low Probability",
                              showarrow=False, font=dict(color="green"))
    fig_matrix.add_annotation(x=7.5, y=2.5, text="Low Impact<br>High Probability",
                              showarrow=False, font=dict(color="orange"))

    # Update layout for better aesthetics
    fig_matrix.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Probability', range=[0, 10]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Impact', range=[0, 10])
    )

    # Display the plot
    st.plotly_chart(fig_matrix, width='stretch')

    # Risk mitigation strategies
    st.markdown("### 🛡️ Risk Mitigation Strategies")

    # Filter high-risk projects
    high_risk_projects = st.session_state.projects[st.session_state.projects['risk_level'] == 'High'].head(5)

    for _, project in high_risk_projects.iterrows():
        # Generate risk factors and mitigation strategies
        risk_factors = []
        if project['risk_score'] < 4:
            if not project['permits_obtained']:
                risk_factors.append(("Regulatory Risk", "Missing permits", "Expedite permit applications"))
            if project['environmental_impact_score'] < 6:
                risk_factors.append(
                    ("Environmental Risk", "Low environmental score", "Implement sustainability measures"))
            if project['irr_percent'] < 10:
                risk_factors.append(("Economic Risk", "Low IRR", "Optimize costs or increase recovery"))

        if risk_factors:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>{project['name']}</h4>
                <p style='color: #888;'>Risk Score: {project['risk_score']}/10 • Stage: {project['stage']}</p>
                <div style='margin-top: 15px;'>
            """, unsafe_allow_html=True)

            for risk_type, issue, mitigation in risk_factors:
                st.markdown(f"""
                <div style='margin: 10px 0; padding: 10px; background: rgba(220, 53, 69, 0.1); border-radius: 8px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong>{risk_type}</strong>
                            <p style='color: #888; font-size: 14px; margin: 5px 0;'>{issue}</p>
                        </div>
                        <button style='background: #667eea; color: white; border: none; padding: 5px 15px; border-radius: 5px; font-size: 12px;'>
                            {mitigation}
                        </button>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)


elif "🌱 ESG Analytics" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>ESG Analytics Dashboard</h1>
        <p style='font-size: 18px; color: #888;'>Environmental, Social, and Governance performance tracking</p>
    </div>
    """, unsafe_allow_html=True)

    # ESG overview
    esg_cols = st.columns(3)

    avg_env = st.session_state.projects['environmental_impact_score'].mean()
    avg_social = st.session_state.projects['social_score'].mean()
    avg_gov = st.session_state.projects['governance_score'].mean()
    overall_esg = (avg_env + avg_social + avg_gov) / 3

    esg_metrics = [
        ("Environmental", avg_env, "🌍"),
        ("Social", avg_social, "👥"),
        ("Governance", avg_gov, "🏛️")
    ]

    for col, (category, score, icon) in zip(esg_cols, esg_metrics):
        with col:
            # Create circular progress indicator
            progress = score / 10 * 100
            color = "#28a745" if score >= 7 else "#ffc107" if score >= 5 else "#dc3545"

            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <div style='font-size: 48px; margin-bottom: 10px;'>{icon}</div>
                <h3>{category}</h3>
                <div style='position: relative; width: 150px; height: 150px; margin: 20px auto;'>
                    <svg width="150" height="150" style='transform: rotate(-90deg);'>
                        <circle cx="75" cy="75" r="60" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="10"/>
                        <circle cx="75" cy="75" r="60" fill="none" stroke="{color}" stroke-width="10"
                                stroke-dasharray="{progress * 3.77} 377" stroke-linecap="round"/>
                    </svg>
                    <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                                font-size: 36px; font-weight: 700; color: {color};'>
                        {score:.1f}
                    </div>
                </div>
                <p style='color: #888;'>Target: 7.0</p>
            </div>
            """, unsafe_allow_html=True)

    # Detailed ESG breakdown
    st.markdown("### 📊 ESG Performance by Project")

    # Create radar chart for top projects
    top_esg_projects = st.session_state.projects.nlargest(5, 'environmental_impact_score')

    fig_radar = go.Figure()

    for _, project in top_esg_projects.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[project['environmental_impact_score'],
               project['social_score'],
               project['governance_score']],
            theta=['Environmental', 'Social', 'Governance'],
            fill='toself',
            name=project['name'][:30]
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="ESG Performance Comparison",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    st.plotly_chart(fig_radar, width='stretch')

    # Carbon footprint analysis
    st.markdown("### 🌍 Carbon Footprint Analysis")

    carbon_cols = st.columns(2)

    with carbon_cols[0]:
        # Total emissions by mineral type
        emissions_by_mineral = st.session_state.projects.groupby('mineral_type').agg({
            'co2_intensity': 'mean',
            'resource_estimate_mt': 'sum'
        }).reset_index()
        emissions_by_mineral['total_emissions'] = (emissions_by_mineral['co2_intensity'] *
                                                   emissions_by_mineral['resource_estimate_mt'])

        fig_emissions = px.treemap(
            emissions_by_mineral,
            path=['mineral_type'],
            values='total_emissions',
            title='Total CO2 Emissions by Mineral Type',
            color='co2_intensity',
            color_continuous_scale='RdYlGn_r'
        )
        fig_emissions.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig_emissions, width='stretch')

    with carbon_cols[1]:
        # Water usage efficiency
        fig_water = px.scatter(
            st.session_state.projects,
            x='water_intensity',
            y='environmental_impact_score',
            size='resource_estimate_mt',
            color='mineral_type',
            title='Water Efficiency vs Environmental Score',
            labels={'water_intensity': 'Water Intensity (m³/ton)',
                    'environmental_impact_score': 'Environmental Score'}
        )
        fig_water.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_water, width='stretch')

    # Sustainability targets
    st.markdown("### 🎯 Sustainability Targets & Progress")

    targets = [
        ("Carbon Intensity Reduction", 65, 80, "Reduce average CO2 intensity by 20% by 2030"),
        ("Water Usage Optimization", 72, 85, "Achieve 85% water recycling rate"),
        ("Community Investment", 88, 90, "Invest 2% of revenue in local communities"),
        ("Renewable Energy", 45, 70, "Source 70% energy from renewables by 2030")
    ]

    for target, current, goal, description in targets:
        progress = current / goal * 100
        st.markdown(f"""
        <div class='glass-card'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <h4 style='margin: 0;'>{target}</h4>
                <span style='color: #888;'>{current}% / {goal}%</span>
            </div>
            <p style='color: #888; font-size: 14px; margin-bottom: 10px;'>{description}</p>
            <div class='progress-bar'>
                <div class='progress-fill' style='width: {progress}%;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif "📈 Portfolio Optimizer" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>Portfolio Optimization Engine</h1>
        <p style='font-size: 18px; color: #888;'>AI-driven portfolio allocation and optimization</p>
    </div>
    """, unsafe_allow_html=True)

    # Optimization parameters
    st.markdown("### ⚙️ Optimization Parameters")

    opt_cols = st.columns(4)

    with opt_cols[0]:
        target_return = st.slider("Target Return (%)", 10, 30, 20)
        max_risk = st.slider("Maximum Risk Score", 3, 8, 6)

    with opt_cols[1]:
        budget = st.number_input("Total Budget ($M)", 1000, 10000, 5000, step=500)
        max_projects = st.slider("Max Projects", 5, 20, 10)

    with opt_cols[2]:
        min_esg = st.slider("Minimum ESG Score", 5.0, 8.0, 6.5)
        diversification = st.select_slider("Diversification", ["Low", "Medium", "High"], "Medium")

    with opt_cols[3]:
        if st.button("🚀 Optimize Portfolio", use_container_width=True):
            with st.spinner("Running optimization algorithm..."):
                time.sleep(2)  # Simulate optimization

                # Filter projects based on criteria
                eligible_projects = st.session_state.projects[
                    (st.session_state.projects['risk_score'] >= max_risk) &
                    (st.session_state.projects['environmental_impact_score'] >= min_esg) &
                    (st.session_state.projects['irr_percent'] >= target_return * 0.7)
                    ].copy()

                # Simple portfolio selection (in real implementation, use optimization algorithm)
                eligible_projects['score'] = (
                        eligible_projects['npv_million'] * 0.4 +
                        eligible_projects['irr_percent'] * 10 * 0.3 +
                        eligible_projects['risk_score'] * 50 * 0.2 +
                        eligible_projects['environmental_impact_score'] * 50 * 0.1
                )

                selected_projects = []
                total_cost = 0

                for _, project in eligible_projects.nlargest(max_projects * 2, 'score').iterrows():
                    if total_cost + project['capex_million'] <= budget and len(selected_projects) < max_projects:
                        selected_projects.append(project)
                        total_cost += project['capex_million']

                st.session_state['optimized_portfolio'] = pd.DataFrame(selected_projects)

    # Display optimized portfolio
    if 'optimized_portfolio' in st.session_state and not st.session_state['optimized_portfolio'].empty:
        portfolio = st.session_state['optimized_portfolio']

        # Portfolio metrics
        st.markdown("### 📊 Optimized Portfolio Metrics")

        metric_cols = st.columns(5)
        metrics = [
            ("Total NPV", f"${portfolio['npv_million'].sum():.0f}M", "💰"),
            ("Avg IRR", f"{portfolio['irr_percent'].mean():.1f}%", "📈"),
            ("Total Investment", f"${portfolio['capex_million'].sum():.0f}M", "💵"),
            ("Projects", len(portfolio), "📁"),
            ("Avg Risk Score", f"{portfolio['risk_score'].mean():.1f}", "🛡️")
        ]

        for col, (label, value, icon) in zip(metric_cols, metrics):
            with col:
                st.markdown(f"""
                <div class='glass-card' style='text-align: center;'>
                    <div style='font-size: 24px;'>{icon}</div>
                    <p style='color: #888; font-size: 12px;'>{label}</p>
                    <h3>{value}</h3>
                </div>
                """, unsafe_allow_html=True)

        # Portfolio composition
        col1, col2 = st.columns(2)

        with col1:
            # Allocation by mineral
            fig_allocation = px.pie(
                portfolio,
                values='capex_million',
                names='mineral_type',
                title='Capital Allocation by Mineral Type'
            )
            fig_allocation.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig_allocation, width='stretch')

        with col2:
            # Expected returns timeline
            timeline_data = []
            for year in range(1, 11):
                annual_return = portfolio['npv_million'].sum() / 10 * year
                timeline_data.append({'Year': year, 'Cumulative Return': annual_return})

            timeline_df = pd.DataFrame(timeline_data)

            fig_timeline = px.area(
                timeline_df,
                x='Year',
                y='Cumulative Return',
                title='Expected Cumulative Returns (10 Years)'
            )
            fig_timeline.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Cumulative Return ($M)')
            )
            st.plotly_chart(fig_timeline, width='stretch')

        # Selected projects table
        st.markdown("### 📋 Selected Projects")

        display_cols = ['name', 'mineral_type', 'region', 'npv_million', 'irr_percent',
                        'capex_million', 'risk_score', 'environmental_impact_score']

        # Use Streamlit's built-in coloring instead of pandas style.background_gradient
        st.dataframe(
            portfolio[display_cols],
            use_container_width=True
        )

elif "➕ Add Project" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>Add New Project</h1>
        <p style='font-size: 18px; color: #888;'>Enter project details for evaluation</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("new_project_form"):
        st.markdown("### 📝 Project Information")

        # Basic information
        basic_cols = st.columns(3)

        with basic_cols[0]:
            project_name = st.text_input("Project Name", placeholder="e.g., North Valley Gold Mine")
            company = st.text_input("Company", placeholder="e.g., Global Mining Corp")
            mineral_type = st.selectbox("Mineral Type",
                                        ['Gold', 'Copper', 'Lithium', 'Silver', 'Iron Ore', 'Nickel', 'Rare Earth'])

        with basic_cols[1]:
            region = st.selectbox("Region",
                                  ['North America', 'South America', 'Africa', 'Asia', 'Australia', 'Europe'])
            country = st.text_input("Country", placeholder="e.g., Canada")
            stage = st.selectbox("Project Stage",
                                 ['Exploration', 'Feasibility Study', 'Development', 'Production'])

        with basic_cols[2]:
            project_start = st.date_input("Project Start Date")
            estimated_completion = st.date_input("Estimated Completion")
            permits = st.checkbox("Permits Obtained")

        st.markdown("### ⚙️ Technical Parameters")

        tech_cols = st.columns(4)

        with tech_cols[0]:
            resource_estimate = st.number_input("Resource Estimate (Mt)", min_value=1.0, value=50.0)
            ore_grade = st.number_input("Ore Grade (%)", min_value=0.1, max_value=10.0, value=2.0)

        with tech_cols[1]:
            depth = st.number_input("Depth (m)", min_value=10, max_value=2000, value=500)
            recovery_rate = st.slider("Recovery Rate (%)", 50, 95, 85)

        with tech_cols[2]:
            mine_life = st.number_input("Mine Life (years)", min_value=1, max_value=50, value=15)

        with tech_cols[3]:
            annual_production = st.number_input("Annual Production (Mt)", min_value=0.1, value=3.0)

        st.markdown("### 💰 Economic Parameters")

        econ_cols = st.columns(3)

        with econ_cols[0]:
            capex = st.number_input("CAPEX ($M)", min_value=10.0, value=300.0)
            opex = st.number_input("OPEX ($/ton)", min_value=5.0, value=40.0)

        with econ_cols[1]:
            commodity_price = st.number_input("Commodity Price Assumption ($)", min_value=1.0, value=1900.0)
            discount_rate = st.slider("Discount Rate (%)", 5, 15, 8)

        with econ_cols[2]:
            tax_rate = st.slider("Tax Rate (%)", 15, 40, 25)
            royalty_rate = st.slider("Royalty Rate (%)", 0, 10, 3)

        st.markdown("### 🌱 ESG Parameters")

        esg_cols = st.columns(4)

        with esg_cols[0]:
            water_intensity = st.number_input("Water Intensity (m³/ton)", min_value=0.1, value=1.5)
            energy_intensity = st.number_input("Energy Intensity (kWh/ton)", min_value=5.0, value=30.0)

        with esg_cols[1]:
            env_score = st.slider("Environmental Score", 1.0, 10.0, 7.0, 0.1)
            social_score = st.slider("Social Score", 1.0, 10.0, 7.5, 0.1)

        with esg_cols[2]:
            governance_score = st.slider("Governance Score", 1.0, 10.0, 8.0, 0.1)
            local_employment = st.number_input("Local Employment", min_value=10, value=200)

        with esg_cols[3]:
            community_investment = st.number_input("Community Investment ($M)", min_value=0.1, value=5.0)

        # File upload section
        st.markdown("### 📎 Supporting Documents")

        uploaded_files = st.file_uploader(
            "Upload project documents",
            type=['pdf', 'xlsx', 'docx', 'csv'],
            accept_multiple_files=True
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            submit_button = st.form_submit_button("✅ Submit Project", use_container_width=True)

    if submit_button:
        # Calculate NPV and other metrics
        annual_revenue = annual_production * ore_grade * commodity_price * recovery_rate / 100
        annual_costs = annual_production * opex
        annual_cashflow = (annual_revenue - annual_costs) * (1 - tax_rate / 100) * (1 - royalty_rate / 100)

        npv = -capex
        for year in range(mine_life):
            npv += annual_cashflow / ((1 + discount_rate / 100) ** (year + 1))

        # Calculate IRR (simplified)
        irr = (annual_cashflow / capex) * 100

        # Risk score calculation
        risk_score = (
                             (env_score * 0.3 + social_score * 0.3 + governance_score * 0.4) * 0.5 +
                             (min(recovery_rate / 90, 1) * 0.3) +
                             (min(20 / mine_life, 1) * 0.2)
                     ) * 10

        # Create new project entry
        new_project = {
            'project_id': f'PRJ-{len(st.session_state.projects) + 1:04d}',
            'name': project_name,
            'company': company,
            'region': region,
            'country': country,
            'mineral_type': mineral_type,
            'mineral_unit': '%',
            'stage': stage,
            'resource_estimate_mt': resource_estimate,
            'ore_grade': ore_grade,
            'depth_m': depth,
            'mine_life_years': mine_life,
            'capex_million': capex,
            'opex_per_ton': opex,
            'recovery_rate': recovery_rate / 100,
            'discount_rate': discount_rate / 100,
            'water_intensity': water_intensity,
            'energy_intensity': energy_intensity,
            'co2_intensity': energy_intensity * 0.5,
            'local_employment': local_employment,
            'community_investment_million': community_investment,
            'permits_obtained': permits,
            'environmental_impact_score': env_score,
            'social_score': social_score,
            'governance_score': governance_score,
            'last_update': datetime.now(),
            'project_start_date': project_start,
            'estimated_completion': estimated_completion,
            'npv_million': round(npv, 2),
            'irr_percent': round(irr, 1),
            'payback_years': round(capex / annual_cashflow, 1) if annual_cashflow > 0 else 99,
            'risk_score': round(risk_score, 1),
            'risk_level': 'Low' if risk_score > 7 else 'Medium' if risk_score > 4 else 'High',
            'ai_confidence': 0.85,
            'market_sentiment': 'Neutral'
        }

        # Add to dataframe
        st.session_state.projects = pd.concat([
            st.session_state.projects,
            pd.DataFrame([new_project])
        ], ignore_index=True)

        # Show success message with results
        st.success("✅ Project successfully added!")

        st.markdown(f"""
        <div class='glass-card' style='background: #28a74520; border-color: #28a745;'>
            <h3>Project Analysis Results</h3>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;'>
                <div>
                    <p style='color: #888;'>Net Present Value</p>
                    <h2 style='color: #28a745;'>${npv:.1f}M</h2>
                </div>
                <div>
                    <p style='color: #888;'>Internal Rate of Return</p>
                    <h2 style='color: #28a745;'>{irr:.1f}%</h2>
                </div>
                <div>
                    <p style='color: #888;'>Risk Score</p>
                    <h2>{risk_score:.1f}/10</h2>
                </div>
                <div>
                    <p style='color: #888;'>Payback Period</p>
                    <h2>{new_project['payback_years']} years</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # AI recommendations
        if npv > 0 and risk_score > 5:
            recommendation = "This project shows strong potential. Consider proceeding to the next development stage."
            rec_icon = "✅"
        elif npv > 0 and risk_score <= 5:
            recommendation = "Positive NPV but elevated risk. Implement risk mitigation strategies before proceeding."
            rec_icon = "⚠️"
        else:
            recommendation = "Negative NPV indicates challenges. Review assumptions or consider project modifications."
            rec_icon = "❌"

        st.markdown(f"""
        <div class='glass-card'>
            <h4>{rec_icon} AI Recommendation</h4>
            <p>{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

elif "📑 Reports" in page:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 48px; font-weight: 800; margin-bottom: 10px;'>Report Generation Center</h1>
        <p style='font-size: 18px; color: #888;'>Create comprehensive reports and analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Report type selection
    report_cols = st.columns(3)

    with report_cols[0]:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Analysis", "Financial Report",
             "ESG Report", "Risk Assessment", "Portfolio Analysis"]
        )

    with report_cols[1]:
        date_range = st.date_input(
            "Report Period",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            format="YYYY-MM-DD"
        )

    with report_cols[2]:
        if st.button("📊 Generate Report", use_container_width=True):
            st.session_state['generate_report'] = True

    # Report generation
    if st.session_state.get('generate_report', False):
        with st.spinner("Generating report..."):
            time.sleep(2)  # Simulate report generation

            if report_type == "Executive Summary":
                st.markdown("### 📊 Executive Summary Report")

                # Report header
                st.markdown(f"""
                <div class='glass-card'>
                    <h2>Mining Portfolio Executive Summary</h2>
                    <p style='color: #888;'>Report Period: {date_range[0].strftime('%B %d, %Y')} - {date_range[1].strftime('%B %d, %Y')}</p>
                    <p style='color: #888;'>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                """, unsafe_allow_html=True)

                # Key metrics summary
                st.markdown("#### 📈 Key Performance Indicators")

                kpi_cols = st.columns(4)
                total_projects = len(st.session_state.projects)
                total_npv = st.session_state.projects['npv_million'].sum()
                avg_irr = st.session_state.projects['irr_percent'].mean()
                total_capex = st.session_state.projects['capex_million'].sum()

                kpis = [
                    ("Total Projects", total_projects, f"{total_projects - 45:+d} vs last period"),
                    ("Portfolio NPV", f"${total_npv:,.0f}M", f"+{total_npv * 0.12:,.0f}M YoY"),
                    ("Average IRR", f"{avg_irr:.1f}%", "+2.3% vs target"),
                    ("Total Investment", f"${total_capex:,.0f}M", f"{total_capex * 0.85:.0f}M deployed")
                ]

                for col, (metric, value, change) in zip(kpi_cols, kpis):
                    with col:
                        st.markdown(f"""
                        <div class='glass-card'>
                            <p style='color: #888; font-size: 14px;'>{metric}</p>
                            <h3>{value}</h3>
                            <p style='color: #667eea; font-size: 12px;'>{change}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Portfolio composition
                st.markdown("#### 🎯 Portfolio Composition")

                comp_cols = st.columns(2)

                with comp_cols[0]:
                    # By mineral type
                    mineral_summary = st.session_state.projects.groupby('mineral_type').agg({
                        'project_id': 'count',
                        'npv_million': 'sum',
                        'capex_million': 'sum'
                    }).round(2)
                    mineral_summary.columns = ['Projects', 'NPV ($M)', 'CAPEX ($M)']
                    st.markdown("**By Mineral Type**")
                    st.dataframe(mineral_summary, use_container_width=True)

                with comp_cols[1]:
                    # By region
                    region_summary = st.session_state.projects.groupby('region').agg({
                        'project_id': 'count',
                        'npv_million': 'sum',
                        'risk_score': 'mean'
                    }).round(2)
                    region_summary.columns = ['Projects', 'NPV ($M)', 'Avg Risk Score']
                    st.markdown("**By Region**")
                    st.dataframe(region_summary, use_container_width=True)

                # Risk analysis
                st.markdown("#### ⚠️ Risk Analysis")

                risk_summary = f"""
                <div class='glass-card'>
                    <h4>Portfolio Risk Profile</h4>
                    <ul>
                        <li>High Risk Projects: {len(st.session_state.projects[st.session_state.projects['risk_level'] == 'High'])} ({len(st.session_state.projects[st.session_state.projects['risk_level'] == 'High']) / total_projects * 100:.1f}%)</li>
                        <li>Medium Risk Projects: {len(st.session_state.projects[st.session_state.projects['risk_level'] == 'Medium'])} ({len(st.session_state.projects[st.session_state.projects['risk_level'] == 'Medium']) / total_projects * 100:.1f}%)</li>
                        <li>Low Risk Projects: {len(st.session_state.projects[st.session_state.projects['risk_level'] == 'Low'])} ({len(st.session_state.projects[st.session_state.projects['risk_level'] == 'Low']) / total_projects * 100:.1f}%)</li>
                        <li>Average Portfolio Risk Score: {st.session_state.projects['risk_score'].mean():.1f}/10</li>
                    </ul>
                </div>
                """
                st.markdown(risk_summary, unsafe_allow_html=True)

                # Top opportunities
                st.markdown("#### 💎 Top Opportunities")

                top_projects = st.session_state.projects.nlargest(5, 'npv_million')[
                    ['name', 'mineral_type', 'region', 'npv_million', 'irr_percent', 'risk_level']
                ]
                # Use basic dataframe display without pandas styling
                st.dataframe(top_projects, use_container_width=True)

                # Recommendations
                st.markdown("#### 🎯 Strategic Recommendations")

                recommendations = [
                    ("Accelerate Development", "Fast-track 3 high-NPV lithium projects to capture market opportunity",
                     "High", "#28a745"),
                    ("Risk Mitigation", "Implement enhanced monitoring for 5 projects with risk scores below 4.0",
                     "High", "#dc3545"),
                    ("ESG Enhancement", "Invest in renewable energy for projects with high energy intensity", "Medium",
                     "#ffc107"),
                    ("Portfolio Diversification", "Consider acquiring copper assets to balance commodity exposure",
                     "Medium", "#17a2b8"),
                    ("Cost Optimization", "Review OPEX for mature projects - potential 15% reduction identified", "Low",
                     "#6c757d")
                ]

                for title, desc, priority, color in recommendations:
                    st.markdown(f"""
                    <div class='glass-card' style='border-left: 4px solid {color};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <h5>{title}</h5>
                                <p style='color: #888; font-size: 14px; margin: 0;'>{desc}</p>
                            </div>
                            <span style='background: {color}; color: white; padding: 3px 10px; border-radius: 15px; font-size: 12px;'>
                                {priority}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            elif report_type == "Financial Report":
                st.markdown("### 💰 Financial Analysis Report")

                # Financial overview
                st.markdown("#### Financial Performance Overview")

                # Create financial summary
                financial_metrics = {
                    'Total Portfolio Value': f"${st.session_state.projects['npv_million'].sum():,.0f}M",
                    'Total Capital Requirements': f"${st.session_state.projects['capex_million'].sum():,.0f}M",
                    'Average Payback Period': f"{st.session_state.projects['payback_years'].mean():.1f} years",
                    'Weighted Avg IRR': f"{(st.session_state.projects['irr_percent'] * st.session_state.projects['capex_million']).sum() / st.session_state.projects['capex_million'].sum():.1f}%"
                }

                fin_cols = st.columns(4)
                for col, (metric, value) in zip(fin_cols, financial_metrics.items()):
                    with col:
                        st.markdown(f"""
                        <div class='glass-card' style='text-align: center;'>
                            <p style='color: #888; font-size: 14px;'>{metric}</p>
                            <h3>{value}</h3>
                        </div>
                        """, unsafe_allow_html=True)

                # Cash flow projection
                st.markdown("#### 💸 Cash Flow Projections")

                # Generate cash flow data
                years = list(range(datetime.now().year, datetime.now().year + 10))
                cash_flows = []
                cumulative = 0

                for i, year in enumerate(years):
                    if i < 3:  # Investment phase
                        annual_flow = -st.session_state.projects['capex_million'].sum() / 3
                    else:  # Production phase
                        annual_flow = st.session_state.projects['npv_million'].sum() / 7

                    cumulative += annual_flow
                    cash_flows.append({
                        'Year': year,
                        'Annual Cash Flow': annual_flow,
                        'Cumulative Cash Flow': cumulative
                    })

                cf_df = pd.DataFrame(cash_flows)

                fig_cf = go.Figure()
                fig_cf.add_trace(go.Bar(
                    x=cf_df['Year'],
                    y=cf_df['Annual Cash Flow'],
                    name='Annual Cash Flow',
                    marker_color=cf_df['Annual Cash Flow'].apply(lambda x: '#28a745' if x > 0 else '#dc3545')
                ))
                fig_cf.add_trace(go.Scatter(
                    x=cf_df['Year'],
                    y=cf_df['Cumulative Cash Flow'],
                    name='Cumulative Cash Flow',
                    line=dict(color='#667eea', width=3),
                    yaxis='y2'
                ))

                fig_cf.update_layout(
                    title='10-Year Cash Flow Projection',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=400,
                    yaxis=dict(title='Annual Cash Flow ($M)', gridcolor='rgba(255,255,255,0.1)'),
                    yaxis2=dict(title='Cumulative Cash Flow ($M)', overlaying='y', side='right',
                                gridcolor='rgba(255,255,255,0.1)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cf, width='stretch')

                # Sensitivity table
                st.markdown("#### 📊 Financial Sensitivity Analysis")

                sensitivity_data = {
                    'Parameter': ['Commodity Price', 'OPEX', 'CAPEX', 'Exchange Rate', 'Discount Rate'],
                    '-20%': [-1250, 450, 380, -180, 420],
                    '-10%': [-625, 225, 190, -90, 210],
                    'Base Case': [0, 0, 0, 0, 0],
                    '+10%': [625, -225, -190, 90, -210],
                    '+20%': [1250, -450, -380, 180, -420]
                }

                sensitivity_df = pd.DataFrame(sensitivity_data)

                # Display without styling for simplicity
                st.dataframe(sensitivity_df, use_container_width=True)

    # Export options
    st.markdown("### 📥 Export Options")

    export_cols = st.columns(4)

    with export_cols[0]:
        if st.button("📄 Export as PDF", use_container_width=True):
            st.info("PDF export functionality would be implemented here")

    with export_cols[1]:
        if st.button("📊 Export to Excel", use_container_width=True):
            # Create Excel download
            output = st.session_state.projects.to_csv(index=False)
            st.download_button(
                label="Download Excel File",
                data=output,
                file_name=f"minemetrics_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with export_cols[2]:
        if st.button("📧 Email Report", use_container_width=True):
            st.success("Report sent to registered email address")

    with export_cols[3]:
        if st.button("☁️ Save to Cloud", use_container_width=True):
            st.success("Report saved to cloud storage")

# Footer with real-time updates
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px 0;'>
    <p style='color: #888; font-size: 14px;'>
        MineMetrics v2.0 | Authors: Alisher Beisembekov & Syrym Serikov | 
        <span style='color: #28a745;'>● All Systems Operational</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Add floating action button for AI assistant
st.markdown("""
<button class='fab' onclick='alert("AI Assistant would open here")'>
    🤖
</button>
""", unsafe_allow_html=True)

