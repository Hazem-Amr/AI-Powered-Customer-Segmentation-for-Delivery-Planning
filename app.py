"""
Customer Delivery Policy Recommendation System
================================================
A professional Streamlit application that predicts whether a wholesale customer
should receive 5 delivery days/week (HoReCa) or 3 delivery days/week (Retail)
based on their annual spending behavior.

Uses a Logistic Regression model trained on GMM-discovered segment labels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Customer Delivery Policy Recommendation System",
    page_icon="CDP",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Styling ────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(0, 0, 0, 0.25);
    }

    .metric-card h3 {
        color: #64b5f6;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.5rem;
    }

    .metric-card .value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Result cards */
    .result-card-horeca {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(52, 211, 153, 0.3);
        box-shadow: 0 8px 32px rgba(45, 106, 79, 0.2);
        margin: 1rem 0;
    }

    .result-card-retail {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(96, 165, 250, 0.3);
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.2);
        margin: 1rem 0;
    }

    .result-card-horeca h2, .result-card-retail h2 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }

    .result-card-horeca .schedule, .result-card-retail .schedule {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .result-card-horeca .schedule {
        color: #34d399;
    }

    .result-card-retail .schedule {
        color: #60a5fa;
    }

    .result-card-horeca p, .result-card-retail p {
        color: #d1d5db;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #a78bfa);
        color: #ffffff;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35);
        letter-spacing: 0.5px;
        margin: 0.5rem 0;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .sidebar-section h3 {
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.75rem;
    }

    .sidebar-section p, .sidebar-section li {
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Info card */
    .info-card {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 14px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        margin: 1rem 0;
    }

    .info-card h3 {
        color: #60a5fa;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-label {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 500;
    }

    .info-value {
        color: #e2e8f0;
        font-size: 0.88rem;
        font-weight: 600;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Number input styling */
    .stNumberInput label {
        font-weight: 600 !important;
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
    }

    /* Segment badge */
    .segment-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    .segment-horeca {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .segment-retail {
        background: rgba(96, 165, 250, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 2rem 0;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load the trained Logistic Regression model from disk."""
    model_path = os.path.join(os.path.dirname(__file__), "customer_segment_classifier.pkl")
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file `customer_segment_classifier.pkl` not found. "
                 "Please ensure it is in the same directory as this application.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_input(fresh, milk, grocery, frozen, detergents_paper, delicatessen):
    """
    Apply log transformation to raw spending values to match training preprocessing.
    Adds a small constant (1) to avoid log(0).
    Returns a DataFrame with feature names to match training data.
    """
    feature_names = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
    raw_values = np.array([[fresh, milk, grocery, frozen, detergents_paper, delicatessen]])
    log_values = np.log(raw_values + 1)  # +1 to handle zero values
    return pd.DataFrame(log_values, columns=feature_names)


def predict_segment(model, processed_input):
    """
    Predict customer segment and return prediction with probabilities.

    Returns:
        segment (int): 0 for HoReCa, 1 for Retail
        probability (float): Confidence score (max of predict_proba)
        probabilities (array): Full probability distribution
    """
    segment = model.predict(processed_input)[0]
    probabilities = model.predict_proba(processed_input)[0]
    confidence = np.max(probabilities) * 100
    return segment, confidence, probabilities


def create_spending_chart(values, labels):
    """Create a professional Plotly bar chart of customer spending profile."""
    colors = ['#34d399', '#60a5fa', '#a78bfa', '#f472b6', '#fbbf24', '#fb923c']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.1)', width=1),
            cornerradius=6,
        ),
        text=[f"${v:,.0f}" for v in values],
        textposition='outside',
        textfont=dict(size=13, color='#e2e8f0', family='Inter'),
        hovertemplate="<b>%{x}</b><br>Annual Spending: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Customer Spending Profile",
            font=dict(size=18, color='#e2e8f0', family='Inter'),
            x=0.5,
            xanchor='center',
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=12, color='#94a3b8', family='Inter'),
            gridcolor='rgba(255,255,255,0.03)',
        ),
        yaxis=dict(
            title=dict(
                text="Annual Spending ($)",
                font=dict(size=13, color='#94a3b8', family='Inter'),
            ),
            tickfont=dict(size=11, color='#94a3b8', family='Inter'),
            gridcolor='rgba(255,255,255,0.05)',
            zeroline=False,
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=30, t=60, b=40),
        height=420,
        hoverlabel=dict(
            bgcolor='#1a1a2e',
            bordercolor='rgba(255,255,255,0.1)',
            font=dict(size=13, color='#e2e8f0', family='Inter'),
        ),
        bargap=0.3,
    )

    return fig


def create_confidence_gauge(confidence):
    """Create a professional gauge chart for prediction confidence."""
    if confidence >= 90:
        bar_color = '#34d399'
    elif confidence >= 75:
        bar_color = '#fbbf24'
    else:
        bar_color = '#f87171'

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        number=dict(
            suffix="%",
            font=dict(size=42, color='#e2e8f0', family='Inter', weight=700),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor='rgba(255,255,255,0.1)',
                dtick=20,
                tickfont=dict(size=11, color='#64748b'),
            ),
            bar=dict(color=bar_color, thickness=0.7),
            bgcolor='rgba(255,255,255,0.03)',
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color='rgba(248, 113, 113, 0.08)'),
                dict(range=[50, 75], color='rgba(251, 191, 36, 0.08)'),
                dict(range=[75, 100], color='rgba(52, 211, 153, 0.08)'),
            ],
            threshold=dict(
                line=dict(color='#a78bfa', width=3),
                thickness=0.85,
                value=confidence,
            ),
        ),
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
    )

    return fig


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Navigation")
    st.markdown("---")

    # Project Overview
    st.markdown("""
    <div class="sidebar-section">
        <h3>Project Overview</h3>
        <p>
            This system uses machine learning to predict the optimal delivery 
            frequency for wholesale customers based on their annual purchasing 
            patterns across six product categories.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model Information
    st.markdown("""
    <div class="sidebar-section">
        <h3>Model Information</h3>
        <div class="info-row">
            <span class="info-label">Model</span>
            <span class="info-value">Logistic Regression</span>
        </div>
        <div class="info-row">
            <span class="info-label">Input Features</span>
            <span class="info-value">6</span>
        </div>
        <div class="info-row">
            <span class="info-label">Target</span>
            <span class="info-value">Customer Segment</span>
        </div>
        <div class="info-row">
            <span class="info-label">Output</span>
            <span class="info-value">Delivery Recommendation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Segment Definitions
    st.markdown("""
    <div class="sidebar-section">
        <h3>Segment Definitions</h3>
        <p><span class="segment-badge segment-horeca">HoReCa</span></p>
        <p style="margin-top: 0.5rem;">
            Hotels, Restaurants, Cafes & Catering<br>
            <strong style="color: #34d399;">→ 5 Days/Week</strong>
        </p>
        <br>
        <p><span class="segment-badge segment-retail">Retail</span></p>
        <p style="margin-top: 0.5rem;">
            Supermarkets, Grocery & Retail Chains<br>
            <strong style="color: #60a5fa;">→ 3 Days/Week</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
        Built with Streamlit & Scikit-Learn<br>
        © 2024 Customer Segmentation Project
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>Customer Delivery Policy Recommendation System</h1>
    <p>Predict the optimal delivery schedule for wholesale customers using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# Load Model
model = load_model()

# ─── Customer Input Form ───────────────────────────────────────────────────────

st.markdown("### Customer Spending Information")
st.markdown("Enter the customer's annual spending (in monetary units) for each product category:")

col1, col2, col3 = st.columns(3)

with col1:
    fresh = st.number_input(
        "Fresh",
        min_value=0,
        value=8000,
        step=100,
        help="Annual spending on fresh products (fruits, vegetables, meat, fish)",
        key="input_fresh",
    )
    frozen = st.number_input(
        "Frozen",
        min_value=0,
        value=2000,
        step=100,
        help="Annual spending on frozen products",
        key="input_frozen",
    )

with col2:
    milk = st.number_input(
        "Milk",
        min_value=0,
        value=4000,
        step=100,
        help="Annual spending on dairy products",
        key="input_milk",
    )
    detergents_paper = st.number_input(
        "Detergents_Paper",
        min_value=0,
        value=1500,
        step=100,
        help="Annual spending on cleaning and paper products",
        key="input_detergents_paper",
    )

with col3:
    grocery = st.number_input(
        "Grocery",
        min_value=0,
        value=5000,
        step=100,
        help="Annual spending on grocery products",
        key="input_grocery",
    )
    delicatessen = st.number_input(
        "Delicatessen",
        min_value=0,
        value=1000,
        step=100,
        help="Annual spending on delicatessen products",
        key="input_delicatessen",
    )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ─── Prediction ────────────────────────────────────────────────────────────────

predict_button = st.button("Predict Recommendation", key="predict_btn")

if predict_button and model is not None:
    # Step 1: Collect values
    feature_names = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
    raw_values = [fresh, milk, grocery, frozen, detergents_paper, delicatessen]

    # Step 2: Preprocess (log transform)
    processed_input = preprocess_input(fresh, milk, grocery, frozen, detergents_paper, delicatessen)

    # Step 3 & 4: Predict
    segment, confidence, probabilities = predict_segment(model, processed_input)

    # Segment mapping
    segment_map = {0: "HoReCa", 1: "Retail"}
    segment_name = segment_map.get(segment, "Unknown")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ─── Prediction Output ──────────────────────────────────────────────────

    st.markdown("### Prediction Results")

    result_col1, result_col2 = st.columns([3, 2])

    with result_col1:
        if segment_name == "HoReCa":
            st.markdown("""
            <div class="result-card-horeca">
                <h2>Recommended Delivery Schedule</h2>
                <div class="schedule">5 Days Per Week</div>
                <p style="margin-top: 1rem;">
                    <strong style="color: #34d399;">Segment:</strong> HoReCa (Hotels, Restaurants, Cafes)
                </p>
                <p style="margin-top: 0.75rem;">
                    <strong style="color: #34d399;">Reason:</strong><br>
                    This customer exhibits purchasing behavior similar to restaurants, hotels, 
                    and food-service businesses that rely on fresh inventory and frequent replenishment.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card-retail">
                <h2>Recommended Delivery Schedule</h2>
                <div class="schedule">3 Days Per Week</div>
                <p style="margin-top: 1rem;">
                    <strong style="color: #60a5fa;">Segment:</strong> Retail (Supermarkets, Grocery Stores)
                </p>
                <p style="margin-top: 0.75rem;">
                    <strong style="color: #60a5fa;">Reason:</strong><br>
                    This customer exhibits purchasing behavior similar to retailers and supermarkets 
                    that can maintain larger inventories and require less frequent deliveries.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with result_col2:
        st.markdown("#### Prediction Confidence")
        confidence_fig = create_confidence_gauge(confidence)
        st.plotly_chart(confidence_fig, key="confidence_gauge")
        st.markdown(
            f'<div style="text-align:center;"><span class="confidence-badge">'
            f'Prediction Confidence: {confidence:.1f}%</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ─── Customer Profile Visualization ─────────────────────────────────────

    st.markdown("### Customer Profile Visualization")

    chart_col, table_col = st.columns([3, 2])

    with chart_col:
        spending_chart = create_spending_chart(raw_values, feature_names)
        st.plotly_chart(spending_chart, key="spending_chart")

    with table_col:
        st.markdown("#### Feature Contribution")
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": [f"${v:,}" for v in raw_values],
        })
        st.dataframe(
            feature_df,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="medium"),
            },
        )

        # Segment probability breakdown
        st.markdown("#### Segment Probabilities")
        prob_df = pd.DataFrame({
            "Segment": ["HoReCa (5 Days)", "Retail (3 Days)"],
            "Probability": [f"{probabilities[0]*100:.1f}%", f"{probabilities[1]*100:.1f}%"],
        })
        st.dataframe(prob_df, hide_index=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ─── Model Information Card ─────────────────────────────────────────────

    st.markdown("### Model Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Clustering Algorithm</h3>
            <div class="value">Gaussian Mixture Model (GMM)</div>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Classification Model</h3>
            <div class="value">Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with info_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Training F1 Score</h3>
            <div class="value" style="color: #34d399;">98.36%</div>
        </div>
        """, unsafe_allow_html=True)

elif predict_button and model is None:
    st.error("Cannot make predictions without a loaded model. "
             "Please check that `customer_segment_classifier.pkl` exists in the application directory.")

# ─── Footer ────────────────────────────────────────────────────────────────────

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-text">
    Customer Delivery Policy Recommendation System | 
    Powered by Machine Learning | 
    Built with Streamlit, Scikit-Learn & Plotly
</div>
""", unsafe_allow_html=True)
