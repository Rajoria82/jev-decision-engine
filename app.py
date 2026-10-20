import streamlit as st
import time
from local_jev import LocalJevEngine

st.set_page_config(
    page_title="System-1 Decision Engine (Jev Architecture)",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ System-1 Decision Engine")
st.caption("Low-latency intent routing and risk triage using Jev Primitives (Choice, Score, Noul)")

# Cache the model so it loads into memory only once
@st.cache_resource(show_spinner=False)
def get_engine():
    print("[LOG] Initializing LocalJevEngine in memory...")
    engine = LocalJevEngine()
    print("[LOG] LocalJevEngine loaded successfully.")
    return engine

# Show visual status in UI while model loads
with st.spinner("Loading System-1 Decision Model into memory... Please wait 5-10 seconds."):
    engine = get_engine()

QUESTIONS_SCHEMA = {
    "department_routing": {
        "type": "choice",
        "instructions": "This customer request belongs to",
        "criteria": {
            "billing and payment": "Invoices, transaction failures, UPI refunds",
            "sales and enterprise": "Bulk discounts, business quotes, subscriptions",
            "technical support": "Errors, system crashes, server bugs, 502 gateway"
        }
    },
    "urgency_rating": {
        "type": "score",
        "instructions": "Assess urgency",
        "levels": ["low", "moderate", "high", "critical"]
    },
    "escalate_to_manager": {
        "type": "noul",
        "instructions": "Does this issue involve financial loss or system outage requiring immediate escalation?"
    }
}

example_prompts = [
    "UPI payment succeeded, money deducted from HDFC account, but booking status shows failed! Please refund or confirm urgently.",
    "Can we get custom pricing or volume licensing if our enterprise team buys 500+ developer seats?",
    "The web application dashboard is throwing a 502 Bad Gateway error when downloading CSV reports.",
    "Just wanted to say the new update is super smooth, thank you team!"
]

selected_example = st.selectbox("Choose a pre-loaded query (or type below):", ["-- Select an example --"] + example_prompts)

default_text = "" if selected_example == "-- Select an example --" else selected_example
query_input = st.text_area("Customer Ticket / Query:", value=default_text, height=100)

if st.button("Evaluate Decision Model", type="primary"):
    if not query_input.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Computing System-1 decision..."):
            res = engine.decide(query_input, QUESTIONS_SCHEMA)
            decisions = res["decisions"]
            latency = res["latency_ms"]

        st.success(f"Decision computed in **{latency:.2f} ms**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Department Route (Choice)",
                value=decisions["department_routing"]["winner"].title(),
                delta=f"{decisions['department_routing']['confidence']*100:.1f}% confidence"
            )

        with col2:
            st.metric(
                label="Urgency Level (Score)",
                value=f"{decisions['urgency_rating']['calculated_score']} / 4.0",
                delta=decisions["urgency_rating"]["dominant_level"].upper()
            )

        with col3:
            escalate = decisions["escalate_to_manager"]["verdict"]
            st.metric(
                label="Immediate Escalation (Noul)",
                value="YES (Action Required)" if escalate else "NO (Standard Flow)",
                delta="Risk Flagged" if escalate else "Normal",
                delta_color="inverse" if escalate else "normal"
            )

        with st.expander("View Raw Probabilities & Jev Schema"):
            st.json(decisions)