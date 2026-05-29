import streamlit as st

st.set_page_config(
    page_title="Gurgaon Real Estate Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hero
st.title("🏠 Gurgaon Real Estate Analytics Platform")
st.caption("End-to-end analytics, price prediction, and recommendations for the Gurgaon property market.")

st.sidebar.success("👈 Select a page above")

st.divider()

# Feature cards
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.subheader("💰 Price Predictor")
    st.write("Estimate a property's price from its attributes — sector, size, BHK, luxury tier, and more.")
    st.caption("Powered by a tuned regression pipeline trained on 3,000+ listings.")

with c2:
    st.subheader("📊 Analysis App")
    st.write("Explore sector-level prices, geomaps, wordclouds, BHK distributions, and correlations.")
    st.caption("Interactive Plotly charts and Streamlit widgets.")

with c3:
    st.subheader("🤖 Recommend Apartments")
    st.write("Find societies similar to a chosen property, or search by proximity to a landmark.")
    st.caption("Content-based filtering with TF-IDF + cosine similarity.")

st.divider()

# Stats row
st.subheader("📈 Platform Snapshot")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Listings analyzed", "3,000+")
s2.metric("Gurgaon Sectors", "100+")
s3.metric("Property Types", "Flats + Houses")
s4.metric("Model", "Regression Pipeline")

st.divider()

# Tech stack
st.markdown(
    """
    ### 🛠️ Tech Stack
    **Python** · **pandas** · **scikit-learn** · **XGBoost** · **Streamlit** · **Plotly** · **Seaborn** · **WordCloud**

    ### 🚀 Get Started
    Click any page in the **sidebar** to begin.
    """
)
