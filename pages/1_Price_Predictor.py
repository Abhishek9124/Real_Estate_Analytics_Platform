import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Price Predictor", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def _find(name):
    for base in (ROOT, HERE):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None

file_path_df = _find('df.pkl')
file_path_pipeline = _find('pipeline.pkl')

if not file_path_df or not file_path_pipeline:
    st.error("Required model files (df.pkl, pipeline.pkl) not found. Run model-selection.ipynb to generate them.")
    st.stop()

try:
    with open(file_path_df, 'rb') as file:
        df = pickle.load(file)
    with open(file_path_pipeline, 'rb') as file:
        pipeline = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

st.title('💰 Property Price Predictor')
st.caption('Estimate a property price in Gurgaon based on its attributes.')
st.divider()

with st.form('predict_form'):
    st.subheader('Property Attributes')

    c1, c2, c3 = st.columns(3)
    with c1:
        property_type = st.selectbox('Property Type', ['flat', 'house'])
        sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
        bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedRoom'].unique().tolist())))
        bathroom = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))
    with c2:
        balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))
        property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))
        built_up_area = float(st.number_input('Built Up Area (sqft)', min_value=0.0, value=1000.0, step=50.0))
        furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
    with c3:
        luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
        floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))
        servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
        store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

    submitted = st.form_submit_button('🎯 Predict Price', use_container_width=True)

if submitted:

    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)

    try:
        with st.spinner('Predicting price...'):
            base_price = np.expm1(pipeline.predict(one_df))[0]
            low = base_price - 0.22
            high = base_price + 0.22
        st.divider()
        st.subheader('📊 Estimated Price Range')
        m1, m2, m3 = st.columns(3)
        m1.metric('Lower Bound', f"₹ {round(low,2)} Cr")
        m2.metric('Estimated Price', f"₹ {round(base_price,2)} Cr")
        m3.metric('Upper Bound', f"₹ {round(high,2)} Cr")
        st.success(f"The estimated price is between **₹ {round(low,2)} Cr** and **₹ {round(high,2)} Cr**.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
