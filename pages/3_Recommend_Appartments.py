import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Recommend Appartments", layout="wide")

st.title('🏠 Property Recommender System')

try:
    location_df = pickle.load(open('location_distance.pkl','rb'))
    cosine_sim1 = pickle.load(open('cosine_sim1.pkl','rb'))
    cosine_sim2 = pickle.load(open('cosine_sim2.pkl','rb'))
    cosine_sim3 = pickle.load(open('cosine_sim3.pkl','rb'))
    
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
    
    st.success("✅ Property data loaded successfully from pickle files.")

except Exception as e:
    st.error(f"❌ Error loading pickle files: {e}. Using dummy data for demonstration.")
    
    properties = [
        'DLF The Camellias', 'Emaar Emerald Hills', 'Godrej Sector 89', 
        'BPTP Park Floors', 'Signature Global Proxima', 'Suncity Heights', 
        'Central Park Flower Valley', 'Sobha International City', 'Ambience Tiverton',
        'Vipul Gardens', 'Tata Primanti', 'M3M Golfestate'
    ]
    locations = ['Cyber Hub', 'Sector 54 Rapid Metro', 'Indira Gandhi Airport', 'Rajiv Chowk']
    
    data = np.random.randint(500, 15000, size=(len(properties), len(locations)))
    location_df = pd.DataFrame(data, index=properties, columns=locations)
    
    size = len(properties)
    cosine_sim_matrix = np.random.rand(size, size)
    cosine_sim_matrix = (cosine_sim_matrix + cosine_sim_matrix.T) / 2
    np.fill_diagonal(cosine_sim_matrix, 1.0)


def recommend_properties_with_scores(property_name, top_n=5):
    if property_name not in location_df.index:
        st.error(f"Property '{property_name}' not found in the database.")
        return pd.DataFrame()

    property_index = location_df.index.get_loc(property_name)
    
    sim_scores = list(enumerate(cosine_sim_matrix[property_index]))

    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices_scores = sorted_scores[1:top_n + 1]
    
    top_indices = [i[0] for i in top_indices_scores]
    top_scores = [i[1] for i in top_indices_scores]

    top_properties = location_df.index[top_indices].tolist()

    recommendations_df = pd.DataFrame({
        'Recommended Property': top_properties,
        'Similarity Score': top_scores
    })

    recommendations_df['Similarity Score'] = (recommendations_df['Similarity Score'] * 100).round(2).astype(str) + '%'

    return recommendations_df

st.header('📍 Search by Location & Radius')

selected_location = st.selectbox('Select Location', sorted(location_df.columns.to_list()))
radius = st.number_input('Radius in Kms', min_value=0.5, value=5.0, step=0.5, format="%.1f")

if st.button('Search Properties', use_container_width=True):
    st.subheader(f"Properties within {radius} km of {selected_location}")
    
    mask = location_df[selected_location] < radius * 1000
    result_ser = location_df[mask][selected_location].sort_values()

    if result_ser.empty:
        st.info("No properties found within this radius.")
    else:
        location_results = pd.DataFrame({
            'Property Name': result_ser.index,
            'Distance (km)': (result_ser / 1000).round(2)
        })
        st.dataframe(location_results, use_container_width=True, hide_index=True)


st.divider()

st.header('🤖 Recommend Similar Apartments')
selected_appartment = st.selectbox('Select a base apartment for recommendation', sorted(location_df.index.to_list()))

if st.button('Get Recommendations', use_container_width=True):
    recommendation_df = recommend_properties_with_scores(selected_appartment)
    
    if not recommendation_df.empty:
        st.markdown("---")
        st.markdown(f"**Step 1: Reference Property Selected:** `{selected_appartment}`")
        st.markdown("---")
        
        st.subheader(f"Step 2: Top 5 Similar Properties Found:")
        
        for i, row in recommendation_df.iterrows():
            property_name = row['Recommended Property']
            score = row['Similarity Score']
            st.markdown(f"**{i+1}.** **{property_name}** (Similarity: **{score}**)")
            
    else:
        st.info("Could not generate recommendations for the selected property.")

st.caption("Note: If the original files failed to load, the data shown is randomly generated dummy data.")



# This Streamlit application implements a property recommender system.
# 1. Data Loading: It attempts to load pre-calculated similarity matrices (cosine_sim1/2/3)
#    and location data (location_df) from pickle files. It uses robust error handling (try/except)
#    to catch common issues like FileNotFoundError or internal TypeError/PickleError (often caused
#    by version mismatch) and falls back to generating dummy data for demonstration purposes.
# 2. Similarity Matrix: A weighted similarity matrix (cosine_sim_matrix) is computed globally
#    by combining the three loaded matrices (0.5 * sim1 + 0.8 * sim2 + 1 * sim3).
# 3. recommend_properties_with_scores(property_name, top_n=5): This function takes a property
#    name, finds its index in the location_df, retrieves its scores from the combined
#    similarity matrix, sorts the scores in descending order, and returns the top N results
#    (excluding the input property itself) as a DataFrame with percentage scores.
# 4. Location Search UI: Allows users to select a location and a radius (in Kms) to filter
#    properties based on their proximity (using data from location_df, which contains
#    distances in meters).
# 5. Recommendation UI: Allows users to select a base property. Upon clicking 'Get Recommendations',
#    the system uses the content-based similarity model to present the 5 most similar
#    properties in a clear, step-wise list.
# 6. UI/Aesthetics: Uses st.set_page_config(layout="wide") for better use of screen space and
#    provides visual feedback (success/error messages, dividers, headers).
# --- END CONSOLIDATED COMMENTS BLOCK ---