import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Added for custom map layers
import pickle
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Plotting Demo", layout="wide")

st.title('🏠 Real Estate Analytics')

# --- Data Loading ---
try:
    new_df = pd.read_csv('data_viz1.csv')
    with open('feature_text.pkl', 'rb') as f:
        feature_text = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Required data files ('data_viz1.csv' or 'feature_text.pkl') not found.")
    st.stop()

group_df = new_df.groupby('sector').mean(numeric_only=True)[['price','price_per_sqft','built_up_area','latitude','longitude']]

# --- 1. Sector Price per Sqft Geomap ---
st.header('🗺️ Sector Price per Sqft Geomap')
fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
                        color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                        mapbox_style="open-street-map", width=1200, height=700, hover_name=group_df.index)

st.plotly_chart(fig, use_container_width=True)

# --- 2. Features Wordcloud ---
st.header('☁️ Features Wordcloud')

wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='black',
                      stopwords = set(['s']),
                      min_font_size = 10).generate(feature_text)

fig_wc, ax_wc = plt.subplots(figsize = (8, 8), facecolor = None)
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
plt.tight_layout(pad = 0)
st.pyplot(fig_wc)

# --- 3. Area Vs Price Scatter Plot ---
st.header('📈 Area Vs Price')

property_type = st.selectbox('Select Property Type', ['flat','house'])

if property_type == 'house':
    df_filtered = new_df[new_df['property_type'] == 'house']
else:
    df_filtered = new_df[new_df['property_type'] == 'flat']

fig1 = px.scatter(df_filtered, x="built_up_area", y="price", color="bedRoom",
                  title=f"Area Vs Price for {property_type.capitalize()}s")

st.plotly_chart(fig1, use_container_width=True)

# --- 4. BHK Pie Chart ---
st.header('🥧 BHK Distribution Pie Chart')

sector_options = sorted(new_df['sector'].unique().tolist())
sector_options.insert(0,'overall')

selected_sector = st.selectbox('Select Sector', sector_options)

if selected_sector == 'overall':
    df_pie = new_df
else:
    df_pie = new_df[new_df['sector'] == selected_sector]

fig2 = px.pie(df_pie, names='bedRoom', title=f'BHK Distribution in {selected_sector.capitalize()}')

st.plotly_chart(fig2, use_container_width=True)

# --- 5. Side by Side BHK price comparison (Box Plot) ---
st.header('📦 Side by Side BHK Price Comparison')

df_box = new_df[new_df['bedRoom'].notna() & (new_df['bedRoom'] <= 4)]

fig3 = px.box(df_box, x='bedRoom', y='price', title='BHK Price Range (up to 4 Bedrooms)')

st.plotly_chart(fig3, use_container_width=True)


# --- 6. Side by Side Distplot for property type ---
st.header('📊 Side by Side Price Distribution')

fig4 = plt.figure(figsize=(10, 4))
sns.kdeplot(new_df[new_df['property_type'] == 'house']['price'], label='house', fill=True)
sns.kdeplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat', fill=True)
plt.title('Price Distribution by Property Type')
plt.xlabel('Price')
plt.legend()
st.pyplot(fig4)

# --- 7. Correlation Heatmap ---
st.header('🔥 Correlation Heatmap')

# Select numerical columns for correlation analysis
corr_cols = ['price', 'price_per_sqft', 'built_up_area', 'bedRoom', 'bathroom', 'luxury_score']
corr_df = new_df[corr_cols].dropna() # Drop rows with NaN values for accurate correlation calculation

plt.figure(figsize=(10, 8))
# Calculate the correlation matrix and plot it using a heatmap
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Key Features')
st.pyplot(plt.gcf())

# --- 8. Property Type Count Plot ---
st.header('🏠 Property Type Count')

plt.figure(figsize=(6, 4))
sns.countplot(x='property_type', data=new_df, palette='viridis')
plt.title('Count of Flats vs. Houses')
plt.xlabel('Property Type')
plt.ylabel('Count')
st.pyplot(plt.gcf())

# --- 9. Luxury Score vs. Price Scatter Plot ---
st.header('💎 Luxury Score vs. Price')

fig5 = px.scatter(new_df, x="luxury_score", y="price",
                  color="property_type",
                  title="Luxury Score vs. Price",
                  hover_data=['sector', 'built_up_area'])

st.plotly_chart(fig5, use_container_width=True)


# --- 10. Choropleth Map: Average Price per Sector ---
st.header('💰 Sector Price Distribution (Choropleth)')

# Calculated based on new_df since global filters aren't active
map_data = new_df.groupby('sector').agg(
    avg_price=('price', 'mean'),
    lat=('latitude', 'mean'),
    lon=('longitude', 'mean'),
    total_properties=('sector', 'count')
).reset_index()

# Attempt to load GeoJSON for polygon rendering
geojson_data = None
try:
    with open('sectors.geojson', 'r') as f:
        geojson_data = json.load(f)
except FileNotFoundError:
    st.warning("⚠️ 'sectors.geojson' file not found. Displaying points instead. To see sector polygons, please add a 'sectors.geojson' file to your directory.")

if geojson_data:
    # TRUE CHOROPLETH: Uses GeoJSON to draw polygons
    fig_choropleth = px.choropleth_mapbox(
        map_data,
        geojson=geojson_data,
        locations='sector',
        featureidkey='properties.Name', 
        color='avg_price',
        hover_name='sector',
        hover_data={'total_properties': True, 'avg_price': ':.2f'},
        color_continuous_scale=px.colors.sequential.Plasma,
        mapbox_style="open-street-map",
        zoom=10,
        center={"lat": map_data['lat'].mean(), "lon": map_data['lon'].mean()},
        opacity=0.5,
        width=1200,
        height=700
    )
    # Add text labels for Sector Numbers on top of the polygons
    fig_choropleth.add_trace(go.Scattermapbox(
        lat=map_data['lat'],
        lon=map_data['lon'],
        mode='text',
        text=map_data['sector'],
        textposition='top center',
        showlegend=False,
        textfont=dict(size=12, color='black')
    ))

else:
    # FALLBACK: Uses Scatter Mapbox (Points) if GeoJSON is missing
    fig_choropleth = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        color="avg_price",
        text="sector", # Show sector name on the points directly
        size="total_properties",
        hover_name="sector",
        hover_data={'avg_price': ':.2f'},
        color_continuous_scale=px.colors.sequential.Plasma,
        mapbox_style="open-street-map",
        zoom=10,
        width=1200,
        height=700
    )

st.plotly_chart(fig_choropleth, use_container_width=True)