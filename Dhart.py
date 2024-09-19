import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px

# Create synthetic datasets
np.random.seed(42)

# Define number of samples
num_samples = 100

# Create synthetic data
rainfall_data = pd.DataFrame({
    'location_id': range(num_samples),
    'rainfall': np.random.uniform(0, 500, num_samples)
})

hydrogeology_data = pd.DataFrame({
    'location_id': range(num_samples),
    'hydrogeology': np.random.uniform(0, 10, num_samples)
})

landuse_data = pd.DataFrame({
    'location_id': range(num_samples),
    'landuse': np.random.uniform(0, 100, num_samples)
})

population_data = pd.DataFrame({
    'location_id': range(num_samples),
    'population_density': np.random.uniform(0, 1000, num_samples)
})

surface_elevation_data = pd.DataFrame({
    'location_id': range(num_samples),
    'surface_elevation': np.random.uniform(0, 3000, num_samples)
})

natural_features_data = pd.DataFrame({
    'location_id': range(num_samples),
    'natural_features': np.random.uniform(0, 10, num_samples)
})

tidal_cycles_data = pd.DataFrame({
    'location_id': range(num_samples),
    'tidal_cycles': np.random.uniform(0, 100, num_samples)
})

# Create a synthetic target variable
groundwater_level_data = pd.DataFrame({
    'location_id': range(num_samples),
    'groundwater_level': np.random.uniform(0, 100, num_samples)
})

# Merge datasets
data = pd.merge(rainfall_data, hydrogeology_data, on='location_id', how='inner')
data = pd.merge(data, landuse_data, on='location_id', how='inner')
data = pd.merge(data, population_data, on='location_id', how='inner')
data = pd.merge(data, surface_elevation_data, on='location_id', how='inner')
data = pd.merge(data, natural_features_data, on='location_id', how='inner')
data = pd.merge(data, tidal_cycles_data, on='location_id', how='inner')
data = pd.merge(data, groundwater_level_data, on='location_id', how='inner')

# Prepare features and target variable
features = ['rainfall', 'hydrogeology', 'landuse', 'population_density', 'surface_elevation', 'natural_features', 'tidal_cycles']
X = data[features]
y = data['groundwater_level']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Streamlit app UI
st.title('🌍 Groundwater Level Prediction App')

# Display introduction with styled header
st.markdown("""
    <div style="background-color: lightblue; padding: 10px; border-radius: 10px;">
        <h2 style="text-align: center;">Predict Groundwater Levels Based on Various Factors</h2>
        <p style="text-align: center;">Use this tool to input details and predict groundwater levels using machine learning.</p>
    </div>
""", unsafe_allow_html=True)

st.write('### Please enter the following details to predict the groundwater level:')

# Arrange input fields in two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input('🌧️ Rainfall (mm)', min_value=0.0, step=0.1)
    population_density = st.number_input('👥 Population Density (people per sq km)', min_value=0.0, step=1.0)
    surface_elevation = st.number_input('🏔️ Surface Elevation (m)', min_value=0.0, step=0.1)

with col2:
    hydrogeology = st.number_input('🧭 Hydrogeology (feature)', min_value=0.0, step=0.1)
    landuse = st.number_input('🌿 Landuse (feature)', min_value=0.0, step=0.1)
    natural_features = st.number_input('🌱 Natural Features (feature)', min_value=0.0, step=0.1)
    tidal_cycles = st.number_input('🌊 Tidal Cycles (feature)', min_value=0.0, step=0.1)

# Predict button
st.markdown("<br>", unsafe_allow_html=True)  # Add space between inputs and button
if st.button('🔮 Predict Groundwater Level'):
    try:
        # Prepare input for model
        input_data = pd.DataFrame([[rainfall, hydrogeology, landuse, population_density, surface_elevation, natural_features, tidal_cycles]],
                                  columns=features)

        # Predict groundwater level
        prediction = model.predict(input_data)[0]
        st.markdown(f"""
            <div style="background-color: #90ee90; padding: 10px; border-radius: 10px; text-align: center;">
                <h3>Predicted Groundwater Level: {prediction:.2f} meters below ground level</h3>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f'Error: {e}')

# Display model performance metric
st.write(f"#### Model Mean Squared Error: `{mse:.2f}`")

# Add Graph: Predicted vs Actual Groundwater Levels (Scatter Plot)
st.subheader("📊 Predicted vs. Actual Groundwater Levels")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Predicted vs Actual Groundwater Levels')
st.pyplot(fig)

# Add Graph: Feature Importance (Bar Chart)
st.subheader("📈 Feature Importance (Coefficients of the Linear Regression Model)")
coefficients = pd.DataFrame({
    'Feature': features,
    'Importance': model.coef_
})
fig = px.bar(coefficients, x='Feature', y='Importance', title='Feature Importance')
st.plotly_chart(fig)
