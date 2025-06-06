import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle  # Ensure pickle is imported for model loading

# Load and clean data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Debugging: Print available columns
    st.sidebar.write("Dataset columns:", list(data.columns))

    return data

# Sidebar input sliders
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave_points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave_points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave_points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        if key in data.columns:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
        else:
            st.sidebar.warning(f"⚠️ Column '{key}' missing! Check dataset.")
            input_dict[key] = 0  # Assigning default value to prevent errors

    return input_dict

# Function to scale values for radar chart
def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        if key in X.columns:
            max_val = X[key].max()
            min_val = X[key].min()
            scaled_value = (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
            scaled_dict[key] = scaled_value
        else:
            scaled_dict[key] = 0  # Handle missing column gracefully

    return scaled_dict

# Utility function for safely fetching values
def get_value(input_data, key):
    return input_data.get(key, 0)

# Generate Radar Chart
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ["Radius", "Texture", "Perimeter", "Area",
                  "Smoothness", "Compactness", "Concavity","Concave points",
                   "Symmetry", "Fractal Dimension"]

    fig = go.Figure()

    # Mean Values
    fig.add_trace(go.Scatterpolar(
        r=[get_value(input_data, key) for key in [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean","concave_points_mean","symmetry_mean", "fractal_dimension_mean"]
          ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    # Standard Error Values
    fig.add_trace(go.Scatterpolar(
        r=[get_value(input_data, key) for key in [
            "radius_se", "texture_se", "perimeter_se", "area_se",
            "smoothness_se", "compactness_se", "concavity_se","concave_points_se","symmetry_se", "fractal_dimension_se"]],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    # Worst Values
    fig.add_trace(go.Scatterpolar(
        r=[get_value(input_data, key) for key in [
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst","concave_points_worst","symmetry_worst", "fractal_dimension_worst"]],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])  # Normalize values
        ),
        showlegend=True
    )

    return fig

# Add Predictions
def add_predictions(input_data):
    # Load model and scaler correctly
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    # Proper NumPy array transformation
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.write("The Cluster is:")
    if prediction[0] ==0:
        st.write("<span class='diagnosis benign'>Benign</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis Malicious'>Malicious</span>",unsafe_allow_html=True)

    st.write("Probability of being benign:",model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Malicious:",model.predict_proba(input_array_scaled)[0][1])
    st.write("(This app can assist medical professional in making a dignosis,but should not be used as a sustitue for a professiional diagnosis.)")
    st.write(input_array_scaled)


    # Scale input data
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)
    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

    # Display prediction in the main view
    st.subheader(f"🔬 Predicted Diagnosis: **{diagnosis}**")

    # Display processed NumPy array in the sidebar for debugging
    st.sidebar.subheader("📊 Processed Input Array:")
    st.sidebar.write(input_scaled)

# Main function
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔬 Breast Cancer Predictor")
    st.write(
        "This app predicts whether a breast mass is benign or malignant based on cytology lab measurements. "
        "Adjust the measurement sliders in the sidebar to see predictions."
    )

    with open("assests/style.css") as f:
     st.markdown("<style>{}</style>".format(f.read()),unsafe_allow_html=True)

    input_data = add_sidebar()
    radar_chart = get_radar_chart(input_data)

    st.plotly_chart(radar_chart, use_container_width=True)

    add_predictions(input_data)

if __name__ == '__main__':
    main()
