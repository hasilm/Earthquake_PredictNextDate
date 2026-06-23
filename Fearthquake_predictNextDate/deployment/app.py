import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from train import predict_pure_spatial_timeline

Folder_name="Fearthquake_predictNextDate"
HF_username="hasilm1"
App_name="earthquake_predictNextDate"

Model_name1="earthquake_predict_date_model_v1.joblib"
Model_name2="earthquake_predict_mag_model_v1.joblib"

# Download and load the model
model_path = hf_hub_download(repo_id=str(HF_username)+"/"+str(App_name), filename=Model_name1) # enter the Hugging Face username here
model_days = joblib.load(model_path)
model_path = hf_hub_download(repo_id=str(HF_username)+"/"+str(App_name), filename=Model_name2) # enter the Hugging Face username here
model_mag = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title(str(App_name)+" App")
st.write("""
This application predicts the data set provided.
Please enter the data below to get a prediction.
""")

# User inputs
latitude = st.number_input("latitude", min_value=-90.0, max_value=+90.0, value=34.05)
longitude = st.number_input("longitude", min_value=-180.0, max_value=+180.0, value=-118.24)

# Execute pure spatial forecast
#print(predict_next_quake_date(34.05, -118.24, "2026-06-18"))
#34.239°N 25.124°E
#40.911°N 47.761°E
#42.360°N 126.573°W
spatial_kmeans = KMeans(n_clusters=25, random_state=42, n_init='auto')

predict_pure_spatial_timeline(
    input_lat=latitude,
    input_lon=longitude,
    historical_grid_map=grid_map_spatial,
    kmeans_obj=spatial_kmeans,
    model_time=model_days,
    model_intensity=model_mag,
    steps=3
)

# Prediction button
if st.button("Predict "):
    det=predict_pure_spatial_timeline(
    input_lat=latitude,
    input_lon=longitude,
    historical_grid_map=grid_map_spatial,
    kmeans_obj=spatial_kmeans,
    model_time=model_days,
    model_intensity=model_mag,
    steps=3)
    
    st.subheader("Prediction Result:")
    st.success(f"The model predict: **{det}**")
