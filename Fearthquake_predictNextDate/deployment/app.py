import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from sklearn.cluster import KMeans
from xgboost import XGBRegressor

# Now you can safely import your function
#from model_building.train import predict_pure_spatial_timeline

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
st.title(str(App_name)+" App [beta version]")
st.write("""
This application predicts the data set provided.
Please enter the data below to get a prediction.
""")

# User inputs
latitude = st.number_input("latitude", min_value=-90.0, max_value=+90.0, value=34.05)
longitude = st.number_input("longitude", min_value=-180.0, max_value=+180.0, value=-118.24)

df_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/df.csv"                      # enter the Hugging Face username here
df = pd.read_csv(df_path)


# 3. Calculate Local Spatial Density Pattern
# Counts how many total historical events share this exact geometric cluster assignment
cluster_density = df['spatial_cluster_id'].value_counts().to_dict()
df['spatial_cluster_density'] = df['spatial_cluster_id'].map(cluster_density)

# Ensure data is sorted chronologically within its geographical grid framework
df['grid_id'] = df['latitude'].round(1).astype(str) + "_" + df['longitude'].round(1).astype(str)
df = df.sort_values(['grid_id', 'time']).reset_index(drop=True)

grid_map_spatial = df.sort_values('time').groupby('grid_id').last().reset_index()
pure_spatial_features = [
    'latitude', 'longitude', 
    'spatial_cluster_id', 'distance_to_center', 'spatial_cluster_density',
    'days_since_last_local', 'avg_local_gap', 'gap_acceleration' # Core sequence tracking
]
grid_map_spatial = grid_map_spatial[pure_spatial_features]

def predict_pure_spatial_timeline(input_lat, input_lon, historical_grid_map, kmeans_obj, model_time, model_intensity, steps=3):
    """
    Forecasts future event dates and magnitudes using purely spatial geometric patterns.
    """
    # 1. Match Input Coordinates to the Global Geometric Pattern
    input_features_base = pd.DataFrame([{'latitude': float(input_lat), 'longitude': float(input_lon)}])
    pred_cluster = int(kmeans_obj.predict(input_features_base[['latitude', 'longitude']])[0])

    centroid = kmeans_obj.cluster_centers_[pred_cluster]
    dist_to_center = float(np.sqrt((float(input_lat) - centroid[0])**2 + (float(input_lon) - centroid[1])**2))
    cluster_density_val = float((historical_grid_map['spatial_cluster_id'] == pred_cluster).sum())

    # 2. Haversine lookup for the nearest local sequence history
    lat1, lon1 = np.radians(historical_grid_map['latitude']), np.radians(historical_grid_map['longitude'])
    lat2, lon2 = np.radians(float(input_lat)), np.radians(float(input_lon))
    closest_idx = (2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2))).idxmin()

    current_state = historical_grid_map.iloc[closest_idx].to_dict()
    # 1. Force the date column to exist and be a clean string column initially
    df['time'] = df['time'].astype(str)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

# 2. Rebuild the grid map while explicitly retaining the string date
    grid_map_spatial = df.sort_values('time').groupby('grid_id').last().reset_index()
    #below newly added,Hasil
    current_state['time']=df.sort_values('time').groupby('grid_id').last().reset_index()['time'].iloc[-1]

# 3. Inside your forecasting function, safely parse that string column:
    current_date_ts = pd.to_datetime(df['time'])

    #current_date_ts=current_date_ts.strftime('%Y-%m-%d')
    #print("date1:",current_date_ts)

    from datetime import date
    try:
        current_date_ts = pd.to_datetime(current_state['time']).tz_localize(None) if pd.to_datetime(current_state['time']).tz is not None else pd.Timestamp(current_state['time'])
        current_date_ts = pd.Timestamp(date.today())
    except Exception as m:

        current_date_ts = pd.Timestamp(date.today())

    # Extract baseline time sequences
    last_gap = float(current_state.get('days_since_last_local', 1))
    prev_gap = last_gap - float(current_state.get('gap_acceleration', 0))
    avg_local_gap = float(current_state.get('avg_local_gap', 1))

    print(f"=== Pure Spatial Pattern Forecast for ({input_lat}, {input_lon}) ===")
    print(f"Assigned Spatial Cluster Profile: ID {pred_cluster} (Density: {int(cluster_density_val)} historical reference points)")
    print(f"Distance to Cluster Core Centroid: {dist_to_center:.4f} degrees")
    print(f"Baseline Event Reference Date:   {current_date_ts}\n")

    # 3. Pure Spatial Recursive Simulation Loop
    for i in range(1, steps + 1):
        input_features = pd.DataFrame([{
            'latitude': float(input_lat),
            'longitude': float(input_lon),
            'spatial_cluster_id': pred_cluster,
            'distance_to_center': dist_to_center,
            'spatial_cluster_density': cluster_density_val,
            'days_since_last_local': float(last_gap),
            'avg_local_gap': float(avg_local_gap),
            'gap_acceleration': float(last_gap - prev_gap)
        }])

        # Predict Timing Sequence step
        pred_log = model_time.predict(input_features)
        pred_days = int(max(1, np.expm1(np.clip(pred_log.item(), 0, 7.5))))

                # Ensure current_date_ts is a single, timezone-naive scalar Timestamp
        if hasattr(current_date_ts, 'iloc'):
            current_date_ts = current_date_ts.iloc[0]

        current_date_ts = pd.Timestamp(current_date_ts)
        if current_date_ts.tz is not None:
            current_date_ts = current_date_ts.tz_localize(None)

        # Create a matching naive max pandas date boundary
        max_pandas_date = pd.Timestamp.max.tz_localize(None) - pd.Timedelta(days=1)

        # FIX: Explicit mathematical subtraction between two clean scalar Timestamps
        days_rem = int((max_pandas_date - current_date_ts).days)

        # Limit predictions to protect boundaries
        safe_days = int(min(pred_days, days_rem))

        # Datetime protection limits
        #days_rem = int((pd.Timestamp.max.tz_localize(None) - pd.Timedelta(days=1) - current_date_ts).days)
        #safe_days = min(pred_days, days_rem)
        future_date = current_date_ts + pd.Timedelta(days=safe_days)

        # Predict Magnitude strictly using the spatial pattern identity
        pred_mag = float(model_intensity.predict(input_features).item())

        print(f"Forecasted Event #{i}:")
        print(f"  -> Date: {future_date.strftime('%Y-%m-%d')} (Interval: {safe_days} days)")
        print(f"  -> Predicted Intensity: {pred_mag:.2f}")
        print()

        # 4. Update Time Sequences for Next Iteration Step
        current_date_ts = future_date
        prev_gap = last_gap
        last_gap = float(safe_days)
        avg_local_gap = (avg_local_gap * 2 + safe_days) / 3

# Execute pure spatial forecast
#print(predict_next_quake_date(34.05, -118.24, "2026-06-18"))
#34.239°N 25.124°E
#40.911°N 47.761°E
#42.360°N 126.573°W
  
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
