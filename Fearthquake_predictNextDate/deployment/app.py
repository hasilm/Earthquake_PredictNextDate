import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import numpy as np

# Now you can safely import your function
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

latitude=0.0
longitude=0.0

import os
from geopy.geocoders import Nominatim

# 1. App Titles and Description
#st.title("earthquake_predictNextDate App [beta version]")

st.markdown("""
    <style>
    /* 1. Makes the main question/label bold */
    div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] p {
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    /* 2. Forces the option choices to remain normal (non-bold) weight */
    div[data-testid="stRadio"] div[role="radiogroup"] label p {
        font-weight: normal !important;
        font-size: 14px !important;
    }
    /* 1. Makes the field label text above the box bold */
    div[data-testid="stTextInput"] label[data-testid="stWidgetLabel"] p {
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    /* 2. Forces the user-typed input text inside the box to be non-bold */
    div[data-testid="stTextInput"] input {
        font-weight: normal !important;
        font-size: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title with an Emoji Anchor
st.title("🌍 Earthquake Prediction App")

# Add a formatted sub-header for the version info
#st.markdown("### :orange[[Beta Version 2.0]]")
st.markdown("<p style='text-align: right; font-size: 14px;'>[ Beta Version 2.0 ]</p>", unsafe_allow_html=True)
            
st.divider() # Adds a clean horizontal line under the title block
st.markdown("<i>This application predicts earthquakes. Please enter the data below to get a prediction.</i>", unsafe_allow_html=True)
msg_textbox = st.empty()

# 2. Fast CSV Data Loader (1,000 rows max)
@st.cache_data # Caches data so it doesn't reload and slow down on every click
def load_earthquake_data():
    csv_filename = "earthquake_data.csv"
    if not os.path.exists(csv_filename):
        return pd.DataFrame(columns=["Latitude", "Longitude", "Date"])
    try:
        return pd.read_csv(csv_filename)
    except Exception as e:
        st.error(f"CSV Load Error: {e}")
        return pd.DataFrame()

#dataset = load_earthquake_data()

# 3. Geocoding Function
def convertToLatLon(address_str):
    if not address_str or not address_str.strip():
        return "0.0", "0.0"
    geolocator = Nominatim(user_agent="my_earthquake_pipeline_app_2026")
    try:
        location = geolocator.geocode(address_str)
        if location:
            return str(location.latitude), str(location.longitude)
    except Exception as e:
        e=e+str("<br>Please try after sometime.")
        st.sidebar.error(f"Geocoding error: {e}")
    return "0.0", "0.0"

# 4. Streamlit Radio Button Component
choice = st.radio(
    "Select Input Method",
    options=["address", "latitude,longitude"],
    index=0 # Sets 'address' as the default choice
)

# 5. Dynamic Conditional Input Fields
# Streamlit automatically handles layout visibility using simple if/else logic!
if choice == "address":
    text1 = st.text_input("Enter Address", value="")
    text2 = "" # Dummy empty value to pass downstream
else:
    text1 = st.text_input("Enter Latitude", value="")
    text2 = st.text_input("Enter Longitude", value="")

msg_textbox.warning("⚠️ wait for app to load...")
        
#latitude = st.number_input("latitude", min_value=-90.0, max_value=+90.0, value=34.05)
#longitude = st.number_input("longitude", min_value=-180.0, max_value=+180.0, value=-118.24)

df_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/df.csv"                      # enter the Hugging Face username here
df = pd.read_csv(df_path)

X_spatial_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/X_spatial.csv"                      # enter the Hugging Face username here
X_spatial = pd.read_csv(X_spatial_path)

spatial_kmeans = KMeans(n_clusters=25, random_state=42, n_init='auto')
df['spatial_cluster_id'] = spatial_kmeans.fit_predict(df[['latitude', 'longitude']])

# 2. Calculate Distance to Cluster Center (Centroid)
# This lets the model know if a point is in the core of a spatial pattern or on the fringe
centroids = spatial_kmeans.cluster_centers_
df['distance_to_center'] = np.sqrt(
    (df['latitude'] - centroids[df['spatial_cluster_id'], 0])**2 + 
    (df['longitude'] - centroids[df['spatial_cluster_id'], 1])**2
)

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
    current_today_ts=""

    from datetime import date
    try:
        current_date_ts = pd.to_datetime(current_state['time']).tz_localize(None) if pd.to_datetime(current_state['time']).tz is not None else pd.Timestamp(current_state['time'])
        #current_date_ts = pd.Timestamp(date.today())
    except Exception as m:
        print(f"Error parsing date: {m}")
        current_today_ts = pd.Timestamp(date.today())

    # Convert inputs to clean, timezone-naive Pandas Timestamps
    date_a = pd.to_datetime(current_today_ts).tz_localize(None)
    date_b = pd.to_datetime(current_date_ts).tz_localize(None)

    days_since_date_a = (date_a - date_b).days
    
    # Extract baseline time sequences
    last_gap = float(current_state.get('days_since_last_local', 1))
    prev_gap = last_gap - float(current_state.get('gap_acceleration', 0))
    avg_local_gap = float(current_state.get('avg_local_gap', 1))

    print(f"=== Pure Spatial Pattern Forecast for ({input_lat}, {input_lon}) ===")
    print(f"Assigned Spatial Cluster Profile: ID {pred_cluster} (Density: {int(cluster_density_val)} historical reference points)")
    print(f"Distance to Cluster Core Centroid: {dist_to_center:.4f} degrees")
    print(f"Baseline Event Reference Date:   {current_date_ts}\n")

    future_date=""
    ret="2"
    ret_desc=""
    # 3. Pure Spatial Recursive Simulation Loop
    for i in range(1, steps + 1):
        input_features = pd.DataFrame([
            {
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
        add_days=0
        safe_days1=safe_days+int(add_days)
        future_date1 = current_date_ts + pd.Timedelta(days=safe_days1)
        future_date = current_date_ts + pd.Timedelta(days=safe_days)

        # Predict Magnitude strictly using the spatial pattern identity
        pred_mag = float(model_intensity.predict(input_features).item())

        add_mag="0.0"
        pred_mag+=float(add_mag)

        days_between = days_since_date_a

        # 4. Update Time Sequences for Next Iteration Step
        current_date_ts = future_date
        prev_gap = last_gap
        last_gap = float(safe_days)
        avg_local_gap = (avg_local_gap * 2 + safe_days) / 3

        date_a = pd.Timestamp(date.today())
        df['future_date']=future_date
        df['future_date'] = pd.to_datetime(df['future_date']).dt.tz_localize(None)
        df['days_since_date_a'] = (df['future_date'] - date_a).dt.days
        days_since_date_a = df['days_since_date_a'].iloc[0]

        print(f"===Forecasted Event #{i}:"+str(steps)+"===")
        print(f"location:", ({input_lat}, {input_lon}) )
        print(f"  -> days away from today : "+str(days_since_date_a))
        print(f"  -> Date: {future_date.strftime('%Y-%m-%d')} (Interval: {safe_days} days)")
        print(f"  -> Predicted Intensity: {pred_mag:.2f}")
        print()

        ret_desc+=str(future_date)+","+str(pred_mag)+"#"

    date_a = pd.Timestamp(date.today())
    df['future_date']=future_date
    df['future_date'] = pd.to_datetime(df['future_date']).dt.tz_localize(None)
    df['days_since_date_a'] = (df['future_date'] - date_a).dt.days
    days_since_date_a = df['days_since_date_a'].iloc[0]

    if days_since_date_a > 400:
        print(" is greater than "+str(days_since_date_a))
        ret="0"

    elif days_since_date_a < -5 :
        print(" is less than "+str(days_since_date_a))
        ret="1"

    else:
        ret="2"

    return ret,ret_desc

# Execute pure spatial forecast
#print(predict_next_quake_date(34.05, -118.24, "2026-06-18"))
#34.239°N 25.124°E
#40.911°N 47.761°E
#42.360°N 126.573°W
#34.291°N 25.107°E, greece
#6.210°S 104.608°E, lampung indonesia.6.27.2016
#1.00,121.0 ,Central Sulawesi,6/16/2026
spatial_kmeans = KMeans(n_clusters=25, random_state=42, n_init='auto')
spatial_kmeans.fit(X_spatial[['latitude', 'longitude']]) # Fit KMeans here

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
#spatial_kmeans = KMeans(n_clusters=25, random_state=42, n_init='auto')

def runModelOutput(latitude,longitude,grid_map_spatial,spatial_kmeans,model_days_spatial,model_mag_spatial):
    cnt=0
    stps=5
    det=""

    while cnt < 50:
        #print("stps:",stps)

        rr,dt=predict_pure_spatial_timeline(
            input_lat=latitude,
            input_lon=longitude,
            historical_grid_map=grid_map_spatial,
            kmeans_obj=spatial_kmeans,
            model_time=model_days_spatial,
            model_intensity=model_mag_spatial,
            steps=stps
        )
        det+=dt

        cnt+=1
        msg_textbox.warning("⚠️ wait for AI model to generate output..., takes longer for quake pron locations, "+str(cnt))
        #st.sidebar.empty()
        #st.sidebar.error(f"det: {det}")

        if rr=="2":
            break
        elif rr == "0":
            stps-=5
        elif rr == "1":
            stps+=5
    
    return det
    
import datetime
import requests

def get_earthquake_date(lat, lon, radius_km=50):

    radius_coverage_km=10

    # 1. Construct the USGS API endpoint URL
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    # 2. Define the geographic search parameters


    cnt=0
    while cnt < radius_coverage_km:
        params = {
        "format": "geojson",
        "latitude": lat,
        "longitude": lon,
        "maxradiuskm": radius_km,
        "orderby": "time"  # Puts the newest earthquake first
    }
    # 3. Request data from the API
        response = requests.get(url, params=params)
        #print(response)
        if response.status_code != 200:
            return f"Error: API returned status code {response.status_code}"

        data = response.json()
        features = data.get("features", [])

        if not features:
            ttt="No earthquakes found within radius:"+str(radius_km)+" km"
            st.markdown("<p style='text-align: left; font-size: 14px;'><i>"+str(ttt)+"</i></p>", unsafe_allow_html=True)

            radius_km+=50
        else:
            ttt=" earthquakes found with radius:"+str(radius_km)+" km"
            st.markdown("<p style='text-align: left; font-size: 14px;'><i>"+str(ttt)+"</i></p>", unsafe_allow_html=True)

            break

        cnt+=1

    # 4. Extract the most recent earthquake event properties
    latest_quake = features[0]["properties"]
    place = latest_quake["place"]
    magnitude = latest_quake["mag"]

    # USGS returns time in milliseconds since Unix epoch; convert to seconds
    epoch_time_ms = latest_quake["time"]
    readable_date = datetime.datetime.fromtimestamp(epoch_time_ms / 1000).strftime('%Y-%m-%d %H:%M:%S')

    return {
        "date_time": readable_date,
        "location_description": place,
        "magnitude": magnitude,
        "radius_km": radius_km
    }

def getDateCloseToOriginal(o_date,p_date):

  from datetime import datetime

  t_diff=0
  close_date=""
  close_mag=""
  d_since=0
  t_diff1=0
  close_date1=""
  close_mag1=""
  d_since1=0
  t_diff2=0
  close_date2=""
  close_mag2=""
  d_since2=0

  first_time=True
  highest_date=pd.to_datetime(o_date).date()
  future_dates_cnt=0

  for inn in p_date.split("#"):
    if inn == "":
      continue

    t_date=inn.split(",")[0]
    mag=inn.split(",")[1]

    clean_date = pd.to_datetime(t_date).date()
    o_date = pd.to_datetime(o_date).date()
    days_since = (clean_date - o_date).days
    
    #print(o_date,days_since,clean_date,t_diff,t_diff1,t_diff2)
    if int(abs(t_diff)) > int(abs(days_since)) or first_time :

        t_diff2=abs(t_diff1)
        close_date2=close_date1
        close_mag2=close_mag1
        d_since2=d_since1

        t_diff1=abs(t_diff)
        close_date1=close_date
        close_mag1=close_mag
        d_since1=d_since

        t_diff=abs(days_since)
        close_date=t_date
        close_mag=mag
        d_since=days_since

        
        a = datetime.strptime(str(highest_date), "%Y-%m-%d")
        b = datetime.strptime(str(clean_date), "%Y-%m-%d")
        if b > a:
            highest_date=clean_date

    elif  int(abs(t_diff1)) > int(abs(days_since)) or first_time :
        t_diff2=abs(t_diff1)
        close_date2=close_date1
        close_mag2=close_mag1
        d_since2=d_since1

        t_diff1=abs(days_since)
        close_date1=t_date
        close_mag1=mag
        d_since1=days_since
        
        a = datetime.strptime(str(highest_date), "%Y-%m-%d")
        b = datetime.strptime(str(clean_date), "%Y-%m-%d")
        if b > a:
            highest_date=clean_date
            
    elif  int(abs(t_diff2)) > int(abs(days_since)) or first_time :
        t_diff2=abs(days_since)
        close_date2=t_date
        close_mag2=mag
        d_since2=days_since
        
        a = datetime.strptime(str(highest_date), "%Y-%m-%d")
        b = datetime.strptime(str(clean_date), "%Y-%m-%d")
        if b > a:
            highest_date=clean_date

    first_time=False

  #print("HD:",highest_date)
  ret="HD:"+str(highest_date) 
  st.sidebar.error(f"det: {ret}")
    
  future_dates=""
  future_dates_cnt=0
  for inn in p_date.split("#"):
    if inn == "":
      continue

    t_date=inn.split(",")[0]
    mag=inn.split(",")[1]

    clean_date = pd.to_datetime(t_date).date()
  
    #print("compare:",highest_date,clean_date)
  
    a = datetime.strptime(str(highest_date), "%Y-%m-%d")
    b = datetime.strptime(str(clean_date), "%Y-%m-%d")
    if a < b:
 
        if int(future_dates_cnt) >= 1:
            break
        future_dates+=str(clean_date)+","+str(mag)+","

        future_dates_cnt+=1
   
  f_date = pd.Timestamp(datetime.today())
  f_date = f_date.date()
  ret="f_HD:"+str(f_date) 
  st.sidebar.error(f"det: {ret}")

  f_dates=""
  future_dates_cnt=0
  for inn in p_date.split("#"):
    if inn == "":
      continue

    t_date=inn.split(",")[0]
    mag=inn.split(",")[1]

    clean_date = pd.to_datetime(t_date).date()
  
    #print("compare:",f_date,clean_date)
  
    a = datetime.strptime(str(f_date), "%Y-%m-%d")
    b = datetime.strptime(str(clean_date), "%Y-%m-%d")
    if a < b:

        if int(future_dates_cnt) >= 1:
            break
        f_dates+=str(clean_date)+","+str(mag)+","

        future_dates_cnt+=1


  pre_dates=str(close_date)+","+str(close_mag)+","+str(d_since)+","+str(close_date1)+","+str(close_mag1)+","+str(d_since1)+","+str(close_date2)+","+str(close_mag2)+","+str(d_since2)
  pre_dates+=","+str(future_dates)
  
  return pre_dates,f_dates
    
def formatOutput(lat,lon,det):
    ret=""
    address="Bandar Lampung, Indonesia"
#address = "1600 Amphitheatre Pkwy, Mountain View, CA"
#address="Central Sulawesi"
#address="12 stryker ct, bridgewater,nj,usa"
#address="North Sulawesi"
#address="Changning, China"
#address="Catuday, Philippines"
    quake_info = get_earthquake_date(lat, lon)
    #print(quake_info)

    dt=quake_info['date_time']
    mag=quake_info['magnitude']
    add=quake_info['location_description']
    rad=quake_info['radius_km']

    original_date = pd.to_datetime(dt).date()

    ret="###################################################################################"
    st.markdown("<span style='font-size: 10px;'>"+str(ret)+"</span>", unsafe_allow_html=True)
    ret="<b>RECENT EARTHQUAKE:</b>"
    st.markdown("<span style='background-color: #FF4B4B;font-size: 18px;'>"+str(ret)+"</span>", unsafe_allow_html=True)
    ret="<b><i>[which is close to the address entered]</i></b>"
    st.markdown("<p style='color: #ff0000;'text-align: left; margin-bottom: 0px;font-size: 12px;'>"+str(ret)+"</p>", unsafe_allow_html=True)

    ret="Address   : <b>"+str(add)+"</>"
    st.markdown("<p style='font-size: 14px;'>"+str(ret)+"</p>", unsafe_allow_html=True)

    ret="Date     :<b>"+str(original_date)+"</b>"
    st.markdown("<p style='font-size: 14px;'>"+str(ret)+"</p>", unsafe_allow_html=True)
    ret="Magnitude :<b>"+str(mag)+"</b>"
    st.markdown("<p style='font-size: 14px;'>"+str(ret)+"</p>", unsafe_allow_html=True)
    ret="radius :<b>"+str(rad)+"</b>"
    st.markdown("<p style='font-size: 14px;'>"+str(ret)+" km</p>", unsafe_allow_html=True)
    ret="------------------------------------------------------------------------------------"
    st.markdown("<span style='font-size: 14px;'>"+str(ret)+"</span>", unsafe_allow_html=True)

    ret="<b>AI PREDICTION:</b>"
    st.markdown("<span style='background-color: #FF4B4B;font-size: 18px;'>"+str(ret)+"</span>", unsafe_allow_html=True)

    dd,fd=getDateCloseToOriginal(original_date,det)
    ret="<b><i>AI prediction close to above earthquake:</i></b>"
    st.markdown("<p style='text-align: left; margin-bottom: 0px;font-size: 16px;'>"+str(ret)+"</p>", unsafe_allow_html=True)

    i=1
    cnt_i=1
    for rr in dd.split(","):

        if rr == "":
            continue
                    
        if i == 1:
            ret=str(cnt_i)+". Date:<b>"+str(rr)+"</b>"
            st.markdown("<p style='font-size: 13px;'>"+str(ret)+"</p>", unsafe_allow_html=True)
            cnt_i+=1
        elif i == 2:
            ret="Magnitude:<b>"+str(rr)+"</b>"
            st.markdown("<p style='font-size: 13px;'>"+str(ret)+"</p>", unsafe_allow_html=True)

        elif i == 3:
            ret="Days away from predicted:<b>"+str(rr)+"</b>"
            st.markdown("<p style='font-size: 13px;'>"+str(ret)+"</p>", unsafe_allow_html=True)

            i=0
        i+=1

    ret="   <b><i> Future AI earthquakes prediction:</i></b>"
    st.markdown("<p style='text-align: left; margin-bottom: 0px;font-size: 16px;'>"+str(ret)+"</p>", unsafe_allow_html=True)
    i=1
    cnt_i=1
    for rr in fd.split(","):
        #ret=str(rr)
        #st.markdown("<p style='font-size: 10px;'><b>"+str(ret)+"</b></p>", unsafe_allow_html=True)
   
        if rr == "":
            continue
                    
        if i == 1:
            ret=str(cnt_i)+". Date:<large><b>"+str(rr)+"</b></large>"
            st.markdown("<p style='font-size: 13px;'>"+str(ret)+"</p>", unsafe_allow_html=True)
            cnt_i+=1
        elif i == 2:
            ret="Magnitude:<large><b>"+str(rr)+"</b></large>"
            st.markdown("<p style='font-size: 13px;'>"+str(ret)+"</p>", unsafe_allow_html=True)
            i=0 
        i+=1
        
    ret="###################################################################################"
    st.markdown("<p style='font-size: 10px;'>"+str(ret)+"</p>", unsafe_allow_html=True)
    
msg_textbox.warning("")

# 6. Action Submission Button
if st.button("click to predict"):
 
    lat=0.0
    longi=0.0
    if choice == "address":
        if not text1.strip():
            st.warning("Please type a valid address before submitting.")
        else:
            with st.spinner("Geocoding address..."):
                latitude, longitude = convertToLatLon(text1)
            ttt=" Address Entered :<b> "+str(text1) +"</b>"
            st.markdown("<p style=' font-size: 14px;'>"+str(ttt)+"</p>", unsafe_allow_html=True)
            ttt=" Lat/Lon         : <b>"+str(latitude)+","+str(longitude)+"</b>"
            st.markdown("<p style=' font-size: 14px;'>"+str(ttt)+"</p>", unsafe_allow_html=True)
            ttt=" Dataset Loaded   : <b>"+str(len(df))+"</b>"
            st.markdown("<p style=' font-size: 14px;'>"+str(ttt)+"</p>", unsafe_allow_html=True)
                    
            lat=latitude
            longi=longitude
    else:
        if not text1.strip() or not text2.strip():
            st.warning("Please enter both Latitude and Longitude values.")
        else:
            lat=text1
            longi=text2

            st.write(f" Latitude       : {text1}")
            st.write(f" Longitude      : {text2}")
            st.write(f" Dataset Loaded : {len(dataset)}")

    msg_textbox.warning("⚠️ wait for AI model to generate output..., takes longer for quake pron locations")
    det=runModelOutput(latitude,longitude,grid_map_spatial,spatial_kmeans,model_days,model_mag)

    msg_textbox.warning("⚠️ loading output...")
    formatOutput(lat,longi,det)
    msg_textbox.warning("")

