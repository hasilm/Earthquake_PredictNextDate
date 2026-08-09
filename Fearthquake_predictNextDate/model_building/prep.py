
# for data manipulation
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

Folder_name="Fearthquake_predictNextDate"
HF_username="hasilm1"
App_name="earthquake_predictNextDate"
Column_name='latitude'
Column_name2='longitude'
Test_size=0.2
Random_state=42
data_filename="earthquakes.csv"

#DATASET_PATH = "/data/"+str(data_filename)                  # enter the Hugging Face username here
DATASET_PATH = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/"+str(data_filename)                  # enter the Hugging Face username here

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
#df = df.drop(columns=['id'])

# Convert time column to datetime format
#df['time'] = pd.to_datetime(df['time'])
df['time'] = pd.to_datetime(df['time'], format='ISO8601')

df = df.dropna()
#################3
# Round to the nearest 0.5 or 1.0 degree to create spatial bins
df['lat_bin'] = df['latitude'].round(1)
df['lon_bin'] = df['longitude'].round(1)

# Group by the bins instead of exact coordinates when calculating intervals
df['next_quake_time'] = df.groupby(['lat_bin', 'lon_bin'])['time'].shift(-1)
##################################
# Sort by location and time to track seismic history chronologically
df = df.sort_values(by=['latitude', 'longitude', 'time','mag','depth']).reset_index(drop=True)

#id,time,year,month,day_of_year,hour,latitude,longitude,depth,mag
#year,month,day_of_year,hour,longitude,depth,mag
target_col = ['latitude', 'longitude']
# 1. Target Variable: Days until the NEXT event
# Round coordinates to create regional grid cells
df['lat_grid'] = df['latitude'].round(1)
df['lon_grid'] = df['longitude'].round(1)
df['grid_id'] = df['lat_grid'].astype(str) + "_" + df['lon_grid'].astype(str)

# Sort chronologically WITHIN each regional grid
df = df.sort_values(['grid_id', 'time']).reset_index(drop=True)

# Calculate time gaps within each specific region
df['days_since_last_local'] = df.groupby('grid_id')['time'].diff().dt.days

# Calculate a rolling average of gaps in that region to give the model a baseline
df['avg_local_gap'] = df.groupby('grid_id')['days_since_last_local'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Drop the first few events per grid that won't have historical data
df = df.dropna(subset=['days_since_last_local', 'avg_local_gap']).reset_index(drop=True)
# Target: Days until the next event IN THIS GRID
df['days_until_next_local'] = df.groupby('grid_id')['time'].shift(-1) - df['time']
df['days_until_next_local'] = df['days_until_next_local'].dt.days

# Magnitude of the immediate past event in this grid
df['last_mag'] = df.groupby('grid_id')['mag'].shift(0)

# Rolling average of past event magnitudes
df['rolling_avg_mag'] = df.groupby('grid_id')['mag'].transform(lambda x: x.rolling(3, min_periods=1).mean())
# The gap before the current one
df['prev_days_since_last'] = df.groupby('grid_id')['days_since_last_local'].shift(1)

# Acceleration: Did the quiet period expand or shrink?
df['gap_acceleration'] = df['days_since_last_local'] - df['prev_days_since_last']
# Count global events within a rolling 30-day window based on the datetime index
df = df.sort_values('time')
df['global_30_day_count'] = df.rolling('30D', on='time')['grid_id'].count()
df = df.sort_values(['grid_id', 'time']).reset_index(drop=True)
df['gap_to_avg_ratio'] = df['days_since_last_local'] / (df['avg_local_gap'] + 1)

# Drop rows that don't have enough historical lag depth
df = df.dropna(subset=['last_mag', 'gap_acceleration', 'days_until_next_local']).reset_index(drop=True)

# Expanded feature matrix
X = df[[
    'latitude', 'longitude', 
    'days_since_last_local', 'avg_local_gap', 'gap_acceleration',
    'last_mag', 'rolling_avg_mag',
    'global_30_day_count',
    'gap_to_avg_ratio'
]]

y = df['days_until_next_local']

# Use your optimized feature matrix and split your target
X_mag = df[[
    'latitude', 'longitude', 
    'days_since_last_local', 'avg_local_gap', 'gap_acceleration',
    'last_mag', 'rolling_avg_mag', 'global_30_day_count', 'gap_to_avg_ratio'
]]
y_mag = df['mag'] # The target is raw magnitude

# 1. Generate Geometric Cluster IDs
# 25 clusters provides a strong balance for mapping spatial groupings across 8,589 rows
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


# Pure spatial feature matrix
pure_spatial_features = [
    'latitude', 'longitude', 
    'spatial_cluster_id', 'distance_to_center', 'spatial_cluster_density',
    'days_since_last_local', 'avg_local_gap', 'gap_acceleration' # Core sequence tracking
]

X_spatial = df[pure_spatial_features]
y_days = np.log1p(df['days_until_next_local'])
y_mag = df['mag']

#only days prediction data is split with training/test
Xtrain, Xtest, ytrain, ytest = train_test_split(X_spatial, y_days, test_size=0.2, random_state=42)

X_spatial.to_csv("X_spatial.csv",index=False)
y_days.to_csv("y_days.csv",index=False)
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)
y_mag.to_csv("ymag.csv",index=False)
df.to_csv("df.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv","ymag.csv","X_spatial.csv","y_days.csv","df.csv"]


for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=str(HF_username)+"/"+str(App_name),                                           # enter the Hugging Face username here
        repo_type="dataset",
    )
