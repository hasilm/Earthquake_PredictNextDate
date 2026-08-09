import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib
import os
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
 
from xgboost import XGBRegressor

from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()
HF_username="hasilm1"
App_name="earthquake_predictNextDate"

Xtrain_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/Xtrain.csv"                    # enter the Hugging Face username here
Xtest_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/Xtest.csv"                      # enter the Hugging Face username here
ytrain_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/ytrain.csv"                    # enter the Hugging Face username here
ytest_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/ytest.csv"                      # enter the Hugging Face username here
ymag_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/ymag.csv"                      # enter the Hugging Face username here

X_spatial_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/X_spatial.csv"                    # enter the Hugging Face username here
y_days_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/y_days.csv"                      # enter the Hugging Face username here
df_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/df.csv"                      # enter the Hugging Face username here

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
X_spatial = pd.read_csv(X_spatial_path)

y_days = pd.read_csv(y_days_path)
ytest = pd.read_csv(ytest_path)
ytrain = pd.read_csv(ytrain_path)
get_df = pd.read_csv(df_path)

y_mag = pd.read_csv(ymag_path)
df=get_df

# Train Spatial Timeline Model
#Normal model
model_days_spatial = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.03, random_state=42)
model_days_spatial.fit(X_spatial, y_days)

# Train Spatial Magnitude Model
model_mag_spatial = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, random_state=42)
model_mag_spatial.fit(X_spatial, y_mag)

#Random Search
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
            'max_depth':[2,6],
            'learning_rate': [0.01, 0.03, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
           'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

search = RandomizedSearchCV(
    estimator=model_days_spatial,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
    )

search.fit(X_spatial, y_days)
model_days_rs_spatial = search.best_estimator_

model_mag_rs_spatial = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, random_state=42)
search = RandomizedSearchCV(
    estimator=model_mag_spatial,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
    )
search.fit(X_spatial, y_mag)
model_mag_rs_spatial = search.best_estimator_
        
#Random Forest
from sklearn.ensemble import RandomForestRegressor

#ymag_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/ymag.csv"                      # enter the Hugging Face username here
#X_spatial_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/X_spatial.csv"                    # enter the Hugging Face username here
#y_days_path = "hf://datasets/"+str(HF_username)+"/"+str(App_name)+"/y_days.csv"                      # enter the Hugging Face username here

#y_mag = pd.read_csv(ymag_path) 
#X_spatial = pd.read_csv(X_spatial_path)
#y_days = pd.read_csv(y_days_path)

model_days_rf_spatial = RandomForestRegressor(random_state=42, n_jobs=-1)
param_distributions = {
            'n_estimators': [100, 200,400, 500],
            'max_depth': [12, 16, 18],
            'min_samples_split': [2,5,10],
            'min_samples_leaf': [1,2,4],
            'max_features': ['sqrt', 'log2'] # Limits features per split to avoid spatial dominance
}
rf_search = RandomizedSearchCV(
            estimator=model_days_rf_spatial, 
            param_distributions=param_distributions, 
            n_iter=10, 
            cv=3, 
            random_state=42, 
            n_jobs=-1,
            scoring='neg_mean_squared_error'
)
        
rf_search.fit(X_spatial, y_days.values.ravel())
model_days_rf_spatial = rf_search.best_estimator_

param_distributions = {
            'max_depth':[2,6],
            'learning_rate': [0.01, 0.03, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
           'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}

model_mag_rf_spatial = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(
            estimator=model_mag_rf_spatial,
            param_distributions=param_distributions,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
)
search.fit(X_spatial, y_mag)
model_mag_rf_spatial = search.best_estimator_

# Save the final reference map snapshot
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
    #grid_map_spatial = df.sort_values('time').groupby('grid_id').last().reset_index()
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

    ret=""
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

        ret=""
        print(f"Forecasted Event #{i}:")
        print(f"  -> Date: {future_date.strftime('%Y-%m-%d')} (Interval: {safe_days} days)")
        print(f"  -> Predicted Intensity: {pred_mag:.2f}")
        print()
        ret+=" Forcasted event :"+str(i)     
        ret+="  -> Date: "+future_date.strftime('%Y-%m-%d')+" (Interval: "+safe_days+" days"
        ret += "  -> Predicted Intensity: "+pred_mag+" \n"
     
        # 4. Update Time Sequences for Next Iteration Step
        current_date_ts = future_date
        prev_gap = last_gap
        last_gap = float(safe_days)
        avg_local_gap = (avg_local_gap * 2 + safe_days) / 3
     
    return ret
     
# Execute pure spatial forecast
#print(predict_next_quake_date(34.05, -118.24, "2026-06-18"))
#34.239°N 25.124°E
#40.911°N 47.761°E
#42.360°N 126.573°W
#predict_pure_spatial_timeline(
#    input_lat=34.05,
#    input_lon=-118.24,
#    historical_grid_map=grid_map_spatial,
#    kmeans_obj=spatial_kmeans,
#    model_time=model_days_spatial,
#    model_intensity=model_mag_spatial,
#    steps=3
#)

#X_spatial, y_days

y_pred_train = model_days_spatial.predict(Xtrain)
y_pred_test = model_days_spatial.predict(Xtest)

# Evaluate the model accuracy
mae = mean_absolute_error(ytest, y_pred_test)
print(f"Mean Absolute Error: {mae:.2f} days")

# Evaluation
print("\nTest Evaluation Report:")

# 5. Evaluate the model
mae = mean_absolute_error(ytest, y_pred_test)
mse = mean_squared_error(ytest, y_pred_test)
r2 = r2_score(ytest, y_pred_test)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared ($R^2$) Score: {r2:.4f}")


print("\nTrain Evaluation Report:")

mae = mean_absolute_error(ytrain, y_pred_train)
mse = mean_squared_error(ytrain, y_pred_train)
r2 = r2_score(ytrain, y_pred_train)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared ($R^2$) Score: {r2:.4f}")

Model_nor_name1="earthquake_predict_date_model_v1.joblib"
Model_nor_name2="earthquake_predict_mag_model_v1.joblib"

Model_rs_name1="earthquake_predict_date_rs_model_v1.joblib"
Model_rs_name2="earthquake_predict_mag_rs_model_v1.joblib"

Model_rf_name1="earthquake_predict_date_rf_model_v1.joblib"
Model_rf_name2="earthquake_predict_mag_rf_model_v1.joblib"

# Save best model
joblib.dump(model_days_spatial, Model_nor_name1)
joblib.dump(model_mag_spatial, Model_nor_name2)

joblib.dump(model_days_rs_spatial, Model_rs_name1)
joblib.dump(model_mag_rs_spatial, Model_rs_name2)

joblib.dump(model_days_rf_spatial, Model_rf_name1)
joblib.dump(model_mag_rf_spatial, Model_rf_name2)

# Upload to Hugging Face
repo_id = str(HF_username)+"/"+str(App_name)                                         # enter the Hugging Face username here
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=Model_nor_name1,
    path_in_repo=Model_nor_name1,
    repo_id=repo_id,
    repo_type=repo_type,
)
# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=Model_nor_name2,
    path_in_repo=Model_nor_name2,
    repo_id=repo_id,
    repo_type=repo_type,
)
 # create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=Model_rs_name1,
    path_in_repo=Model_rs_name1,
    repo_id=repo_id,
    repo_type=repo_type,
)
# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=Model_rs_name2,
    path_in_repo=Model_rs_name2,
    repo_id=repo_id,
    repo_type=repo_type,
)
# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=Model_rf_name1,
    path_in_repo=Model_rf_name1,
    repo_id=repo_id,
    repo_type=repo_type,
)
# create_repo("best_machine_failure_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj=Model_rf_name2,
    path_in_repo=Model_rf_name2,
    repo_id=repo_id,
    repo_type=repo_type,
)
