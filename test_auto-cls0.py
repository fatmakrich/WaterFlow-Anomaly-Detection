import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Define file paths
model_path = "autoencoder_models/autoencoder_cluster_0.h5"
csv_path = "clusters_Anomaly/cluster_0.csv"
output_csv_path = "detected_anomalies_cluster_0.csv"

# Load the pre-trained autoencoder model
print("Loading the autoencoder model...")
autoencoder = load_model(model_path, compile=False)  # Use compile=False if loss causes issues

# Load the CSV data
print("Loading the CSV data...")
df = pd.read_csv(csv_path)

# Drop the 'anomalie' and 'type' columns if they exist
if 'anomalie' in df.columns and 'type' in df.columns:
    df = df.drop(columns=['anomalie', 'type'])

# Normalize the 'Consommation' column
print("Normalizing data...")
scaler = MinMaxScaler()
df['Consommation_scaled'] = scaler.fit_transform(df[['Consommation']])

# Prepare the data for prediction
X = df['Consommation_scaled'].values.reshape(-1, 1)

# Get reconstruction errors from the autoencoder
print("Calculating reconstruction errors...")
X_reconstructed = autoencoder.predict(X)
reconstruction_errors = np.mean(np.square(X - X_reconstructed), axis=1)

# Add reconstruction errors to the dataframe
df['Reconstruction_Error'] = reconstruction_errors

# Define an anomaly detection threshold (you can fine-tune this value)
threshold = df['Reconstruction_Error'].quantile(0.99)  # 99th percentile
print(f"Using threshold for anomaly detection: {threshold}")

# Flag anomalies
df['Anomaly'] = df['Reconstruction_Error'] > threshold

# Save anomalies to a new CSV
anomalies = df[df['Anomaly'] == True]
print(f"Number of anomalies detected: {len(anomalies)}")

# Remove the scaled and intermediate columns
anomalies = anomalies.drop(columns=['Consommation_scaled', 'Reconstruction_Error', 'Anomaly'])
anomalies.to_csv(output_csv_path, index=False)

print(f"Anomalies saved to {output_csv_path}")
