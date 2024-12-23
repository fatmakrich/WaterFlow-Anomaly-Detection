import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
model_path = "autoencoder_models/autoencoder_cluster_0.h5"
csv_path = "clusters_Anomaly/cluster_0_anomaly.csv"

# Load the pre-trained autoencoder model
print("Loading the autoencoder model...")
autoencoder = load_model(model_path, compile=False)  # Use compile=False if loss causes issues

# Load the CSV data
print("Loading the CSV data...")
df = pd.read_csv(csv_path)

# Drop the 'type' column if it exists (keep 'anomalie' for ground truth)
if 'type' in df.columns:
    df = df.drop(columns=['type'])

# Ensure 'anomalie' column is present for ground truth
if 'anomalie' not in df.columns:
    raise ValueError("The 'anomalie' column is required for evaluation but is missing.")

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

# Define an anomaly detection threshold
mean_error = df['Reconstruction_Error'].mean()
std_error = df['Reconstruction_Error'].std()
threshold = mean_error + 3 * std_error  # Trois écarts-types au-dessus de la moyenne

print(f"Using threshold for anomaly detection: {threshold}")

# Flag anomalies
df['Predicted_Anomaly'] = df['Reconstruction_Error'] > threshold

# Generate confusion matrix and normalize it
y_true = df['anomalie']  # Ground truth (0: Normal, 1: Anomaly)
y_pred = df['Predicted_Anomaly'].astype(int)  # Predictions (convert bool to int)

conf_matrix = confusion_matrix(y_true, y_pred)
conf_matrix_percentage = conf_matrix / conf_matrix.sum() * 100  # Normalize to percentages

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_percentage,
    annot=True,                # Show values inside the cells
    fmt=".2f",                 # Format values to 2 decimal places
    cmap="Blues",              # Use a blue colormap
    xticklabels=["Normal", "Anomalie"],  # X-axis labels
    yticklabels=["Normal", "Anomalie"]   # Y-axis labels
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (in %)")
plt.show()

# Print classification report
class_report = classification_report(y_true, y_pred)
print("\nClassification Report:")
print(class_report)

# Visualize reconstruction error distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Reconstruction_Error'], bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Distribution des erreurs de reconstruction")
plt.xlabel("Erreur de reconstruction")
plt.ylabel("Fréquence")
plt.legend()
plt.show()
