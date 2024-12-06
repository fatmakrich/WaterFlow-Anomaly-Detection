import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Chargement des données
data = pd.read_csv('data/data_clusters_anomaly.csv')

# Filtrer pour le cluster 2 uniquement (modification ici)
data_cluster_2 = data[data['cluster'] == 2.0]  # Modification du cluster 1 en cluster 2
print(f"Nombre de lignes pour le cluster 2 : {len(data_cluster_2)}")

# Nettoyer la colonne 'anomalie' (convertir en entier)
data_cluster_2['anomalie'] = data_cluster_2['anomalie'].astype(int)

# Créer les fenêtres glissantes
def create_sliding_windows(data, window_size, step_size):
    """
    Génère des fenêtres glissantes à partir des données avec chevauchement.
    """
    X, y = [], []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data.iloc[start:end]
        # Les caractéristiques sont la consommation
        X.append(window['Consommation'].values)
        # L'étiquette est 1 si une anomalie est présente dans la fenêtre
        y.append(window['anomalie'].max())
    return np.array(X), np.array(y)

# Paramètres des fenêtres glissantes
window_size = 3  # Taille de la fenêtre (nombre de lignes par fenêtre)
step_size = window_size // 2  # Pas (chevauchement de 50 %)

# Générer les données pour le modèle
X, y = create_sliding_windows(data_cluster_2, window_size, step_size)
print(f"Nombre d'échantillons générés pour le cluster 2 : {len(X)}")

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Prédictions
y_pred = clf.predict(X_test)

# Évaluation du modèle
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
print(f"Exactitude : {accuracy_score(y_test, y_pred)}")

# Précision pour la classe 1 (anomalie)
precision_class_1 = (y_pred == 1) & (y_test == 1)
print(f"Précision pour la classe '2' (anomalie) : {precision_class_1.sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0}")
