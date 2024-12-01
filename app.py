import gradio as gr
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Charger le modèle entraîné
model = load_model('modele_consommation_electricite_avec_plus_de_couches.keras')

# Charger et préparer les données de prévisions météorologiques
df_meteo_previsions = pd.read_csv('meteo_previsions.csv')
df_meteo_previsions['time'] = pd.to_datetime(df_meteo_previsions['time'])
df_meteo_previsions.rename(columns={'temperature_2m (°C)': 'temperature'}, inplace=True)

# Normaliser les données de température
scaler_X = MinMaxScaler()
df_meteo_previsions['temperature_scaled'] = scaler_X.fit_transform(df_meteo_previsions[['temperature']])

# Fonction de prédiction
def predict_consumption():
    sequence_length = 48  # Utiliser les 48 dernières heures pour prédire les 72 prochaines
    # Vérifier qu'il y a suffisamment de données
    if len(df_meteo_previsions) < sequence_length + 72:
        return "Pas assez de données météorologiques pour effectuer la prédiction."
    
    # Préparer les données pour le modèle
    X_future = []
    for i in range(sequence_length, sequence_length + 72):
        seq = df_meteo_previsions['temperature_scaled'].values[i-sequence_length:i]
        X_future.append(seq)
    
    X_future = np.array(X_future)
    X_future = X_future.reshape((X_future.shape[0], X_future.shape[1], 1))  # (samples, timesteps, features)
    
    # Faire les prédictions
    y_pred_scaled = model.predict(X_future)
    
    # Revenir à l'échelle originale
    y_pred = scaler_X.inverse_transform(y_pred_scaled)
    
    # Créer une plage de dates pour les prochaines 72 heures
    last_time = df_meteo_previsions['time'].iloc[sequence_length + 72 -1]
    future_times = [last_time + timedelta(hours=i+1) for i in range(72)]
    
    # Créer un DataFrame pour les prédictions
    df_future = pd.DataFrame({
        'time': future_times,
        'consommation_prévue': y_pred.flatten()
    })
    
    # Visualiser les prédictions
    plt.figure(figsize=(10,5))
    plt.plot(df_future['time'], df_future['consommation_prévue'], marker='o')
    plt.title('Prédiction de la Consommation d\'Électricité pour les 72 Prochaines Heures')
    plt.xlabel('Date et Heure')
    plt.ylabel('Consommation (MW)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig('prediction_plot.png')
    plt.close()
    
    return df_future, 'prediction_plot.png'

# Définir l'interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Prédiction de la Consommation d'Électricité en France pour les Prochaines 72 Heures")
    
    with gr.Row():
        with gr.Column():
            btn_predict = gr.Button("Lancer la Prédiction")
            output_table = gr.Dataframe(label="Consommation Prévue (MW)")
        with gr.Column():
            output_plot = gr.Image(label="Graphique des Prédictions")
    
    btn_predict.click(
        fn=predict_consumption,
        inputs=None,
        outputs=[output_table, output_plot]
    )

# Lancer l'application
demo.launch()
