from builtins import Exception, int
import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Charger les modèles
model_ann = joblib.load('modelRF.pkl')  # Modèle ANN
model_lgbm = joblib.load('modelLGBM.pkl')  # Modèle LGBM
model_xgb = joblib.load('modelXGB.pkl')  # Modèle XGBoost

# Liste des produits
produits = {
    0: "SAFIA PET 6X1.5 L",
    1: "SAFIA VER.RET 12X90 CLB",
}

@app.route('/')
def home():
    return render_template('index.html', produits=produits, prediction=None, produit_nom=None, model_name=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        produit_id = int(request.form['produit'])
        date_str = request.form['date']
        model_choice = request.form['model']  # Récupérer le choix du modèle

        # Extraire l'année, le mois et le jour de la date
        date = pd.to_datetime(date_str)
        année = date.year
        mois = date.month
        jour = date.day

        # Préparer les données pour la prédiction
        data = np.array([[année, mois, jour]])

        # Faire la prédiction selon le modèle choisi
        if model_choice == 'Random Forest':
            prediction = model_ann.predict(data)[0]  # Prédiction avec le modèle ANN
            model_name = "Random Forest"
        elif model_choice == 'LGBM':
            prediction = model_lgbm.predict(data)[0]  # Prédiction avec le modèle LGBM
            model_name = "LGBM"
        else:  # XGBoost
            prediction = model_xgb.predict(data)[0]  # Prédiction avec le modèle XGBoost
            model_name = "XGBoost"

        # Récupérer le nom du produit à partir de l'ID
        produit_nom = produits[produit_id]

        return render_template('index.html', produits=produits, prediction=prediction, produit_nom=produit_nom, model_name=model_name)
    except Exception as e:
        return render_template('index.html', produits=produits, prediction=None, produit_nom=None, model_name=None)

if __name__ == '__main__':
    app.run(debug=True)