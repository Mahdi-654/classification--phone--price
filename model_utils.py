import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X, y, model_type='random_forest'):
    """
    Fonction générale pour entraîner un modèle.
    Modèles supportés : 'random_forest', 'logistic_regression'
    
    :param X: Les données d'entraînement (features)
    :param y: Les étiquettes cibles
    :param model_type: Le type de modèle à entraîner ('random_forest' ou 'logistic_regression')
    :return: Le modèle entraîné, la précision, le rapport de classification, la matrice de confusion
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sélection du modèle en fonction de l'option choisie
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Modèle {model_type} non supporté")

    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Prédictions sur les données de test
    y_pred = model.predict(X_test)
    
    # Calcul de la précision et des métriques
    accuracy = model.score(X_test, y_test)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, classification_rep, confusion_mat


def predict_price(model, user_input_df):
    """
    Prédire la plage de prix d'un téléphone en fonction de ses caractéristiques.
    
    :param model: Le modèle d'apprentissage automatique déjà entraîné.
    :param user_input_df: Un DataFrame contenant les caractéristiques du téléphone.
    
    :return: La prédiction du prix (ici, la classe prédite).
    """
    # Utilisation du modèle pour prédire la classe du prix
    prediction = model.predict(user_input_df)
    
    # Vous pouvez adapter cette partie selon votre façon de représenter les plages de prix
    # Par exemple, vous pouvez utiliser un mapping pour les classes en une plage spécifique.
    
    return prediction[0]
