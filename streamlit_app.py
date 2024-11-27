import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Charger les données avec le système de mise en cache
@st.cache_data
def load_data():
    # Chemins relatifs des fichiers CSV
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    # Charger les fichiers CSV
    train_df = pd.read_csv(train_path, sep=',')
    test_df = pd.read_csv(test_path, sep=',')
    return train_df, test_df

# Fonction de prétraitement des données
def preprocess_data(train_df, test_df):
    # Vérifier si la colonne 'id' existe avant de la supprimer dans test_df
    if 'id' in test_df.columns:
        test_df = test_df.drop(columns=["id"])

    # Vérifier si la colonne 'price_range' existe dans train_df
    if 'price_range' not in train_df.columns:
        st.error("La colonne 'price_range' est manquante dans les données d'entraînement.")
        return None, None, None

    # Encoder les variables catégorielles
    train_df = pd.get_dummies(train_df, drop_first=True)
    test_df = pd.get_dummies(test_df, drop_first=True)

    # Aligner les colonnes entre train et test
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0
    test_df = test_df[train_df.columns]

    # Séparer les caractéristiques (X) et la cible (y) pour l'entraînement
    X = train_df.drop(columns=["price_range"])
    y = train_df["price_range"]
    
    return X, y, test_df

# Entraîner le modèle Random Forest
def train_random_forest(X, y):
    # Diviser les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul de la précision et autres métriques
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, classification_rep, confusion_mat

# Entraîner le modèle de régression logistique
def train_logistic_regression(X, y):
    # Diviser les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calcul de la précision et autres métriques
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, classification_rep, confusion_mat

# Prédiction de la plage de prix pour un téléphone donné
def predict_price(model, features):
    # Assurez-vous que les colonnes sont bien alignées avec le modèle
    prediction = model.predict(features)
    return prediction[0]

# Visualiser l'importance des caractéristiques pour Random Forest
def plot_feature_importance(model):
    """
    Fonction pour afficher l'importance des caractéristiques du modèle RandomForest.
    """
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importance des Caractéristiques")
    plt.barh(range(len(indices)), feature_importances[indices], align="center")
    plt.yticks(range(len(indices)), np.array(model.feature_names_in_)[indices])
    plt.xlabel("Importance")
    st.pyplot(plt)

# Fonction principale
def main():
    # Titre de l'application
    st.set_page_config(page_title="Prédiction de Plage de Prix des Téléphones", page_icon="📱", layout="wide")
    
    # Utilisation de la barre latérale pour la navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choisissez une option", ["Prédiction", "Évaluation du modèle"])

    # Vérification si le modèle est déjà chargé dans session_state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.accuracy = None
        st.session_state.classification_rep = None
        st.session_state.confusion_mat = None

    if app_mode == "Prédiction":
        # Titre principal
        st.title("Prédiction de la Plage de Prix des Téléphones")
        st.markdown(
            """
            Cette application permet de prédire la **plage de prix** des téléphones en fonction de leurs caractéristiques.
            Sélectionnez les caractéristiques ci-dessous pour prédire la plage de prix d'un téléphone.
            """
        )
        
        # Charger les données
        train_df, test_df = load_data()

        # Prétraiter les données
        X, y, test_df_processed = preprocess_data(train_df, test_df)
        
        if X is None or y is None:
            st.stop()  # Arrêter l'exécution si les données sont incorrectes

        # Interface utilisateur pour prédire le prix d'un téléphone
        st.write("### Entrez les caractéristiques du téléphone pour prédire sa plage de prix")

        # Champs de saisie pour les caractéristiques du téléphone
        battery_power = st.number_input("Battery Power", min_value=0, help="La puissance de la batterie du téléphone.")
        blue = st.selectbox("Bluetooth (blue)", [0, 1], help="Indique si le téléphone dispose de Bluetooth.")
        clock_speed = st.number_input("Clock Speed", min_value=0.0, format="%.2f", help="Vitesse d'horloge du téléphone.")
        dual_sim = st.selectbox("Dual SIM", [0, 1], help="Indique si le téléphone est dual SIM.")
        fc = st.number_input("Front Camera", min_value=0, help="Nombre de mégapixels de la caméra frontale.")
        four_g = st.selectbox("4G", [0, 1], help="Indique si le téléphone prend en charge la 4G.")
        int_memory = st.number_input("Internal Memory", min_value=0, help="Mémoire interne du téléphone (en Go).")
        m_dep = st.number_input("Mobile Depth", min_value=0.0, format="%.2f", help="Profondeur du téléphone (en mm).")
        mobile_wt = st.number_input("Mobile Weight", min_value=0, help="Poids du téléphone (en grammes).")
        n_cores = st.number_input("Number of Cores", min_value=0, help="Nombre de cœurs du processeur du téléphone.")
        pc = st.number_input("Primary Camera", min_value=0, help="Nombre de mégapixels de la caméra principale.")
        px_height = st.number_input("Pixel Height", min_value=0, help="Hauteur de l'écran en pixels.")
        px_width = st.number_input("Pixel Width", min_value=0, help="Largeur de l'écran en pixels.")
        ram = st.number_input("RAM", min_value=0, help="Mémoire vive du téléphone (en Mo).")
        sc_h = st.number_input("Screen Height", min_value=0, help="Hauteur de l'écran du téléphone (en pouces).")
        sc_w = st.number_input("Screen Width", min_value=0, help="Largeur de l'écran du téléphone (en pouces).")
        talk_time = st.number_input("Talk Time", min_value=0, help="Autonomie en mode conversation (en heures).")
        three_g = st.selectbox("3G", [0, 1], help="Indique si le téléphone prend en charge la 3G.")
        touch_screen = st.selectbox("Touch Screen", [0, 1], help="Indique si le téléphone a un écran tactile.")
        wifi = st.selectbox("WiFi", [0, 1], help="Indique si le téléphone prend en charge le WiFi.")

        # Créer un dictionnaire avec les valeurs saisies par l'utilisateur
        user_input = {
            "battery_power": battery_power,
            "blue": blue,
            "clock_speed": clock_speed,
            "dual_sim": dual_sim,
            "fc": fc,
            "four_g": four_g,
            "int_memory": int_memory,
            "m_dep": m_dep,
            "mobile_wt": mobile_wt,
            "n_cores": n_cores,
            "pc": pc,
            "px_height": px_height,
            "px_width": px_width,
            "ram": ram,
            "sc_h": sc_h,
            "sc_w": sc_w,
            "talk_time": talk_time,
            "three_g": three_g,
            "touch_screen": touch_screen,
            "wifi": wifi,
        }

        # Convertir le dictionnaire en DataFrame
        input_df = pd.DataFrame(user_input, index=[0])

        # Vérifier si le modèle est chargé
        if st.session_state.model:
            model = st.session_state.model
            prediction = predict_price(model, input_df)
            st.write(f"La plage de prix prédite pour ce téléphone est : **{prediction}**")

        else:
            st.write("Modèle non disponible. Veuillez entraîner un modèle dans l'onglet 'Évaluation du modèle'.")

    if app_mode == "Évaluation du modèle":
        st.title("Évaluation du Modèle")
        st.markdown(
            """
            Cette section vous permet d'évaluer la performance du modèle en termes de précision et d'autres métriques.
            """
        )

        # Charger les données
        train_df, test_df = load_data()

        # Prétraiter les données
        X, y, test_df_processed = preprocess_data(train_df, test_df)
        
        if X is None or y is None:
            st.stop()  # Arrêter l'exécution si les données sont incorrectes

        # Entraîner les modèles
        st.write("### Entraînement du modèle Random Forest")
        model_rf, accuracy_rf, classification_rep_rf, confusion_mat_rf = train_random_forest(X, y)
        st.write(f"Précision du modèle Random Forest : {accuracy_rf * 100:.2f}%")
        st.write("### Rapport de Classification - Random Forest")
        st.write(classification_rep_rf)
        st.write("### Matrice de Confusion - Random Forest")
        fig_rf, ax_rf = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_mat_rf, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3"], yticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3"])
        st.pyplot(fig_rf)

        # Visualisation de l'importance des caractéristiques pour Random Forest
        st.write("### Importance des Caractéristiques - Random Forest")
        plot_feature_importance(model_rf)

        st.write("### Entraînement du modèle de Régression Logistique")
        model_lr, accuracy_lr, classification_rep_lr, confusion_mat_lr = train_logistic_regression(X, y)
        st.write(f"Précision du modèle de Régression Logistique : {accuracy_lr * 100:.2f}%")
        st.write("### Rapport de Classification - Régression Logistique")
        st.write(classification_rep_lr)
        st.write("### Matrice de Confusion - Régression Logistique")
        fig_lr, ax_lr = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_mat_lr, annot=True, fmt="d", cmap="Blues", xticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3"], yticklabels=["Classe 0", "Classe 1", "Classe 2", "Classe 3"])
        st.pyplot(fig_lr)

        # Enregistrer les modèles dans session_state pour prédictions ultérieures
        st.session_state.model_rf = model_rf
        st.session_state.model_lr = model_lr

if __name__ == "__main__":
    main()
