import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Entraîner le modèle
def train_model(X, y):
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

# Prédiction de la plage de prix pour un téléphone donné
def predict_price(model, features):
    # Assurez-vous que les colonnes sont bien alignées avec le modèle
    prediction = model.predict(features)
    return prediction[0]

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

        # Convertir les valeurs saisies par l'utilisateur en DataFrame
        user_input_df = pd.DataFrame(user_input, index=[0])

        # Si le modèle est disponible, faire la prédiction
        if st.session_state.model:
            prediction = predict_price(st.session_state.model, user_input_df)
            st.write(f"La plage de prix prédite est : {prediction}")
        else:
            st.error("Le modèle n'est pas encore entraîné. Veuillez d'abord évaluer le modèle.")

    elif app_mode == "Évaluation du modèle":
        st.title("Évaluation du Modèle")
        st.markdown(
            """
            Cette section vous permet d'évaluer le modèle d'apprentissage automatique entraîné
            en calculant la précision, en affichant le rapport de classification et la matrice de confusion.
            """
        )

        # Charger les données et entraîner le modèle
        train_df, test_df = load_data()
        X, y, test_df_processed = preprocess_data(train_df, test_df)
        
        if X is None or y is None:
            st.stop()  # Arrêter l'exécution si les données sont incorrectes

        model, accuracy, classification_rep, confusion_mat = train_model(X, y)
        
        # Enregistrer le modèle dans la session
        st.session_state.model = model
        st.session_state.accuracy = accuracy
        st.session_state.classification_rep = classification_rep
        st.session_state.confusion_mat = confusion_mat

        # Afficher les résultats
        st.write(f"### Précision du modèle: {accuracy:.2f}")
        st.write("### Rapport de classification:")
        st.text(classification_rep)
        
        st.write("### Matrice de confusion:")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Prédictions')
        plt.ylabel('Réel')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
