import pandas as pd

def load_data(train_path, test_path):
    """
    Charge les ensembles de données d'entraînement et de test à partir des fichiers CSV.
    
    Args:
    - train_path : chemin vers le fichier CSV d'entraînement
    - test_path : chemin vers le fichier CSV de test
    
    Retourne :
    - train_df : DataFrame d'entraînement
    - test_df : DataFrame de test
    """
    # Charger les données depuis les fichiers CSV
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Retourner les DataFrames
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Prétraitement des données pour préparer les ensembles d'entraînement et de test.
    
    Args:
    - train_df : DataFrame d'entraînement
    - test_df : DataFrame de test
    
    Retourne :
    - X : Caractéristiques (features) de l'ensemble d'entraînement
    - y : Cible (target) de l'ensemble d'entraînement
    - test_df_processed : Ensemble de test préparé
    """
    # Vérifier si la colonne 'id' existe avant de la supprimer dans test_df
    if 'id' in test_df.columns:
        test_df = test_df.drop(columns=["id"])

    # Vérifier si la colonne 'price_range' existe dans train_df
    if 'price_range' not in train_df.columns:
        raise ValueError("La colonne 'price_range' est manquante dans les données d'entraînement.")

    # Vérifier si 'price_range' existe dans test_df (même si elle n'est pas utilisée dans le test)
    if 'price_range' in test_df.columns:
        test_df = test_df.drop(columns=["price_range"])

    # Encoder les variables catégorielles si nécessaire (utilisation de pd.get_dummies)
    train_df = pd.get_dummies(train_df, drop_first=True)
    test_df = pd.get_dummies(test_df, drop_first=True)

    # Aligner les colonnes entre train_df et test_df
    missing_cols = set(train_df.columns) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0  # Ajouter les colonnes manquantes dans test_df

    # Réorganiser les colonnes de test_df pour qu'elles correspondent à celles de train_df
    test_df = test_df[train_df.columns]

    # Séparer les caractéristiques (X) et la cible (y) dans l'ensemble d'entraînement
    X = train_df.drop(columns=["price_range"])  # Caractéristiques
    y = train_df["price_range"]  # Cible

    # L'ensemble de test finalisé (test_df) est prêt à être utilisé
    test_df_processed = test_df

    return X, y, test_df_processed
