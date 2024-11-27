import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st
def plot_confusion_matrix(model, X_test, y_test):
    # Vérification des tailles
    if len(X_test) != len(y_test):
        st.error(f"Les tailles de X_test ({len(X_test)}) et y_test ({len(y_test)}) ne correspondent pas.")
        return

    # Prédiction
    y_pred = model.predict(X_test)

    # Vérification de la taille de y_pred
    if len(y_test) != len(y_pred):
        st.error(f"Les tailles de y_test ({len(y_test)}) et y_pred ({len(y_pred)}) ne correspondent pas.")
        return

    # Création de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Création de la figure pour la matrice de confusion
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)

    # Titre et affichage
    ax.set_title("Matrice de Confusion")
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Réel')
    st.pyplot(fig)  # Affiche la figure dans Streamlit
    plt.clf()  # Nettoyer la figure après affichage


def plot_feature_importance(model, X_test=None):
    # Vérification si l'attribut `feature_importances_` existe
    if not hasattr(model, 'feature_importances_'):
        st.error("Le modèle ne possède pas d'attribut 'feature_importances_' (pas un modèle basé sur l'arbre).")
        return

    # Extraire l'importance des caractéristiques
    importance = model.feature_importances_

    # Si X_test est passé, utiliser ses colonnes comme noms de caractéristiques
    if X_test is not None and hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # Utiliser les noms de caractéristiques de X_test si ce dernier est fourni, sinon générer des noms génériques
        feature_names = X_test.columns if X_test is not None and hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(len(importance))]

    # Création de la figure pour l'importance des caractéristiques
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer l'importance des caractéristiques
    sns.barplot(x=importance, y=feature_names, ax=ax)

    # Titre et affichage
    ax.set_title("Importance des caractéristiques")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Caractéristiques")
    st.pyplot(fig)  # Affiche la figure dans Streamlit
    plt.clf()  # Nettoyer la figure après affichage
