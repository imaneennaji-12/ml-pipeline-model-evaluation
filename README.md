💳 Détection de Fraude Bancaire avec Machine Learning

📌 Présentation

Ce projet est un système complet de Machine Learning permettant de détecter les transactions frauduleuses.

Il comprend :

Le prétraitement des données
L'entraînement de plusieurs modèles
L’évaluation et la sélection du meilleur modèle
Une application web interactive avec Streamlit
L’explication des prédictions avec SHAP

🎯 Objectif

L’objectif est d’identifier les transactions frauduleuses avec une haute précision tout en réduisant les faux positifs.

📊 Données

Le dataset contient des transactions bancaires simulées avec des variables telles que :

Montant de la transaction
Heure de transaction
Score de confiance du device
Nombre de transactions sur 24h
Âge du client
Catégorie du commerçant
Transactions étrangères / anomalies de localisation
🧠 Pipeline Machine Learning

Le projet suit un pipeline complet :

Nettoyage des données
Prétraitement (encodage, normalisation)
Entraînement de plusieurs modèles
Évaluation des performances
Comparaison des modèles
Sélection du modèle optimal
Sauvegarde du modèle final (.pkl)

🤖 Modèles testés

Régression Logistique
Arbre de Décision
Random Forest
Gradient Boosting / XGBoost (si utilisé)
📈 Métriques d’évaluation
Accuracy
Precision
Recall
F1-score
ROC-AUC
🌐 Application Streamlit

L’application permet de :

Simuler une transaction bancaire
Prédire si elle est frauduleuse
Afficher un score de risque
Visualiser l’explication du modèle avec SHAP
