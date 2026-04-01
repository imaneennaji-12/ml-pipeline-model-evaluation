# 🛡️ Détection de Fraude Bancaire — Simulation ML

> Projet académique de Machine Learning appliqué à la détection de transactions frauduleuses par carte bancaire, avec interface de simulation interactive.

---

## 📌 Description

Ce projet simule un système de détection de fraude en temps réel dans un contexte fintech.  
Il comprend une phase complète d'analyse exploratoire, un entraînement de modèles de classification, une sélection du meilleur modèle en production, et une interface utilisateur développée avec **Streamlit**.

> ⚠️ **Simulation académique uniquement** — Les données sont synthétiques et le modèle est destiné à des fins pédagogiques.

---

## 📁 Structure du projet

```
├── PROJET3.ipynb                  # Notebook principal (EDA, modélisation, évaluation)
├── fraud_model_production1.pkl    # Modèle XGBoost entraîné (pipeline complet)
├── credit_card_fraud_10k.csv      # Dataset simulé (10 000 transactions)
├── app2.py                        # Interface Streamlit de simulation
└── README.md
```

---

## 📊 Dataset

| Caractéristique | Détail |
|---|---|
| Taille | 10 000 transactions |
| Variables | 9 (8 features + 1 cible) |
| Taux de fraude | ~1,51 % (151 fraudes / 9 849 légitimes) |
| Valeurs manquantes | Aucune |

### Variables du dataset

| Variable | Type | Description |
|---|---|---|
| `amount` | Numérique | Montant de la transaction |
| `transaction_hour` | Numérique | Heure de la transaction (0–23) |
| `merchant_category` | Catégorielle | Catégorie du marchand (Grocery, Food, Travel, Electronics, Clothing) |
| `foreign_transaction` | Binaire | Transaction à l'étranger (0/1) |
| `location_mismatch` | Binaire | Incohérence de localisation (0/1) |
| `device_trust_score` | Numérique | Score de confiance du terminal (0.0–1.0) |
| `velocity_last_24h` | Numérique | Nombre de transactions dans les dernières 24h |
| `cardholder_age` | Numérique | Âge du porteur de carte |
| `is_fraud` | Binaire | **Cible** — 0 : légitime, 1 : frauduleuse |

---

## 🔍 Analyse Exploratoire (EDA)

Quelques observations clés :

- **Déséquilibre de classes sévère** : seulement 1,51 % de fraudes → traité avec **SMOTE**
- **Pic de fraude nocturne** : taux atteignant ~9 % entre 00h et 03h du matin
- **Catégories à risque** : Grocery (2,00 %) > Food (1,67 %) > Travel (1,46 %)
- Les montants des transactions frauduleuses présentent une distribution distincte des transactions légitimes

---

## 🤖 Modélisation

### Pipeline d'entraînement

```
Preprocessing → SMOTE → Modèle de classification
```

Quatre modèles ont été comparés via **validation croisée** (métriques : PR-AUC, ROC-AUC, Recall, Precision, F1) :

| Modèle | Notes |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Bonne robustesse |
| XGBoost | ✅ **Sélectionné en production** |
| LightGBM | Concurrent principal |

### Modèle de production

Le pipeline de production utilise **XGBoost** avec préprocessing intégré, sérialisé via `joblib` :

```python
Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", XGBClassifier(eval_metric="logloss", random_state=42))
])
```

---

## 🖥️ Interface Streamlit

L'application `app2.py` permet de simuler la prédiction d'une transaction en temps réel.

### Fonctionnalités

- **Formulaire de saisie** : identifiants, montant, heure, caractéristiques comportementales et contextuelles
- **Prédiction en temps réel** : classe prédite + score de risque (probabilité)
- **Explicabilité SHAP** : waterfall plot pour visualiser la contribution de chaque variable à la décision
- **Enregistrement simulé** : affichage d'un objet JSON représentant l'entrée en base de données

### Lancement

```bash
streamlit run app2.py
```

---

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/<votre-username>/<nom-du-repo>.git
cd <nom-du-repo>

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app2.py
```

### Dépendances principales

```
streamlit
pandas
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
joblib
matplotlib
numpy
```

---

## 📈 Résultats

Les performances détaillées par modèle (matrices de confusion, courbes ROC/PR, scores de validation croisée) sont disponibles dans le notebook `PROJET3.ipynb`.
