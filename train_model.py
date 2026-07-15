import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Charger le dataset
df = pd.read_csv("diabetes.csv")

# Colonnes à nettoyer
columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Remplacer les 0 par des valeurs manquantes
df[columns] = df[columns].replace(0, pd.NA)

# Remplacer les valeurs manquantes par la médiane
for col in columns:
    df[col] = df[col].fillna(df[col].median())

print("Nettoyage terminé !")

# ==========================
# Séparer X et y
# ==========================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ==========================
# Diviser le dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Taille de X_train :", X_train.shape)
print("Taille de X_test :", X_test.shape)

# ==========================
# Entraîner le modèle
# ==========================

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ==========================
# Faire des prédictions
# ==========================

predictions = model.predict(X_test)

# ==========================
# Évaluer le modèle
# ==========================

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"Accuracy  : {accuracy:.2%}")
print(f"Precision : {precision:.2%}")
print(f"Recall    : {recall:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix :")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Rapport complet
print("\nClassification Report :")
print(classification_report(y_test, predictions))