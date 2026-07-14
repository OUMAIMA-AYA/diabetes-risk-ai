import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

print(f"Précision : {accuracy * 100:.2f}%")