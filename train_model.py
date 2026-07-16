import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression #un algo de classification 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
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

model = RandomForestClassifier(random_state=42) #la creation du model (il est vide , il ne connait encore rien)
model.fit(X_train, y_train)

# ==========================
# Faire des prédictions
# ==========================

predictions = model.predict(X_test)

# ==========================
# "Évaluer le modèle"
# ==========================

accuracy = accuracy_score(y_test, predictions) #on compare
precision = precision_score(y_test, predictions)#quand le modele dit diabetique a t il raison ?
recall = recall_score(y_test, predictions)#Parmi tous les vrais diabétiques, combien le modèle a-t-il détectés ?

print(f"Accuracy  : {accuracy:.2%}")
print(f"Precision : {precision:.2%}")
print(f"Recall    : {recall:.2%}")

# Matrice de confusion
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix :")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#cette partie affiche la matrice sous forme de graphique
disp.plot()
plt.show()

# Rapport complet
print("\nClassification Report :")
print(classification_report(y_test, predictions))

#ces 4 modeles essaienet tous de resoudre le meme prb mais ils utilisent des methodes diff
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),#il cherche une formule mathematique qui separe les personnes diabetiques des personnes non diabetiques 
    "Decision Tree": DecisionTreeClassifier(random_state=42),#il fonctionne comme une serie de questions , il construit un arbre de decisions
    "Random Forest": RandomForestClassifier(random_state=42),#au lieu d'un seul arbre , il cree beaucoup d'arbres
    "KNN": KNeighborsClassifier()#il ne construit presque rien , il cherche les patients qui ressemble le plus le nouveau passient qui vient d'arriver
}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} : {accuracy:.2%}")