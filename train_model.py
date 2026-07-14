import pandas as pd

# Charger le dataset
df = pd.read_csv("diabetes.csv")

print("===== Aperçu des données =====")
print(df.head())

print("\n===== Dimensions =====")
print(df.shape)

print("\n===== Colonnes =====")
print(df.columns)

print("\n===== Informations =====")
print(df.info())

print("\n===== Statistiques =====")
print(df.describe())

print("\n===== Valeurs manquantes =====")
print(df.isnull().sum())