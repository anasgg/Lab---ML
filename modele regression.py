import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Charger les données depuis le fichier texte avec une tabulation comme délimiteur
data = np.loadtxt('votre_fichier.txt', delimiter='\t')

# Diviser les données en caractéristiques (X) et variable cible (Y)
X = data[:, 0].reshape(-1, 1)  # Première colonne du fichier texte
Y = data[:, 1]  # Deuxième colonne du fichier texte

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Créer un modèle de régression linéaire
regressor = LinearRegression()

# Entraîner le modèle sur l'ensemble d'entraînement
regressor.fit(X_train, Y_train)

# Faire des prédictions sur l'ensemble de test
Y_pred = regressor.predict(X_test)

# Tracer les données réelles et les prédictions
plt.scatter(X_test, Y_test, color='blue', label='Données réelles')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Régression linéaire')
plt.xlabel('Caractéristiques (X)')
plt.ylabel('Variable cible (Y)')
plt.legend()
plt.show()
