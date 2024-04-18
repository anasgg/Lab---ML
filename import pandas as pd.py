import pandas as pd

# Spécifiez le chemin vers votre fichier texte
chemin_fichier = "C:/Users/Anas/Desktop/farms1.txt"

# Lisez le fichier texte en utilisant pandas et spécifiez le séparateur si nécessaire
# Par exemple, si le fichier est séparé par des virgules (CSV), utilisez sep=','.
# Si le fichier est tabulé, vous pouvez utiliser sep='\t'.
dataframe = pd.read_csv(chemin_fichier, sep='\t')  # Remplacez '\t' par le séparateur approprié

# Vous avez maintenant votre fichier texte dans un DataFrame, et vous pouvez effectuer
# différentes opérations de manipulation de données avec pandas.
print(dataframe)
# Par exemple, vous pouvez afficher les premières lignes du DataFrame

colonnes_selectionnees = dataframe[["DIFF", "R2", "R3", "R7", "R14" , "R17" , "R18" , "R21" , "R32" , "R36"]]

print(colonnes_selectionnees.head())
# Vous pouvez également effectuer diverses opérations de filtrage, de tri, etc.
# sur le DataFrame selon vos besoins.

# N'oubliez pas de sauvegarder le DataFrame modifié si nécessaire.
# dataframe.to_csv("nouveau_fichier.csv", index=False)  # Pour sauvegarder au format CSV

# Assurez-vous de personnaliser ce code en fonction de votre fichier texte et de vos besoins spécifiques.
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Charger votre DataFrame (remplacez 'votre_dataframe.csv' par le chemin vers votre fichier CSV)
df = colonnes_selectionnees

# Suppression des valeurs manquantes
df.dropna(inplace=True)

# Diviser les données en fonction de vos caractéristiques et de votre variable cible
X = df.drop('DIFF', axis=1)  # Remplacez 'cible' par le nom de votre variable cible
y = df['DIFF']

# Créer une instance de LDA
lda = LinearDiscriminantAnalysis()

# Appliquer LDA pour réduire la dimensionnalité
X_lda = lda.fit_transform(X, y)

# Afficher les composantes discriminantes
print(X_lda)

# Vous pouvez également accéder aux coefficients discriminants
print(lda.coef_)
# Retrieve the coefficients β0 and β'
beta0 = lda.intercept_
beta_prime = lda.coef_

print("β0 (Intercept):", beta0)
print("β' (Coefficients):", beta_prime)
scores = lda.decision_function(X) 
# 'scores' now contains the LDA scores for each farm in 'input_data'
import matplotlib.pyplot as plt

# You can add the scores to your DataFrame if needed
print(scores)
# Create a scatter plot of LDA scores
plt.figure(figsize=(10, 6))
plt.scatter(scores, range(len(scores)), c='b', marker='o', alpha=0.5)
plt.xlabel('LDA Score')
plt.ylabel('Farm Index')
plt.title('Scatter Plot of LDA Scores for 1260 Farms')
plt.grid(True)

# Show the plot
plt.show()
# Assuming you have already loaded your dataset into a DataFrame 'input_data'
# 'DIFF' is the target variable
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split your dataset into features (X) and the target variable (y)


# Split your data into a training set and a testing set (you may adjust the split ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of QDA
qda = QuadraticDiscriminantAnalysis()

# Fit the QDA model to your training data
qda.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = qda.predict(X_test)

# Evaluate the performance of the QDA model (e.g., accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of QDA: {accuracy * 100:.2f}%")
qda = QuadraticDiscriminantAnalysis()

# Fit the QDA model to your data
qda.fit(X, y)

# Calculate posterior probabilities for each farm
posterior_probs = qda.predict_proba(X)

# 'posterior_probs' is a NumPy array where each row corresponds to a farm
# and each column corresponds to the probability of that farm belonging to a specific class

# Interpretation:
# For each farm (row), you have a set of probabilities representing the likelihood of it belonging
# to each class. The class with the highest probability is the predicted class for that farm.

# If you want to associate these probabilities with the farm data in your DataFrame,
# you can add them as new columns, one column for each class:
class_names = qda.classes_  # Get the class names

for i, class_name in enumerate(class_names):
    dataframe[f'Probability_{class_name}'] = posterior_probs[:, i]

# Now, your DataFrame 'input_data' contains new columns with the posterior probabilities
# for each class associated with each farm.

# You can access the posterior probabilities for a specific farm by its row index in 'posterior_probs'.
# For example, if you want the posterior probabilities for the first farm:
farm_index = 0
probabilities_for_first_farm = posterior_probs[farm_index]

# 'probabilities_for_first_farm' contains the probabilities for each class for the first farm.

