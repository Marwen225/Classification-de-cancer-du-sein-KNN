import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px  

# Titre de l'application
st.title("Performance de K-Nearest Neighbors sur types de cancer du sein")

# Chargement des données
df = pd.read_csv('data.csv')
Y = pd.read_excel('x1.xlsx')
y = Y["di"]
df.drop("diagnosis", axis=1, inplace=True)
x = df

st.sidebar.header("Options")
k_values = st.sidebar.slider("Nombre de voisins (k)", 1, 20, (1, 5))

# Imputation des valeurs manquantes
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Normalisation des données
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Train Test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Liste pour stocker les performances en fonction de k
k_range = list(range(k_values[0], k_values[1] + 1))
r2_scores = []
mae_scores = []

# Boucle pour tester différentes valeurs de k
for k in k_range:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_pred = knn_classifier.predict(X_test)
    r2 = r2_score(y_test, knn_pred)
    mae = mean_absolute_error(y_test, knn_pred)
    r2_scores.append(r2)
    mae_scores.append(mae)

#les performances
data = pd.DataFrame({'k': k_range, 'R2 Score': r2_scores, 'Mean Absolute Error': mae_scores})

# Utilisez Plotly Express pour créer un graphique interactif
fig = px.line(data, x='k', y=['R2 Score', 'Mean Absolute Error'], labels={'value': 'Performance'}, title='Performance de K-Nearest Neighbors en fonction de k')
st.plotly_chart(fig)
