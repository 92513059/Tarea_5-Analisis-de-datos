import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset
file_path = '/mnt/data/Titanic-Dataset.csv'
titanic_df = pd.read_csv(file_path)

# Step 1: Análisis exploratorio de los datos
# Mostrar información general del dataset
print(titanic_df.info())

# Describir el dataset
print(titanic_df.describe())

# Mostrar distribución de las variables categóricas
for column in ['Sexo', 'Embarcado', 'Clase/billete']:
    print(titanic_df[column].value_counts())

# Graficar las distribuciones de las variables numéricas
plt.figure(figsize=(15, 10))
titanic_df.hist(bins=30, figsize=(20, 15), edgecolor='black')
plt.show()

# Step 2: Preprocesamiento de los datos
# Convertir 'Tarifa/pasajero' a numérico, manejando posibles errores
titanic_df['Tarifa/pasajero'] = pd.to_numeric(titanic_df['Tarifa/pasajero'], errors='coerce')

# Imputar valores faltantes en 'Edad' y 'Tarifa/pasajero' con la media
imputer = SimpleImputer(strategy='mean')
titanic_df[['Edad', 'Tarifa/pasajero']] = imputer.fit_transform(titanic_df[['Edad', 'Tarifa/pasajero']])

# Imputar valores faltantes en 'Embarcado' con la moda
imputer = SimpleImputer(strategy='most_frequent')
titanic_df['Embarcado'] = imputer.fit_transform(titanic_df[['Embarcado']]).ravel()

# Convertir variables categóricas a numéricas
label_encoder = LabelEncoder()
titanic_df['Sexo'] = label_encoder.fit_transform(titanic_df['Sexo'])
titanic_df['Embarcado'] = label_encoder.fit_transform(titanic_df['Embarcado'])

# Selección de características relevantes
features = ['Clase/billete', 'Sexo', 'Edad', 'Número de hermanos/cónyuges a bordo del Titanic', 'Número de padres/hijos a bordo del Titanic', 'Tarifa/pasajero', 'Embarcado']
X = titanic_df[features]
y = titanic_df['Sobreviviente']

# Step 4: Dividir el dataset en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Entrenar el modelo de Árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Evaluar el desempeño del modelo
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Mostrar los resultados
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", cm)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Step 7: Graficar la importancia de las características
plt.figure(figsize=(12, 6))
feature_importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Visualizing Important Features')
plt.show()

# Step 8: Interpretar, analizar y documentar los resultados obtenidos
# Mostramos las métricas y discutimos los resultados

report_text = classification_report(y_test, y_pred)
print("Classification Report:\n", report_text)
