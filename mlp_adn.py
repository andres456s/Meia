#abrir archivo q contiene datos
#datos de test
import pandas as pd
import dat
#defining my worksheet


Drosophila_test_path = 'Drosophila_test.csv'
df_test = pd.read_csv(Drosophila_test_path, delimiter=';')
#print(df_test)
# Establecer la primera columna como el índice de fila
df_test = df_test.set_index(df_test.columns[0])

#print(df_test)

Drosophila_train_path = 'Drosophila_train.csv'
df_train = pd.read_csv(Drosophila_train_path, delimiter=';')
#print(df_train)
# Establecer la primera columna como el índice de fila
df_train = df_train.set_index(df_train.columns[0])

Y_train = df_train["SPP"]
#print(Y_train)


X_train = df_train.loc[:, 'S1':'S663']
#print(X_train.head)


#print(df_train)


#implementar modelo de clasificacion 

#importacion de dependencias
from sklearn import datasets, model_selection, metrics, ensemble

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


#------------------------------PREPROCESAMIENTO ONECODE-----------------------

#x_train, x_test, y_train, y_test=Proc_Data()

print("Training Dataset:", dat.x_train.shape)
print("Test Dataset:", dat.x_test.shape)

#-------------------------------------------MODELO-------------------------------

OPT_LAYERS=[7,]
data = datasets.load_iris()
X = data['data']
y = data['target']

print(y)

training_X, testing_X, training_y, testing_y = dat.x_train, dat.x_test, dat.y_train, dat.y_test

myscaler = StandardScaler()
myscaler.fit(training_X)
training_X = myscaler.transform(training_X)
testing_X = myscaler.transform(testing_X)


mlp = MLPClassifier(hidden_layer_sizes=OPT_LAYERS, activation='relu', solver='lbfgs',alpha=1.611189582022303,learning_rate='invscaling')
mlp.fit(training_X, training_y)

predicted_values = mlp.predict(testing_X)

print("prueba:    ", testing_y)
print("predicción:", predicted_values)
print(confusion_matrix(testing_y, predicted_values))
print(classification_report(testing_y, predicted_values))
