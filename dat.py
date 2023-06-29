#abrir archivo q contiene datos
#datos de test
import pandas as pd
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

#------------------------------PREPROCESAMIENTO ONECODE-----------------------

# list(data) or
#print(df_train.columns)
def Proc_Data():
    
    df_X = df_train.loc[:, 'S1':'S663']
    #print(df_X.columns)

    df = df_train.loc[:, 'SPP':'S663']
    #print(df.columns)

    one_hot_encoded_data = pd.get_dummies(df, columns = df_X.columns,dtype=int)
    #print(one_hot_encoded_data)

    one_hot_encoded_data = one_hot_encoded_data.iloc[:-1, :]
    one_hot_encoded_data = one_hot_encoded_data.iloc[:-1, :]
    one_hot_encoded_data = one_hot_encoded_data.iloc[:-1, :]
    one_hot_encoded_data = one_hot_encoded_data.iloc[:-1, :]
    one_hot_encoded_data = one_hot_encoded_data.iloc[:-1, :]
    #print(one_hot_encoded_data)


    le = preprocessing.LabelEncoder()
    le.fit(one_hot_encoded_data.SPP)
    list(le.classes_)
    one_hot_encoded_data['SPP'] = le.transform(one_hot_encoded_data.SPP)
    #print(one_hot_encoded_data)

    #print(le.inverse_transform(one_hot_encoded_data['SPP']))

    target = one_hot_encoded_data['SPP']
    one_hot_encoded_data_x = one_hot_encoded_data.loc[:, 'S1_A':'S663_T']
    x_train, x_test, y_train, y_test = model_selection.train_test_split(one_hot_encoded_data_x, target, test_size = 0.25, random_state =0)
    return  x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test=Proc_Data()
if __name__ == "__main__":
    
    print(x_train)
   