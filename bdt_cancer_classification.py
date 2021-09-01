#%%

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import tree

#%%

""" 
Cargamos los datos desde scikit-learn y los combinamos en un solo dataframe. La columna ‘target’ representa el tipo de tumor: 1 benigno, 0 maligno 
"""

cancer = load_breast_cancer()
data = np.c_[getattr(cancer, "data"), getattr(cancer, "target")]
columns = np.append(getattr(cancer, "feature_names"), ["target"])
sizeMeasurements = pd.DataFrame(data, columns=columns)

# %%

""" 
Dividimos los datos en dos partes, 80% para entrenamiento 
y 20% para el testeo del modelo
"""

X = sizeMeasurements[sizeMeasurements.columns[:-1]]
y = sizeMeasurements.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# %%

""" 
Creamos nuestro árbol binario de decisión. En este caso lo vamos a usar de
tipo clasificación y configurado de tal forma de que la máxima profundidad 
que el árbol pueda alcanzar sea de 3. También que la mínima cantidad de 
muestras que una hoja pueda tener sea de 12. 

Nota: estas configuraciones son para el entrenamiento del árbol
"""

clf1 = tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=12)
clf1.fit(X_train, Y_train)

# %%

""" 
Comprobamos la precisión del modelo usando los mismos datos con los que entrenamos y los de prueba 
"""

print(
    "Precisión del modelo usando el conjunto de entrenamiento : {:.2f}".format(
        clf1.score(X_train, Y_train)
    )
)
print(
    "Precisión del modelo usando el conjunto de prueba : {:.2f}".format(
        clf1.score(X_test, Y_test)
    )
)

# %%
