# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 10:44:05 2024

@author: carol
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, MaxAbsScaler, RobustScaler, MinMaxScaler

# Load the data
data =  pd.read_csv('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/new_theropod_teeth_data.csv')
# classification
y = data['Classification position']
# Inventory number
z= data['Inventory number']
# select the variables
X=data[["CBL", "CBW","CH", "CBR", "CHR","DC", "DDL" ]]
X1=X
#load standardization
sc=joblib.load('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/5LogModelResultsnew_clade_ne/scaler_Random Forest_QuantileTransformer_RandomOverSampler.joblib')
# standardize the dataset
X = sc.transform(X[["Log_CBL", "Log_CBW","Log_CH", "Log_CBR", "Log_CHR","Log_DC", "Log_DDL" ]])

X = pd.DataFrame(X, columns=X1.columns)

#load model
rf_model=joblib.load('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/5LogModelResultsnew_clade_ne/model_Random Forest_QuantileTransformer_RandomOverSampler.joblib')

#re-order the variables
X = X[rf_model.feature_names_in_]


# rf_model is the trained Random Forest model
# Predict probabilities
proba = rf_model.predict_proba(X)
# Predict most probable class
predicted_labels = rf_model.predict(X)

# Get the top 3 predicted classes and their probabilities
top_3_indices = np.argsort(proba, axis=1)[:, -3:][:, ::-1]  # Indices of top 3 classes
top_3_probs = np.sort(proba, axis=1)[:, -3:][:, ::-1]       # Probabilities of top 3 classes

# Convert indices to class labels
classes = rf_model.classes_
top_3_labels = classes[top_3_indices]

# Create a DataFrame
results = pd.DataFrame({
    'Inventory number':z,
    'True Label': y,
    'Most Probable Label': top_3_labels[:, 0],
    'Most Probable Probability': top_3_probs[:, 0],
    'Second Probable Label': top_3_labels[:, 1],
    'Second Probable Probability': top_3_probs[:, 1],
    'Third Probable Label': top_3_labels[:, 2],
    'Third Probable Probability': top_3_probs[:, 2]
})
results.to_excel(
    'C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/results_new_teeth_guimarota_clade.xlsx', 
    index=False
)

# Display results
print(results)