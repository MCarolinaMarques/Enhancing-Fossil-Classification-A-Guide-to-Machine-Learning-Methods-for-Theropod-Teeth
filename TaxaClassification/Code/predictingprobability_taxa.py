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
data =  pd.read_csv('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/teeth_data_taxa_epoch1.csv')
#select the classification
y = data['TaxonToothtype']
#select the variables
X=data[["CBL", "CBW","CH", "CBR", "CHR","DC", "DDL" ]]
X1=X
#load standardization
sc=joblib.load('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/5ModelResultsnew_ne/scaler_Random Forest_StandardScaler_RandomOverSampler.joblib')
#standardize the data
X = sc.transform(X[["CBL", "CBW","CH", "CBR", "CHR","DC", "DDL" ]])

X = pd.DataFrame(X, columns=X1.columns)

#load mtrained model
rf_model=joblib.load('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/5ModelResultsnew_ne/model_Random Forest_StandardScaler_RandomOverSampler.joblib')

#re-order the variables
X = X[rf_model.feature_names_in_]


#  rf_model is the trained Random Forest model 
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
    'True Label': y,
    'Most Probable Label': top_3_labels[:, 0],
    'Most Probable Probability': top_3_probs[:, 0],
    'Second Probable Label': top_3_labels[:, 1],
    'Second Probable Probability': top_3_probs[:, 1],
    'Third Probable Label': top_3_labels[:, 2],
    'Third Probable Probability': top_3_probs[:, 2]
})

# Filter for incorrectly predicted observations
incorrect_predictions = results[results['True Label'] != results['Most Probable Label']]

# Save to a CSV file
incorrect_predictions.to_csv('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/incorrect_predictions_best_model_taxa.csv', index=False)
incorrect_predictions.to_excel(
    'C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/incorrect_predictions_best_model_taxa.xlsx', 
    index=False
)


# Display results
print(incorrect_predictions)


# Predict probabilities
probs = rf_model.predict_proba(X)
top_3_indices = np.argsort(probs, axis=1)[:, -3:]  # Indices of top 3 classes
top_3_classes = rf_model.classes_[top_3_indices]  # Corresponding class labels


#  Check if the true class is in the top 3 predictions
y= np.array(y)  # Ensure y_test is a NumPy array
in_top_3 = np.array([true_label in top_classes for true_label, top_classes in zip(y, top_3_classes)])

# Compute the proportion
proportion_in_top_3 = np.mean(in_top_3)
print(f"Proportion of true classes in top 3 predictions: {proportion_in_top_3:.2%}")
