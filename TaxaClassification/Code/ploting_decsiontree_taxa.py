#Proportion of true classes in top 3 predictions: 97.84%
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import plot_tree


# Load the data
data =  pd.read_csv('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/teeth_data_taxa_epoch1.csv')

y = data['TaxonToothtype']

X=data[["CBL", "CBW","CH", "CBR", "CHR","DC", "DDL" ]]

rf_model=joblib.load('C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/5ModelResultsnew_ne/model_Random Forest_StandardScaler_RandomOverSampler.joblib')


# Visualize the first tree in the Random Forest
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns, class_names=rf_model.classes_, filled=True)
plt.show()

# Visualize the first tree in the Random Forest
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[1], feature_names=X.columns, class_names=rf_model.classes_, filled=True)
plt.show()


# Visualize the first tree in the Random Forest
plt.figure(figsize=(45, 15))  # Increased figure size
plot_tree(
    rf_model.estimators_[0],  # First tree from the Random Forest
    feature_names=X.columns,  # Feature names from your dataset
    class_names=rf_model.classes_,  # Class names for the target variable
    filled=True,  # Fill nodes with color based on class
    max_depth=3,  # Restrict to 3 levels for better visualization
    fontsize=12  # Increased font size for clarity
    
)

# Save the plot to a file
plt.savefig("C:/Users/carol/Dropbox/PhD_Paleo_stuff/Dentes/decision_tree_visualizationtaxa.png", dpi=300, bbox_inches='tight')  # High resolution
plt.show()

np.max([estimator.get_depth() for estimator in rf_model.estimators_])

