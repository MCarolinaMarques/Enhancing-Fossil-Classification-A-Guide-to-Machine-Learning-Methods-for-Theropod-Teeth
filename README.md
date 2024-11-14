# Enhancing-Fossil-Classification-A-Guide-to-Machine-Learning-Methods-for-Theropod-Teeth
Here you can find all of the information on the article "Enhancing Fossil Classification: A Guide to Machine Learning Methods for Theropod Teeth". Including the newest version of the article (PDF), the figures included, the code that creates them and the pre-processing done in R and the analysis done in Python.

## To apply the trained models to classify new theropod teeth follow the following steps:

1- Import libraries:
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, RobustScaler, MinMaxScaler

2- Read the xlsx file with the new data:
X= pd.read_excel('C:/path/to/your/folder/.../filedata.xlsx')

3- Select the variables needed:
X=X[["Epoch","CBL", "CBW", "CH", "CBR", "CHR", "DC", "DDL"]]

4-  Open the trained standadization method:
sc=joblib.load('CC:/path/to/your/folder/.../scaler_Random Forest_QuantileTransformer_KMeansSMOTE.joblib')

5- Load the trained model:
model=joblib.load('C:/path/to/your/folder/.../model_Random Forest_QuantileTransformer_KMeansSMOTE.joblib')


6- Mapping the epochs to their label encodings:
epoch_mapping = {
    "Late Triassic": 0,
    "Early Jurassic": 1,
    "Middle Jurassic":2,
    "Late Jurassic": 3,
    "Late Jurassic - Early Cretaceous": 3.5,
    "Early Cretaceous": 4,
    "Middle Cretaceous": 5,
    "Late Cretaceous":6
}

7- Apply the mapping to the 'Epoch' column:
X['Epoch'] = X['Epoch'].map(epoch_mapping)

8- Select the numeric columns to be standardized:
numeric_columns = ["CBL", "CBW", "CH", "CBR", "CHR", "DC", "DDL"]

8- Re-order the columns to mach the order used in training:
X=X.reindex(["Epoch","CBL", "CBW", "CH", "CBR", "CHR", "DC", "DDL"], axis=1)
X1=X

9- Standardize the numeric columns:
X[numeric_columns] = sc.transform(X[numeric_columns])

10- Transform the data to a DataFrame
X = pd.DataFrame(X, columns=X1.columns)

11- Predict the classification probabilities for the new data:
predictions = model.predict(X)

12- Printing predictions:
print(f'{predictions}')

13- Get the predicted class probabilities
predictions_proba = model.predict_proba(X)

14- Convert the results to a DataFrame for better readability:
predictions_df = pd.DataFrame(predictions_proba, columns=model.classes_)

15- Create a function to get the top 3 classes with the highest probabilities:
def get_top_n_classes(row, n=3):
    sorted_indices = row.values.argsort()[::-1][:n]  # Get the indices of the top n probabilities
    top_n_classes = row.index[sorted_indices]        # Get the class labels for these indices
    top_n_probs = row.values[sorted_indices]         # Get the probabilities for these indices
        result = {}
    for idx in range(n):
        result[f'Top{idx+1}_Class'] = top_n_classes[idx]
        result[f'Top{idx+1}_Prob'] = top_n_probs[idx]
    return pd.Series(result)

16-  Apply the function to each row/ teeth:
top_predictions_df = predictions_df.apply(get_top_n_classes, axis=1)

17- Print the results
print(top_predictions_df)
