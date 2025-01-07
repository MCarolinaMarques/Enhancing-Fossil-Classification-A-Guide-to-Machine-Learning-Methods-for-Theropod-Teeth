from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, MaxAbsScaler, RobustScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, KMeansSMOTE, SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from collections import deque
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_predict,StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline 
from collections import defaultdict  # Import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# Plotting Confusion Matrices



# Load the data
data = pd.read_csv('teeth_data_log_clade_epoch1.csv')



# Mapping the epochs to their label encodings
epoch_mapping = {
    "Late Triassic": 0,
    "Early Jurassic": 1,
    "Middle Jurassic":2,
    "Late Jurassic": 3,
    "Late Jurassic - Early Cretaceous": 3.5,
    "Early Cretaceous": 4,
    "Middle Cretaceous": 5,
    "Late Cretaceous": 6
}

# Apply the mapping to the 'epch' column
data['Epoch'] = data['Epoch'].map(epoch_mapping)

# Separate features and target
X = data.drop(columns='CladeToothtype')

X = X.drop(columns='Epoch')

y = data['CladeToothtype']


# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Verify the number of unique classes
unique_classes = label_encoder.classes_
print("Unique classes:", unique_classes)
print("Number of unique classes:", len(unique_classes))


# Calculate the number of observations for each class
class_counts = y.value_counts()

# Plot the bar plot
plt.figure(figsize=(12, 8))
sns.set(font_scale=1.5)  # Increase font scale
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
# Adding titles and labels
plt.title('Number of Observations for Each Class', fontsize=20)
plt.xlabel('Class', fontsize=19)
plt.ylabel('Number of Observations', fontsize=19)
# Rotate class names and make them italic
plt.xticks(rotation=90, fontstyle='italic')
plt.tight_layout()
# Show the plot
plt.show()


# Define classifiers or models
# Compile the model
# List of columns to standardize (excluding 'Epoch')
numeric_columns = ["Log_CBL" ,         "Log_CBW"   ,       "Log_CH", "Log_CBR"  ,        "Log_CHR"  ,        "Log_DC" ,"Log_DDL"]
#Model methods
methods=[
    ('Random Forest',RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Support Vector Machine',SVC(kernel='rbf', probability=True, random_state=42)),
    ('K-Nearest Neighbors',KNeighborsClassifier(n_neighbors=5)),
    ('Gradient Boosting',GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('Discriminant Analysis',QuadraticDiscriminantAnalysis(reg_param=0.1))
]
# Define scaling techniques
scalers = [
    ('NoScaler',None),
    ('StandardScaler', StandardScaler()),
    ('RobustScaler', RobustScaler()),
    ('QuantileTransformer', QuantileTransformer(output_distribution='normal'))
]

#Oversamplers
oversamplers = [
    ('NoOversampling', None),
    ('SMOTE', SMOTE(random_state=42, k_neighbors=4)),
    ('RandomOverSampler', RandomOverSampler(sampling_strategy='auto',random_state=42)),
    ('BorderlineSMOTE', BorderlineSMOTE(random_state=42, k_neighbors=4)),
    ('KMeansSMOTE', KMeansSMOTE(random_state=42,cluster_balance_threshold=0.01))
]
# Initialize a deque to keep track of the top 3 models (with maxlen 3)
top_models = deque(maxlen=3)

all_models = []

# Create k-fold cross-validation
k_folds = 5  # Number of folds
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Variable to track the best model
best_accuracy = -np.inf  # Start with a very low value
best_model = None
best_method = None
best_scaler = None
best_sampler = None
second_accuracy = -np.inf  # Start with a very low value
second_model = None
second_method = None
second_scaler = None
second_sampler = None

def average_confusion_matrices(conf_matrices):
    """Average confusion matrices ensuring consistent dimensions."""
    # Determine the maximum shape
    max_rows = max(cm.shape[0] for cm in conf_matrices)
    max_cols = max(cm.shape[1] for cm in conf_matrices)
    
    # Create a list of matrices with the same shape
    padded_matrices = []
    for cm in conf_matrices:
        # Pad confusion matrices to the largest shape
        padded_cm = np.zeros((max_rows, max_cols), dtype=cm.dtype)
        padded_cm[:cm.shape[0], :cm.shape[1]] = cm
        padded_matrices.append(padded_cm)
    
    # Convert list to numpy array
    padded_matrices = np.array(padded_matrices)
    
    # Calculate mean and std
    mean_cm = np.mean(padded_matrices, axis=0)
    std_cm = np.std(padded_matrices, axis=0)
    
    return mean_cm, std_cm
tresults=defaultdict(list)    
# Initialize results storage
results = defaultdict(list)
# Iterate over each combination of scaler and oversampler
for method_name, method in methods:
    clf= method
    for scaler_name, scaler in scalers:
        for sampler_name, sampler in oversamplers:
            
            accuracies=[]  
            f1_scores=[]
            precisions=[]
            recalls=[]
            conf_matrices=[]
            tconf_matricesp=[]
            conf_matricesp=[]
            taccuracies=[]  
            tf1_scores=[]
            tprecisions=[]
            tconf_matrices=[]
            trecalls=[]

            dimportances=[]
            # Perform cross-validation
            for train_idx, test_idx in kf.split(X,y):
            
                #X_train, X_test = X1[train_idx], X1[test_idx]
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # Use .iloc for index-based selection
                
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]  # Use .iloc for index-based selection

                # Handle oversampling
                if sampler is None:
                    X_train = X_train  # No scaling, use the data as is
                    sampleri=None
                    X_test=X_test
    
                   # oversampling_flag = pd.Series([0] * len(X_train), index=X_train.index)  # No rows were oversampled
                else:
                    X_train, y_train = sampler.fit_resample(X_train, y_train)  # Apply oversampling
                    X_train = pd.DataFrame(X_train, columns=X.columns) 


                                                            # Handle scaling
                if scaler is None:
                    X_train = X_train  # No scaling, use the data as is
                    scaleri=None
                    X_test=X_test
                else:
                    scaler.fit(X_train[numeric_columns])
                    
                    scaleri=scaler

                                        # Transform the training data
                    X_train.loc[:, numeric_columns] = scaler.transform(X_train[numeric_columns])
                    
                    # Transform the test data using the same scaler
                    X_test.loc[:, numeric_columns] = scaler.transform(X_test[numeric_columns])

                X_train = pd.DataFrame(X_train, columns=X.columns)
                X_test = pd.DataFrame(X_test, columns=X.columns) 
                feature_names = X_train.columns

                

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                ty_pred = clf.predict(X_train)

                
                # Get permutation importances
                resultss = permutation_importance(clf, X_test, y_test, scoring='accuracy')
                importances = resultss.importances_mean

                    # Print sample sizes
                print(f"  Training sample size (X_train): {X_train.shape[0]}")
                print(f"  Testing sample size (y_test): {y_test.shape[0]}")
                                            
                # Test set metrics
                fold_accuracy = accuracy_score(y_test, y_pred)
                fold_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
                fold_precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
                fold_recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)  # Recall
                fold_conf_matrix = confusion_matrix(y_test, y_pred)
                df_cm = pd.DataFrame(fold_conf_matrix / np.sum(fold_conf_matrix, axis=1)[:, None], index = [i for i in unique_classes], columns = [i for i in unique_classes])
                df_cm = df_cm.to_numpy()  # Convert DataFrame to NumPy array if necessary    
                # Training set metrics
                tfold_accuracy = accuracy_score(y_train, ty_pred)
                tfold_f1 = f1_score(y_train, ty_pred, average='weighted', zero_division=1)
                tfold_precision = precision_score(y_train, ty_pred, average='weighted', zero_division=1)
                tfold_recall = recall_score(y_train, ty_pred, average='weighted', zero_division=1)  # Recall
                tfold_conf_matrix = confusion_matrix(y_train, ty_pred)
                tdf_cm = pd.DataFrame(tfold_conf_matrix / np.sum(tfold_conf_matrix, axis=1)[:, None], index = [i for i in unique_classes], columns = [i for i in unique_classes])
                tdf_cm = tdf_cm.to_numpy()  # Convert DataFrame to NumPy array if necessary
                dimportances.append(importances)
                
                # Append the metrics to lists
                accuracies.append(fold_accuracy)
                f1_scores.append(fold_f1)
                precisions.append(fold_precision)
                recalls.append(fold_recall)  # Append recall
                conf_matrices.append(fold_conf_matrix)
                conf_matricesp.append(df_cm)
                
                taccuracies.append(tfold_accuracy)
                tf1_scores.append(tfold_f1)
                tprecisions.append(tfold_precision)
                trecalls.append(tfold_recall)  # Append recall
                tconf_matrices.append(tfold_conf_matrix)
                tconf_matricesp.append(tdf_cm)
            # Concatenate train and test sets with labels (y)
                X_train['y'] = y_train
                X_test['y'] = y_test

                # Add metadata columns for method, scaler, oversampler, and oversampling flag
                X_train['Method'] = method_name
                X_test['Method'] = method_name
                X_train['Scaler'] = scaler_name
                X_test['Scaler'] = scaler_name
                X_train['Oversampler'] = sampler_name
                X_test['Oversampler'] = sampler_name

                # Append the results to the combined data list
                data_combined=(pd.concat([X_train, X_test]))
                table_save_path = f"5LogTables_clade/table_{method_name}_{scaler_name}_{sampler_name}.csv"

                data_combined.to_csv(table_save_path, index=False)

            

                                    # Calculate mean accuracy across all folds
            mean_accuracy = np.mean(accuracies)
            
      
            # Store the model, scaler, and metrics in a dictionary
            model_info = {
                'accuracy': mean_accuracy,
                'model': clf,
                'scaler': scaleri,
                'method': method_name,
                'scaler_name': scaler_name,
                'sampler_name': sampler_name
            }
            all_models.append(model_info)
            
            # Save the model and scaler
            model_save_path = f"5LogModelResultsnew_clade/model_{method_name}_{scaler_name}_{sampler_name}.joblib"
            scaler_save_path = f"5LogModelResultsnew_clade/scaler_{method_name}_{scaler_name}_{sampler_name}.joblib"
            
            joblib.dump(clf, model_save_path)
            joblib.dump(scaler, scaler_save_path)
            
            print(f"Saved model with accuracy {mean_accuracy} to {model_save_path} and corresponding scaler to {scaler_save_path}")


            
            dimportances=np.array(dimportances)
                # Convert lists to numpy arrays for mean and std calculations
            accuracies = np.array(accuracies)
            precisions = np.array(precisions)
            f1_scores = np.array(f1_scores)
            recalls = np.array(recalls)  # Convert recalls to array
            conf_matrices = np.array(conf_matrices)
            conf_matricesp = np.array(conf_matricesp)            
            
            taccuracies = np.array(taccuracies)
            tprecisions = np.array(tprecisions)
            tf1_scores = np.array(tf1_scores)
            trecalls = np.array(trecalls)  # Convert recalls to array
            tconf_matrices = np.array(tconf_matrices)
            tconf_matricesp = np.array(tconf_matricesp)
            # Calculate mean and standard deviation of metrics
            mean_accuracy = np.mean(accuracies)
            mean_precision = np.mean(precisions)
            mean_f1 = np.mean(f1_scores)
            mean_recall = np.mean(recalls)  # Calculate mean recall
            mean_conf_matrix, sd_conf_matrix = average_confusion_matrices(conf_matrices)
            mean_conf_matrixp, sd_conf_matrixp = average_confusion_matrices(conf_matricesp)
           

            # Calculate the mean importances for each feature across all folds
            mean_importances = np.mean(dimportances, axis=0)
            
            # Sort the mean importances and get the corresponding feature names
            indices = np.argsort(mean_importances)[::-1]
            sorted_importances = mean_importances[indices]
            sorted_feature_names = np.array(feature_names)[indices]
            
            # Plot the sorted mean feature importances
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
            plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Mean Importance')
            plt.title(f'{method_name}+{scaler_name} + {sampler_name} Mean Feature Importances Across Folds')
            filename = f"5LogPlotsnew_clade/{method_name}_{scaler_name}_{sampler_name}_importances.png"
            plt.savefig(filename)
            plt.show()

            
            sd_accuracy = np.std(accuracies)
            sd_precision = np.std(precisions)
            sd_f1 = np.std(f1_scores)
            sd_recall = np.std(recalls)  # Calculate SD of recall
            
            tmean_accuracy = np.mean(taccuracies)
            tmean_precision = np.mean(tprecisions)
            tmean_f1 = np.mean(tf1_scores)
            tmean_recall = np.mean(trecalls)  # Calculate mean recall for training
            tmean_conf_matrix, tsd_conf_matrix = average_confusion_matrices(tconf_matrices)
            tmean_conf_matrixp, tsd_conf_matrixp = average_confusion_matrices(tconf_matricesp)

            
            tsd_accuracy = np.std(taccuracies)
            tsd_precision = np.std(tprecisions)
            tsd_f1 = np.std(tf1_scores)
            tsd_recall = np.std(trecalls)  # Calculate SD of recall for training
            
            # Store the results
            results['Method'].append(method_name)
            results['Scaler'].append(scaler_name)
            results['Oversampler'].append(sampler_name)
            results['MeanAccuracy'].append(mean_accuracy)
            results['SdAccuracy'].append(sd_accuracy)
            results['MeanPrecision'].append(mean_precision)
            results['SdPrecision'].append(sd_precision)
            results['MeanF1-Score'].append(mean_f1)
            results['SdF1-Score'].append(sd_f1)
            results['MeanRecall'].append(mean_recall)
            results['SdRecall'].append(sd_recall)
            results['MeanConfusionMatrix'].append(mean_conf_matrix.tolist())
            results['SdConfusionMatrix'].append(sd_conf_matrix.tolist())
            results['MeanConfusionMatrixProp'].append(mean_conf_matrixp.tolist())
            results['SdConfusionMatrixProp'].append(sd_conf_matrixp.tolist())
            
            tresults['Method'].append(method_name)
            tresults['Scaler'].append(scaler_name)
            tresults['Oversampler'].append(sampler_name)
            tresults['MeanAccuracy'].append(tmean_accuracy)
            tresults['SdAccuracy'].append(tsd_accuracy)
            tresults['MeanPrecision'].append(tmean_precision)
            tresults['SdPrecision'].append(tsd_precision)
            tresults['MeanF1-Score'].append(tmean_f1)
            tresults['SdF1-Score'].append(tsd_f1)
            tresults['MeanRecall'].append(tmean_recall)
            tresults['SdRecall'].append(tsd_recall)
            tresults['MeanConfusionMatrix'].append(tmean_conf_matrix.tolist())
            tresults['SdConfusionMatrix'].append(tsd_conf_matrix.tolist())
            tresults['MeanConfusionMatrixProp'].append(tmean_conf_matrixp.tolist())
            tresults['SdConfusionMatrixProp'].append(tsd_conf_matrixp.tolist())  
            

# Debugging Step: Ensure all lists in results have the same length
for key in results:
    print(f"{key}: {len(results[key])}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

tresults_df = pd.DataFrame(tresults)
print(tresults_df)
# List of columns to exclude
columns_to_exclude = ['MeanConfusionMatrix', 'SdConfusionMatrix','SdConfusionMatrixProp','MeanConfusionMatrixProp']

# Select columns to keep
columns_to_keep = [col for col in results_df.columns if col not in columns_to_exclude]
tcolumns_to_keep = [col for col in tresults_df.columns if col not in columns_to_exclude]

# Create a new DataFrame with only the columns to keep
df_results_filtered = results_df[columns_to_keep]
tdf_results_filtered = tresults_df[tcolumns_to_keep]

# Define the path where you want to save the CSV file
csv_file_path = '5teeth_data_log_clade_resultsnew.csv'
tcsv_file_path = '5teeth_data_log_clade_results_trainnew.csv'

# Save the DataFrame to a CSV file
df_results_filtered.to_csv(csv_file_path, index=False)
tdf_results_filtered.to_csv(tcsv_file_path, index=False)


# Loop through each combination of scaler and oversampler
for k, (method_name, method) in enumerate(methods):
    for i, (scaler_name, scaler) in enumerate(scalers):
        for j, (sampler_name, sampler) in enumerate(oversamplers):
            # Get mean and std confusion matrices for the current combination
            mean_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'MeanConfusionMatrix'
            ].values[0]
            
            std_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'SdConfusionMatrix'
            ].values[0]
            
            # Create a figure with 2 subplots: one for mean, one for std
            plt.figure(figsize=(30, 26))
            
            # Plot the mean confusion matrix
            plt.subplot(1, 1, 1)
            
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(mean_cm, annot=True, fmt='.0f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
       #     plt.title(f'{method_name}+ {scaler_name} + {sampler_name} (Mean)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
                        
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/mean_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()
            
            # Plot the standard deviation confusion matrix
            plt.figure(figsize=(30, 26))
            plt.subplot(1, 1, 1)
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(std_cm, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
           # plt.title(f'{method_name}+{scaler_name} + {sampler_name} (Std Dev)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
            
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/sd_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()


            pmean_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'MeanConfusionMatrixProp'
            ].values[0]
            
            pstd_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'SdConfusionMatrixProp'
            ].values[0]
            
            # Create a figure with 2 subplots: one for mean, one for std
            plt.figure(figsize=(30, 26))
            
            # Plot the mean confusion matrix
            plt.subplot(1, 1, 1)
            
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(pmean_cm, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
       #     plt.title(f'{method_name}+ {scaler_name} + {sampler_name} (Mean)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
                        
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/prop_mean_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()
            
            # Plot the standard deviation confusion matrix
            plt.figure(figsize=(30, 26))
            plt.subplot(1, 1, 1)
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(pstd_cm, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
           # plt.title(f'{method_name}+{scaler_name} + {sampler_name} (Std Dev)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
            
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/prop_sd_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()
            

            from sklearn.base import BaseEstimator, ClassifierMixin



all_models = []
methods=[
    ('Neural Network')
]

# Create k-fold cross-validation
k_folds = 5  # Number of folds
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)


def average_confusion_matrices(conf_matrices):
    """Average confusion matrices ensuring consistent dimensions."""
    # Determine the maximum shape
    max_rows = max(cm.shape[0] for cm in conf_matrices)
    max_cols = max(cm.shape[1] for cm in conf_matrices)
    
    # Create a list of matrices with the same shape
    padded_matrices = []
    for cm in conf_matrices:
        # Pad confusion matrices to the largest shape
        padded_cm = np.zeros((max_rows, max_cols), dtype=cm.dtype)
        padded_cm[:cm.shape[0], :cm.shape[1]] = cm
        padded_matrices.append(padded_cm)
    
    # Convert list to numpy array
    padded_matrices = np.array(padded_matrices)
    
    # Calculate mean and std
    mean_cm = np.mean(padded_matrices, axis=0)
    std_cm = np.std(padded_matrices, axis=0)
    
    return mean_cm, std_cm
    
# Initialize results storage
results = defaultdict(list)
# Iterate over each combination of scaler and oversampler
tresults=defaultdict(list)  


class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=50, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.classes_ = None  # Initialize classes_ attribute
    
    def build_model(self, input_shape, num_classes):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def fit(self, X, y):
        num_classes = len(np.unique(y))
        self.model = self.build_model(X.shape[1], num_classes)
        self.classes_ = np.unique(y)  # Set the classes_ attribute
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self
    
    def predict(self, X):
        y_pred_prob = self.model.predict(X)
        return np.argmax(y_pred_prob, axis=1)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
# Example usage of the KerasClassifierWrapper
methods = [('Neural Network')]

# Create k-fold cross-validation
k_folds = 5  # Number of folds
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize results storage
results = defaultdict(list)
tresults = defaultdict(list)

# Iterate over each combination of scaler and oversampler
for method_name in methods:
    for scaler_name, scaler in scalers:
        for sampler_name, sampler in oversamplers:
    
            # Lists to store performance metrics
            accuracies = []  
            f1_scores = []
            precisions = []
            recalls = []
            conf_matrices = []
            taccuracies = []  
            tf1_scores = []
            tprecisions = []
            trecalls = []
            tconf_matrices = []
            tconf_matricesp=[]
            conf_matricesp=[]
            dimportances = []
            
            # Perform cross-validation
            for train_idx, test_idx in kf.split(X,y):
                
                # Use .iloc for index-based selection
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Handle oversampling
                if sampler is None:
                    X_train = X_train  # No scaling, use the data as is
                    sampleri=None
                    X_test=X_test
    
                   # oversampling_flag = pd.Series([0] * len(X_train), index=X_train.index)  # No rows were oversampled
                else:
                    X_train, y_train = sampler.fit_resample(X_train, y_train)  # Apply oversampling
                    X_train = pd.DataFrame(X_train, columns=X.columns) 
                 

                                                            # Handle scaling
                if scaler is None:
                    X_train = X_train  # No scaling, use the data as is
                    X_test=X_test
                else:
                    scaler.fit(X_train[numeric_columns])
                    
                    scaleri=scaler

                                # Transform the training data
                    X_train.loc[:, numeric_columns] = scaler.transform(X_train[numeric_columns])
                    
                    # Transform the test data using the same scaler
                    X_test.loc[:, numeric_columns] = scaler.transform(X_test[numeric_columns])
                
                X_train = pd.DataFrame(X_train, columns=X.columns)
                X_test = pd.DataFrame(X_test, columns=X.columns) 
                
                feature_names = X_train.columns
                # Encode labels
                label_encoder = LabelEncoder()
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)
                
                # Build and train the model
                clf = KerasClassifierWrapper(epochs=50, batch_size=32)
                clf.fit(X_train, y_train)
                
                # Predict on the test set
                y_pred = clf.predict(X_test)
                
                # Training set predictions
                ty_pred = clf.predict(X_train)
                
                # Get permutation importances
                resultss = permutation_importance(clf, X_test, y_test, scoring='accuracy')
                importances = resultss.importances_mean
                
                # Test set metrics
                fold_accuracy = accuracy_score(y_test, y_pred)
                fold_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
                fold_precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
                fold_recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)  # Recall
                fold_conf_matrix = confusion_matrix(y_test, y_pred)
                df_cm = pd.DataFrame(fold_conf_matrix / np.sum(fold_conf_matrix, axis=1)[:, None], index = [i for i in unique_classes], columns = [i for i in unique_classes])
                df_cm = df_cm.to_numpy()  # Convert DataFrame to NumPy array if necessary
                # Training set metrics
                tfold_accuracy = accuracy_score(y_train, ty_pred)
                tfold_f1 = f1_score(y_train, ty_pred, average='weighted', zero_division=1)
                tfold_precision = precision_score(y_train, ty_pred, average='weighted', zero_division=1)
                tfold_recall = recall_score(y_train, ty_pred, average='weighted', zero_division=1)  # Recall
                tfold_conf_matrix = confusion_matrix(y_train, ty_pred)
                tdf_cm = pd.DataFrame(fold_conf_matrix / np.sum(tfold_conf_matrix, axis=1)[:, None], index = [i for i in unique_classes], columns = [i for i in unique_classes])
                tdf_cm = tdf_cm.to_numpy()  # Convert DataFrame to NumPy array if necessary
                
                
                # Append the metrics to lists
                accuracies.append(fold_accuracy)
                f1_scores.append(fold_f1)
                precisions.append(fold_precision)
                recalls.append(fold_recall)
                conf_matrices.append(fold_conf_matrix)
                conf_matricesp.append(df_cm)
                
                taccuracies.append(tfold_accuracy)
                tf1_scores.append(tfold_f1)
                tprecisions.append(tfold_precision)
                trecalls.append(tfold_recall)
                tconf_matrices.append(tfold_conf_matrix)
                tconf_matricesp.append(tdf_cm)
                dimportances.append(importances)

             # Concatenate train and test sets with labels (y)
                X_train['y'] = y_train
                X_test['y'] = y_test

                # Add metadata columns for method, scaler, oversampler, and oversampling flag
                X_train['Method'] = method_name
                X_test['Method'] = method_name
                X_train['Scaler'] = scaler_name
                X_test['Scaler'] = scaler_name
                X_train['Oversampler'] = sampler_name
                X_test['Oversampler'] = sampler_name

                # Append the results to the combined data list
                data_combined=(pd.concat([X_train, X_test]))
                table_save_path = f"5LogTables_clade/table_{method_name}_{scaler_name}_{sampler_name}.csv"

                data_combined.to_csv(table_save_path, index=False)

                         # Calculate mean accuracy across all folds
            mean_accuracy = np.mean(accuracies)
            
      
            # Store the model, scaler, and metrics in a dictionary
            model_info = {
                'accuracy': mean_accuracy,
                'model': clf,
                'scaler': scaleri,
                'method': method_name,
                'scaler_name': scaler_name,
                'sampler_name': sampler_name
            }
            all_models.append(model_info)
            
            # Save the model and scaler
            model_save_path = f"5LogModelResultsnew_clade/model_{method_name}_{scaler_name}_{sampler_name}.joblib"
            scaler_save_path = f"5LogModelResultsnew_clade/scaler_{method_name}_{scaler_name}_{sampler_name}.joblib"
            
            joblib.dump(clf, model_save_path)
            joblib.dump(scaler, scaler_save_path)
            
            print(f"Saved model with accuracy {mean_accuracy} to {model_save_path} and corresponding scaler to {scaler_save_path}")

            

            dimportances = np.array(dimportances)
            # Convert lists to numpy arrays for mean and std calculations
            accuracies = np.array(accuracies)
            precisions = np.array(precisions)
            f1_scores = np.array(f1_scores)
            recalls = np.array(recalls)
            conf_matrices = np.array(conf_matrices)
            conf_matricesp = np.array(conf_matricesp)
            
            taccuracies = np.array(taccuracies)
            tprecisions = np.array(tprecisions)
            tf1_scores = np.array(tf1_scores)
            trecalls = np.array(trecalls)
            tconf_matrices = np.array(tconf_matrices)
            conf_matricesp = np.array(conf_matricesp)           
            
            # Calculate mean and standard deviation of metrics
            mean_accuracy = np.mean(accuracies)
            mean_precision = np.mean(precisions)
            mean_f1 = np.mean(f1_scores)
            mean_recall = np.mean(recalls)
            mean_conf_matrix, sd_conf_matrix = average_confusion_matrices(conf_matrices)
            mean_conf_matrixp, sd_conf_matrixp = average_confusion_matrices(conf_matricesp)
           
            sd_accuracy = np.std(accuracies)
            sd_precision = np.std(precisions)
            sd_f1 = np.std(f1_scores)
            sd_recall = np.std(recalls)
            
            tmean_accuracy = np.mean(taccuracies)
            tmean_precision = np.mean(tprecisions)
            tmean_f1 = np.mean(tf1_scores)
            tmean_recall = np.mean(trecalls)
            tmean_conf_matrix, tsd_conf_matrix = average_confusion_matrices(tconf_matrices)
            tmean_conf_matrixp, tsd_conf_matrixp = average_confusion_matrices(tconf_matricesp)
           

            # Calculate the mean importances for each feature across all folds
            mean_importances = np.mean(dimportances, axis=0)
            
            # Sort the mean importances and get the corresponding feature names
            indices = np.argsort(mean_importances)[::-1]
            sorted_importances = mean_importances[indices]
            sorted_feature_names = np.array(feature_names)[indices]
            
            # Plot the sorted mean feature importances
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
            plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Mean Importance')
            plt.title(f'{method_name}+{scaler_name} + {sampler_name} Mean Feature Importances Across Folds')
            filename = f"5LogPlotsnew_clade/{method_name}_{scaler_name}_{sampler_name}_importances.png"
            plt.savefig(filename)
            plt.show()

            tsd_accuracy = np.std(taccuracies)
            tsd_precision = np.std(tprecisions)
            tsd_f1 = np.std(tf1_scores)
            tsd_recall = np.std(trecalls)
            
            # Store the results
            results['Method'].append(method_name)
            results['Scaler'].append(scaler_name)
            results['Oversampler'].append(sampler_name)
            results['MeanAccuracy'].append(mean_accuracy)
            results['SdAccuracy'].append(sd_accuracy)
            results['MeanPrecision'].append(mean_precision)
            results['SdPrecision'].append(sd_precision)
            results['MeanF1-Score'].append(mean_f1)
            results['SdF1-Score'].append(sd_f1)
            results['MeanRecall'].append(mean_recall)
            results['SdRecall'].append(sd_recall)
            results['MeanConfusionMatrix'].append(mean_conf_matrix.tolist())
            results['SdConfusionMatrix'].append(sd_conf_matrix.tolist())
            results['MeanConfusionMatrixProp'].append(mean_conf_matrixp.tolist())
            results['SdConfusionMatrixProp'].append(sd_conf_matrixp.tolist())
            
            tresults['Method'].append(method_name)
            tresults['Scaler'].append(scaler_name)
            tresults['Oversampler'].append(sampler_name)
            tresults['MeanAccuracy'].append(tmean_accuracy)
            tresults['SdAccuracy'].append(tsd_accuracy)
            tresults['MeanF1-Score'].append(tmean_f1)
            tresults['SdF1-Score'].append(tsd_f1)
            tresults['MeanRecall'].append(tmean_recall)
            tresults['SdRecall'].append(tsd_recall)
            tresults['MeanConfusionMatrix'].append(tmean_conf_matrix.tolist())
            tresults['SdConfusionMatrix'].append(tsd_conf_matrix.tolist())
            tresults['MeanConfusionMatrixProp'].append(tmean_conf_matrixp.tolist())
            tresults['SdConfusionMatrixProp'].append(tsd_conf_matrixp.tolist())



# Debugging Step: Ensure all lists in results have the same length
for key in results:
    print(f"{key}: {len(results[key])}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

tresults_df = pd.DataFrame(tresults)
print(tresults_df)
# List of columns to exclude
columns_to_exclude = ['MeanConfusionMatrix', 'SdConfusionMatrix','SdConfusionMatrixProp','MeanConfusionMatrixProp']

# Select columns to keep
columns_to_keep = [col for col in results_df.columns if col not in columns_to_exclude]
tcolumns_to_keep = [col for col in tresults_df.columns if col not in columns_to_exclude]

# Create a new DataFrame with only the columns to keep
df_results_filtered = results_df[columns_to_keep]
tdf_results_filtered = tresults_df[tcolumns_to_keep]

# Define the path where you want to save the CSV file
csv_file_path = '5teeth_data_log_clade_results_nnnew.csv'
tcsv_file_path = '5teeth_data_log_clade_results_train_nnnew.csv'

# Save the DataFrame to a CSV file
df_results_filtered.to_csv(csv_file_path, index=False)
tdf_results_filtered.to_csv(tcsv_file_path, index=False)



# Loop through each combination of scaler and oversampler
for k, (method_name) in enumerate(methods):
    for i, (scaler_name, scaler) in enumerate(scalers):
        for j, (sampler_name, sampler) in enumerate(oversamplers):
            # Get mean and std confusion matrices for the current combination
            mean_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'MeanConfusionMatrix'
            ].values[0]
            
            std_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'SdConfusionMatrix'
            ].values[0]
            
            # Create a figure with 2 subplots: one for mean, one for std
            plt.figure(figsize=(30, 26))
            
            # Plot the mean confusion matrix
            plt.subplot(1, 1, 1)
            
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(mean_cm, annot=True, fmt='.0f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
       #     plt.title(f'{method_name}+ {scaler_name} + {sampler_name} (Mean)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
           
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/mean_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()
            
            # Plot the standard deviation confusion matrix
            plt.figure(figsize=(30, 26))
            plt.subplot(1, 1, 1)
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(std_cm, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
           # plt.title(f'{method_name}+{scaler_name} + {sampler_name} (Std Dev)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
            
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/sd_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()


            pmean_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'MeanConfusionMatrixProp'
            ].values[0]
            
            pstd_cm = results_df.loc[
                (results_df['Scaler'] == scaler_name) & 
                (results_df['Method'] == method_name) & 
                (results_df['Oversampler'] == sampler_name),
                'SdConfusionMatrixProp'
            ].values[0]
            
            # Create a figure with 2 subplots: one for mean, one for std
            plt.figure(figsize=(30, 26))
            
            # Plot the mean confusion matrix
            plt.subplot(1, 1, 1)
            
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(pmean_cm, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
       #     plt.title(f'{method_name}+ {scaler_name} + {sampler_name} (Mean)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
                        
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/prop_mean_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()
            
            # Plot the standard deviation confusion matrix
            plt.figure(figsize=(30, 26))
            plt.subplot(1, 1, 1)
            sns.set(font_scale=3.5)  # Increase font scale
            sns.heatmap(pstd_cm, annot=True, fmt='.1f', cmap='Blues', annot_kws={"size":21},
                        cbar=False, square=True, 
                        xticklabels=unique_classes,  # Adjust according to your classes
                        yticklabels=unique_classes)  # Adjust according to your classes
           # plt.title(f'{method_name}+{scaler_name} + {sampler_name} (Std Dev)')
            plt.xlabel('Predicted class', fontsize=22)
            plt.ylabel('True class', fontsize=22)
            plt.xticks(rotation=90, fontstyle='italic')
            plt.yticks(fontstyle='italic')
            
            # Adjust layout and show the plot
            plt.tight_layout()
            filename = f"5LogPlotsnew_clade/prop_sd_{method_name}_{scaler_name}_{sampler_name}_cm.png"
            plt.savefig(filename)
            plt.show()
            