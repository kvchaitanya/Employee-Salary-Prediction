# -*- coding: utf-8 -*-
"""
Author: Venkata Chaitanya Kanakamedala

**Phase 1**
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from prettytable import PrettyTable
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from prettytable import PrettyTable
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

"""Reading Data from CSV"""

df = pd.read_csv('batch2_jobID_00B80TR.csv')
df.head()

"""Checking for Missing Data"""

count_missing_data=df.isnull().sum()
print(count_missing_data)

"""Check for Null values"""

df.replace({"NONE": np.nan}, inplace=True)
df.dropna(inplace=True,axis=0)

"""Check for Duplicates"""

# Check for duplicates
duplicates = df[df.duplicated()]

# Print the duplicates
print("Duplicate Rows except first occurrence:")
print(duplicates)

print("\nNumber of duplicate rows (excluding the first occurrence):", duplicates.shape[0])

# Remove duplicates from the original DataFrame
df = df.drop_duplicates()

# Print the DataFrame after removing duplicates
print(df)

duplicates = df[df.duplicated()]

# Print the duplicates
print("Duplicate Rows except first occurrence:")
print(duplicates)

"""Adding classification variable"""

# Add high salary column
threshold_salary = df['salary'].mean()
df['high_salary'] = df['salary'].apply(lambda x: 1 if x > threshold_salary else 0)

# Plot the distribution of high salary
print(df.head())
plt.figure(figsize=(10, 6))
# plt.hist(df['high_salary'], bins=10, color='skyblue', edgecolor='black')
plt.hist(df['high_salary'], bins=10, facecolor='orange', edgecolor='blue', alpha=0.7)
plt.title('Distribution of High Salary')
plt.xlabel('High Salary (1: Yes, 0: No)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

"""Undersampling"""


X = df.drop(columns=['high_salary'])
y = df['high_salary']
under_sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = under_sampler.fit_resample(X, y)

print("Class distribution after undersampling:")
print(y_resampled.value_counts())

# Plot the distribution of high salary

plt.figure(figsize=(10, 6))
plt.title('Distribution of High Salary')
plt.xlabel('High Salary (1: Yes, 0: No)')
plt.ylabel('Frequency')
plt.hist(y_resampled, bins=10, color='skyblue', edgecolor='black')
plt.grid()
plt.show()

final_df = df[df["high_salary"] == 0].sample(100000)
final_df = pd.concat([final_df, df[df["high_salary"] == 1].sample(100000)])
final_df=final_df.drop('companyId',axis=1)

# Plot the distribution of high salary
plt.figure(figsize=(10, 6))
plt.title('Distribution of High Salary')
plt.xlabel('High Salary (1: Yes, 0: No)')
plt.ylabel('Frequency')
plt.hist(final_df["high_salary"], bins=10, color='green', edgecolor='black')
plt.grid()
plt.show()

"""Skewness"""

# Assuming final_df is your DataFrame
numerical_features = final_df['salary']

# Calculate skewness for each numerical feature
skewness = numerical_features.skew()

# Display skewness summary
print("Skewness Summary:")
print(skewness)

"""One hot Encoding"""

for columns in final_df:
    print(columns)

# One hot encoding
categorical_features = final_df.select_dtypes(include=object).columns.tolist()
numerical_features = final_df.select_dtypes(include='number').columns.tolist()
print("\ncategorical_features\n", categorical_features)
print("\nnumerical_features\n", numerical_features)

# final_df.head()

# final_df.columns

# Perform one-hot encoding
df_encoded = pd.get_dummies(final_df, columns=categorical_features, drop_first=True)
df_encoded = df_encoded.astype(int)

# df_encoded.columns

print(len(df_encoded.columns))

regression_df=df_encoded.drop('high_salary',axis=1)

"""Splitting the dataset"""

# Separate features (X) and target variable (y)
y = regression_df['salary']
X = regression_df.drop(columns=['salary'])

# X
#
# y.name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

"""Standardization for Regression"""

x_scaler = StandardScaler()
X_train[['yearsExperience', 'milesFromMetropolis']] = x_scaler.fit_transform(X_train[['yearsExperience', 'milesFromMetropolis']])
X_test[['yearsExperience', 'milesFromMetropolis']] = x_scaler.transform(X_test[['yearsExperience', 'milesFromMetropolis']])
X_train_std = X_train
X_test_std = X_test

# Target standardization
y_scaler = StandardScaler()
y_train_std = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_std = y_scaler.transform(y_test.values.reshape(-1, 1))

columns_num=['yearsExperience','milesFromMetropolis']
sns.boxplot(data=X_train_std[columns_num])
plt.ylabel("values")
plt.title("Boxplot for the yearsExperience & milesFromMetropolis")
plt.show()

"""Outliers"""

import pandas as pd
# Calculate the interquartile range (IQR)
First_Quartile = df_encoded['yearsExperience'].quantile(0.25)
Third_Quartile = df_encoded['yearsExperience'].quantile(0.75)
IQR = Third_Quartile - First_Quartile

# Define a threshold for outliers (e.g., 1.5 times the IQR)
threshold = 1.5 * IQR

# Identify outliers
outliers = df_encoded[(df_encoded['yearsExperience'] < First_Quartile - threshold) | (df_encoded['yearsExperience'] > Third_Quartile + threshold)]

# Print the number of outliers
print(f'Number of outliers: {len(outliers)}')

import pandas as pd
# Calculate the interquartile range (IQR)
First_Quartile = df_encoded['milesFromMetropolis'].quantile(0.25)
Third_Quartile = df_encoded['milesFromMetropolis'].quantile(0.75)
IQR = Third_Quartile - First_Quartile

# Define a threshold for outliers (e.g., 1.5 times the IQR)
threshold = 1.5 * IQR

# Identify outliers
outliers = df_encoded[(df_encoded['milesFromMetropolis'] < First_Quartile - threshold) | (df_encoded['milesFromMetropolis'] > Third_Quartile + threshold)]

# Print the number of outliers
print(f'Number of outliers: {len(outliers)}')

columns_num=['yearsExperience','milesFromMetropolis']
sns.boxplot(data=X_train_std[columns_num])
plt.ylabel("values")
plt.title("Boxplot for the yearsExperience & milesFromMetropolis")
plt.show()

"""Random forest regressor"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

# RFA
model = RandomForestRegressor()
X_train_std = pd.DataFrame(data=X_train_std, columns=X_train_std.columns)
standardized_flat_y_train = y_train_std.ravel()
standardized_flat_y_train = pd.DataFrame(standardized_flat_y_train)
rf_model=model.fit(X_train_std, standardized_flat_y_train)

feature_imp = model.feature_importances_
data_dict = {'features_train': X_train_std.columns, 'Importance': feature_imp}
df_data = pd.DataFrame(data_dict)
df_data.sort_values('Importance', ascending=False, inplace=True)


Variance_factor_attributes = 0.95
total_variance_value = df_data['Importance'].cumsum()
Best_features_high_variance = df_data[total_variance_value <= Variance_factor_attributes]
print(f"Selected Features with Cumulative Importance <= {Variance_factor_attributes}:")
for feature, importance in zip(Best_features_high_variance['features_train'], Best_features_high_variance['Importance']):
    print(f"{feature}: {importance}")

# Plot the cumulative importance
plt.plot(range(1, len(df_data) + 1), total_variance_value, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Importance of Features')
plt.axhline(y=Variance_factor_attributes, color='r', linestyle='--', label=f'Threshold ({Variance_factor_attributes * 100}%)')
plt.axvline(x=19, color='red', linestyle='--', label='Vertical Line at x=19')
plt.legend()

# Plotting the horizontal bar graph for selected features
plt.figure(figsize=(12, 10))
plt.xlabel('Importance')
plt.ylabel('features_train')
plt.title(f'Horizontal Bar graph for features with Cumulative Importance <= {Variance_factor_attributes} in descending order')
plt.barh(Best_features_high_variance['features_train'], Best_features_high_variance['Importance'], color='skyblue')
plt.show()


"""Principal Component Analysis"""

print(f'PCA: condition number for reduced data (important features): {np.linalg.cond(X_train_std):.2f}')

len(X_train_std.columns)

# PCA
from sklearn.decomposition import PCA
pca = PCA()
pca_X_trained_vale = pca.fit_transform(X_train_std)
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.xlabel('Number of features')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance vs. Number of features')
plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o')
num_features_95_percent = np.argmax(explained_variance_ratio_cumsum > 0.95) + 1
plt.legend()
plt.grid()
plt.show()

print(f"\nNumber of features_train needed to explain more than 95% of the variance: {num_features_95_percent}")
print(f"Cumulative explained variance with {num_features_95_percent} features_train: {explained_variance_ratio_cumsum[num_features_95_percent - 1]:.3f}")
print("\nExplained Variance of Principal Components:")
print([f"{ratio:.3f}" for ratio in pca.explained_variance_ratio_])

# PCA
from sklearn.decomposition import PCA
# num_features_95_percent = np.argmax(explained_variance_ratio_cumsum > 0.95) + 1
pca = PCA(n_components=19, svd_solver='full')
pca_X_trained_vale = pca.fit_transform(X_train_std)
print(f'PCA: condition number for reduced data (important features): {np.linalg.cond(pca_X_trained_vale):.2f}')

plt.figure(figsize=(16, 6))

# Subplot 2: Explained Variance Ratio of Principal Components
plt.subplot(1, 2, 2)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='orange')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.tight_layout()
plt.grid()
plt.title('Explained Variance Ratio for Each Principal Component')

plt.tight_layout()
plt.show()

"""SVD (Singular Value Decomposition)"""

from prettytable import PrettyTable

# Singular Value Decomposition Analysis
print("\n SVD \n")
sin_val = np.linalg.svd(X_train_std, compute_uv=False)

# Print singular values
singualr_value_table = PrettyTable()
singualr_value_table.field_names = ['Singular Value']
singualr_value_table.add_row(['{:.2f}'.format(sin_val[0])])
singualr_value_table.add_row(['{:.2f}'.format(sin_val[1])])
singualr_value_table.title = 'Singular Values'
print(singualr_value_table)

"""VIF analysis"""
print("\n VIF Analysis \n")

dropto_columns=['salary']
vif_df=df_encoded.drop(columns=dropto_columns)
from statsmodels.stats.outliers_influence import variance_inflation_factor
data_for_vif_analysis = pd.DataFrame()
data_for_vif_analysis['feature'] = vif_df.columns
data_for_vif_analysis['VIF'] = [variance_inflation_factor(vif_df.values, i) for i in
range(len(vif_df.columns))]
print(data_for_vif_analysis)

"""Heatmap"""

import seaborn as sns
import matplotlib.pyplot as plt

numerical_features = final_df.select_dtypes(include='number')
covariance_matrix_num_fea = numerical_features.cov()

# Create a heatmap of the covariance matrix
plt.figure(figsize=(8, 8))
sns.heatmap(covariance_matrix_num_fea, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5)
plt.title('Covariance Matrix Heatmap for Numerical Features')
plt.show()

correlation_matrix = X_train_std.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0.5)
plt.title('Pearson Correlation Coefficients Heatmap')
plt.show()

# # df.columns
#
# """**`Phase 2 `**
#
# ---
#
#
# """
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy import stats

imp_columns = [
    'milesFromMetropolis',
    'yearsExperience',
    'jobType_JUNIOR',
    'jobType_SENIOR',
    'degree_MASTERS',
    'degree_DOCTORAL',
    'jobType_MANAGER',
    'industry_EDUCATION',
    'major_BUSINESS',
    'major_ENGINEERING',
    'major_MATH',
    'major_PHYSICS',
    'major_CHEMISTRY',
    'industry_SERVICE',
    'major_COMPSCI',
    'major_LITERATURE',
    'industry_FINANCE',
    'industry_OIL',
    'jobType_CFO',
]

# milesFromMetropolis: 0.30015863857564884
# yearsExperience: 0.22230081988648398
# jobType_JUNIOR: 0.05955302062221151
# jobType_SENIOR: 0.04274268423012232
# degree_MASTERS: 0.030957701204619617
# degree_DOCTORAL: 0.02983924922631353
# jobType_MANAGER: 0.028294485025432862
# industry_EDUCATION: 0.02585874981536761
# major_BUSINESS: 0.02178348859311928
# major_ENGINEERING: 0.021681824835159184
# major_MATH: 0.020773931600956603
# major_PHYSICS: 0.020411676244295383
# major_CHEMISTRY: 0.020363250095257364
# industry_SERVICE: 0.02017365609583438
# major_COMPSCI: 0.02016097574435642
# major_LITERATURE: 0.01907610117405134
# industry_FINANCE: 0.01614150766907083
# industry_OIL: 0.01610671485069441
# jobType_CFO

# X_train_std.columns

imp_fea_df=df_encoded[imp_columns]
imp_fea_df.head()

print("\n2 Printing OLS summary \n)")
model_initial = sm.OLS(y_train_std, X_train_std).fit()
X_train_std_copy = X_train_std.copy()

# Initial model_initial Summary
print("Initial model_initial Summary:")
print(model_initial.summary())

"""Linear Regression"""
print("\n Linear Regression \n")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_std, y_train_std)
y_train_pred = model.predict(X_train_std)
y_test_pred = model.predict(X_test_std)
X_train_selected=sm.add_constant(X_train_std)
ols_model=sm.OLS(y_train_std,X_train_selected).fit()
print(ols_model.summary())
X_train_selected = sm.add_constant(X_train_std)
ols_model = sm.OLS(y_train_std, X_train_selected).fit()

# Obtain AIC, BIC, and MSE values
aic_value = ols_model.aic
bic_value = ols_model.bic
mse_value = ((ols_model.resid) ** 2).mean()
# Obtain R-squared value
r_squared_value = ols_model.rsquared
# Print the values
print(f'AIC: {aic_value}')
print(f'BIC: {bic_value}')
print(f'R-squared: {r_squared_value}')
print(f'MSE: {mse_value}')
confidence_intervals = ols_model.conf_int()

# Print confidence intervals
print("Confidence Intervals:")
print(confidence_intervals)

# # # T test Analysis
t_test_results = model_initial.summary().tables[1]
print("T Test Results:")
print(t_test_results)

# # # F-test analysis
f_test_results = model_initial.f_pvalue
print("F Test Results:")
print(f_test_results)
#
# # X_train_std.drop(['industry_HEALTH'],axis=1,inplace=True)
# # model_initial = sm.OLS(y_train_std,X_train_std).fit()
# # print(model_initial.summary()) Need to check if we can remove any values
# #%%

print("\n Linear Regression done \n")
"""Polynomial Regression"""
print("\n Polynomial Regression \n")
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
multi_non_linear_features = PolynomialFeatures(degree=3)
fitted_features = multi_non_linear_features.fit_transform(X_train_std)
model = sm.OLS(y_train_std, fitted_features)
non_linear_polynomial = model.fit()
print(non_linear_polynomial.summary())
xt = multi_non_linear_features.fit_transform(X_test_std)
predicting_y= non_linear_polynomial.predict(xt)

# """Backward Stepwise regression or forward stepwise regression"""


"""**Phase 3**

"""

classification_df=df_encoded.drop('salary',axis=1)

# Separate features (X) and target variable (y)
X = classification_df.drop(columns=['high_salary'])
y = classification_df['high_salary']

"""Standardization"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)
x_scaler = StandardScaler()
X_train[['yearsExperience', 'milesFromMetropolis']] = x_scaler.fit_transform(X_train[['yearsExperience', 'milesFromMetropolis']])
X_test[['yearsExperience', 'milesFromMetropolis']] = x_scaler.transform(X_test[['yearsExperience', 'milesFromMetropolis']])
X_train_std = X_train
X_test_std = X_test

# Target standardization
y_train_std = y_train
y_test_std = y_test

"""Selected important cloumns from RFA & VIF"""

columns_to_drop = ['industry_HEALTH', 'jobType_VICE_PRESIDENT', 'industry_WEB', 'jobType_CTO']
opt_X_train = X_train_std.drop(columns=columns_to_drop)

# Drop the same columns from X_test
opt_X_test = X_test.drop(columns=columns_to_drop)
#

def cross_val_score_without_parallel(model, X, y, cv):
    scores = []
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y

    for train, test in cv.split(X_array, y_array):
        model.fit(X_array[train], y_array[train])
        y_pred = model.predict(X_array[test])
        accuracy = accuracy_score(y_array[test], y_pred)
        scores.append(accuracy)
    return scores


# Pre pruning
"""Trial -2 prepruning"""
print("\n Pre pruning \n")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

tuned_parameters = [{'max_depth': [5, 10],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4],
                     'max_features': [1, 20],
                     'splitter': ['best', 'random'],
                     'criterion': ['gini', 'entropy', 'log_loss']}]
dt_classifier = DecisionTreeClassifier()
grid_search = GridSearchCV(dt_classifier, tuned_parameters, cv=5, scoring='accuracy')
grid_search.fit(opt_X_train, y_train_std)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_model = grid_search.best_estimator_
y_test_predicted = best_model.predict(opt_X_test)
test_accuracy = accuracy_score(y_test_std, y_test_predicted)

print(f'Test accuracy: {round(test_accuracy, 2)}')
# plt.figure(figsize=(20, 12))
# plot_tree(best_model, rounded=True, filled=True, feature_names=X.columns.tolist())
# plt.show()

plt.figure(figsize=(20, 12))
plot_tree(best_model, rounded=True, filled=True, feature_names=opt_X_train.columns.tolist())
plt.show()


# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score

conf_matrix = confusion_matrix(y_test, y_test_predicted)
print("Confusion Matrix:")
print(conf_matrix)

# Display Confusion Matrix
confusion_matrix_neural_network = confusion_matrix(y_test_std, y_test_predicted)
sns.heatmap(confusion_matrix_neural_network, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Decision Tree Confusion Matrix")
plt.show()


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_std, y_test_predicted))

# Extract precision, recall, f1-score, and support individually
precision, recall, fscore, support = precision_recall_fscore_support(y_test_std, y_test_predicted)
accuracy = accuracy_score(y_test_std, y_test_predicted)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)

# Print individual values
print("\nIndividual Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", fscore)
print("Accuracy:", accuracy)
print("Specificity:", specificity)

from sklearn.metrics import roc_curve, auc

# Assuming X_test_std and y_test_std are your standardized test data
y_test_prob = best_model.predict_proba(opt_X_test)[:, 1]
# Calculate false positive rate (fpr), true positive rate (tpr), and thresholds
fpr, tpr, thresholds = roc_curve(y_test_std, y_test_prob)
# Calculate AUC
roc_auc = auc(fpr, tpr)
# Plot both ROC curve and AUC on the same plot
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve with AUC')
plt.legend(loc="lower right")
# Mark the AUC value on the plot
plt.text(0.6, 0.2, 'AUC = {:.2f}'.format(roc_auc), fontsize=12)
plt.show()

n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score(dt_classifier, opt_X_train, y_train_std, cv=stratified_kfold, scoring='accuracy')

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", cross_val_results.mean())

# # Post pruning
#
# """Decision Trees"""
print("\n Post pruning \n")
model = DecisionTreeClassifier(random_state=5805)
path = model.cost_complexity_pruning_path(opt_X_train,y_train_std)
alphas = path['ccp_alphas']
print(len(alphas))

"""test

"""


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np


# Tune max_depth
max_depths = range(1, 21)
max_test_accuracy_depth = 0
optimal_max_depth = None

for max_depth in max_depths:
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=5805)
    model.fit(opt_X_train, y_train_std)

    test_acc = cross_val_score(model, opt_X_test, y_test_std, cv=5).mean()

    if test_acc > max_test_accuracy_depth:
        max_test_accuracy_depth = test_acc
        optimal_max_depth = max_depth

# Tune ccp_alpha
alphas = np.linspace(0, 0.05, 20)  # Adjust the range of alphas based on your needs
max_test_accuracy_alpha = 0
optimal_alpha = None

for alpha in alphas:
    model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=5805)
    model.fit(opt_X_train, y_train_std)

    # Testing accuracy using cross-validation
    test_acc = cross_val_score(model, opt_X_test, y_test_std, cv=5).mean()

    # Update optimal values if current test accuracy is higher
    if test_acc > max_test_accuracy_alpha:
        max_test_accuracy_alpha = test_acc
        optimal_alpha = alpha

# Print optimal values
print(f"Optimal max_depth: {optimal_max_depth} with max test accuracy: {max_test_accuracy_depth:.4f}")
print(f"Optimal alpha: {optimal_alpha} with max test accuracy: {max_test_accuracy_alpha:.4f}")
# Post-pruning Decision Tree
post_pruning_model = DecisionTreeClassifier(ccp_alpha=optimal_alpha, random_state=5805)
post_pruning_model.fit(opt_X_train, y_train_std)

# Plot the tree
plt.figure(figsize=(20, 12))
plot_tree(post_pruning_model, rounded=True, filled=True, feature_names=X.columns.tolist())
plt.show()

#
"""Logistic Regression"""
print("\n Logistic Regression \n")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Define the logistic regression model
logreg = LogisticRegression(random_state=0)

# Define the parameter grid for grid search
hyper_parameters = {
    'C': [0.01, 0.1, 1, 10],  # regularization parameter
    'penalty': ['l1', 'l2']  # regularization type
}

grid_search = GridSearchCV(logreg, hyper_parameters, cv=5, scoring='accuracy')
grid_search.fit(opt_X_train, y_train_std)
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
best_logreg = grid_search.best_estimator_
y_train_pred = best_logreg.predict(opt_X_train)
y_test_pred = best_logreg.predict(opt_X_test)
accuracy_train = accuracy_score(y_train_std, y_train_pred)
accuracy_test = accuracy_score(y_test_std, y_test_pred)

print(f"Training Accuracy: {accuracy_train}")
print(f"Testing Accuracy: {accuracy_test}")

# Additional evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test_std, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test_std, y_test_pred))

precision, recall, fscore, support = precision_recall_fscore_support(y_test_std, y_test_pred)

tn, fp, fn, tp = confusion_matrix(y_test_std, y_test_pred).ravel()

specificity = tn / (tn + fp)

# Print individual values
print("\nIndividual Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", fscore)
print("Accuracy:", accuracy_test)
print("Specificity:", specificity)

# ROC Curve and AUC
y_test_prob = best_logreg.predict_proba(opt_X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_std, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve with AUC')
plt.legend(loc="lower right")
plt.text(0.6, 0.2, 'AUC = {:.2f}'.format(roc_auc), fontsize=12)
plt.show()

def cross_val_score_without_parallel(model, X, y, cv):
    scores = []
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y

    for train, test in cv.split(X_array, y_array):
        model.fit(X_array[train], y_array[train])
        y_pred = model.predict(X_array[test])
        accuracy = accuracy_score(y_array[test], y_pred)
        scores.append(accuracy)

    return scores

# Cross-validation without parallelization
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(best_logreg, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))

"""KNN Clustering"""
print("\n KNN clustering \n")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, auc, roc_curve,precision_score,recall_score,f1_score
# Choose a range of K values
k_values = range(1, 21)

# Evaluate performance for each K
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(opt_X_train, y_train_std)
    y_pred = knn.predict(opt_X_test)
    accuracy = accuracy_score(y_test_std, y_pred)
    accuracy_scores.append(accuracy)

# Plot the results
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN: Optimal K Value')
plt.show()

best_k = k_values[np.argmax(accuracy_scores)]
print(f"Best K value: {best_k}")

Knn_best_model = KNeighborsClassifier(n_neighbors=best_k)
Knn_best_model.fit(opt_X_train, y_train_std)
y_pred_best = Knn_best_model.predict(opt_X_test)

conf_matrix = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

precision = precision_score(y_test_std, y_pred_best, average='weighted')
recall = recall_score(y_test_std, y_pred_best, average='weighted')
specificity = accuracy_score(y_test_std, y_pred_best)
f_score = f1_score(y_test_std, y_pred_best, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F-score: {f_score:.2f}")

y_scores = Knn_best_model.predict_proba(opt_X_test)[:, 1]  # KNN does not have predict_proba, so use decision_function
fpr, tpr, thresholds = roc_curve(y_test_std, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(Knn_best_model, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))

"""SVM"""
print("\n SVM \n")
# Linear Kernel
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

C_value = 0.01  # You can adjust this value

linear_svm = svm.SVC(kernel='linear', C=C_value)
linear_svm.fit(opt_X_train, y_train_std)

y_pred = linear_svm.predict(opt_X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Linear SVM Accuracy: {accuracy}')

conf_matrix = confusion_matrix(y_test_std, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

precision = precision_score(y_test_std, y_pred, average='weighted')
recall = recall_score(y_test_std, y_pred, average='weighted')
specificity = accuracy_score(y_test_std, y_pred)
f_score = f1_score(y_test_std, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F-score: {f_score:.2f}")

# Display ROC and AUC Curve
y_scores = linear_svm.decision_function(opt_X_test)
fpr, tpr, thresholds = roc_curve(y_test_std, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(linear_svm, opt_X_train, y_train_std, cv=stratified_kfold)

print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))

# Polynomial Kernel
poly_svm = svm.SVC(kernel='poly', degree=3,  C=C_value)  # You can adjust the degree parameter
poly_svm.fit(opt_X_train, y_train_std)

# Make predictions on the test set
y_pred_poly = poly_svm.predict(opt_X_test)

# Evaluate the accuracy
accuracy_poly = accuracy_score(y_test_std, y_pred_poly)
print(f'Polynomial SVM Accuracy: {accuracy_poly}')

# Display Confusion Matrix
conf_matrix_poly = confusion_matrix(y_test_std, y_pred_poly)
sns.heatmap(conf_matrix_poly, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Polynomial SVM Confusion Matrix")
plt.show()

# Display Precision, Recall, Specificity, and F-score
precision_poly = precision_score(y_test_std, y_pred_poly, average='weighted')
recall_poly = recall_score(y_test_std, y_pred_poly, average='weighted')
specificity_poly = accuracy_score(y_test_std, y_pred_poly)
f_score_poly = f1_score(y_test_std, y_pred_poly, average='weighted')

print(f"Precision: {precision_poly:.2f}")
print(f"Recall (Sensitivity): {recall_poly:.2f}")
print(f"Specificity: {specificity_poly:.2f}")
print(f"F-score: {f_score_poly:.2f}")

# Display ROC and AUC Curve
y_scores_poly = poly_svm.decision_function(opt_X_test)
fpr_poly, tpr_poly, thresholds_poly = roc_curve(y_test_std, y_scores_poly)
roc_auc_poly = auc(fpr_poly, tpr_poly)

plt.figure(figsize=(8, 8))
plt.plot(fpr_poly, tpr_poly, label=f'AUC = {roc_auc_poly:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Polynomial SVM ROC Curve')
plt.legend()
plt.show()


n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(poly_svm, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))

# Radial Basis Function (RBF) Kernel
# Create an SVM model with an RBF kernel
rbf_svm = svm.SVC(kernel='rbf',  C=C_value)
rbf_svm.fit(opt_X_train, y_train_std)

# Make predictions on the test set
y_pred_rbf = rbf_svm.predict(opt_X_test)

# Evaluate the accuracy
accuracy_rbf = accuracy_score(y_test_std, y_pred_rbf)
print(f'RBF SVM Accuracy: {accuracy_rbf}')

# Display Confusion Matrix
conf_matrix_rbf = confusion_matrix(y_test_std, y_pred_rbf)
sns.heatmap(conf_matrix_rbf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("RBF SVM Confusion Matrix")
plt.show()

# Display Precision, Recall, Specificity, and F-score
precision_rbf = precision_score(y_test_std, y_pred_rbf, average='weighted')
recall_rbf = recall_score(y_test_std, y_pred_rbf, average='weighted')
specificity_rbf = accuracy_score(y_test_std, y_pred_rbf)
f_score_rbf = f1_score(y_test_std, y_pred_rbf, average='weighted')

print(f"Precision: {precision_rbf:.2f}")
print(f"Recall (Sensitivity): {recall_rbf:.2f}")
print(f"Specificity: {specificity_rbf:.2f}")
print(f"F-score: {f_score_rbf:.2f}")

# Display ROC and AUC Curve
y_scores_rbf = rbf_svm.decision_function(opt_X_test)
fpr_rbf, tpr_rbf, thresholds_rbf = roc_curve(y_test_std, y_scores_rbf)
roc_auc_rbf = auc(fpr_rbf, tpr_rbf)

plt.figure(figsize=(8, 8))
plt.plot(fpr_rbf, tpr_rbf, label=f'AUC = {roc_auc_rbf:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RBF SVM ROC Curve')
plt.legend()
plt.show()


# Cross-validation without parallelization
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(best_logreg, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))

"""Grid Search"""
print("\n Grid Search \n")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# Hyperparameter tuning using GridSearchCV
hyper_parameters = {
    "var_smoothing": np.logspace(0, -9, num=100),
}

nb_classifier = GaussianNB()
grid_search = GridSearchCV(
    nb_classifier, hyper_parameters, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy"
)

grid_search.fit(opt_X_train, y_train_std)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Use the best model from the grid search
best_classifier_Naive_bayes = grid_search.best_estimator_

# Make predictions on the test set
y_test_pred = best_classifier_Naive_bayes.predict(opt_X_test)

# Display Confusion matrix
conf_matrix = confusion_matrix(y_test_std, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Display Precision
precision = precision_score(y_test_std, y_test_pred)
print("Precision:", precision)

# Display Sensitivity or Recall
recall = recall_score(y_test_std, y_test_pred)
print("Recall (Sensitivity):", recall)

# Display Specificity
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)

# Display F-score
f_score = f1_score(y_test_std, y_test_pred)
print("F-score:", f_score)

# Display ROC and AUC curve
y_test_prob = best_classifier_Naive_bayes.predict_proba(opt_X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_std, y_test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (AUC = {:.2f})".format(roc_auc))
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve with AUC")
plt.legend(loc="lower right")
plt.text(0.6, 0.2, "AUC = {:.2f}".format(roc_auc), fontsize=12)
plt.show()

# Stratified K-fold cross-validation
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_predict(best_classifier_Naive_bayes, opt_X_train, y_train_std, cv=stratified_kfold, method="predict")

# Print cross-validation results
print("Stratified K-fold Cross-Validation Results:")
print("Individual Fold Predictions:", cross_val_results)

"""Random Forest Classifier

Bagging boosting stacking
"""
print("\n Random Forest \n")
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


max_depth_value = 5  # You can adjust this value

# Create a Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=max_depth_value, random_state=42)
rf_classifier.fit(opt_X_train, y_train_std)

# Make predictions on the test set
y_pred_rf = rf_classifier.predict(opt_X_test)

# Evaluate the accuracy
accuracy_rf = accuracy_score(y_test_std, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')

# Display Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test_std, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Display Precision, Recall, Specificity, and F-score
precision_rf = precision_score(y_test_std, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test_std, y_pred_rf, average='weighted')
specificity_rf = accuracy_score(y_test_std, y_pred_rf)
f_score_rf = f1_score(y_test_std, y_pred_rf, average='weighted')

print(f"Precision: {precision_rf:.2f}")
print(f"Recall (Sensitivity): {recall_rf:.2f}")
print(f"Specificity: {specificity_rf:.2f}")
print(f"F-score: {f_score_rf:.2f}")

# Display ROC and AUC Curve
y_scores_rf = rf_classifier.predict_proba(opt_X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_std, y_scores_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 8))
plt.plot(fpr_rf, tpr_rf, label=f'AUC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.show()

# Cross-validation without parallelization
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(rf_classifier, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))



"""Random Forest"""
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, auc, roc_curve, precision_score,f1_score,recall_score
from sklearn.metrics import accuracy_score
#
max_depth_value = 5  # You can adjust this value

# Create a Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=max_depth_value, random_state=42)
rf_classifier.fit(opt_X_train, y_train_std)

# Make predictions on the test set
y_pred_rf = rf_classifier.predict(opt_X_test)

# Evaluate the accuracy
accuracy_rf = accuracy_score(y_test_std, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')

# Display Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test_std, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Display Precision, Recall, Specificity, and F-score
precision_rf = precision_score(y_test_std, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test_std, y_pred_rf, average='weighted')
specificity_rf = accuracy_score(y_test_std, y_pred_rf)
f_score_rf = f1_score(y_test_std, y_pred_rf, average='weighted')

print(f"Precision: {precision_rf:.2f}")
print(f"Recall (Sensitivity): {recall_rf:.2f}")
print(f"Specificity: {specificity_rf:.2f}")
print(f"F-score: {f_score_rf:.2f}")

# Display ROC and AUC Curve
y_scores_rf = rf_classifier.predict_proba(opt_X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_std, y_scores_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 8))
plt.plot(fpr_rf, tpr_rf, label=f'AUC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend()
plt.show()

# Cross-validation without parallelization
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(rf_classifier, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))

"""Bagging"""
print("\n Bagging \n")
# Bagging (Bootstrap Aggregating)
from sklearn.ensemble import BaggingClassifier

# Create a BaggingClassifier with a base Random Forest model
bagging_classifier = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10, random_state=42)
bagging_classifier.fit(opt_X_train, y_train_std)

# Make predictions on the test set
predicted_y_probability_bagging = bagging_classifier.predict(opt_X_test)

# Evaluate the accuracy
accuracy_bagging = accuracy_score(y_test_std, predicted_y_probability_bagging)
print(f'Bagging Accuracy: {accuracy_bagging}')

# Display Confusion Matrix
conf_matrix_bagging = confusion_matrix(y_test_std, predicted_y_probability_bagging)
sns.heatmap(conf_matrix_bagging, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Bagging Confusion Matrix")
plt.show()

# Display Precision, Recall, Specificity, and F-score
precision_bagging = precision_score(y_test_std, predicted_y_probability_bagging, average='weighted')
recall_bagging = recall_score(y_test_std, predicted_y_probability_bagging, average='weighted')
specificity_bagging = accuracy_score(y_test_std, predicted_y_probability_bagging)
f_score_bagging = f1_score(y_test_std, predicted_y_probability_bagging, average='weighted')

print(f"Precision: {precision_bagging:.2f}")
print(f"Recall (Sensitivity): {recall_bagging:.2f}")
print(f"Specificity: {specificity_bagging:.2f}")
print(f"F-score: {f_score_bagging:.2f}")

# Display ROC and AUC Curve
y_scores_bagging = bagging_classifier.predict_proba(opt_X_test)[:, 1]
fpr_bagging, tpr_bagging, thresholds_bagging = roc_curve(y_test_std, y_scores_bagging)
roc_auc_bagging = auc(fpr_bagging, tpr_bagging)

plt.figure(figsize=(8, 8))
plt.plot(fpr_bagging, tpr_bagging, label=f'AUC = {roc_auc_bagging:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Bagging ROC Curve')
plt.legend()
plt.show()


# Cross-validation without parallelization
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(bagging_classifier, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))


# Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking_classifier = StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                                                     ('dt', DecisionTreeClassifier(random_state=42))],
                                         final_estimator=LogisticRegression())
stacking_classifier.fit(opt_X_train, y_train_std)

# Make predictions on the test set
predicted_y_probability_stacking = stacking_classifier.predict(opt_X_test)

# Evaluate the accuracy
accuracy_stacking = accuracy_score(y_test_std, predicted_y_probability_stacking)
print(f'Stacking Accuracy: {accuracy_stacking}')

# Display Confusion Matrix
conf_matrix_stacking = confusion_matrix(y_test, predicted_y_probability_stacking)
sns.heatmap(conf_matrix_stacking, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Stacking Confusion Matrix")
plt.show()

# Display Precision, Recall, Specificity, and F-score
precision_stacking = precision_score(y_test_std, predicted_y_probability_stacking, average='weighted')
recall_stacking = recall_score(y_test_std, predicted_y_probability_stacking, average='weighted')
specificity_stacking = accuracy_score(y_test_std, predicted_y_probability_stacking)
f_score_stacking = f1_score(y_test_std, predicted_y_probability_stacking, average='weighted')

print(f"Precision: {precision_stacking:.2f}")
print(f"Recall (Sensitivity): {recall_stacking:.2f}")
print(f"Specificity: {specificity_stacking:.2f}")
print(f"F-score: {f_score_stacking:.2f}")

# Display ROC and AUC Curve
y_scores_stacking = stacking_classifier.predict_proba(opt_X_test)[:, 1]
fpr_stacking, tpr_stacking, thresholds_stacking = roc_curve(y_test_std, y_scores_stacking)
roc_auc_stacking = auc(fpr_stacking, tpr_stacking)

plt.figure(figsize=(8, 8))
plt.plot(fpr_stacking, tpr_stacking, label=f'AUC = {roc_auc_stacking:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stacking ROC Curve')
plt.legend()
plt.show()

# Cross-validation without parallelization
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(stacking_classifier, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))


# Boosting (Expensive computation) Running perfectly in notebook, in python requires a little more ram usage .
print("\n Boosting \n")
from sklearn.ensemble import AdaBoostClassifier

# Create an AdaBoost classifier with a base Random Forest model
adaboost_classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
                                         n_estimators=50, random_state=42)
adaboost_classifier.fit(opt_X_train, y_train_std)

y_pred_adaboost = adaboost_classifier.predict(opt_X_test)

accuracy_adaboost = accuracy_score(y_test_std, y_pred_adaboost)
print(f'AdaBoost Accuracy: {accuracy_adaboost}')

# Display Confusion Matrix
conf_matrix_adaboost = confusion_matrix(y_test_std, y_pred_adaboost)
sns.heatmap(conf_matrix_adaboost, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("AdaBoost Confusion Matrix")
plt.show()

precision_adaboost = precision_score(y_test_std, y_pred_adaboost, average='weighted')
recall_adaboost = recall_score(y_test_std, y_pred_adaboost, average='weighted')
specificity_adaboost = accuracy_score(y_test_std, y_pred_adaboost)
f_score_adaboost = f1_score(y_test_std, y_pred_adaboost, average='weighted')

print(f"Precision: {precision_adaboost:.2f}")
print(f"Recall (Sensitivity): {recall_adaboost:.2f}")
print(f"Specificity: {specificity_adaboost:.2f}")
print(f"F-score: {f_score_adaboost:.2f}")

y_scores_adaboost = adaboost_classifier.predict_proba(opt_X_test)[:, 1]
fpr_adaboost, tpr_adaboost, thresholds_adaboost = roc_curve(y_test_std, y_scores_adaboost)
roc_auc_adaboost = auc(fpr_adaboost, tpr_adaboost)

plt.figure(figsize=(8, 8))
plt.plot(fpr_adaboost, tpr_adaboost, label=f'AUC = {roc_auc_adaboost:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AdaBoost ROC Curve')
plt.legend()
plt.show()


# Cross-validation without parallelization
n_folds = 5
stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
cross_val_results = cross_val_score_without_parallel(adaboost_classifier, opt_X_train, y_train_std, cv=stratified_kfold)

# Print cross-validation results
print("Cross-Validation Results:")
print("Individual Fold Accuracies:", cross_val_results)
print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))



"""Neural Networks"""
print("\n Neural Networks \n")
# Neural Network
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(opt_X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons for 3 classes
])

model.compile(optimizer='adam',  # You can use other optimizers like 'sgd' or 'rmsprop'
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

model.fit(opt_X_train, y_train_std, epochs=50, batch_size=32, validation_data=(opt_X_test, y_test_std))

# Evaluate the model
predicted_y_probability = model.predict(opt_X_test)
y_pred = tf.argmax(predicted_y_probability, axis=1).numpy()
accuracy = accuracy_score(y_test_std, y_pred)
print(f'MLP Accuracy: {accuracy}')

confusion_matrix_neural_network = confusion_matrix(y_test_std, y_pred)
sns.heatmap(confusion_matrix_neural_network, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Neural Network Confusion Matrix")
plt.show()

# Display Precision, Recall, Specificity, and F-score
precision_nn = precision_score(y_test_std, y_pred, average='weighted')
recall_nn = recall_score(y_test_std, y_pred, average='weighted')
specificity_nn = accuracy_score(y_test_std, y_pred)
f_score_nn = f1_score(y_test_std, y_pred, average='weighted')

print(f"Precision: {precision_nn:.2f}")
print(f"Recall (Sensitivity): {recall_nn:.2f}")
print(f"Specificity: {specificity_nn:.2f}")
print(f"F-score: {f_score_nn:.2f}")

# Display ROC and AUC Curve
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test_std, predicted_y_probability[:, 1])
roc_auc_nn = auc(fpr_nn, tpr_nn)

plt.figure(figsize=(8, 8))
plt.plot(fpr_nn, tpr_nn, label=f'AUC = {roc_auc_nn:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network ROC Curve')
plt.legend()
plt.show()


# Cross-validation without parallelization
# n_folds = 5
# stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
# cross_val_results = cross_val_score_without_parallel(model, opt_X_train, y_train_std, cv=stratified_kfold)
#
# # Print cross-validation results
# print("Cross-Validation Results:")
# print("Individual Fold Accuracies:", cross_val_results)
# print("Mean Accuracy:", sum(cross_val_results) / len(cross_val_results))


"""
**Phase 4**

---

"""

df_encoded.columns

# new_df_encoded=df_encoded.drop('salary',axis=1)
new_df_encoded=df_encoded
# Separate features (X) and target variable (y)
y = new_df_encoded['high_salary']
X = new_df_encoded.drop(columns=['high_salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)
x_scaler = StandardScaler()
X_train[['yearsExperience', 'milesFromMetropolis']] = x_scaler.fit_transform(X_train[['yearsExperience', 'milesFromMetropolis']])
X_test[['yearsExperience', 'milesFromMetropolis']] = x_scaler.transform(X_test[['yearsExperience', 'milesFromMetropolis']])
X_train_std = X_train
X_test_std = X_test

# Target standardization
y_scaler = StandardScaler()
y_train_std = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_std = y_scaler.transform(y_test.values.reshape(-1, 1))

"""K Means clustering"""

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

print("------------------ Clustering --------------------------------")

# flight_data = flight_data.drop(['price_category'], axis=1)
categorical = [var for var in X_train_std.columns if X_train_std[var].dtype == 'O']

def category_encoding(dataset):
    for var in categorical:
        ordered_labels = dataset.groupby([var])['salary'].mean().sort_values().index
        ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}
        print(f'Encoding for {var}: {ordinal_label}')
        dataset[var] = dataset[var].map(ordinal_label)

# Applying encoding to the dataset
category_encoding(X_train_std)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X_train_std)

print("------------------ K-Mean Clustering --------------------------------")

silhouette_scores = []
inertia_scores = []

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    inertia_scores.append(kmeans.inertia_)


plt.figure(figsize=(10, 5))
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.subplot(1, 2, 1)
plt.title('Silhouette Scores for Different Clusters')
plt.plot(range(2, 11), silhouette_scores, marker='o')


# Choose the number of clusters with the highest silhouette score
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2


# Plotting within-cluster variation
plt.subplot(1, 2, 2)
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Variation')
plt.plot(range(2, 11), inertia_scores, marker='o')
plt.title('Within-Cluster Variation for Different Clusters')
plt.show()

"""Apriori Algorithm"""

from itertools import combinations
from collections import defaultdict

relevant_columns = [
    'yearsExperience', 'milesFromMetropolis', 'jobType_CFO',
    'jobType_JUNIOR', 'jobType_MANAGER', 'jobType_SENIOR',
    'degree_DOCTORAL', 'degree_MASTERS', 'major_BUSINESS', 'major_CHEMISTRY',
    'major_COMPSCI', 'major_ENGINEERING', 'industry_EDUCATION', 'industry_FINANCE', 'industry_OIL', 'industry_SERVICE'
]

def convert_to_transactions(data, relevant_columns):
    itemset_for_orders = []
    for _, row in data[relevant_columns].iterrows():
        transaction = [f"{col}_{row[col]}" for col in relevant_columns]
        itemset_for_orders.append(transaction)
    return itemset_for_orders

def apriori_custom(itemset_for_orders, min_support, max_length=2):
    item_count = defaultdict(int)
    num_transactions = len(itemset_for_orders)

    for transaction in itemset_for_orders:
        for item in transaction:
            item_count[frozenset([item])] += 1

    item_count = {item: count for item, count in item_count.items() if count / num_transactions >= min_support}

    frequent_itemsets = list(item_count.keys())

    for length in range(2, max_length + 1):
        combos = combinations(frequent_itemsets, length)
        current_itemsets = defaultdict(int)

        for combo in combos:
            current_set = frozenset.union(*combo)

            if current_set not in current_itemsets:
                for transaction in itemset_for_orders:
                    if current_set.issubset(transaction):
                        current_itemsets[current_set] += 1

        current_itemsets = {item: count for item, count in current_itemsets.items() if count / num_transactions >= min_support}
        frequent_itemsets.extend(current_itemsets.keys())

    return [(itemset, count / num_transactions) for itemset, count in item_count.items()]

def apply_apriori_and_display(data, relevant_columns, min_support=0.01, max_length=2, display_count=10):
    itemset_for_orders = convert_to_transactions(data, relevant_columns)
    frequent_itemsets_custom = apriori_custom(itemset_for_orders, min_support, max_length)

    print(f"Top {display_count} frequent itemsets:")
    for itemset, support in frequent_itemsets_custom[:display_count]:
        print(f"{itemset}: {support}")

# Applying the custom Apriori algorithm on the itemset_for_orders
apply_apriori_and_display(opt_X_train, relevant_columns, min_support=0.01, max_length=2, display_count=10)


"""DBScan"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=50)
cluster_labels_dbscan = dbscan.fit_predict(scaled_data)

# Add the cluster labels to the original DataFrame
X_train_std['Cluster_KMeans'] = kmeans.labels_
X_train_std['Cluster_DBSCAN'] = cluster_labels_dbscan


plt.scatter(X_train_std['yearsExperience'], X_train_std['milesFromMetropolis'], c=X_train_std['Cluster_KMeans'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('yearsExperience')
plt.ylabel('milesFromMetropolis')
plt.show()

plt.scatter(X_train_std['yearsExperience'], X_train_std['milesFromMetropolis'], c=X_train_std['Cluster_DBSCAN'], cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('yearsExperience')
plt.ylabel('milesFromMetropolis')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_train_std['yearsExperience'], X_train_std['milesFromMetropolis'], X_train_std['salary'],
           c=X_train_std['Cluster_DBSCAN'], cmap='viridis')

ax.set_xlabel('yearsExperience')
ax.set_ylabel('milesFromMetropolis')
ax.set_zlabel('salary')

plt.title('DBSCAN Clustering (3D Scatter Plot)')
plt.show()

"""Author : Venkata Chaitanya K"""

