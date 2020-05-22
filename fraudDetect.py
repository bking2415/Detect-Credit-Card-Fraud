from pyexpat import features

import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import random
import numpy as np
import seaborn as sns
import graphviz

import matplotlib.pyplot as plt
from seaborn.external.six import StringIO
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.graphics.gofplots import ProbPlot

'Step 1: Importing Data Sets'

creditCardData = pd.read_csv('creditcard.csv')

'Step 2: Data Exploration'

# Print the first 6 rows
# print(creditCardData.head(6))

# Print the last 6 rows
# print(creditCardData.tail(6))

# Frequency Table of Class count
# print(creditCardData["Class"].value_counts())

# Statistics summary of Amount column
# print(creditCardData["Amount"].describe())

# Print list of column names
# print(list(creditCardData.columns.values))
dataColumns = list(creditCardData.columns.values)

# Variance of Amount column
# print(creditCardData["Amount"].var())

# Standard deviation of Amount column
# print(creditCardData["Amount"].std())

'Step 3: Data Manipulation'

# Standardize a dataset along column Amount
# print("Before Scaling...")
# print(creditCardData["Amount"].head(5))
creditCardData["Amount"] = sklearn.preprocessing.scale(creditCardData["Amount"])
# print("After Scaling...")
# print(creditCardData["Amount"].head(5))

# Drop the first column in the data frame
newData = creditCardData.drop(columns=["Time"])
# print(newData.head(6))

'Step 4: Data Modelling'

# Creating train and test sets
# Set seed
random.seed(123)
# sss = sklearn.model_selection.StratifiedShuffleSplit(newData["Class"], n_splits=1, test_size=0.80, random_state=123)
# Create a data frame of train and test sets
# for train_index, test_index in sss.split(features, labels):
#     dataTrain = newData.iloc[train_index, :]
#     dataTest = newData.iloc[test_index, :]

train, test = sklearn.model_selection.train_test_split(newData, train_size=0.80, random_state=123)
# Create a data frame of train and test sets
dataTrain = pd.DataFrame(train, columns=newData.columns)
dataTest = pd.DataFrame(test, columns=newData.columns)

# Print the dimensions
# print(dataTrain.shape)
# print(dataTest.shape)

import statsmodels.api as sm
import statsmodels.formula.api as smf
import createFormula as cf

# R like formula for python model
# formula = 'Class ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V14+V15+V16+V17+V18+V19+V20+V21+V22+V23+V24+V25+V26+V27' \
#           '+V28+Amount '
formula = cf.createFormula(list(dataTrain.columns.values))
# print(formula)

# Instantiate a logistic family model with the default link function.
# exog, endog = sm.add_constant(dataTest), dataTest["Class"]
logistic_model = smf.glm(formula=formula, data=dataTrain, family=sm.families.Binomial())
result = logistic_model.fit()
print(result.summary())

# create dataframe from X, y for easier plot handling
dataframe = pd.concat([dataTrain[dataTrain.columns[:-1]], dataTrain["Class"]], axis=1)
# print(dataframe.head(4))
# Binomial Model

# binom = smf.logit(formula=formula, data=dataTest).fit()
# print(binom.summary())

"Step 5: Fitting Logistic Regression Model"

# Creating 4 plots examine a few different assumptions about the model and the data
# Similar to plot() function in R

# Residual vs Fitted

# model values
model_fitted_y = result.fittedvalues
y = dataTrain["Class"].tolist()
# print(model_fitted_y)
model_fitted_y = model_fitted_y.values.tolist()
# print(model_fitted_y[0:4])
# print(type(model_fitted_y))
# print(y)
# # print(type(y))
# # print(y)
# # model residuals
# # Output list initialization
model_residuals = []

for i in range(len(y)):
    model_residuals.append(y[i] - model_fitted_y[i])

# print(model_residuals[0:4])
# normalized residuals
model_norm_residuals = []
amin, amax = min(model_residuals), max(model_residuals)
for i in range(len(model_residuals)):
    model_norm_residuals.append(
        (model_residuals[i] - amin) / (amax - amin))

# print(model_norm_residuals[0:4])
# # model_norm_residuals = result.get_influence().resid_studentized_internal
# # absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# print(model_norm_residuals_abs_sqrt[0:4])
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# print(model_abs_resid[0:4])
# leverage, from statsmodels internals
model_leverage = result.get_influence().hat_matrix_diag
# print(model_leverage[0:4])
# cook's distance, from statsmodels internals
model_cooks = result.get_influence().cooks_distance[0]
# print(model_cooks[0:4])

'Uncomment to plot diagnostics plots similar to plot() ' \
'function in R'
# plot_lm_1 = plt.figure()
# plot_lm_1.axes[0] = sns.residplot(model_fitted_y, dataframe.columns[-1], data=dataframe,
#                                   lowess=True,
#                                   scatter_kws={'alpha': 0.5},
#                                   line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
#
# plot_lm_1.axes[0].set_title('Residuals vs Fitted')
# plot_lm_1.axes[0].set_xlabel('Fitted values')
# plot_lm_1.axes[0].set_ylabel('Residuals')
#
# # Show plotted data
# plt.show()
#
# # # Normal Q-Q Plot
# model_norm_residuals = np.array(model_norm_residuals)
# QQ = ProbPlot(model_norm_residuals)
# plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
# plot_lm_2.axes[0].set_title('Normal Q-Q')
# plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
# plot_lm_2.axes[0].set_ylabel('Standardized Residuals')
# # annotations
# abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
# abs_norm_resid_top_3 = abs_norm_resid[:3]
# for r, i in enumerate(abs_norm_resid_top_3):
#     plot_lm_2.axes[0].annotate(i,
#                                xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
#                                    model_norm_residuals[i]))
#
# # Show plotted data
# plt.show()
#
# Scale location
#
# plot_lm_3 = plt.figure()
# plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
# sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
#             scatter=False,
#             ci=False,
#             lowess=True,
#             line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
# plot_lm_3.axes[0].set_title('Scale-Location')
# plot_lm_3.axes[0].set_xlabel('Fitted values')
# plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')
#
# # annotations
# abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
# abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
# for i in abs_sq_norm_resid_top_3:
#     plot_lm_3.axes[0].annotate(i,
#                                xy=(model_fitted_y[i],
#                                    model_norm_residuals_abs_sqrt[i]))
#
# # Show plotted data
# plt.show()
#
# # Residuals vs Leverage
#
# plot_lm_4 = plt.figure();
# plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
# sns.regplot(model_leverage, model_norm_residuals,
#             scatter=False,
#             ci=False,
#             lowess=True,
#             line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
# plot_lm_4.axes[0].set_xlim(0, max(model_leverage) + 0.01)
# plot_lm_4.axes[0].set_ylim(-3, 5)
# plot_lm_4.axes[0].set_title('Residuals vs Leverage')
# plot_lm_4.axes[0].set_xlabel('Leverage')
# plot_lm_4.axes[0].set_ylabel('Standardized Residuals')
#
# # annotations
# leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
# for i in leverage_top_3:
#     plot_lm_4.axes[0].annotate(i,
#                                xy=(model_leverage[i],
#                                    model_norm_residuals[i]))
#
# # Show plotted data
# plt.show()

# Predict Logistic model
# logPredict = logistic_model.predict(x=dataTrain)
# plt.scatter(logPredict,trees['Volume'][20:31])
# Predict probabilities for the test data.
probs = result.predict(dataTest)
# keep probabilities of positive class only
# print(probs)
# Compute the AUC Score.
auc = roc_auc_score(dataTest["Class"], probs)
print('Logistic Regression Accuracy: %.2f' % auc)
# Get the ROC curve
fpr, tpr, thresholds = roc_curve(dataTest["Class"], probs)

'Uncomment to Plot ROC Curve'
# Plot ROC Curve
# def plot_roc_curve(fpr, tpr):
#     plt.plot(fpr, tpr, color='orange', label='ROC')
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.show()
#
#
# plot_roc_curve(fpr, tpr)

"Step 6: Fitting a Decision Tree Model"

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(dataTrain[dataTrain.columns[:-1]], dataTrain["Class"])

# Predict the response for test dataset
y_pred = clf.predict(dataTest[dataTest.columns[:-1]])

# Model Accuracy, how often is the classifier correct?
print("Decision Tree Accuracy:", sklearn.metrics.accuracy_score(dataTest["Class"], y_pred))

'Uncomment to draw decision tree output'
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

# Create features column
# feature_cols = list(dataTest[dataTest.columns[:-1]].columns.values)
#
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('fraud.png')
# Image(graph.create_png())

"Step 7: Artificial Neural Network"

# Create your first MLP in SKlearn

ann_clf = MLPClassifier(activation='identity', solver='lbfgs', alpha=1e-5, random_state=1, max_iter=300)

# Train MLP Classifer
ann_clf = ann_clf.fit(dataTrain[dataTrain.columns[:-1]], dataTrain["Class"])

# Predict the response for test dataset
y_pred = ann_clf.predict(dataTest[dataTest.columns[:-1]])

# Model Accuracy, how often is the classifier correct?
print("MLP Accuracy:", sklearn.metrics.accuracy_score(dataTest["Class"], y_pred))

"Step 8: Gradient Boosting (GBM)"

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# from sklearn.grid_search import GridSearchCV

baseline = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, max_depth=3, min_samples_split=200,
                                      min_samples_leaf=100, subsample=1, max_features='sqrt', random_state=10)
baseline.fit(dataTrain[dataTrain.columns[:-1]], dataTrain["Class"])
# predictors = list(dataTrain[dataTrain.columns[:-1]].columns.values)
# feat_imp = pd.Series(baseline.feature_importances_, predictors).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Importance of Features')
# plt.ylabel('Feature Importance Score')
# Accuracy of the model
print('Accuracy of the GBM on test set: {:.3f}'.format(
    baseline.score(dataTest[dataTest.columns[:-1]], dataTest["Class"])))
# Predict new data
pred = baseline.predict(dataTest[dataTest.columns[:-1]])
# Print the precision and recall table
print(classification_report(dataTest["Class"], pred))

# Plot important features
# plt.show()
