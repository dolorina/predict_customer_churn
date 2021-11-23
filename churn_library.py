'''
Module to predict customer churn with ML Methods
(Random Forest Classificator and Logisctic Regression)

Author: Marina
Date: November 2021
'''
# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import dataframe_image as dfi
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth="./data/bank_data.csv"):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe

    output:
            None
    '''
    # path where to save images
    pth = './images/eda/'

    dfi.export(dataframe.describe(), pth + 'dataframe_describe.png')
    dfi.export(dataframe.head(), pth + 'dataframe_head.png')

    # save df.isnull().sum() as pd DataFrame and save it as figure to image
    # folder
    columns, index = [], []
    for key, val in dataframe.isnull().sum().iteritems():
        index.append(key)
        columns.append(val)
    isnull_sum = pd.DataFrame(index=index)

    for cols in columns:
        isnull_sum[cols] = pd.Series(dtype=int)
        isnull_sum[cols] = cols
    dfi.export(isnull_sum, pth + 'isnull_sum.png')

    # save churn histogram
    plt.figure(figsize=(20, 10))
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    dataframe['Churn'].hist()
    plt.savefig(pth + 'histogram_churn.png')
    plt.close()

    # save customer age histogram
    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig(pth + 'histogram_age.png')
    plt.close()

    # save barplot of maritial status
    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(pth + 'barplot_marital.png')
    plt.close()

    # save correlation as heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(pth + 'heatmap_correlation.png')
    plt.close()


def encoder_helper(dataframe, category_lst, response='_Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe:           pandas dataframe
            category_lst: list of columns that contain categorical features
            response:     string of response name [optional argument that could be used for
                          naming variables or index y column]

    output:
            dataframe: pandas dataframe with new columns for
    '''

    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    for cat in category_lst:
        # cat encoded column
        cat_list = []
        cat_group = dataframe.groupby(cat).mean()['Churn']
        for val in dataframe[cat]:
            cat_list.append(cat_group.loc[val])
        dataframe[cat + response] = cat_list
    return dataframe


def perform_feature_engineering(dataframe, response):
    '''
    input:
              dataframe:       pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    X[response] = dataframe[response]
    y = dataframe['Churn']

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # path where to save images
    pth = './images/results/'

    # scores stored in pd dataframes and saved in images file
    report_rfc_test = classification_report(
        y_test, y_test_preds_rf, output_dict=True)
    dataframe_report_rfc_test = pd.DataFrame(report_rfc_test).transpose()
    dfi.export(dataframe_report_rfc_test, pth + 'results_rfc_test.png')

    report_rfc_train = classification_report(
        y_train, y_train_preds_rf, output_dict=True)
    dataframe_report_rfc_train = pd.DataFrame(report_rfc_train).transpose()
    dfi.export(dataframe_report_rfc_train, pth + 'results_rfc_train.png')

    report_lrc_test = classification_report(
        y_test, y_test_preds_lr, output_dict=True)
    dataframe_report_lrc_test = pd.DataFrame(report_lrc_test).transpose()
    dfi.export(dataframe_report_lrc_test, pth + 'results_lrc_test.png')

    report_lrc_train = classification_report(
        y_train, y_train_preds_lr, output_dict=True)
    dataframe_report_lrc_train = pd.DataFrame(report_lrc_train).transpose()
    dfi.export(dataframe_report_lrc_train, pth + 'results_lrc_train.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Store figure
    plt.savefig(output_pth + '/feature_importance.png')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve.png')


if __name__ == "__main__":

    DF = import_data()
    perform_eda(DF)

    CAT_COLUMNS = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    DF = encoder_helper(DF, CAT_COLUMNS)

    RESPONSE = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DF, RESPONSE)

    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    RFC_MODEL = joblib.load('./models/rfc_model.pkl')
    feature_importance_plot(RFC_MODEL, DF[RESPONSE], './images/results')
