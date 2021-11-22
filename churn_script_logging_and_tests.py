import logging
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    dataframe = import_data()
    try:
        perform_eda(dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except (FileNotFoundError, AssertionError) as err:
        if isinstance(err, FileNotFoundError):
            logging.error(
                "Testing perform_eda: The file or directory wasn't found")
        elif isinstance(err, AssertionError):
            logging.error(
                "Testing perform_data: The file doesn't appear to have rows and columns")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    dataframe = import_data()
    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    try:
        encoder_helper(dataframe, category_lst)
        logging.info("Testing encoder_helper: SUCCESS")
        assert category_lst == [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
    except (FileNotFoundError, AssertionError, KeyError) as err:
        if isinstance(err, FileNotFoundError):
            logging.error(
                "Testing encoder_helper: The file or directory wasn't found")
        elif isinstance(err, AssertionError):
            logging.error(
                "Testing encoder_helper: The file doesn't appear to have rows and columns")
        elif isinstance(err, KeyError):
            logging.error(
                "Testing encoder_helper: Either the category_list \
					or a key value seams to be incorrect.")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    dataframe = import_data()
    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    dataframe = encoder_helper(dataframe, category_lst)
    response = [
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

    try:
        perform_feature_engineering(dataframe, response)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        assert len(response) == 19
    except (AssertionError, ValueError, KeyError) as err:
        if isinstance(err, AssertionError):
            logging.error(
                "Testing perform_feature_engineering: \
					The file doesn't appear to have rows and columns")
        elif isinstance(err, ValueError):
            logging.error(
                "Testing perform_feature_engineering: \
					The response list seams to be empty")
        elif isinstance(err, KeyError):
            logging.error(
                "Testing encoder_helper: category_lst, \
					response or a key value of the dataframe seams to be incorrect.")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    dataframe = import_data()
    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    dataframe = encoder_helper(dataframe, category_lst)
    response = [
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
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataframe, response)

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    except (ValueError, KeyError) as err:
        if isinstance(err, ValueError):
            logging.error(
                "Testing train_models: X_train and y_train or \
					X_test and y_test seam to have different lenghts.")
        elif isinstance(err, KeyError):
            logging.error(
                "Testing encoder_helper: Either the category_lst, \
					response or a key value of the dataframe seams to be incorrect.")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
