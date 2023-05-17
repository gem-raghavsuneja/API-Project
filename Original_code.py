# import pandas as pd
# import numpy as np
# # reading the required files into python using pandas 
# df = pd.read_csv('C:\\Users\\raghav.suneja\\OneDrive - Gemini Solutions\\Desktop\\file.csv')
# test = pd.read_csv('C:\\Users\\raghav.suneja\\OneDrive - Gemini Solutions\\Desktop\\test.csv')
# # Fill the NaN values with 'None'
# df = df.fillna('None')
# # now since our machine learning model can only understand numeric values we'll have to convert the strings to numbers/indicator variables
# X_train = pd.get_dummies(df.drop('element',axis=1))
# # returns all the unique Elements stored in the training data
# df['element'].unique()
# # creating a dictionary of elements 
# element_dict = dict(zip(df['element'].unique(), range(df['element'].nunique())))
# # >>>{'_token': 0, 'email': 1, 'password': 2, 'LOGIN': 3, 'on': 4}
# # replacing dictionary values into dataframe as we meed to convert this into numbers
# y_train = df['element'].replace(element_dict)
# # Now we need to train our model , we can prefer any model which provides accurate results -
# # Random Forest Model
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=50, random_state=0)
# rf.fit(X_train, y_train)
# # ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# def predict_elements():
#     num_of_records = len(test)
#     test_ = test.fillna('None')
#     concatenated = pd.concat([df, test_], axis=0).drop('element',     axis=1)
#     if num_of_records == 1:
#         processed_test = pd.DataFrame(pd.get_dummies(concatenated).iloc[-num_of_records]).T
#         probabilites = list(rf.predict_proba(processed_test)[0])
#         element_name = list(element_dict.keys())[np.argmax(probabilites)]
#         element_name = 'Hence, the name of our predicted element is {}'.format(element_name)
#         score = list(zip(df['element'].unique(), probabilites))

#     elif num_of_records > 1:
#         processed_test = pd.get_dummies(concatenated).iloc[-num_of_records:]
#         probabilites = list(rf.predict_proba(processed_test))

#         score = []
#         for i in range(len(probabilites)):
#             score.append(list(zip(df['element'].unique(), list(probabilites[i]))))

#         element_index = np.argmax(probabilites, axis=1)
#         element_name = []
#         for ind_, i in enumerate(element_index):
#             element_name.append(
#                 (ind_, 'Hence, the name of our predicted element is {}'.format(list(element_dict.keys())[i])))
#     return score, element_name, test
# # Calling the predict_elements method to return

# scores, element_name, test_df = predict_elements()
# print(scores)
# print(element_name)
# print(test_df)
# # ///////////////////////////////////////////////////////////////////////////////////////////////

# def predict_elements():













import pandas as pd
import numpy as np
import sys
# reading the required files into python using pandas 
df = pd.read_csv('C:\\Users\\raghav.suneja\\OneDrive - Gemini Solutions\\Desktop\\file.csv')
test = pd.read_csv('C:\\Users\\raghav.suneja\\OneDrive - Gemini Solutions\\Desktop\\test.csv')
# Fill the NaN values with 'None'
df = df.fillna('None')
sys.path.append("./")
# now since our machine learning model can only understand numeric values we'll have to convert the strings to numbers/indicator variables
X_train = pd.get_dummies(df.drop('element',axis=1))
# returns all the unique Elements stored in the training data
df['element'].unique()
# creating a dictionary of elements 
element_dict = dict(zip(df['element'].unique(), range(df['element'].nunique())))
# >>>{'_token': 0, 'email': 1, 'password': 2, 'LOGIN': 3, 'on': 4}
# replacing dictionary values into dataframe as we meed to convert this into numbers
y_train = df['element'].replace(element_dict)
# Now we need to train our model , we can prefer any model which provides accurate results -
# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(X_train, y_train)
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def predict_elements():
    num_of_records = len(test)
    test_ = test.fillna('None')
    
    # concatenate the training and test data
    concatenated = pd.concat([df, test_], axis=0)
    
    # apply one-hot encoding to the concatenated data
    processed = pd.get_dummies(concatenated.drop('element', axis=1))
    
    # separate the training and test data
    X_train = processed.iloc[:len(df)]
    X_test = processed.iloc[len(df):]
    
    # fit the random forest model
    y_train = df['element'].replace(element_dict)
    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    rf.fit(X_train, y_train)
    
    if num_of_records == 1:
        # get the probabilities for the test record
        processed_test = pd.DataFrame(X_test.iloc[-1]).T
        probabilites = list(rf.predict_proba(processed_test)[0])
        
        # get the name of the predicted element
        element_name = list(element_dict.keys())[np.argmax(probabilites)]
        element_name = 'Hence, the name of our predicted element is {}'.format(element_name)
        
        score = list(zip(df['element'].unique(), probabilites))

    elif num_of_records > 1:
        # get the probabilities for each test record
        processed_test = X_test
        probabilites = list(rf.predict_proba(processed_test))

        score = []
        for i in range(len(probabilites)):
            score.append(list(zip(df['element'].unique(), list(probabilites[i]))))

        # get the name of the predicted element for each test record
        element_index = np.argmax(probabilites, axis=1)
        element_name = []
        for ind_, i in enumerate(element_index):
            element_name.append(
                (ind_, 'Hence, the name of our predicted element is {}'.format(list(element_dict.keys())[i])))
    return score, element_name, test

# Calling the predict_elements method to return

scores, element_name, test_df = predict_elements()
print(scores)
print(element_name)
print(test_df)

























