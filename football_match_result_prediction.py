"""
===============================================================================
Multiclass Classification Project: Prediction of the result of a football match
 using an API application
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
4. Models Serialisation
"""
# Standard libraries
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import pandas as pd
import sweetviz as sv
import ydata_profiling
import sklearn
import pickle


from sweetviz import analyze
from ydata_profiling import ProfileReport
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from functions import *


# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))



# Constants
SEED = 0
MAX_ROWS_DISPLAY = 300
MAX_COLUMNS_DISPLAY = 150

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Data Analysis
===============================================================================
"""
# Load datasets
print('\n\n\nLoad the dataset: ')
INPUT_CSV_1 = 'https://www.football-data.co.uk/mmz4281/2223/E0.csv'
INPUT_CSV_2 = 'https://www.football-data.co.uk/mmz4281/2223/E1.csv'
INPUT_CSV_3 = 'https://www.football-data.co.uk/mmz4281/2223/E2.csv'
raw_dataset_e0 = pd.read_csv(INPUT_CSV_1)
raw_dataset_e1 = pd.read_csv(INPUT_CSV_2)
raw_dataset_e2 = pd.read_csv(INPUT_CSV_3)
raw_dataset = pd.concat([raw_dataset_e0, raw_dataset_e1, raw_dataset_e2])
raw_dataset = raw_dataset.reset_index(inplace=False, drop=True)

# Display the raw dataset's dimensions
print('\nDimensions of the raw dataset: {}'.format(raw_dataset.shape))

# Display the raw dataset's information
print('\nInformation about the raw dataset:')
print(raw_dataset.info())

# Description of the raw dataset
print('\nDescription of the raw dataset:')
print(raw_dataset.describe(include='all'))

# Display the head and the tail of the raw dataset
print(f'\nRaw dataset shape: {raw_dataset.shape}')
print(raw_dataset.info())
print(pd.concat([raw_dataset.head(150), raw_dataset.tail(150)]))


# Dispaly the raw dataset report
report = analyze(source=raw_dataset)
report.show_html('raw_dataset_report.html')
#report_ydp = ProfileReport(df=raw_dataset, title='Raw Dataset Report')
#report_ydp.to_file('raw_dataset_report_ydp.html')


# Cleanse the dataset
dataset = raw_dataset.copy()
dataset['FTR'] = dataset['FTR'].map(
    {'H': 'Home team wins', 'A': 'Away team wins', 'D': 'Draw match'})
dataset = dataset.rename(columns={
    'FTR': 'Result', 'FTHG': 'HomeTeamScore', 'FTAG': 'AwayTeamScore'})

# Management of duplicates
print('\n\nManagement of duplicates:')
duplicate = dataset[dataset.duplicated()]
print('Dimensions of the duplicates dataset: {}'.format(duplicate.shape))
if duplicate.shape[0] > 0:
    dataset = dataset.drop_duplicates()

# Management of missing data
dataset = dataset.dropna()
dataset.reset_index(inplace=True, drop=True)

# Display the dataset's dimensions
print('\nDimensions of the dataset: {}'.format(dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(dataset.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(dataset.describe(include='all'))

# Display the head and the tail of the dataset
print(f'\nDataset shape: {dataset.shape}')
print(dataset.info())
print(pd.concat([dataset.head(150), dataset.tail(150)]))


# Dispaly the dataset report
report = analyze(source=dataset)
report.show_html('dataset_report.html')
#report_ydp = ProfileReport(df=dataset, title='Dataset Report')
#report_ydp.to_file('dataset_report_ydp.html')


# Display the label categories
display_pie_chart(dataset, 'Result', (5, 5))
display_barplot(dataset, 'Result', (10, 5))



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
# Feature selection
y = dataset['Result'].to_numpy()
X = dataset[['HomeTeamScore', 'AwayTeamScore']]

# Display the head and the tail of the dataset
print(f'\n\n\nX dataset shape: {X.shape}')
print(X.info())
print(pd.concat([X.head(150), X.tail(150)]))


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Display the head and the tail of the train set
print(f'\n\nTrain set shape: {X_train.shape}')
print(X_train.info())
print(pd.concat([X_train.head(150), X_train.tail(150)]))

# Display the head and the tail of the test set
print(f'\nTest set shape: {X_test.shape}')
print(X_test.info())
print(pd.concat([X_test.head(150), X_test.tail(150)]))


# Encode the label
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
print(f'\n\nLabels count: {Counter(y_test)}')
y_test = encoder.transform(y_test)
print(f'\nTrain label shape: {y_train.shape}')
print(f'Test label shape: {y_test.shape}')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# Classes and labels
print(f'\n\n\nTrain classes count: {Counter(y_train)}')
print(f'Test classes count: {Counter(y_test)}')
labels = list(set(dataset['Result']))
print(f'Labels: {labels}')


# Instantiate the model
model = LogisticRegression(class_weight='balanced', random_state=SEED)

# Fit the model
model.fit(np.array(X_train), y_train)

# Make predictions
y_pred = model.predict(np.array(X_test))

# Evaluation
evaluate_multiclass_classification(y_test, y_pred, labels)



"""
===============================================================================
4. Models Serialisation
===============================================================================
"""
# Serialise encoder
pickle.dump({'encoder': encoder}, open('models/encoder' + '.pkl', 'wb'))

# Serialise the model
pickle.dump({'model': model}, open('models/model' + '.pkl', 'wb'))
