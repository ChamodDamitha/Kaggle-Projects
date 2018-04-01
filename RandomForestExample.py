import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
print(melbourne_data.describe())

print(melbourne_data.columns)

# store the series of prices separately as melbourne_price_data.
melbourne_price_data = melbourne_data.Price
# the head command returns the top few lines of data.
print(melbourne_price_data.head())

columns_of_interest = ['Landsize', 'BuildingArea']
two_columns_of_data = melbourne_data[columns_of_interest]
print(two_columns_of_data.describe())

y = melbourne_data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_predictors]

# make copy to avoid changing original data (when Imputing)
new_data = x.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns 
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()


from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X = pd.DataFrame(my_imputer.fit_transform(new_data))

# print(melbourne_data.isnull().sum())

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(X, y)
preds = forest_model.predict(X)
print(mean_absolute_error(y, preds))
