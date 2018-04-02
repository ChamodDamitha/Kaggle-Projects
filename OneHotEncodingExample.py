import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

y = melbourne_data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude', 'Suburb']
X = melbourne_data[melbourne_predictors]

#One hot encoding categoricals
one_hot_encoded_X = pd.get_dummies(X)

#without categoricals
X = X.select_dtypes(exclude=['object'])
#print(one_hot_encoded_X.head())

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X = pd.DataFrame(my_imputer.fit_transform(X))
one_hot_encoded_X = pd.DataFrame(my_imputer.fit_transform(one_hot_encoded_X))

#print(one_hot_encoded_X.head())

# print(melbourne_data.isnull().sum())
#print(one_hot_encoded_X.dtypes)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

mae_without_categoricals = get_mae(X, y)

mae_one_hot_encoded = get_mae(one_hot_encoded_X, y)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
