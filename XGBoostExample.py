import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

y = melbourne_data.Price
melbourne_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude', 'Suburb']
X = melbourne_data[melbourne_predictors]

#without categoricals
X = X.select_dtypes(exclude=['object'])

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X = pd.DataFrame(my_imputer.fit_transform(X))

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

from xgboost import XGBRegressor

#my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
#my_model.fit(train_X, train_y, verbose=False)

from sklearn.metrics import mean_absolute_error

def getMAE(est) :
	#model tuning
	#my_model = XGBRegressor(n_estimators=est)
             
	#add learning rate
	my_model = XGBRegressor(n_estimators=est, learning_rate=0.05)
	my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)


	# make predictions
	predictions = my_model.predict(test_X)
	print("estimators = " + str(est) +" Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

for i in range(100, 2001, 200):
	getMAE(i) 
