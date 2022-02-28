from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from statistics import mean
from sklearn.model_selection import GridSearchCV
import numpy as np

# variable that contain .csv file
full_file = "steel.csv"

# arrays of String's that contain independent data and target data
independant_columns = ["normalising_temperature","tempering_temperature","percent_silicon","percent_chromium",
                       "percent_copper","percent_nickel","percent_sulphur","percent_carbon","percent_manganese"]
dependant_column = ["tensile_strength"]

# variable for reading the data from the csv file
df_full = pd.read_csv(full_file)

# seperator for columns in the independent variables dataframe
x = df_full.loc[:, independant_columns]

# seperator for columns in the target variable dataframe
y = df_full.loc[:, dependant_column]

# KFold with number of folds, shuffle used to shuffle data, this produced better results
kf = KFold(n_splits=10,shuffle=True)

# domain independant & domain specific measures of error chosen empty lists
average_r_squared = []
average_mean_squared_error = []

# lists containing hyperperameter values
criterion_val = ["squared_error","friedman_mse"]
# criterion = ["absolute_error", "poisson"]
max_depth= list(range(1, 30))

# parameters list created for uses with chosen regressor algorithm
parameters = [{'criterion': criterion_val,'max_depth':max_depth}]

# for loop used to perform 10 fold cross validation
# gridsearch function used to autotune hyperparametrs
# rmse and R^2 used for error measurs, used numpy to gey sqrt() of mean squared error
for train_index, test_index in kf.split(x):
   x_train, x_test = x.loc[train_index], x.loc[test_index]
   y_train, y_test = y.loc[train_index], y.loc[test_index]
   grid_search = GridSearchCV(DecisionTreeRegressor(),param_grid=parameters)
   grid_search.fit(x_train,y_train)
   #print("Training X:", x_train)
   #print("Training Y:", y_train)
   predict_x = grid_search.predict(x_test)
   rmse_error = metrics.mean_squared_error(predict_x,y_test)
   r_squared_error = metrics.r2_score(predict_x,y_test)
   #print("mean squared Error: ",mse_error)
   #print("R Squared Error: ",r_squared_error)
   average_r_squared.append(r_squared_error)
   average_mean_squared_error.append(np.sqrt(rmse_error))

# print results to screen
print("average for R squared values: ",mean(average_r_squared))
print("average for Mean Squared Error",mean(average_mean_squared_error))
#print(average_r_squared)

# References
# Mannion, D., 2021. Machine Learning - Lecture Notes. [PDF] Blackboard, Galway.