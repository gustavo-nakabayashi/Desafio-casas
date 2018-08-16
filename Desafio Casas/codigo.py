import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import sys
sys._enablelegacywindowsfsencoding()

#função erro absoluto medio
def get_mae(predictors_train, predictors_val, targ_train, targ_val):
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(predictors_train, targ_train)
    preds_val = forest_model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

train_file_path = Path('C:/Users/Bancada/Documents/Gustavo/Programação/Desafio Casas/Dados/train.csv') # this is the path to the Iowa data that you will use
test_file_path = Path('C:/Users/Bancada/Documents/Gustavo/Programação/Desafio Casas/Dados/test.csv') # this is the path to the Iowa data that you will use
train = pd.read_csv(train_file_path)#le o arquivo de treino
test = pd.read_csv(test_file_path)#le o arquivo de treino

y = train.SalePrice
predictors = train.drop(['SalePrice'], axis=1)
X = predictors.select_dtypes(exclude=['object'])
#X = train[numeric_predictors]

#test_y = test.SalePrice
#test_x = test[predictors]

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(val_X)


my_mae = get_mae(imputed_X_train, imputed_X_test, train_y, val_y)
print(my_mae)

    


# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
#print()
