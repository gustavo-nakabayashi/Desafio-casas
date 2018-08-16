import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import sys

sys._enablelegacywindowsfsencoding()
#função erro absoluto medio
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(predictors_train, targ_train)
    preds_val = forest_model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

train_file_path = Path('C:/Users/Bancada/Documents/Gustavo/Programação/Desafio Casas/Dados/train.csv') # this is the path to the Iowa data that you will use
test_file_path = Path('C:/Users/Bancada/Documents/Gustavo/Programação/Desafio Casas/Dados/test.csv') # this is the path to the Iowa data that you will use
train = pd.read_csv(train_file_path)#le o arquivo de treino
test = pd.read_csv(test_file_path)#le o arquivo de treino

predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd', 'YearRemodAdd', 'TotalBsmtSF']
y = train.SalePrice
X = train[predictors]
#test_y = test.SalePrice
#test_x = test[predictors]

train_X, val_X, train_y, val_y = train_test_split(X, y)
best_one = 99999999
for i in range(5,100,2):
    max_leaf_nodes = i
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    #print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    if my_mae < best_one:
        best_one = my_mae
        folhas = max_leaf_nodes
    
print(best_one)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
#print()
