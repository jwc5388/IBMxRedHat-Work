import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score



#1 data
x,y = load_digits(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=42, train_size=0.8, stratify=y)


learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3,4,5,6,7]


best_score = 0
best_parameters = ''

for i, lr in enumerate(learning_rate):
    for j, md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(learning_rate= lr,
                                               max_depth=md)
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(lr, md, '의 정답률:', result)
        
        if result > best_score:
            best_score = result
            best_parameters = lr, md
        print(i, ',', j, '번쨰 score:', round(result,4), '도는중...', '현재 최고점:', best_parameters, ':', best_score)
            
        
        
        
        
print('최고 점수: {:.2f}'.format(best_score))
print('최적 매개변수: ', best_parameters)


# 최고 점수: 0.97
# 최적 매개변수:  [0.1, 5]


# 최고 점수: 0.98
# 최적 매개변수:  [0.1, 7]