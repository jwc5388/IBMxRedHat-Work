#traintestsplit 후  scaling 후 pca


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

#1.data

dataset = load_iris()
x = dataset['data']
y = dataset.target

print(x.shape, y.shape) #(150, 4) (150,)

#어떤놈들이 scaler는 pca 전에 하는게 좋다 했다




x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=337, stratify=y)

scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_features = x.shape(1)
for i in range(1,n_features+1):
    
    pca = PCA(n_components= i)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    
    model = RandomForestClassifier(random_state=333)

    #3 train
    model.fit(x_train_pca, y_train)

    #4 evalutate
    result = model.score(x_test_pca, y_test)
    
    print(f"n_components={i} → model score: {result:.4f}")


# print(x)
# print(x.shape) #(150,3)

# #2 model
# model = RandomForestClassifier(random_state=333)

# #3 train
# model.fit(x_train, y_train)

# #4 evalutate
# result = model.score(x_test, y_test)

# print(x.shape)


# # (150, 3)


# #(150, 3) 의 model score: 0.8666666666666667
# # (150, 2) 의 model score: 0.9



