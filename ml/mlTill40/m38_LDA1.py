#traintestsplit 후  scaling 후 pca
#LDA n_component 는 y 라벨의 갯수 -1 이하로 만들수 있다.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

n_features = x.shape[1]
for i in range(1,n_features+1):
    
    # pca = PCA(n_components= i)
    lda = LinearDiscriminantAnalysis(n_components=2)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test)
    
    print(x_train_lda.shape)
    # exit()
    
    model = RandomForestClassifier(random_state=333)

    #3 train
    model.fit(x_train_lda, y_train)

    #4 evalutate
    result = model.score(x_test_lda, y_test)
    
    print(f"n_components={i} → model score: {result:.4f}")

# n_components=1 → model score: 1.0000
# (120, 2)
# n_components=2 → model score: 1.0000
# (120, 2)
# n_components=3 → model score: 1.0000
# (120, 2)
# n_components=4 → model score: 1.0000