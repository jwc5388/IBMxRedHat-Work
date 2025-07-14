from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

model_list = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

# === Load Data ===
path = './Study25/_data/dacon/cancer/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target Label ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']


# === One-Hot Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=categorical_cols)
test_csv = pd.get_dummies(test_csv, columns=categorical_cols)
x, test_csv = x.align(test_csv, join='left', axis=1, fill_value=0)

# === Split Train/Validation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

for i, model in enumerate(model_list):
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    
    print(f"[{i}] {model.__class__.__name__}")
    print(f"    ✅ Validation Accuracy: {result:.4f}")



# [0] LinearSVC
#     ✅ Validation Accuracy: 0.8794
# [1] LogisticRegression
#     ✅ Validation Accuracy: 0.8792
# [2] DecisionTreeClassifier
#     ✅ Validation Accuracy: 0.8181
# [3] RandomForestClassifier
#     ✅ Validation Accuracy: 0.8802