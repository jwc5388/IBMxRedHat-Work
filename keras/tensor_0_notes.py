# --- numpy (수치 계산, 배열 처리) ---
import numpy as np
from numpy import array, arange, reshape, zeros, ones
from numpy.random import randn, randint, seed

# --- pandas (데이터 처리) ---
import pandas as pd

# --- scikit-learn (머신러닝 핵심) ---
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

# 주요 분류 모델
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 주요 회귀 모델
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 클러스터링
from sklearn.cluster import KMeans, DBSCAN

# --- imbalanced-learn (불균형 데이터 처리) ---
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# --- keras (딥러닝, 텐서플로우 기반) ---
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, GRU, Embedding, BatchNormalization, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import to_categorical, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# --- matplotlib / seaborn (시각화) ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- 기타 ---
import os
import random
import datetime



# problem_type_checker.py

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def guess_problem_type(y):
    """
    Guess the type of machine learning problem based on the target vector y.
    """
    unique_values = np.unique(y)
    if y.dtype.kind in 'fc':  # float or complex
        return "Regression"
    elif len(unique_values) == 2:
        return "Binary Classification"
    elif len(unique_values) > 2 and y.dtype.kind in 'iu':  # int or unsigned int
        return "Multiclass Classification"
    else:
        return "Unknown or Multilabel"

def print_target_summary(y):
    print("\n=== Target Summary ===")
    print("dtype:", y.dtype)
    print("Unique values:", np.unique(y))
    print("Value counts:\n", pd.Series(y).value_counts())
    print("\nGuessed problem type:", guess_problem_type(y))

def plot_target_distribution(y):
    print("\n=== Showing Distribution Plot ===")
    if y.dtype.kind in 'fc':
        sns.histplot(y, kde=True)
        plt.title("Target Distribution (Regression)")
    else:
        sns.countplot(x=y)
        plt.title("Target Class Distribution (Classification)")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.show()

# Example usage:
if __name__ == "__main__":
    import pandas as pd

    # Example 1: Binary Classification
    y_binary = np.array([0, 1, 1, 0, 0, 1, 1, 0])
    print_target_summary(y_binary)
    plot_target_distribution(y_binary)

    # Example 2: Regression
    y_regression = np.array([10.5, 11.2, 12.3, 10.8, 13.5, 11.9])
    print_target_summary(y_regression)
    plot_target_distribution(y_regression)

    # Example 3: Multiclass Classification
    y_multiclass = np.array([0, 1, 2, 1, 2, 0, 1, 2, 2, 0])
    print_target_summary(y_multiclass)
    plot_target_distribution(y_multiclass)
