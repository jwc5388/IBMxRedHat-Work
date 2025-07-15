# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_california_housing, load_diabetes

# # 1. 데이터 로딩
# dataset = load_diabetes()
# x = dataset.data
# y = dataset.target  # median house value ($100,000 단위)

# # 2. 로그 변환 (지수분포형 y 대상)
# print(y)
# print(y.shape)
# print(np.min(y), np.max(y))  # 0.14999 ~ 5.00001 등


# # (442,)
# # 25.0 346.0
# # exit()

# log_y = np.log1p(y)  # log(1 + y) 변환 (0 대비 안전함)

# # 3. 히스토그램 시각화
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.hist(y, bins=50, color='blue', alpha=0.5)
# plt.title('Original Target (House Value)')

# plt.subplot(1, 2, 2)
# plt.hist(log_y, bins=50, color='red', alpha=0.5)
# plt.title('Log Transformed Target')

# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 1. 데이터 로딩
dataset = load_diabetes()
x = dataset.data
y = dataset.target

# 2. 로그 변환
# x는 음수값이 있어 NaN 발생 가능 → clip 후 처리
x_clipped = np.clip(x, a_min=0, a_max=None)  # 음수 0으로
log_x = np.log1p(x_clipped)
log_y = np.log1p(y)  # y는 항상 양수

# # 3. NaN 보정 (혹시라도)
# log_x = SimpleImputer(strategy='constant', fill_value=0).fit_transform(log_x)

# 4. 모델 정의
model = LinearRegression()

# 5. 모델 평가 함수
def evaluate_model(x_data, y_data, description):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # 로그 y를 예측한 경우 → 역변환
    if "로그 y" in description:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)

    score = r2_score(y_test, y_pred)
    print(f"{description:<30} → R² Score: {score:.4f}")

# 6. 네 가지 케이스 출력
print("\n✅ 모델 성능 비교 (R² Score)")
evaluate_model(x, y, "1. 원본 x, 원본 y")
evaluate_model(x, log_y, "2. 원본 x, 로그 y")
evaluate_model(log_x, y, "3. 로그 x, 원본 y")
evaluate_model(log_x, log_y, "4. 로그 x, 로그 y")



# 3. 히스토그램 시각화
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(y, bins=50, color='blue', alpha=0.5)
plt.title('Original Target (House Value)')

plt.subplot(1, 2, 2)
plt.hist(log_y, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed Target')

plt.tight_layout()
plt.show()



# ✅ 모델 성능 비교 (R² Score)
# 1. 원본 x, 원본 y                  → R² Score: 0.4526
# 2. 원본 x, 로그 y                  → R² Score: 0.4272
# 3. 로그 x, 원본 y                  → R² Score: 0.4444
# 4. 로그 x, 로그 y                  → R² Score: 0.2913