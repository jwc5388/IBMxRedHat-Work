from xgboost import XGBClassifier, XGBRegressor  # 또는 XGBRegressor

model = XGBClassifier(             # 회귀는 XGBRegressor
    tree_method='gpu_hist',        # ✅ GPU 기반 트리 생성
    predictor='gpu_predictor',     # ✅ 예측도 GPU에서 수행
    gpu_id=0                       # ✅ 사용할 GPU 번호 (기본은 0)
)

##### CPU 사용 정보 출력 #####

import psutil

print("CPU 개수 (논리):", psutil.cpu_count(logical=True))
print("CPU 개수 (물리):", psutil.cpu_count(logical=False))
print("현재 전체 CPU 사용률 (%):", psutil.cpu_percent(interval=1))



    # tree_method='gpu_hist',         # ✅ GPU 학습 방식 지정
    # predictor='gpu_predictor',      # ✅ 예측도 GPU로 (optional)
    # gpu_id=0,                       # ✅ 첫 번째 GPU 사용
    # n_jobs=-5,
    # verbosity=1)
    
    
    
    
xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
lg = LGBMRegressor(device='gpu', gpu_use_dp=False)
cat = CatBoostRegressor(task_type='GPU', devices='0', verbose=0)
# rf = RandomForestRegressor() ← CPU 전용