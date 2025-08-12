

# import os
# import zipfile
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler

# try:
#     from statsmodels.tsa.seasonal import STL
#     HAS_STL = True
# except:
#     HAS_STL = False

# # ===== 0) 경로/입력 =====
# BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
#     else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
# PATH = os.path.join(BASE_PATH, "_data/dacon/electricity/")
# OUT_DIR = os.path.join(PATH, "outlier_report_advanced")
# os.makedirs(OUT_DIR, exist_ok=True)

# TRAIN_PATH = os.path.join(PATH, "train.csv")
# TEST_PATH = os.path.join(PATH, "test.csv")
# BI_PATH = os.path.join(PATH, "building_info.csv")

# # ===== 1) 데이터 로드 =====
# train = pd.read_csv(TRAIN_PATH)
# test = pd.read_csv(TEST_PATH)
# bi = pd.read_csv(BI_PATH)

# # ===== 2) datetime 변환 =====
# for df in (train, test):
#     if "일시" in df.columns:
#         df["일시"] = pd.to_datetime(df["일시"], format="%Y%m%d %H", errors="coerce")

# # ===== 3) building_info 수치형 변환 =====
# for col in ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)", "연면적(m2)", "냉방면적(m2)"]:
#     if col in bi.columns:
#         bi[col] = pd.to_numeric(bi[col].replace("-", np.nan), errors="coerce")

# # ===== 4) 병합 =====
# if "건물번호" in train.columns:
#     train = train.merge(bi.drop_duplicates("건물번호"), on="건물번호", how="left")
# if "건물번호" in test.columns:
#     test = test.merge(bi.drop_duplicates("건물번호"), on="건물번호", how="left")

# # ===== 공통 함수 =====
# def ensure_numeric(df, cols):
#     for c in cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#     return [c for c in cols if c in df.columns]

# def save_df(df, name):
#     df.to_csv(os.path.join(OUT_DIR, name), index=False)

# # ===== 물리적 범위 체크 =====
# def physical_checks(df, name, has_target):
#     checks = []
#     def flag(cond, label):
#         if cond.sum() > 0:
#             sub = df.loc[cond, ["건물번호","일시"]].copy()
#             sub[label] = 1
#             save_df(sub, f"{name}_{label}.csv")
#         checks.append((label, int(cond.sum())))
#     if "습도(%)" in df.columns:
#         flag((df["습도(%)"]<0)|(df["습도(%)"]>100), "bad_humidity")
#     if "풍속(m/s)" in df.columns:
#         flag(df["풍속(m/s)"]<0, "bad_wind")
#     if has_target and "전력소비량(kWh)" in df.columns:
#         flag(df["전력소비량(kWh)"]<0, "bad_target")
#     save_df(pd.DataFrame(checks, columns=["check","count"]), f"{name}_physical_checks.csv")

# # ===== IQR =====
# def iqr_outliers(df, name, cols):
#     cols = ensure_numeric(df, cols)
#     out_list = []
#     for b, g in df.groupby("건물번호"):
#         for c in cols:
#             s = g[c].dropna()
#             if len(s) < 10: continue
#             q1,q3 = np.quantile(s,[0.25,0.75])
#             iqr = q3-q1
#             low, high = q1-1.5*iqr, q3+1.5*iqr
#             mask = (g[c] < low) | (g[c] > high)
#             if mask.any():
#                 tmp = g.loc[mask, ["건물번호","일시", c]].copy()
#                 tmp["outlier_col"] = c
#                 out_list.append(tmp)
#     if out_list:
#         save_df(pd.concat(out_list), f"{name}_iqr_outliers.csv")

# # ===== IsolationForest =====
# def isolation_forest(df, name, cols):
#     cols = ensure_numeric(df, cols)
#     res_list = []
#     for b, g in df.groupby("건물번호"):
#         X = g[cols].dropna()
#         if len(X) < 50: continue
#         Xs = StandardScaler().fit_transform(X)
#         iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
#         pred = iso.fit_predict(Xs)
#         mask = pred == -1
#         if mask.any():
#             tmp = g.loc[X.index[mask], ["건물번호","일시"]+cols].copy()
#             res_list.append(tmp)
#     if res_list:
#         save_df(pd.concat(res_list), f"{name}_iso_outliers.csv")

# # ===== STL =====
# def stl_residual_outliers(df, name, col="전력소비량(kWh)"):
#     if not HAS_STL or col not in df.columns: return
#     out_list = []
#     for b, g in df.groupby("건물번호"):
#         s = g[col].dropna()
#         if len(s) < 48: continue
#         try:
#             stl = STL(s, period=24, robust=True)
#             res = stl.fit()
#             resid = res.resid
#             zscore = (resid - resid.mean()) / resid.std(ddof=0)
#             mask = zscore.abs() > 4
#             if mask.any():
#                 tmp = g.loc[s.index[mask], ["건물번호","일시", col]].copy()
#                 out_list.append(tmp)
#         except:
#             pass
#     if out_list:
#         save_df(pd.concat(out_list), f"{name}_stl_resid_outliers.csv")

# # ===== 실행 =====
# NUM_CANDS = [
#     "전력소비량(kWh)","기온(°C)","강수량(mm)","풍속(m/s)","습도(%)","일조(hr)","일사(MJ/m2)",
#     "연면적(m2)","냉방면적(m2)","태양광용량(kW)","ESS저장용량(kWh)","PCS용량(kW)"
# ]
# NUM_CANDS = [c for c in NUM_CANDS if c in train.columns]

# physical_checks(train, "train", True)
# physical_checks(test, "test", False)
# iqr_outliers(train, "train", NUM_CANDS)
# iqr_outliers(test, "test", [c for c in NUM_CANDS if c != "전력소비량(kWh)"])
# isolation_forest(train, "train", NUM_CANDS)
# isolation_forest(test, "test", [c for c in NUM_CANDS if c != "전력소비량(kWh)"])
# stl_residual_outliers(train, "train")

# # ===== 결과 ZIP =====
# zip_path = os.path.join(PATH, "outlier_report_advanced.zip")
# with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
#     for root, _, files in os.walk(OUT_DIR):
#         for f in files:
#             fp = os.path.join(root, f)
#             zf.write(fp, arcname=os.path.relpath(fp, OUT_DIR))

# print(f"ZIP 생성 완료: {zip_path}")


# import os
# import pandas as pd
# from sklearn.ensemble import IsolationForest

# # ===== 경로 설정 =====
# BASE_PATH = "/workspace/TensorJae/Study25/" if os.path.exists("/workspace/TensorJae/Study25/") \
#     else os.path.expanduser("~/Desktop/IBM:RedHat/Study25/")
# PATH = os.path.join(BASE_PATH, "_data/dacon/electricity/")
# OUT_PATH = os.path.join(PATH, "outlier_report")
# os.makedirs(OUT_PATH, exist_ok=True)

# # ===== 데이터 로드 =====
# train = pd.read_csv(os.path.join(PATH, "train.csv"))
# train["일시"] = pd.to_datetime(train["일시"], format="%Y%m%d %H", errors="coerce")

# # 분석 대상 피처 (숫자형만)
# features = ["기온(°C)", "강수량(mm)", "풍속(m/s)", "습도(%)", 
#             "일조(hr)", "일사(MJ/m2)", "전력소비량(kWh)"]

# features = [c for c in features if c in train.columns]

# # ===== Isolation Forest 모델 =====
# iso = IsolationForest(
#     contamination=0.01,  # 전체의 1%를 이상치로 간주
#     random_state=42
# )
# iso.fit(train[features])

# # 예측 결과 추가
# train["iso_pred"] = iso.predict(train[features])    # 1=정상, -1=이상치
# train["iso_score"] = iso.decision_function(train[features])  # 낮을수록 이상치

# # 이상치만 필터링
# outliers = train[train["iso_pred"] == -1].copy()

# # 각 시점의 모든 피처 값을 한 열에 묶어서 보기 좋게 저장
# outliers["이상치컬럼값"] = outliers[features].to_dict(orient="records")

# # 저장
# outliers.to_csv(os.path.join(OUT_PATH, "train_iso_outliers_detailed.csv"), index=False)

# print(f"[완료] {len(outliers)}개의 이상치가 저장되었습니다.")
