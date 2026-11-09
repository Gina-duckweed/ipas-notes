"""
鐵達尼號生存預測 Titanic Survival Prediction — 完整 Python 專案
iPAS AI 應用規劃師 中級｜資料分析與機器學習實作範例

資料來源：Kaggle Titanic train.csv（891 筆，12 個欄位）
執行方式：pip install pandas numpy matplotlib seaborn scikit-learn
          python titanic_project.py
"""
import sqlite3

import matplotlib
matplotlib.use("Agg")  # 不開視窗，直接存圖
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 中文字型（Windows 用微軟正黑體；Mac 可改 "PingFang TC"）
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. 讀取與初步檢視
# ---------------------------------------------------------------
try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    df = pd.read_csv(URL)

print("資料形狀:", df.shape)
df.info()
print(df.describe())
print("\n缺失值:\n", df.isna().sum().sort_values(ascending=False).head(4))
print("\n整體生存率: %.1f%%" % (df["Survived"].mean() * 100))

# ---------------------------------------------------------------
# 2. 探索式資料分析 EDA
# ---------------------------------------------------------------
print(df.groupby("Sex")["Survived"].mean().round(3))
print(df.groupby("Pclass")["Survived"].mean().round(3))
print(pd.crosstab(df["Pclass"], df["Sex"], values=df["Survived"], aggfunc="mean").round(3))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.countplot(data=df, x="Survived", ax=axes[0, 0])
axes[0, 0].set_title("生存人數（0=罹難, 1=生還）")
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", ax=axes[0, 1])
axes[0, 1].set_title("艙等 × 性別 生存率")
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("年齡分布（依生存）")
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f",
            cmap="coolwarm", ax=axes[1, 1])
axes[1, 1].set_title("數值欄位相關係數")
fig.tight_layout()
fig.savefig("titanic_eda.png", dpi=150)

# ---------------------------------------------------------------
# 3. 特徵工程 Feature Engineering
# ---------------------------------------------------------------
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    # 從姓名擷取稱謂 Title（Mr / Mrs / Miss / Master / Rare）
    d["Title"] = d["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    d["Title"] = d["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    d.loc[~d["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Title"] = "Rare"
    # 家庭人數與是否獨自旅行
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    # 是否有艙房號碼（Cabin 缺 77%，直接填補沒有意義，改成有/無）
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    # 票價取對數，降低右偏
    d["LogFare"] = np.log1p(d["Fare"])
    return d

df = add_features(df)
print(df.groupby("Title")["Survived"].agg(["mean", "count"]).round(3))

FEATURES_NUM = ["Age", "SibSp", "Parch", "LogFare", "FamilySize"]
FEATURES_CAT = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "HasCabin"]
X = df[FEATURES_NUM + FEATURES_CAT]
y = df["Survived"]

# ---------------------------------------------------------------
# 4. 切分資料：先切，再前處理（避免資料洩漏 Data Leakage）
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), FEATURES_NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), FEATURES_CAT),
])

# ---------------------------------------------------------------
# 5. 建模與交叉驗證
# ---------------------------------------------------------------
models = {
    "Baseline（全猜罹難）": None,
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6,
                                           random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\n5 折交叉驗證 Accuracy：")
for name, model in models.items():
    if model is None:
        print(f"  {name:22s} {1 - y_train.mean():.3f}")
        continue
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"  {name:22s} {s.mean():.3f} ± {s.std():.3f}")

# ---------------------------------------------------------------
# 6. 超參數調整（只在訓練集上做）
# ---------------------------------------------------------------
rf_pipe = Pipeline([("prep", preprocess),
                    ("model", RandomForestClassifier(random_state=RANDOM_STATE))])
grid = GridSearchCV(rf_pipe, {
    "model__n_estimators": [200, 400],
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [1, 3],
}, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print("\n最佳參數:", grid.best_params_, "CV=%.3f" % grid.best_score_)

# ---------------------------------------------------------------
# 7. 測試集評估
# ---------------------------------------------------------------
best = grid.best_estimator_
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
print("\n測試集 Accuracy: %.3f  ROC-AUC: %.3f" % (accuracy_score(y_test, pred),
                                               roc_auc_score(y_test, proba)))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=3))

# 特徵重要性
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(best.named_steps["model"].feature_importances_, index=names)
print(imp.sort_values(ascending=False).head(8).round(3))

fig, ax = plt.subplots(figsize=(7, 4))
imp.sort_values().tail(10).plot.barh(ax=ax)
ax.set_title("隨機森林 特徵重要性 Top 10")
fig.tight_layout()
fig.savefig("titanic_importance.png", dpi=150)

# ---------------------------------------------------------------
# 8. 同樣的分析用 SQL 做一次（SQLite 記憶體資料庫）
# ---------------------------------------------------------------
con = sqlite3.connect(":memory:")
df.drop(columns=["Title"]).to_sql("titanic", con, index=False)
print(pd.read_sql("""
    SELECT Pclass, Sex,
           COUNT(*)                     AS n,
           ROUND(AVG(Survived) * 100, 1) AS survival_pct
    FROM titanic
    GROUP BY Pclass, Sex
    HAVING COUNT(*) >= 50
    ORDER BY survival_pct DESC;
""", con))
con.close()
