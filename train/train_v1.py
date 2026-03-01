import os
import django
import polars as pl
import lightgbm as lgb
import numpy as np
from django.conf import settings
from sklearn.preprocessing import LabelEncoder

# Django初期化
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gemini_and_keiba.settings")
django.setup()

def fetch_training_data():
    """中山・芝1200mの過去データを取得"""
    query = """
        SELECT horse_number, handicap, sex, age, odds 
        FROM keibadatabase_race 
        WHERE venue_code = '06' AND distance = 1200 AND surface = '芝'
        AND odds IS NOT NULL
    """
    db_config = settings.DATABASES["app_db"]
    db_url = f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"
    df = pl.read_database_uri(query, db_url)
    return df

def parse_target_race(file_path):
    """target_race.txt をパースして特徴量に変換"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            
            horse_num = int(parts[0])
            name = parts[1]
            gender_age = parts[2] # "牡4"
            handicap = float(parts[3])
            jockey = parts[4]
            
            sex = gender_age[0]
            try:
                age = int(gender_age[1:])
            except ValueError:
                age = 0
            
            data.append({
                "horse_number": horse_num,
                "name": name,
                "sex": sex,
                "age": age,
                "handicap": handicap,
                "jockey": jockey
            })
    return pl.DataFrame(data)

def train_cycle_1():
    # 1. 訓練データの取得
    print("Fetching training data...")
    train_df = fetch_training_data()
    
    # 2. 前処理 (性別の数値化)
    le_sex = LabelEncoder()
    # 訓練データの性別をユニークにするために変換
    train_df = train_df.with_columns([
        pl.Series("sex_encoded", le_sex.fit_transform(train_df["sex"].to_numpy()))
    ])
    
    # 3. モデル構築 (LightGBM)
    features = ["horse_number", "handicap", "age", "sex_encoded"]
    X = train_df[features].to_numpy()
    y = train_df["odds"].to_numpy()
    
    print(f"Training on {len(X)} records...")
    model = lgb.LGBMRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42,
        verbosity=-1 # ログ出力を抑制
    )
    model.fit(X, y)
    
    # 4. ターゲットデータの予測
    print("Predicting target race...")
    target_df = parse_target_race("target_race.txt")
    
    # 性別エンコーディング (未知のラベル対応)
    target_sex_np = target_df["sex"].to_numpy()
    # 学習に含まれていない性別があれば '牡' (0番目) にフォールバックするなどの処置
    known_sex = set(le_sex.classes_)
    target_sex_np = np.array([s if s in known_sex else le_sex.classes_[0] for s in target_sex_np])
    
    target_df = target_df.with_columns([
        pl.Series("sex_encoded", le_sex.transform(target_sex_np))
    ])
    
    X_target = target_df[features].to_numpy()
    preds = model.predict(X_target)
    
    # 予測値が0以下にならないようにクリップ
    preds = np.maximum(preds, 1.1)
    
    # 5. 結果表示
    target_df = target_df.with_columns(pl.Series("predicted_odds", preds))
    print("\n--- Prediction Results (Cycle 1) ---")
    print(target_df.select(["horse_number", "name", "predicted_odds"]).sort("predicted_odds"))

if __name__ == "__main__":
    train_cycle_1()
