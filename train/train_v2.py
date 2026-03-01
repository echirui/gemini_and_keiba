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

def get_db_url():
    db_config = settings.DATABASES["app_db"]
    return f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"

def fetch_jockey_stats():
    """直近4年(2022年以降)のジョッキー成績を集計"""
    print("Fetching jockey stats (2022-2026)...")
    query = """
        SELECT j.name as jockey_name, r.odds, r.finish_position
        FROM keibadatabase_race r
        JOIN keibadatabase_jockey j ON r.jockey_id = j.id
        WHERE r.date_text >= '2022/01/01' AND r.odds IS NOT NULL
    """
    df = pl.read_database_uri(query, get_db_url())
    
    # ジョッキーごとの集計
    stats = df.group_by("jockey_name").agg([
        pl.col("odds").mean().alias("jockey_avg_odds"),
        (pl.col("finish_position") == 1).mean().alias("jockey_win_rate"),
        pl.len().alias("jockey_ride_count")
    ]).filter(pl.col("jockey_ride_count") > 10)
    
    return stats

def fetch_training_data_with_jockeys(jockey_stats):
    """中山・芝1200mの過去データにジョッキー実績を紐付け"""
    query = """
        SELECT r.horse_number, r.handicap, r.sex, r.age, r.odds, j.name as jockey_name
        FROM keibadatabase_race r
        JOIN keibadatabase_jockey j ON r.jockey_id = j.id
        WHERE r.venue_code = '06' AND r.distance = 1200 AND r.surface = '芝'
        AND r.odds IS NOT NULL
    """
    df = pl.read_database_uri(query, get_db_url())
    
    # ジョッキー統計を結合
    df = df.join(jockey_stats, on="jockey_name", how="left")
    
    # 統計がない場合の補完
    df = df.with_columns([
        pl.col("jockey_avg_odds").fill_null(df["jockey_avg_odds"].mean()),
        pl.col("jockey_win_rate").fill_null(df["jockey_win_rate"].mean())
    ])
    return df

def parse_target_race(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            if len(parts) < 5: continue
            
            sex = parts[2][0]
            try:
                age = int(parts[2][1:])
            except:
                age = 0
                
            data.append({
                "horse_number": int(parts[0]),
                "name": parts[1],
                "sex": sex,
                "age": age,
                "handicap": float(parts[3]),
                "jockey_name": parts[4]
            })
    return pl.DataFrame(data)

def train_cycle_2():
    # 1. データの取得
    j_stats = fetch_jockey_stats()
    train_df = fetch_training_data_with_jockeys(j_stats)
    
    # 2. 前処理
    le_sex = LabelEncoder()
    train_df = train_df.with_columns([
        pl.Series("sex_encoded", le_sex.fit_transform(train_df["sex"].to_numpy()))
    ])
    
    # 3. モデル構築
    features = ["horse_number", "handicap", "age", "sex_encoded", "jockey_avg_odds", "jockey_win_rate"]
    X = train_df[features].to_pandas()
    y = train_df["odds"].to_numpy()
    
    print(f"Training Cycle 2 on {len(X)} records...")
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
    model.fit(X, y)
    
    # 4. ターゲット予測
    target_df = parse_target_race("target_race.txt")
    target_df = target_df.join(j_stats, on="jockey_name", how="left")
    
    target_df = target_df.with_columns([
        pl.col("jockey_avg_odds").fill_null(train_df["jockey_avg_odds"].mean()),
        pl.col("jockey_win_rate").fill_null(train_df["jockey_win_rate"].mean())
    ])
    
    target_sex_np = np.array([s if s in set(le_sex.classes_) else le_sex.classes_[0] for s in target_df["sex"].to_numpy()])
    target_df = target_df.with_columns([
        pl.Series("sex_encoded", le_sex.transform(target_sex_np))
    ])
    
    X_target = target_df[features].to_pandas()
    preds = model.predict(X_target)
    preds = np.maximum(preds, 1.1)
    
    # 5. 結果表示
    target_df = target_df.with_columns(pl.Series("predicted_odds", preds))
    print("\n--- Prediction Results (Cycle 2: Jockey added) ---")
    print(target_df.select(["horse_number", "name", "jockey_name", "predicted_odds"]).sort("predicted_odds"))

if __name__ == "__main__":
    train_cycle_2()
