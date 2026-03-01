"""
# Cycle 履歴
- **Cycle 1**: 基礎的な数値特徴量を追加
    - 特徴量: `horse_number`, `handicap`, `age`, `sex_encoded`
- **Cycle 2**: ジョッキー要素を追加 (直近4年の実績)
    - 特徴量: `jockey_avg_odds`, `jockey_win_rate`
- **Cycle 3**: 馬の実績要素を追加 (全期間の実績)
    - 特徴量: `horse_avg_finish_pos`, `horse_avg_popularity`, `horse_avg_odds`
- **Cycle 4**: 近走の勢いとJRA払戻率(80%)の正規化を追加
    - 特徴量: `horse_recent_3_avg_finish_pos` (直近3走の平均着順)
    - ロジック: 予測値を支持率として扱い、全馬の合計が払戻率80%に収束するように正規化。
"""

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
    stats = df.group_by("jockey_name").agg([
        pl.col("odds").mean().alias("jockey_avg_odds"),
        (pl.col("finish_position") == 1).mean().alias("jockey_win_rate"),
        pl.len().alias("jockey_ride_count")
    ]).filter(pl.col("jockey_ride_count") > 5)
    return stats

def fetch_horse_stats():
    """馬ごとの全期間および近走(直近3走)の実績を集計"""
    print("Fetching horse stats (all time & recent 3)...")
    query = """
        SELECT h.name as horse_name, r.finish_position, r.popularity, r.odds, r.date_text
        FROM keibadatabase_race r
        JOIN keibadatabase_horse h ON r.horse_id = h.id
        WHERE r.odds IS NOT NULL
    """
    df = pl.read_database_uri(query, get_db_url())
    
    # 全期間
    all_time = df.group_by("horse_name").agg([
        pl.col("finish_position").mean().alias("horse_avg_finish_pos"),
        pl.col("popularity").mean().alias("horse_avg_popularity"),
        pl.col("odds").mean().alias("horse_avg_odds")
    ])
    
    # 直近3走
    recent_3 = (
        df.sort(["horse_name", "date_text"], descending=[False, True])
        .group_by("horse_name")
        .head(3)
        .group_by("horse_name")
        .agg([
            pl.col("finish_position").mean().alias("horse_recent_3_avg_finish_pos")
        ])
    )
    
    return all_time.join(recent_3, on="horse_name", how="left")

def fetch_training_data_with_all_stats(jockey_stats, horse_stats):
    query = """
        SELECT r.horse_number, r.handicap, r.sex, r.age, r.odds, 
               j.name as jockey_name, h.name as horse_name
        FROM keibadatabase_race r
        JOIN keibadatabase_jockey j ON r.jockey_id = j.id
        JOIN keibadatabase_horse h ON r.horse_id = h.id
        WHERE r.venue_code = '06' AND r.distance = 1200 AND r.surface = '芝'
        AND r.odds IS NOT NULL
    """
    df = pl.read_database_uri(query, get_db_url())
    df = df.join(jockey_stats, on="jockey_name", how="left").join(horse_stats, on="horse_name", how="left")
    
    # 欠損値補完
    fill_vals = {
        "jockey_avg_odds": df["jockey_avg_odds"].mean(),
        "jockey_win_rate": df["jockey_win_rate"].mean(),
        "horse_avg_finish_pos": df["horse_avg_finish_pos"].mean(),
        "horse_avg_popularity": df["horse_avg_popularity"].mean(),
        "horse_avg_odds": df["horse_avg_odds"].mean(),
        "horse_recent_3_avg_finish_pos": df["horse_recent_3_avg_finish_pos"].mean()
    }
    df = df.with_columns([pl.col(c).fill_null(v) for c, v in fill_vals.items()])
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
            try: age = int(parts[2][1:])
            except: age = 0
            data.append({
                "horse_number": int(parts[0]), "horse_name": parts[1],
                "sex": sex, "age": age, "handicap": float(parts[3]), "jockey_name": parts[4]
            })
    return pl.DataFrame(data)

def train_cycle_4():
    j_stats = fetch_jockey_stats()
    h_stats = fetch_horse_stats()
    train_df = fetch_training_data_with_all_stats(j_stats, h_stats)
    
    le_sex = LabelEncoder()
    train_df = train_df.with_columns([pl.Series("sex_encoded", le_sex.fit_transform(train_df["sex"].to_numpy()))])
    
    features = [
        "horse_number", "handicap", "age", "sex_encoded", 
        "jockey_avg_odds", "jockey_win_rate",
        "horse_avg_finish_pos", "horse_avg_popularity", "horse_avg_odds",
        "horse_recent_3_avg_finish_pos"
    ]
    
    # 支持率(Probability)を目的変数にする: Prob = 0.8 / Odds
    X = train_df[features].to_pandas()
    y_prob = 0.8 / train_df["odds"].to_numpy()
    
    print(f"Training Cycle 4 on {len(X)} records...")
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
    model.fit(X, y_prob)
    
    # 予測
    target_df = parse_target_race("target_race.txt")
    target_df = target_df.join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left")
    
    # 欠損値補完
    cols_to_fill = ["jockey_avg_odds", "jockey_win_rate", "horse_avg_finish_pos", "horse_avg_popularity", "horse_avg_odds", "horse_recent_3_avg_finish_pos"]
    target_df = target_df.with_columns([pl.col(c).fill_null(train_df[c].mean()) for c in cols_to_fill])
    
    known_sex = set(le_sex.classes_)
    target_sex_np = np.array([s if s in known_sex else le_sex.classes_[0] for s in target_df["sex"].to_numpy()])
    target_df = target_df.with_columns([pl.Series("sex_encoded", le_sex.transform(target_sex_np))])
    
    X_target = target_df[features].to_pandas()
    raw_probs = model.predict(X_target)
    raw_probs = np.maximum(raw_probs, 0.001)
    
    # 正規化: 合計を0.8にする
    normalized_probs = (raw_probs / raw_probs.sum()) * 0.8
    predicted_odds = 0.8 / normalized_probs
    predicted_odds = np.maximum(predicted_odds, 1.0)
    
    target_df = target_df.with_columns(pl.Series("predicted_odds", predicted_odds))
    print("\n--- Prediction Results (Cycle 4: JRA 80% Rule applied) ---")
    print(target_df.select(["horse_number", "horse_name", "jockey_name", "predicted_odds"]).sort("predicted_odds"))

if __name__ == "__main__":
    train_cycle_4()
