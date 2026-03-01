"""
# Cycle 履歴
- **Cycle 1**: 基礎的な数値特徴量を追加
    - 特徴量: `horse_number`, `handicap`, `age`, `sex_encoded`
- **Cycle 2**: ジョッキー要素を追加 (直近4年の実績)
    - 特徴量: `jockey_avg_odds`, `jockey_win_rate`
- **Cycle 3**: 馬の実績要素を追加 (全期間の実績)
    - 特徴量: `horse_avg_finish_pos`, `horse_avg_popularity`, `horse_avg_odds`
- **Cycle 4**: 近走の勢いとJRA払戻率(80%)の正規化を追加
    - 特徴量: `horse_recent_3_avg_finish_pos`
- **Cycle 5**: コース適性と斤量増減を追加
    - 特徴量: `horse_nakayama_avg_finish_pos`, `handicap_diff`
- **Cycle 6**: 距離適性と馬場状態適性を追加
    - 特徴量: `horse_dist_avg_finish_pos`, `horse_firm_avg_finish_pos`
"""

import os
import django
import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
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

def fetch_horse_v6_stats():
    """馬ごとの詳細実績を集計"""
    print("Fetching highly complex horse stats...")
    query = """
        SELECT h.name as horse_name, r.finish_position, r.popularity, r.odds, 
               r.date_text, r.venue_code, r.handicap, r.distance, r.surface, r.track_condition
        FROM keibadatabase_race r
        JOIN keibadatabase_horse h ON r.horse_id = h.id
        WHERE r.odds IS NOT NULL
    """
    df = pl.read_database_uri(query, get_db_url())
    df = df.sort(["horse_name", "date_text"], descending=[False, True])
    
    # 1. 全期間
    all_time = df.group_by("horse_name").agg([
        pl.col("finish_position").mean().alias("horse_avg_finish_pos"),
        pl.col("popularity").mean().alias("horse_avg_popularity"),
        pl.col("odds").mean().alias("horse_avg_odds")
    ])
    
    # 2. 中山適性
    nakayama = df.filter(pl.col("venue_code") == "06").group_by("horse_name").agg([
        pl.col("finish_position").mean().alias("horse_nakayama_avg_finish_pos")
    ])
    
    # 3. 距離適性 (芝1200m)
    dist_1200 = df.filter((pl.col("distance") == 1200) & (pl.col("surface") == "芝")).group_by("horse_name").agg([
        pl.col("finish_position").mean().alias("horse_dist_avg_finish_pos")
    ])
    
    # 4. 良馬場適性
    firm_track = df.filter((pl.col("surface") == "芝") & (pl.col("track_condition") == "良")).group_by("horse_name").agg([
        pl.col("finish_position").mean().alias("horse_firm_avg_finish_pos")
    ])
    
    # 5. 直近3走
    recent_3 = df.group_by("horse_name").head(3).group_by("horse_name").agg([
        pl.col("finish_position").mean().alias("horse_recent_3_avg_finish_pos")
    ])
    
    # 6. 前走斤量
    last_handicap = df.group_by("horse_name").head(1).select(["horse_name", "handicap"]).rename({"handicap": "prev_handicap"})
    
    # 結合
    res = all_time
    res = res.join(nakayama, on="horse_name", how="left")
    res = res.join(dist_1200, on="horse_name", how="left")
    res = res.join(firm_track, on="horse_name", how="left")
    res = res.join(recent_3, on="horse_name", how="left")
    res = res.join(last_handicap, on="horse_name", how="left")
    return res

def fetch_training_data_cycle_6(j_stats, h_stats):
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
    df = df.join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left")
    
    df = df.with_columns(pl.lit(0.0).alias("handicap_diff"))
    
    fill_cols = [c for c in df.columns if c not in ["horse_number", "handicap", "age", "sex", "jockey_name", "horse_name", "odds"]]
    df = df.with_columns([pl.col(c).fill_null(df[c].mean()) for c in fill_cols])
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

def train_cycle_6():
    j_stats = fetch_jockey_stats()
    h_stats = fetch_horse_v6_stats()
    train_df = fetch_training_data_cycle_6(j_stats, h_stats)
    
    le_sex = LabelEncoder()
    train_df = train_df.with_columns([pl.Series("sex_encoded", le_sex.fit_transform(train_df["sex"].to_numpy()))])
    
    features = [
        "horse_number", "handicap", "age", "sex_encoded", 
        "jockey_avg_odds", "jockey_win_rate",
        "horse_avg_finish_pos", "horse_avg_popularity", "horse_avg_odds",
        "horse_recent_3_avg_finish_pos", "horse_nakayama_avg_finish_pos",
        "horse_dist_avg_finish_pos", "horse_firm_avg_finish_pos", "handicap_diff"
    ]
    
    X = train_df[features].to_pandas()
    y_prob = 0.8 / train_df["odds"].to_numpy()
    
    print(f"Training Cycle 6 on {len(X)} records...")
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
    model.fit(X, y_prob)
    
    # 重要度の表示
    importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    print("\n--- Feature Importance ---")
    print(importance)
    
    # 予測
    target_df = parse_target_race("target_race.txt")
    target_df = target_df.join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left")
    target_df = target_df.with_columns([(pl.col("handicap") - pl.col("prev_handicap")).fill_null(0.0).alias("handicap_diff")])
    
    fill_targets = [f for f in features if f not in ["horse_number", "handicap", "age", "sex_encoded", "handicap_diff"]]
    target_df = target_df.with_columns([pl.col(c).fill_null(train_df[c].mean()) for c in fill_targets])
    
    known_sex = set(le_sex.classes_)
    target_sex_np = np.array([s if s in known_sex else le_sex.classes_[0] for s in target_df["sex"].to_numpy()])
    target_df = target_df.with_columns([pl.Series("sex_encoded", le_sex.transform(target_sex_np))])
    
    X_target = target_df[features].to_pandas()
    raw_probs = model.predict(X_target)
    raw_probs = np.maximum(raw_probs, 0.001)
    
    normalized_probs = (raw_probs / raw_probs.sum()) * 0.8
    predicted_odds = 0.8 / normalized_probs
    predicted_odds = np.maximum(predicted_odds, 1.0)
    
    target_df = target_df.with_columns(pl.Series("predicted_odds", predicted_odds))
    print("\n--- Prediction Results (Cycle 6) ---")
    print(target_df.select(["horse_number", "horse_name", "jockey_name", "predicted_odds"]).sort("predicted_odds"))

if __name__ == "__main__":
    train_cycle_6()
