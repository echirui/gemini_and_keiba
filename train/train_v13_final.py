"""
# Cycle 履歴
- Cycle 1-12: 基礎から重賞実績まで追加
- Cycle 13: 当日の馬体重注入と支持率分布の調整
"""

import os
import django
import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Django初期化
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gemini_and_keiba.settings")
django.setup()

def get_db_url():
    db_config = settings.DATABASES["app_db"]
    return f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"

def time_to_seconds(t_str):
    try:
        if not t_str or ":" not in str(t_str): return float(t_str) if t_str else None
        m, s = str(t_str).split(":")
        return float(m) * 60 + float(s)
    except:
        return None

def get_grade(name, cls):
    s = f"{name} {cls}"
    if "G1" in s or "GI " in s or " GI" in s: return 4
    if "G2" in s or "GII" in s: return 3
    if "G3" in s or "GIII" in s: return 2
    if "オープン" in s or "リステッド" in s or "(L)" in s: return 1
    return 0

def fetch_stats_v13(target_date):
    print("Fetching statistics...")
    query = f"""
        SELECT h.name as horse_name, j.name as jockey_name, r.finish_position, r.popularity, r.odds, 
               r.date_text, r.venue_code, r.handicap, r.distance, r.surface, r.track_condition,
               r.running_time, r.final_600m_time, r.race_name, r.race_class, r.weight, r.weight_change
        FROM keibadatabase_race r
        JOIN keibadatabase_horse h ON r.horse_id = h.id
        JOIN keibadatabase_jockey j ON r.jockey_id = j.id
        WHERE r.date_text < '{target_date}' AND r.odds IS NOT NULL
    """
    df = pl.read_database_uri(query, get_db_url())
    df = df.sort(["horse_name", "date_text"], descending=[False, True])
    
    df = df.with_columns([
        pl.col("running_time").map_elements(time_to_seconds, return_dtype=pl.Float64).alias("time_secs"),
        pl.struct(["race_name", "race_class"]).map_elements(lambda x: get_grade(x["race_name"], x["race_class"]), return_dtype=pl.Int32).alias("race_grade")
    ])
    df = df.with_columns([(pl.col("distance") / pl.col("time_secs")).alias("speed")])

    h_grade_stats = df.group_by("horse_name").agg([
        pl.col("race_grade").max().alias("horse_max_grade"),
        pl.when((pl.col("race_grade") == 4) & (pl.col("finish_position") <= 3)).then(1).otherwise(0).max().alias("horse_is_g1_performer")
    ])

    j_stats = df.filter(pl.col("date_text") >= "2021/01/01").group_by("jockey_name").agg([
        pl.col("odds").mean().alias("jockey_avg_odds"),
        (pl.col("finish_position") == 1).mean().alias("jockey_win_rate"),
        pl.len().alias("jockey_ride_count")
    ]).filter(pl.col("jockey_ride_count") > 5)

    all_time = df.group_by("horse_name").agg([
        pl.col("finish_position").mean().alias("horse_avg_finish_pos"),
        pl.col("popularity").mean().alias("horse_avg_popularity"),
        pl.col("odds").mean().alias("horse_avg_odds")
    ])
    
    prev_race = df.group_by("horse_name").head(1).select(["horse_name", "finish_position", "popularity", "odds", "jockey_name", "handicap"])
    prev_race = prev_race.rename({"finish_position": "prev_finish_pos", "popularity": "prev_popularity", "odds": "prev_odds", "jockey_name": "prev_jockey_name", "handicap": "prev_handicap"})
    
    recent_2y = df.filter((pl.col("date_text") >= "2023/01/01") & (pl.col("surface") == "芝"))
    speed_stats = recent_2y.with_columns(pl.col("distance").sub(1200).abs().alias("dist_diff")).sort(["horse_name", "dist_diff", "speed"], descending=[False, False, True]).group_by("horse_name").head(1).select(["horse_name", "speed"]).rename({"speed": "horse_recent_max_speed"})
    f600_stats = recent_2y.group_by("horse_name").agg(pl.col("final_600m_time").mean().alias("horse_recent_avg_final_600m"))

    nakayama = df.filter(pl.col("venue_code") == "06").group_by("horse_name").agg(pl.col("finish_position").mean().alias("horse_nakayama_avg_finish_pos"))
    dist_1200 = df.filter((pl.col("distance") == 1200) & (pl.col("surface") == "芝")).group_by("horse_name").agg(pl.col("finish_position").mean().alias("horse_dist_avg_finish_pos"))
    recent_3 = df.group_by("horse_name").head(3).group_by("horse_name").agg(pl.col("finish_position").mean().alias("horse_recent_3_avg_finish_pos"))

    h_combined = all_time.join(h_grade_stats, on="horse_name", how="left")
    h_combined = h_combined.join(prev_race, on="horse_name", how="left")
    h_combined = h_combined.join(speed_stats, on="horse_name", how="left")
    h_combined = h_combined.join(f600_stats, on="horse_name", how="left")
    h_combined = h_combined.join(nakayama, on="horse_name", how="left")
    h_combined = h_combined.join(dist_1200, on="horse_name", how="left")
    h_combined = h_combined.join(recent_3, on="horse_name", how="left")
    
    return j_stats, h_combined

def train_cycle_13():
    target_date = "2025/03/01"
    j_stats, h_stats = fetch_stats_v13(target_date)
    
    query = f"""
        SELECT r.horse_number, r.handicap, r.sex, r.age, r.odds, r.weight, r.weight_change,
               j.name as jockey_name, h.name as horse_name
        FROM keibadatabase_race r
        JOIN keibadatabase_jockey j ON r.jockey_id = j.id
        JOIN keibadatabase_horse h ON r.horse_id = h.id
        WHERE r.venue_code = '06' AND r.distance = 1200 AND r.surface = '芝'
        AND r.date_text < '{target_date}' AND r.odds IS NOT NULL
    """
    train_df = pl.read_database_uri(query, get_db_url())
    train_df = train_df.join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left")
    
    train_df = train_df.with_columns([
        (pl.col("handicap") - pl.col("prev_handicap")).fill_null(0.0).alias("handicap_diff"),
        pl.when(pl.col("jockey_name") == pl.col("prev_jockey_name")).then(0).otherwise(1).alias("jockey_changed")
    ])
    
    features = [
        "horse_number", "handicap", "age", "weight", "weight_change",
        "jockey_avg_odds", "jockey_win_rate", "horse_avg_finish_pos", "horse_avg_popularity", 
        "horse_avg_odds", "horse_recent_3_avg_finish_pos", "horse_nakayama_avg_finish_pos",
        "horse_dist_avg_finish_pos", "handicap_diff", "horse_recent_max_speed", 
        "horse_recent_avg_final_600m", "jockey_changed",
        "prev_finish_pos", "prev_popularity", "prev_odds",
        "horse_max_grade", "horse_is_g1_performer"
    ]
    
    for f in features:
        m = train_df.select(pl.col(f).mean()).item()
        train_df = train_df.with_columns(pl.col(f).fill_null(m if m is not None else 0))
    
    X = train_df[features].to_pandas()
    y_log_prob = np.log(0.8 / train_df["odds"].to_numpy())
    
    print("Training Final Cycle 13 model...")
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=42, verbosity=-1)
    model.fit(X, y_log_prob)
    
    today_weights = {
        1: (522, 16), 2: (516, 4), 3: (466, -2), 4: (476, 4), 5: (480, -10),
        6: (498, 8), 7: (492, -18), 8: (456, 0), 9: (492, -10), 10: (504, 6),
        11: (494, -2), 12: (472, -8), 13: (518, -10), 14: (478, 4), 15: (520, 2)
    }
    actual_odds = {1: 12.3, 2: 9.9, 3: 5.9, 4: 133.8, 5: 45.3, 6: 60.5, 7: 34.9, 8: 11.2, 9: 90.4, 10: 8.0, 11: 2.7, 12: 8.0, 13: 9.6, 14: 379.5, 15: 132.6}
    
    target_data = []
    with open("target_race_2025.txt", "r") as f:
        for line in f:
            p = line.strip().split(",")
            num = int(p[0])
            w, wc = today_weights[num]
            target_data.append({
                "horse_number": num, "horse_name": p[1], "age": int(p[2][1:]), "handicap": float(p[3]), 
                "jockey_name": p[4], "weight": w, "weight_change": wc, "actual_odds": actual_odds[num]
            })
    
    target_df = pl.DataFrame(target_data)
    target_df = target_df.join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left")
    target_df = target_df.with_columns([
        (pl.col("handicap") - pl.col("prev_handicap")).fill_null(0.0).alias("handicap_diff"),
        pl.when(pl.col("jockey_name") == pl.col("prev_jockey_name")).then(0).otherwise(1).alias("jockey_changed")
    ])
    
    for f in features:
        if f in ["horse_number", "handicap", "age", "weight", "weight_change", "handicap_diff", "jockey_changed"]: continue
        m = train_df.select(pl.col(f).mean()).item()
        target_df = target_df.with_columns(pl.col(f).fill_null(m if m is not None else 0))
    
    X_target = target_df[features].to_pandas()
    log_preds = model.predict(X_target)
    raw_probs = np.exp(log_preds)
    
    # Calibration
    calibrated_probs = np.power(raw_probs, 1.5)
    normalized_probs = (calibrated_probs / calibrated_probs.sum()) * 0.8
    predicted_odds = 0.8 / normalized_probs
    
    target_df = target_df.with_columns(pl.Series("predicted_odds", predicted_odds))
    print("Final Results (Cycle 13):")
    print(target_df.select(["horse_number", "horse_name", "predicted_odds", "actual_odds"]).sort("actual_odds"))
    
    rmse = np.sqrt(mean_squared_error(np.log(target_df["actual_odds"]), np.log(target_df["predicted_odds"])))
    print(f"Log-RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_cycle_13()
