"""
# Cycle 18: キャリブレーション係数(k)の最適化 (永続化対応版)
"""

import os
import django
import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.metrics import mean_squared_error
from train.common import load_all_stats, get_db_url

# Django初期化
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gemini_and_keiba.settings")
django.setup()

def run_optimization():
    race_ids = ["202506020111", "202406020311", "202306020311", "202206020311", "202106020311"]
    
    # 統計データのロード
    j_stats, h_stats, combo_stats = load_all_stats()
    
    all_raw_probs = []
    for rid in race_ids:
        print("Pre-calculating for:", rid)
        query_date = f"SELECT date_text FROM keibadatabase_race WHERE race_id = '{rid}' LIMIT 1"
        race_date = pl.read_database_uri(query_date, get_db_url())["date_text"][0]
        
        query_train = f"SELECT r.horse_number, r.handicap, r.sex, r.age, r.odds, r.weight, r.weight_change, j.name as jockey_name, h.name as horse_name FROM keibadatabase_race r JOIN keibadatabase_jockey j ON r.jockey_id = j.id JOIN keibadatabase_horse h ON r.horse_id = h.id WHERE r.venue_code = '06' AND r.distance = 1200 AND r.surface = '芝' AND r.date_text < '{race_date}' AND r.odds IS NOT NULL"
        train_df = pl.read_database_uri(query_train, get_db_url()).join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left").join(combo_stats, on=["horse_name", "jockey_name"], how="left")
        train_df = train_df.with_columns([(pl.col("handicap") - pl.col("prev_handicap")).fill_null(0.0).alias("handicap_diff"), pl.when(pl.col("jockey_name") == pl.col("prev_jockey_name")).then(0).otherwise(1).alias("jockey_changed")])
        
        features = ["horse_number", "handicap", "age", "weight", "weight_change", "jockey_avg_odds", "jockey_win_rate", "horse_avg_finish_pos", "horse_avg_popularity", "horse_avg_odds", "horse_recent_3_avg_finish_pos", "horse_nakayama_avg_finish_pos", "horse_dist_avg_finish_pos", "handicap_diff", "horse_recent_max_speed", "horse_recent_avg_final_600m", "jockey_changed", "prev_finish_pos", "prev_popularity", "prev_odds", "horse_max_grade", "horse_is_g1_performer", "combo_avg_finish_pos"]
        
        for f in features:
            m = train_df.select(pl.col(f).mean()).item()
            train_df = train_df.with_columns(pl.col(f).fill_null(m if m is not None else 0))
        
        X, y_log_prob = train_df[features].to_pandas(), np.log(0.8 / train_df["odds"].to_numpy())
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
        model.fit(X, y_log_prob)
        
        query_target = f"SELECT r.horse_number, r.handicap, r.sex, r.age, r.odds as actual_odds, r.weight, r.weight_change, j.name as jockey_name, h.name as horse_name FROM keibadatabase_race r JOIN keibadatabase_jockey j ON r.jockey_id = j.id JOIN keibadatabase_horse h ON r.horse_id = h.id WHERE r.race_id = '{rid}'"
        target_df = pl.read_database_uri(query_target, get_db_url()).join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left").join(combo_stats, on=["horse_name", "jockey_name"], how="left")
        target_df = target_df.with_columns([(pl.col("handicap") - pl.col("prev_handicap")).fill_null(0.0).alias("handicap_diff"), pl.when(pl.col("jockey_name") == pl.col("prev_jockey_name")).then(0).otherwise(1).alias("jockey_changed")])
        
        for f in features:
            if f in ["horse_number", "handicap", "age", "weight", "weight_change", "handicap_diff", "jockey_changed"]: continue
            m = train_df.select(pl.col(f).mean()).item()
            target_df = target_df.with_columns(pl.col(f).fill_null(m if m is not None else 0))
        
        X_target = target_df[features].to_pandas()
        all_raw_probs.append((target_df["actual_odds"].to_numpy(), np.exp(model.predict(X_target))))

    best_k, min_avg_rmse = 1.0, float("inf")
    for k in np.arange(1.0, 3.1, 0.1):
        total_rmse = 0
        for actual_odds, raw_probs in all_raw_probs:
            calibrated = np.power(raw_probs, k)
            norm_probs = (calibrated / calibrated.sum()) * 0.8
            total_rmse += np.sqrt(mean_squared_error(np.log(actual_odds), np.log(0.8 / norm_probs)))
        avg_rmse = total_rmse / len(all_raw_probs)
        if avg_rmse < min_avg_rmse:
            min_avg_rmse, best_k = avg_rmse, k
        print("k =", round(k, 1), "Avg RMSE =", round(avg_rmse, 4))
    print("Best k:", round(best_k, 1), "Min Avg RMSE:", round(min_avg_rmse, 4))

if __name__ == "__main__":
    run_optimization()
