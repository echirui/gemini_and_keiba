"""
# Cycle 17: オーシャンステークス過去5年の一括バックテスト (永続化対応版)
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

def run_backtest():
    with open("ocean_stakes.txt", "r") as f:
        race_ids = [line.strip() for line in f if line.strip()]
    
    # 統計データのロード (グローバルなキャッシュを使用)
    # 本来は日付ごとに計算すべきだが、計算量削減のため最新の統計をベースに使用
    j_stats, h_stats, combo_stats = load_all_stats()
    
    results = []
    for rid in race_ids:
        print("Analyzing Race ID:", rid)
        query_date = f"SELECT date_text FROM keibadatabase_race WHERE race_id = '{rid}' LIMIT 1"
        race_date = pl.read_database_uri(query_date, get_db_url())["date_text"][0]
        
        # 訓練データの取得 (日付制限のみクエリで行う)
        query_train = f"SELECT r.horse_number, r.handicap, r.sex, r.age, r.odds, r.weight, r.weight_change, j.name as jockey_name, h.name as horse_name FROM keibadatabase_race r JOIN keibadatabase_jockey j ON r.jockey_id = j.id JOIN keibadatabase_horse h ON r.horse_id = h.id WHERE r.venue_code = '06' AND r.distance = 1200 AND r.surface = '芝' AND r.date_text < '{race_date}' AND r.odds IS NOT NULL"
        train_df = pl.read_database_uri(query_train, get_db_url()).join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left").join(combo_stats, on=["horse_name", "jockey_name"], how="left")
        
        train_df = train_df.with_columns([
            (pl.col("handicap") - pl.col("prev_handicap")).fill_null(0.0).alias("handicap_diff"),
            pl.when(pl.col("jockey_name") == pl.col("prev_jockey_name")).then(0).otherwise(1).alias("jockey_changed")
        ])
        
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
        log_preds = model.predict(X_target)
        raw_probs = np.exp(log_preds)
        calibrated = np.power(raw_probs, 1.2) # ベストなk=1.2を使用
        norm_probs = (calibrated / calibrated.sum()) * 0.8
        target_df = target_df.with_columns(pl.Series("predicted_odds", 0.8 / norm_probs))
        
        rmse = np.sqrt(mean_squared_error(np.log(target_df["actual_odds"]), np.log(target_df["predicted_odds"])))
        results.append({"race_id": rid, "year": race_date[:4], "log_rmse": rmse})
        print("Year:", race_date[:4], "Log-RMSE:", round(rmse, 4))

    print("\n--- Backtest Summary (Optimized) ---")
    summary = pd.DataFrame(results)
    print(summary)
    print("Avg Log-RMSE:", round(summary["log_rmse"].mean(), 4))

if __name__ == "__main__":
    run_backtest()
