"""
# Cycle 19: 2026年ターゲットレース 最終確定予測 (永続化対応版)
"""

import os
import django
import polars as pl
import lightgbm as lgb
import numpy as np
import pandas as pd
from django.conf import settings
from train.common import load_all_stats, get_db_url

# Django初期化
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gemini_and_keiba.settings")
django.setup()

def final_predict():
    # 統計データのロード (キャッシュがあればファイルから)
    j_stats, h_stats, combo_stats = load_all_stats()
    
    query_train = "SELECT r.horse_number, r.handicap, r.sex, r.age, r.odds, r.weight, r.weight_change, j.name as jockey_name, h.name as horse_name FROM keibadatabase_race r JOIN keibadatabase_jockey j ON r.jockey_id = j.id JOIN keibadatabase_horse h ON r.horse_id = h.id WHERE r.venue_code = '06' AND r.distance = 1200 AND r.surface = '芝' AND r.odds IS NOT NULL"
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
    
    # モデルの学習 (ここではシンプルに毎回学習するが、common.load_or_train_modelで永続化も可能)
    print("Training Model...")
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42, verbosity=-1)
    model.fit(X, y_log_prob)
    
    weights_2026 = {'ファンダム':(470,0), 'レイピア':(506,4), 'ペアポルックス':(472,0), 'ウイングレイテスト':(510,-2), 'ルガル':(532,0), 'カリボール':(512,14), 'フリームファクシ':(524,-2), 'フィオライア':(472,8), 'インビンシブルパパ':(512,-2), 'ピューロマジック':(462,16), 'ルージュラナキラ':(458,0), 'オタルエバー':(502,2), 'ビッグシーザー':(520,2), 'ママコチャ':(500,2), 'フリッカージャブ':(494,4), 'ヨシノイースター':(496,0)}
    target_data = []
    with open("target_race.txt", "r") as f:
        for line in f:
            p = line.strip().split(",")
            name = p[1]
            w, wc = weights_2026.get(name, (490, 0))
            target_data.append({"horse_number": int(p[0]), "horse_name": name, "age": int(p[2][1:]), "handicap": float(p[3]), "jockey_name": p[4], "weight": w, "weight_change": wc})
    
    target_df = pl.DataFrame(target_data).join(j_stats, on="jockey_name", how="left").join(h_stats, on="horse_name", how="left").join(combo_stats, on=["horse_name", "jockey_name"], how="left")
    target_df = target_df.with_columns([(pl.col("handicap") - pl.col("prev_handicap")).fill_null(0.0).alias("handicap_diff"), pl.when(pl.col("jockey_name") == pl.col("prev_jockey_name")).then(0).otherwise(1).alias("jockey_changed")])
    
    for f in features:
        if f in ["horse_number", "handicap", "age", "weight", "weight_change", "handicap_diff", "jockey_changed"]: continue
        m = train_df.select(pl.col(f).mean()).item()
        target_df = target_df.with_columns(pl.col(f).fill_null(m if m is not None else 0))
    
    X_target = target_df[features].to_pandas()
    log_preds = model.predict(X_target)
    raw_probs = np.exp(log_preds)
    calibrated = np.power(raw_probs, 1.2)
    norm_probs = (calibrated / calibrated.sum()) * 0.8
    pred_odds = np.maximum(0.8 / norm_probs, 1.0)
    target_df = target_df.with_columns(pl.Series("predicted_odds", pred_odds))
    
    print("--- 2026 Final Optimized Results (k=1.2) ---")
    with pl.Config(tbl_rows=20):
        print(target_df.select(["horse_number", "horse_name", "jockey_name", "predicted_odds"]).sort("predicted_odds"))

if __name__ == "__main__":
    final_predict()
