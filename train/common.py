import os
import polars as pl
import lightgbm as lgb
import numpy as np
from django.conf import settings

# キャッシュディレクトリ
CACHE_DIR = "data/processed"
MODEL_DIR = "models"

def get_db_url():
    db_config = settings.DATABASES["app_db"]
    return f"postgresql://{db_config['USER']}:{db_config['PASSWORD']}@{db_config['HOST']}:{db_config['PORT']}/{db_config['NAME']}"

def time_to_seconds(t_str):
    try:
        if not t_str or ":" not in str(t_str): return float(t_str) if t_str else None
        m, s = str(t_str).split(":")
        return float(m) * 60 + float(s)
    except: return None

def get_grade(name, cls):
    s = f"{name} {cls}"
    if "G1" in s or "GI " in s or " GI" in s: return 4
    if "G2" in s or "GII" in s: return 3
    if "G3" in s or "GIII" in s: return 2
    if "オープン" in s or "リステッド" in s or "(L)" in s: return 1
    return 0

def fetch_all_stats_from_db():
    print("Fetching comprehensive stats from database (this may take a while)...")
    query = """
        SELECT h.name as horse_name, j.name as jockey_name, r.finish_position, r.popularity, r.odds, 
               r.date_text, r.venue_code, r.handicap, r.distance, r.surface, r.track_condition,
               r.running_time, r.final_600m_time, r.race_name, r.race_class, r.weight, r.weight_change
        FROM keibadatabase_race r
        JOIN keibadatabase_horse h ON r.horse_id = h.id
        JOIN keibadatabase_jockey j ON r.jockey_id = j.id
        WHERE r.odds IS NOT NULL
    """
    df = pl.read_database_uri(query, get_db_url()).sort(["horse_name", "date_text"], descending=[False, True])
    
    df = df.with_columns([
        pl.col("running_time").map_elements(time_to_seconds, return_dtype=pl.Float64).alias("time_secs"),
        pl.struct(["race_name", "race_class"]).map_elements(lambda x: get_grade(x["race_name"], x["race_class"]), return_dtype=pl.Int32).alias("race_grade")
    ])
    df = df.with_columns([(pl.col("distance") / pl.col("time_secs")).alias("speed")])

    h_grade = df.group_by("horse_name").agg([
        pl.col("race_grade").max().alias("horse_max_grade"),
        pl.when((pl.col("race_grade") == 4) & (pl.col("finish_position") <= 3)).then(1).otherwise(0).max().alias("horse_is_g1_performer")
    ])

    j_stats = df.filter(pl.col("date_text") >= "2022/01/01").group_by("jockey_name").agg([
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
    
    recent_2y = df.filter((pl.col("date_text") >= "2024/01/01") & (pl.col("surface") == "芝"))
    speed_stats = recent_2y.with_columns(pl.col("distance").sub(1200).abs().alias("dist_diff")).sort(["horse_name", "dist_diff", "speed"], descending=[False, False, True]).group_by("horse_name").head(1).select(["horse_name", "speed"]).rename({"speed": "horse_recent_max_speed"})
    f600_stats = recent_2y.group_by("horse_name").agg(pl.col("final_600m_time").mean().alias("horse_recent_avg_final_600m"))

    nakayama = df.filter(pl.col("venue_code") == "06").group_by("horse_name").agg(pl.col("finish_position").mean().alias("horse_nakayama_avg_finish_pos"))
    dist_1200 = df.filter((pl.col("distance") == 1200) & (pl.col("surface") == "芝")).group_by("horse_name").agg(pl.col("finish_position").mean().alias("horse_dist_avg_finish_pos"))
    recent_3 = df.group_by("horse_name").head(3).group_by("horse_name").agg(pl.col("finish_position").mean().alias("horse_recent_3_avg_finish_pos"))

    h_combined = all_time.join(h_grade, on="horse_name", how="left") 
                         .join(prev_race, on="horse_name", how="left") 
                         .join(speed_stats, on="horse_name", how="left") 
                         .join(f600_stats, on="horse_name", how="left") 
                         .join(nakayama, on="horse_name", how="left") 
                         .join(dist_1200, on="horse_name", how="left") 
                         .join(recent_3, on="horse_name", how="left")
    
    combo_stats = df.group_by(["horse_name", "jockey_name"]).agg([pl.col("finish_position").mean().alias("combo_avg_finish_pos")])
    
    return j_stats, h_combined, combo_stats

def load_all_stats(force_refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    j_path = os.path.join(CACHE_DIR, "j_stats_global.parquet")
    h_path = os.path.join(CACHE_DIR, "h_stats_global.parquet")
    c_path = os.path.join(CACHE_DIR, "c_stats_global.parquet")
    
    if not force_refresh and all(os.path.exists(p) for p in [j_path, h_path, c_path]):
        print("Loading cached statistics...")
        return pl.read_parquet(j_path), pl.read_parquet(h_path), pl.read_parquet(c_path)
    
    j_stats, h_stats, combo_stats = fetch_all_stats_from_db()
    j_stats.write_parquet(j_path)
    h_stats.write_parquet(h_path)
    combo_stats.write_parquet(c_path)
    return j_stats, h_stats, combo_stats

def load_or_train_model(model_name, train_func, features, force_train=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.txt")
    
    if not force_train and os.path.exists(model_path):
        print(f"Loading cached model: {model_name}")
        return lgb.Booster(model_file=model_path)
    
    print(f"Training new model: {model_name}")
    model = train_func()
    model.booster_.save_model(model_path)
    return model.booster_
