import json
import pandas as pd

def load_candles_from_json(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["datetime_point"] = pd.to_datetime(df["datetime_point"])
    df = df.sort_values("datetime_point").reset_index(drop=True)
    return df
