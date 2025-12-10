import os
import argparse
import glob
import pandas as pd
import numpy as np



def get_cutoff_timestamp(file_paths):
    print("PASS 1: Calculating global time cutoff...")
    dates = []
    for i, path in enumerate(file_paths):
        try:
            df_chunk = pd.read_parquet(path)
            if 'session_start' not in df_chunk.columns:
                continue
            dates.append(pd.to_datetime(df_chunk['session_start']))
            del df_chunk
        except Exception as e:
            print(f"Skipping file {path}: {e}")

    if not dates:
        raise ValueError("No valid data found.")

    all_dates = pd.concat(dates, ignore_index=True)
    all_dates = all_dates.sort_values()
    cutoff_index = int(len(all_dates) * 0.8)
    cutoff_date = all_dates.iloc[cutoff_index]
    
    print(f"Global Time Cutoff found: {cutoff_date}")
    return cutoff_date

def process_file(path, output_dir, cutoff_date):
    try:
        filename = os.path.basename(path)
        df = pd.read_parquet(path)
        
        # Feature Engineering
        if "session_start" not in df.columns: return 
        
        df["session_start"] = pd.to_datetime(df["session_start"])
        df["session_hour"] = df["session_start"].dt.hour
        df["session_weekday"] = df["session_start"].dt.weekday
        df["is_weekend"] = df["session_weekday"].isin([5, 6]).astype(int)

        # Ratios
        df["cart_to_view_ratio"] = df["n_cart"] / df["n_views"].replace(0, 1)

        feature_cols = [
            "n_views", 
            "n_cart", 
            "n_unique_product", 
            "n_unique_category", 
            "session_hour", 
            "session_weekday", 
            "is_weekend",
            "cart_to_view_ratio"
        ]
        
        label_col = "did_purchase"
        
        if label_col not in df.columns: return

        # Split
        train_mask = df["session_start"] < cutoff_date
        
        # Select Columns
        final_cols = feature_cols + [label_col]
        df_final = df[final_cols]
        
        train_chunk = df_final[train_mask]
        test_chunk = df_final[~train_mask]

        # Write
        if not train_chunk.empty:
            train_chunk.to_csv(f"{output_dir}/train/train_{filename}.csv", index=False, header=False)
        if not test_chunk.empty:
            test_chunk.to_csv(f"{output_dir}/test/test_{filename}.csv", index=False, header=False)
            
    except Exception as e:
        print(f"Error processing {path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-data", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.input_data, "*"))
    input_files = [f for f in all_files if os.path.isfile(f) and not f.endswith(".py")]
    
    if not input_files:
        raise RuntimeError(f"No files found")

    os.makedirs(os.path.join(args.output_data, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_data, "test"), exist_ok=True)

    cutoff_date = get_cutoff_timestamp(input_files)
    
    print(f"Starting processing...")
    for i, f in enumerate(input_files):
        process_file(f, args.output_data, cutoff_date)

    print("Job Complete.")
