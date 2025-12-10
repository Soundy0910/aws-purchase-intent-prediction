
import argparse
import os
import glob
import pandas as pd
import xgboost as xgb
import pickle
import gc

def load_dataset(path):
    print(f"Loading CSV files from {path}...")
    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {path}")
    
    dfs = []
    for f in files:
        # Optimization: Use float32 to save RAM
        dfs.append(pd.read_csv(f, header=None, dtype='float32'))
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Label is the LAST column
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--num_round', type=int, default=50)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    
    args = parser.parse_args()

    # 1. Load Data
    print("Loading Train Data...")
    X_train, y_train = load_dataset(args.train)
    
    print("Loading Test Data...")
    X_test, y_test = load_dataset(args.test)
    
    # 2. Convert to DMatrix (Memory Intensive Step)
    print("Converting to DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Free RAM immediately
    del X_train, y_train, X_test, y_test
    gc.collect()
    
    # 3. Train
    print("Starting Training...")
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'objective': args.objective,
        'eval_metric': 'auc',
        'verbosity': 1
    }
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=args.num_round,
        evals=[(dtest, "test")]
    )
    
    # 4. Save
    model_path = os.path.join(args.model_dir, "xgboost-model")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Training Complete. Model saved.")
