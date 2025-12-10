import os
import pickle
import xgboost as xgb

def model_fn(model_dir):
    """
    Deserializes the model. 
    SageMaker calls this immediately after the endpoint starts.
    """
    model_file = os.path.join(model_dir, "xgboost-model")
    print(f"Loading model from {model_file} using pickle...")
    
    with open(model_file, "rb") as f:
        model = pickle.load(f)
        
    return model
