# train.py
import pandas as pd
import numpy as np
import os
import time
import json

from utils.data_loader import (
    load_and_preprocess_data,
    get_surprise_data,
    create_tf_train_test_split,
    PROCESSED_DATA_DIR
)
from models.matrix_factorization import MatrixFactorizationModel
from models.deep_cross_network import DeepCrossNetworkModel


MAX_RATINGS_TO_PROCESS = 5_000_000 # Example: 5 million ratings

# SVD Hyperparameters (Reduced Complexity)
SVD_PARAMS = {
    'n_factors': 50,       # Reduced from 150
    'n_epochs': 15,        # Reduced from 25
    'lr_all': 0.005,       # Keep learning rate reasonable
    'reg_all': 0.02,       # Keep regularization reasonable
    'random_state': 42,
    'biased': True
}
# DCN Hyperparameters (Reduced Complexity)
DCN_PARAMS = {
    'embedding_dim': 16,   # Reduced from 64
    'cross_layers': 2,     # Reduced from 3
    'deep_layers': [32, 16], # Reduced from [128, 64]
    'dropout_rate': 0.1    # Can keep dropout lower with smaller network
}
# DCN Training Parameters (Reduced Batch Size)
DCN_TRAIN_PARAMS = {
    'epochs': 10,          # Reduced from 15
    'batch_size': 512,     # **** Drastically reduced from 2048 ****
    'learning_rate': 0.001
}

# Data splitting for DCN training/validation
TF_VALIDATION_SIZE = 0.15 # Use slightly more for validation if training data is smaller

# Filtering parameters for load_and_preprocess_data
# Adjust these based on MAX_RATINGS_TO_PROCESS. If limiting rows, filtering might be less crucial.
MIN_USER_RATINGS = 10
MIN_MOVIE_RATINGS = 10


# --- Main Training Function ---
def main():
    print("--- Starting Training Pipeline (Low Memory Adjustments) ---")
    # ... (start timer) ...
    pipeline_start_time = time.time()

    # === Phase 1: Data Loading and Preprocessing ===
    print("\n[Phase 1/4] Loading and Preprocessing Data...")
    phase_start_time = time.time()
    try:
        print(f"*** Applying row limit: {MAX_RATINGS_TO_PROCESS if MAX_RATINGS_TO_PROCESS else 'None (Full Dataset)'} ***")
        movies_df, ratings_df, user_map, item_map, inverse_item_map = load_and_preprocess_data(
            min_ratings_user=MIN_USER_RATINGS,
            min_ratings_movie=MIN_MOVIE_RATINGS,
            max_ratings_rows=MAX_RATINGS_TO_PROCESS # Pass the limit
        )
        # ... (rest of phase 1 logging) ...
        num_users = len(user_map)
        num_items = len(item_map)
        print(f"Data loaded/processed: {num_users} users, {num_items} items.")
        print(f"Movies DataFrame shape: {movies_df.shape}")
        print(f"Ratings DataFrame shape: {ratings_df.shape}")

    # ... (rest of phase 1 error handling) ...
    except FileNotFoundError as e: print(f"Error: Raw data file not found. {e}"); return
    except ValueError as e: print(f"Error during data processing: {e}"); return
    except Exception as e: print(f"An unexpected error occurred: {e}"); import traceback; traceback.print_exc(); return
    phase_end_time = time.time(); print(f"[Phase 1 Complete] Time: {phase_end_time - phase_start_time:.2f}s.")


    # === Phase 2: Train Matrix Factorization (SVD) Model ===
    print("\n[Phase 2/4] Training Matrix Factorization (SVD)...")
    phase_start_time = time.time()
    try:
        print("Preparing Surprise data...")
        surprise_data = get_surprise_data(ratings_df)
        print(f"Training SVD with params: {SVD_PARAMS}")
        mf_model = MatrixFactorizationModel(**SVD_PARAMS)
        mf_model.fit(surprise_data=surprise_data)
        mf_model.save()
    # ... (rest of phase 2 error handling) ...
    except Exception as e: print(f"An error occurred during SVD: {e}"); import traceback; traceback.print_exc(); print("Continuing...")
    phase_end_time = time.time(); print(f"[Phase 2 Complete] Time: {phase_end_time - phase_start_time:.2f}s.")


    # === Phase 3: Train Deep Cross Network (DCN) Model ===
    print("\n[Phase 3/4] Training Deep Cross Network (DCN)...")
    phase_start_time = time.time()
    try:
        print(f"Creating train/validation split (Validation size: {TF_VALIDATION_SIZE:.0%})...")
        train_df, val_df = create_tf_train_test_split(ratings_df, test_size=TF_VALIDATION_SIZE, random_state=42)
        print(f"TF Training data size: {len(train_df)}")
        print(f"TF Validation data size: {len(val_df)}")
        if train_df.empty or val_df.empty: raise ValueError("Train or validation DataFrame empty.")

        print(f"Initializing DCN with params: {DCN_PARAMS}")
        dcn_model = DeepCrossNetworkModel(num_users=num_users, num_items=num_items, **DCN_PARAMS)

        print(f"Training DCN with params: {DCN_TRAIN_PARAMS}")
        dcn_model.fit(
            train_df, val_df=val_df, user_map=user_map, item_map=item_map,
            inverse_item_map=inverse_item_map, **DCN_TRAIN_PARAMS
        )
        dcn_model.save()
    # ... (rest of phase 3 error handling) ...
    except ValueError as e: print(f"ValueError during DCN setup/training: {e}"); import traceback; traceback.print_exc()
    except Exception as e: print(f"An unexpected error occurred during DCN: {e}"); import traceback; traceback.print_exc()
    phase_end_time = time.time(); print(f"[Phase 3 Complete] Time: {phase_end_time - phase_start_time:.2f}s.")


    # === Phase 4: Pipeline Completion ===
    # ... (rest of phase 4 logging) ...
    pipeline_end_time = time.time()
    print("\n[Phase 4/4] Training Pipeline Finished.")
    print(f"Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds.")
    # ...

if __name__ == "__main__":
    # Optional: Add basic Tensorflow memory configuration
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Enabled memory growth for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        else:
            print("No GPU found, running on CPU.")
        
    except ImportError:
        print("TensorFlow not found, skipping GPU configuration.")

    main()