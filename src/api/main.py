from fastapi import FastAPI, HTTPException
import torch
import torch.nn.functional as F
import faiss
import pandas as pd
import numpy as np
import sys
import os

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.two_tower import TwoTowerModel

app = FastAPI(title="Video Recommendation Engine")

# ==========================================
# ONLINE FEATURE STORE (Simulating Redis/DynamoDB)
# ==========================================
# In a real system, this would be a connection to a Redis cluster: redis_client = redis.Redis(...)
online_kv_store = {}

def hydrate_online_store():
    """
    Simulates a batch pipeline pushing the latest user features 
    from the Data Lake into the low-latency Online Store.
    """
    print("Hydrating Online Feature Store...")
    # We grab the latest known age for each user from our raw data
    df = pd.read_csv('data/raw/historical_logs.csv')
    df = df.sort_values('timestamp')
    latest_features = df.groupby('user_id').last().reset_index()
    
    for _, row in latest_features.iterrows():
        # Storing data as Key (user_id) -> Value (Dictionary of features)
        online_kv_store[int(row['user_id'])] = {
            "age": float(row['age']),
            "last_watched_video_id": int(row['video_id'])
        }
    print(f"Loaded features for {len(online_kv_store)} users into the KV store.")

# ==========================================
# STARTUP: Load Models and Indexes into Memory
# ==========================================
print("Loading model and vector database...")
try:
    # 1. Initialize the model and load weights
    model = TwoTowerModel(num_users=5001, num_videos=10001)
    model.load_state_dict(torch.load('data/raw/two_tower_weights.pth', weights_only=True))
    model.eval() # Set to inference mode

    # 2. Load the FAISS index
    index = faiss.read_index('data/raw/video_index.faiss')

    # 3. Initialize our mock Redis database
    hydrate_online_store()

    print(f"System Ready. Index contains {index.ntotal} videos.")
except Exception as e:
    print(f"Startup Error: {e}")
    sys.exit(1)

# ==========================================
# STREAMING CONSUMER: Real-Time State Update
# ==========================================
def process_streaming_event(user_id: int, new_video_id: int):
    """
    Simulates a Kafka consumer listening to the 'Video_Clicked' topic.
    Updates the low-latency feature store in real-time without an ETL job.
    """
    user_state = online_kv_store.get(user_id)
    
    if user_state:
        # INSTANT OVERWRITE: The state is updated in memory immediately
        user_state["last_watched_video_id"] = new_video_id
        print(f"STREAM UPDATE: User {user_id} state updated to video {new_video_id}")
    else:
        # If user didn't exist, create a new record on the fly
        online_kv_store[user_id] = {
            "age": 25.0, # Default or inferred age
            "last_watched_video_id": new_video_id
        }
        print(f"STREAM UPDATE: New User {user_id} created with video {new_video_id}")

# ==========================================
# ENDPOINT: Generate Recommendations
# ==========================================
@app.get("/recommend")
def get_recommendations(user_id: int, k: int = 5):
    
    # 1. Look up user features in the Online Key-Value Store (Simulated Redis)
    user_features = online_kv_store.get(user_id)
    
    if not user_features:
        raise HTTPException(status_code=404, detail="User not found in Online Store")
        
    age = user_features["age"]
    last_watched = user_features["last_watched_video_id"]

    # 2. Convert inputs to PyTorch tensors (Shapes: [1] and [1, 1])
    user_tensor = torch.tensor([user_id], dtype=torch.long)
    age_tensor = torch.tensor([[age]], dtype=torch.float32)
    last_watched_tensor = torch.tensor([last_watched], dtype=torch.long)

    # 3. Push through the User Tower to get the 32D Vector
    with torch.no_grad():
        user_embed = model.user_embedding(user_tensor)
        age_lay = F.relu(model.age_layer(age_tensor))
        last_video_embed = model.last_watched_embedding(last_watched_tensor)
        final_user_lay = torch.cat((user_embed, age_lay, last_video_embed), dim=1)
        
        # The final 32-dimensional user representation
        user_vector = model.user_projection(final_user_lay)

    # 4. Query FAISS
    # Convert PyTorch tensor to numpy float32 array for FAISS
    user_vector_np = user_vector.numpy().astype(np.float32)
    distances, video_ids = index.search(user_vector_np, k)

    # 5. Format the JSON Response
    recommendations = []
    for vid_id, score in zip(video_ids[0], distances[0]):
        recommendations.append({
            "video_id": int(vid_id),
            "match_score": float(score)
        })

    return {
        "user_id": user_id,
        "features_used": {"age": age,
                          "last_watched_video_id": last_watched},

        "results": recommendations
    }