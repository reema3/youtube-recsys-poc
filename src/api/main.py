from fastapi import FastAPI, HTTPException
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import sys
import os

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.two_tower import TwoTowerModel

app = FastAPI(title="Video Recommendation Engine")

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
    print(f"System Ready. Index contains {index.ntotal} videos.")
except Exception as e:
    print(f"Startup Error: {e}")
    sys.exit(1)


# ==========================================
# ENDPOINT: Generate Recommendations
# ==========================================
@app.get("/recommend")
def get_recommendations(user_id: int, age: float, k: int = 5):
    # Basic validation
    if user_id > 5000 or user_id < 1:
        raise HTTPException(status_code=400, detail="User ID must be between 1 and 5000")

    # 1. Convert inputs to PyTorch tensors (Shapes: [1] and [1, 1])
    user_tensor = torch.tensor([user_id], dtype=torch.long)
    age_tensor = torch.tensor([[age]], dtype=torch.float32)

    # 2. Push through the User Tower to get the 32D Vector
    with torch.no_grad():
        user_embed = model.user_embedding(user_tensor)
        age_lay = F.relu(model.age_layer(age_tensor))
        final_user_lay = torch.cat((user_embed, age_lay), dim=1)
        
        # The final 32-dimensional user representation
        user_vector = model.user_projection(final_user_lay)

    # 3. Query FAISS
    # Convert PyTorch tensor to numpy float32 array for FAISS
    user_vector_np = user_vector.numpy().astype(np.float32)
    distances, video_ids = index.search(user_vector_np, k)

    # 4. Format the JSON Response
    recommendations = []
    for vid_id, score in zip(video_ids[0], distances[0]):
        recommendations.append({
            "video_id": int(vid_id),
            "match_score": float(score)
        })

    return {
        "user_id": user_id,
        "age": age,
        "results": recommendations
    }