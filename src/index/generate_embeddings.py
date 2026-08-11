import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from src.model.two_tower import TwoTowerModel

def extract_video_embeddings():
    print("1. Reconstructing the video catalog...")
    # We didn't save a separate videos.csv, so we extract the unique videos 
    # and their lengths directly from the historical interactions.
    df = pd.read_csv('data/raw/historical_logs.csv')
    video_catalog = df[['video_id', 'video_length_seconds']].drop_duplicates().sort_values('video_id')
    
    # Convert to PyTorch tensors. 
    # .unsqueeze(1) reshapes the 1D length array into the [B, 1] shape the linear layer requires.
    video_ids = torch.tensor(video_catalog['video_id'].values, dtype=torch.long)
    video_lengths = torch.tensor(video_catalog['video_length_seconds'].values, dtype=torch.float32).unsqueeze(1)

    print("2. Loading the trained weights...")
    model = TwoTowerModel(num_users=5001, num_videos=10001)
    
    # Load the state dictionary (the learned weights) into the blueprint
    model.load_state_dict(torch.load('data/raw/two_tower_weights.pth'))
    
    # Freeze the model. This disables dropout and batch norm updates.
    model.eval() 

    print("3. Pushing data through the Video Tower...")
    # torch.no_grad() tells PyTorch to stop tracking gradients for backpropagation.
    # This strictly uses memory for inference, making it incredibly fast.
    with torch.no_grad():
        video_embed = model.video_embedding(video_ids)
        length_lay = F.relu(model.length_layer(video_lengths))
        final_video_lay = torch.cat((video_embed, length_lay), dim=1)
        
        # This is the final 32-dimensional output!
        video_vectors = model.video_projection(final_video_lay)

    print(f"Success: Extracted {video_vectors.shape[0]} embeddings of dimension {video_vectors.shape[1]}")

    print("4. Saving to disk...")
    # FAISS requires mathematical vectors to be in raw NumPy format, not PyTorch tensors.
    np.save('data/raw/video_ids.npy', video_catalog['video_id'].values)
    np.save('data/raw/video_embeddings.npy', video_vectors.numpy())
    print("Saved video_ids.npy and video_embeddings.npy to data/raw/")

if __name__ == "__main__":
    extract_video_embeddings()