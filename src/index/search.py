import faiss
import numpy as np

def test_faiss_search():
    print("1. Loading the FAISS index...")
    # Read the pre-built index from disk
    index = faiss.read_index('data/raw/video_index.faiss')
    
    print(f"Index loaded successfully. Total videos in database: {index.ntotal}")

    print("\n2. Simulating a User Request...")
    # FAISS strictly requires a 2D numpy array of type float32 for searching.
    # We are simulating 1 user with a 32-dimensional embedding: shape (1, 32).
    # (In production, this vector would come from the User Tower of your PyTorch model).
    dummy_user_vector = np.random.rand(1, 32).astype(np.float32)

    print("\n3. Querying the Vector Database...")
    # k is the number of top recommendations we want to retrieve
    k = 5 
    
    # The search function returns two arrays:
    # 1. distances: The actual Dot Product scores
    # 2. indices: The mapped Video IDs
    distances, video_ids = index.search(dummy_user_vector, k)

    print("\n=================================")
    print("        TOP 5 RECOMMENDATIONS      ")
    print("=================================")
    
    # We grab index 0 because we only queried 1 user
    for rank, (vid_id, score) in enumerate(zip(video_ids[0], distances[0])):
        print(f"Rank {rank + 1} | Video ID: {vid_id:<6} | Dot Product Score: {score:.4f}")

if __name__ == "__main__":
    test_faiss_search()