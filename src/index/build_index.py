import faiss
import numpy as np

def create_faiss_index():
    print("1. Loading embeddings and IDs...")
    embeddings = np.load('data/raw/video_embeddings.npy')
    video_ids = np.load('data/raw/video_ids.npy')
    
    # Check the dimension (should be 32 based on our architecture)
    dimension = embeddings.shape[1]
    print(f"Loaded {embeddings.shape[0]} vectors of dimension {dimension}")

    print("2. Initializing the FAISS Index...")
    # IndexFlatIP calculates the exact Inner Product (Dot Product) it returning the videos with the highest scores
    base_index = faiss.IndexFlatIP(dimension)
    
    # By default, FAISS just assigns sequential IDs (0, 1, 2...). 
    # We want it to use our actual video_ids, so we wrap it in an IndexIDMap.
    index = faiss.IndexIDMap(base_index)
    
    print("3. Adding vectors to the index...")
    # FAISS expects IDs to be strictly 64-bit integers
    video_ids = video_ids.astype(np.int64)
    index.add_with_ids(embeddings, video_ids)
    
    print(f"Total vectors in index: {index.ntotal}")
    
    print("4. Saving the index to disk...")
    faiss.write_index(index, 'data/raw/video_index.faiss')
    print("Success: Index saved to data/raw/video_index.faiss")

if __name__ == "__main__":
    create_faiss_index()