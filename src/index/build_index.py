import faiss
import numpy as np

def create_faiss_index():
    print("1. Loading embeddings and IDs...")
    embeddings = np.load('data/raw/video_embeddings.npy')
    video_ids = np.load('data/raw/video_ids.npy')
    
    # Check the dimension (should be 32 based on our architecture)
    dimension = embeddings.shape[1]
    print(f"Loaded {embeddings.shape[0]} vectors of dimension {dimension}")

    print("2. Initializing the FAISS ANN Index...")
    # 1. Define how many clusters (nlist) to partition the data into
    nlist = 100 
    
    # The quantizer still uses exact search just to find the right cluster
    quantizer = faiss.IndexFlatIP(dimension) 
    
    # Explicitly tell the IVFFlat index to use Inner Product (Dot Product)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    print("3. Training the ANN clusters...")
    # ANN indexes MUST be trained on the data first to figure out where the clusters are
    index.train(embeddings)
    
    print("4. Adding vectors to the index...")
    # FAISS expects IDs to be strictly 64-bit integers
    video_ids = video_ids.astype(np.int64)
    
    # IndexIVFFlat supports add_with_ids natively! No IndexIDMap wrapper needed.
    index.add_with_ids(embeddings, video_ids)
    
    print(f"Total vectors in index: {index.ntotal}")
    
    print("5. Saving the index to disk...")
    faiss.write_index(index, 'data/raw/video_index_ann.faiss')
    print("Success: Index saved to data/raw/video_index_ann.faiss")

if __name__ == "__main__":
    create_faiss_index()