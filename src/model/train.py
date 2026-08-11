# src/model/train.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from two_tower import TwoTowerModel

class VideoInteractionDataset(Dataset):
    def __init__(self, csv_file):
        # 1. Load the historical data into memory
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        # 2. Return the total number of interactions
        return len(self.data)

    def __getitem__(self, idx):
        # 3. Fetch exactly one row by its index
        row = self.data.iloc[idx]

        # 4. Convert Pandas columns to PyTorch Tensors
        # Categorical IDs must be integers (torch.long)
        user_id = torch.tensor(row['user_id'], dtype=torch.long)
        video_id = torch.tensor(row['video_id'], dtype=torch.long)

        # Continuous values must be floats (torch.float32).
        # Notice the brackets around row['age']? This forces the shape 
        # from a 0D scalar to a 1D vector of shape (1,), which becomes (B, 1) in the batch!
        age = torch.tensor([row['age']], dtype=torch.float32)
        video_length = torch.tensor([row['video_length_seconds']], dtype=torch.float32)

        # The target variable we want to predict
        clicked = torch.tensor(row['clicked'], dtype=torch.float32)

        return user_id, age, video_id, video_length, clicked
    
if __name__ == "__main__":
    
    print("Loading dataset...")
    dataset = VideoInteractionDataset('data/raw/historical_logs.csv')

    # Wrap it in a DataLoader
    # shuffle=True ensures the model doesn't memorize the chronological order of clicks
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    print(f"Total batches per epoch: {len(dataloader)}")

    # ------------------------------------------
    # 1. Initialize the Model, Loss, and Optimizer
    # ------------------------------------------
    # num_users and num_videos must be max_id + 1 because IDs start at 1, but PyTorch indexes at 0.
    model = TwoTowerModel(num_users=5001, num_videos=10001)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ------------------------------------------
    # 2. The Training Loop
    # ------------------------------------------
    epochs = 5
    print("\nStarting Training...")

    for epoch in range(epochs):
        model.train() # Put the model in training mode
        total_loss = 0
        
        for batch in dataloader:
            # Unpack the batch
            user_ids, ages, video_ids, video_lengths, clicks = batch
            
            # Step 1: Clear the gradients from the previous batch
            optimizer.zero_grad()
            
            # Step 2: Forward Pass (Make a prediction)
            predictions = model(user_ids, ages, video_ids, video_lengths)
            
            # Step 3: Calculate the Error (Loss)
            loss = criterion(predictions, clicks)
            
            # Step 4: Backward Pass (Calculate the gradients)
            loss.backward()
            
            # Step 5: Update the Weights
            optimizer.step()
            
            total_loss += loss.item()
            
        # Calculate and print the average loss for this epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

    # Save the trained weights to disk
    torch.save(model.state_dict(), "data/raw/two_tower_weights.pth")
    print("Model weights successfully saved to data/raw/two_tower_weights.pth")