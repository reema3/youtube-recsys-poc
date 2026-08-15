import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_videos, embed_dim=32):
        super().__init__()

        # ==========================================
        # TOWER 1: THE USER TOWER
        # ==========================================
        # 1. The ID Lookup Table (B*32)
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        
        # 2. The Continuous Feature Processor (Takes 1 number, expands to 16)
        self.age_layer = nn.Linear(in_features=1, out_features=16)
        self.last_watched_embedding = nn.Embedding(num_videos, embed_dim)
        
        # 3. The Squeezer (embed_dim + 16 -> embed_dim)
        self.user_projection = nn.Linear(in_features=2*embed_dim + 16, out_features=embed_dim)


        # ==========================================
        # TOWER 2: THE VIDEO TOWER
        # ==========================================
        self.video_embedding = nn.Embedding(num_embeddings=num_videos, embedding_dim=embed_dim)
        self.length_layer = nn.Linear(in_features=1, out_features=16)
        self.video_projection = nn.Linear(in_features=embed_dim + 16, out_features=embed_dim)


    def forward(self, user_ids, ages,last_watched_video_id, video_ids, video_lengths):
        # ------------------------------------------
        # STEP 1: Process User Data (B= Batch Size)
        # ------------------------------------------
        user_embed = self.user_embedding(user_ids) #B*32
        age_lay =  self.age_layer(ages) #B*16
        age_lay = F.relu(age_lay) #B*16
        last_video_embed = self.last_watched_embedding(last_watched_video_id) #B*32
        final_user_lay =  torch.cat((user_embed,age_lay), dim=1) #B*32,B*16 ->B*48
        final_user_lay =  torch.cat((final_user_lay,last_video_embed), dim=1) #B*48,B*32 ->B*80
        user_vec = self.user_projection(final_user_lay) #B*80->B*32

        # ------------------------------------------
        # STEP 2: Process Video Data
        # ------------------------------------------
        video_embed = self.video_embedding(video_ids)
        length_lay =  self.length_layer(video_lengths) 
        length_lay = F.relu(length_lay)
        final_video_lay =  torch.cat((video_embed,length_lay),dim=1)
        video_vec = self.video_projection(final_video_lay) #B*32

        # ------------------------------------------
        # STEP 3: Compute Similarity (Dot Product)
        # ------------------------------------------
        # Element-wise multiplication, then sum across dim=1
        similar = torch.sum(user_vec * video_vec, dim=1) #[B]
        
        return similar