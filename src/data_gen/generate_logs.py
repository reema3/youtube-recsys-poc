import numpy as np
import pandas as pd

#Creating a User pool of 5000
#normally distrubuted age with mean of 30 and std. dev of 8
#means we are saying 68% population will be between 22 and 38 
#and 95% between 14 and 46
user_pool_size = 5000
user_age_mean = 30.0
user_age_std_dev = 8.0
video_categories = ['Music','News','Travel','Comedy','Sports','Education']

user_df = pd.DataFrame({'user_id': np.arange(1, user_pool_size + 1)})
rng = np.random.default_rng()
user_df['age'] = rng.normal(loc=user_age_mean,scale=user_age_std_dev,size=user_pool_size).astype(int) 
user_df['preferred_category'] = np.random.choice(video_categories,size=user_pool_size)

print("User Pool:")
print(user_df.head())
print("-" * 30)

#Creating a Video pool
#on an avergae youtube videos are 3 to 10min long
#we have taken mean as 5 min meaning 300s whose natural log is 5.7 so rounding to 5
video_pool_size = 10000

videos_df = pd.DataFrame({'video_id': np.arange(1, video_pool_size + 1)})
videos_df['video_length_seconds'] = np.random.lognormal(mean=5.0, sigma=1.0, size = video_pool_size).astype(int)
videos_df['category'] = np.random.choice(video_categories,size=video_pool_size)

print("Video Pool:")
print(videos_df.head())
print("-" * 30)

#Creating intereaction df
interaction_count = 100000

#Dataset doesn't represent a unique combination of a user and a video;
#It represents an impression (the moment the algorithm showed the video on the user's screen).
#Randomly sample Users and Videos with replacement
interaction_df = pd.DataFrame(
    {
        'user_id': np.random.choice(user_df['user_id'],size=interaction_count),
        'video_id': np.random.choice(videos_df['video_id'],size=interaction_count)
    }
)

#By setting periods=interaction_count * 2 (creating 200,000 available timestamp slots) and then 
#using np.random.choice to randomly grab 100,000 of them, we leave gaps.
#It randomly skips some timestamps and picks others, creating a much more natural, uneven distribution of events.
end_date = pd.Timestamp.now()
start_date = end_date - pd.Timedelta(days=180)
timestamps = pd.date_range(start=start_date,end=end_date, periods=interaction_count*2)
interaction_df['timestamp'] = np.random.choice(timestamps,size =interaction_count )

# Extract the 'is_weekend' feature from the timestamp
# dt.dayofweek returns 0-6 (Monday=0, Sunday=6). So >= 5 is Saturday/Sunday.
interaction_df['is_weekend'] = interaction_df['timestamp'].dt.dayofweek >= 5

# Sort by timestamp to simulate chronological data collection
interaction_df = interaction_df.sort_values('timestamp').reset_index(drop=True)

#Merge the context (Bring in User and Video features)
interaction_df = interaction_df.merge(user_df,on='user_id', how='left')
interaction_df = interaction_df.merge(videos_df,on='video_id', how ='left')

# Setting the base probability to 10% (0.1) mimics this reality. Most impressions result in a "skip" (0).
# By setting matched probability to 0.75, we create a massive mathematical "signal." The model will quickly realize, "Ah, when these categories align, clicks explode!"
# Adding rule like Users under 25 have an artificially high probability of clicking "Comedy" videos.
# If a video is in the "News" category, give it a high probability of being clicked on weekends.
conditions = [
    (interaction_df['preferred_category'] == interaction_df['category']),
    (interaction_df['category'] == 'News') & (interaction_df['is_weekend'] == True),
    (interaction_df['age'] <= 25) & (interaction_df['category'] == 'Comedy')
]

choices = [0.75, 0.6, 0.6]

interaction_df['probability'] = np.select(conditions, choices, default=0.1)

interaction_df['clicked'] = np.random.uniform(0,1,size=interaction_count)
interaction_df['clicked'] = np.where(interaction_df['probability']>=interaction_df['clicked'],1,0).astype(int)

interaction_df.drop(columns=['probability'],inplace=True)

print("Interaction Logs:")
print(interaction_df.head())
print(f"{interaction_df['clicked'].sum()} clicked out of {interaction_count} interactions")
print("-" * 30)

# --- Splitting the Data ---

# We take the first 90,000 rows as our historical batch data.
# This will be used to train our PyTorch model and feed the Feast Offline Store.
batch_df = interaction_df.iloc[:90000]

# We take the last 10,000 rows as our "live" streaming data.
# We will use this in Phase 4 to simulate real-time traffic hitting our Kafka queue.
streaming_df = interaction_df.iloc[90000:]

# Save them to the data folder you created earlier
import os
os.makedirs('data/raw', exist_ok=True)

batch_df.to_csv('data/raw/historical_logs.csv', index=False)
streaming_df.to_csv('data/raw/streaming_logs.csv', index=False)

print("\nData successfully split and saved to data/raw/")
