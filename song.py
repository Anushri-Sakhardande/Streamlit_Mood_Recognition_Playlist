import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import json

scaler = StandardScaler()

df = pd.read_csv('genres_v2.csv')

df_f=df.drop(['type','id','uri','track_href','analysis_url','duration_ms','time_signature','song_name'],axis=1)

num_features =df_f.select_dtypes(include=[np.number]).columns.tolist()
cat_features=df_f.select_dtypes(exclude=[np.number]).columns.tolist()

for feature in num_features:
    Q1 = df_f[feature].quantile(0.25)
    Q3 = df_f[feature].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df_f[feature] >= Q1 - 1.5 * IQR) & (df_f[feature] <= Q3 + 1.5 * IQR)
    df_f = df_f[filter]

spotify_data_scaled = df_f.copy()
spotify_data_scaled[num_features] = scaler.fit_transform(df_f[num_features])


label_encoders = {}
for feature in cat_features:
    le = LabelEncoder()
    df_f[feature] = le.fit_transform(df_f[feature])
    label_encoders[feature] = le

n_components = 4 
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(df_f)

pca_columns = [f'PC{i+1}' for i in range(n_components)]
df_pca = pd.DataFrame(data=principal_components, columns=pca_columns)

# Apply K-means clustering to the PCA-transformed data
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(principal_components)

df_pca['Cluster'] = clusters

cluster_moods = {}
for cluster in range(4):
    cluster_data = df_pca[df_pca['Cluster'] == cluster]
    avg_valence = cluster_data['PC1'].mean()
    avg_energy = cluster_data['PC2'].mean()
    
    if avg_valence > 0 and avg_energy > 0:
        cluster_moods[cluster] = 'Happy'
    elif avg_valence < 0 and avg_energy > 0:
        cluster_moods[cluster] = 'Angry'
    elif avg_valence < 0 and avg_energy < 0:
        cluster_moods[cluster] = 'Sad'
    else:
        cluster_moods[cluster] = 'Calm'

# Map moods to the DataFrame
df_pca['Mood'] = df_pca['Cluster'].map(cluster_moods)
df_pca['song_name'] = df['song_name']  # Ensure the song_name column is included
df_pca['uri'] = df['uri']  # Include uri if needed
df_pca = df_pca.join(df[[ 'energy', 'valence']])


# Define mood clusters
cluster_moods = {0: 'Happy', 1: 'Angry', 2: 'Sad', 3: 'Calm'}
df_pca['Mood'] = df_pca['Cluster'].map(cluster_moods)

# Open and read the JSON file
with open('last_detected_emotion.json', 'r') as f:
    data = json.load(f)

# Access the data
user_mood = data["emotion"] 

# Filter by mood
filtered_by_mood = df_pca[df_pca['Mood'] == user_mood]
filtered_by_mood.to_csv('filtered_by_mood.csv', index=False)


