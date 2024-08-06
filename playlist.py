import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json

# Load data
filtered_by_mood = pd.read_csv('filtered_by_mood.csv')

# Open and read JSON files
with open('last_detected_emotion.json', 'r') as f:
    data = json.load(f)
user_mood = data.get("emotion")

with open('selected_song.json', 'r') as f:
    data = json.load(f)
user_selected_song = data.get("song")

pca_columns = ['PC1', 'PC2', 'PC3', 'PC4']

top_recommendation_uri = None
top_3_recommendations = None

def on_song_selection(song):
    global top_recommendation_uri, top_3_recommendations

    if song in filtered_by_mood['song_name'].values:
        selected_song_features = filtered_by_mood.loc[filtered_by_mood['song_name'] == song, ['energy', 'valence']].values.flatten()
        all_songs_features = filtered_by_mood[['energy', 'valence']].values

        print("selected_song_features shape:", selected_song_features.shape)
        print("all_songs_features shape:", all_songs_features.shape)

        similarities = cosine_similarity([selected_song_features], all_songs_features)
        result_df = pd.DataFrame({
            'song_name': filtered_by_mood['song_name'],
            'similarity': similarities.flatten(),
            'uri': filtered_by_mood['uri']
        })
        result_df = result_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
        result_df = result_df[result_df['song_name'] != song]
        top_3_recommendations = result_df.head(3)

        print("Top 3 recommendations based on energy and valence:")
        print(top_3_recommendations[['song_name', 'similarity']])

        top_recommendation_uri = top_3_recommendations.iloc[0]['uri']
        print(f"URI of top recommendation: {top_recommendation_uri}")

    else:
        print("Selected song not found in filtered data.")

on_song_selection(user_selected_song)

# Spotify credentials
username = '9s2xr23gs18xgqj2i67demsbw'
client_id = 'fdc0136fdcf045f59ed2fe404dbdfbb2'
client_secret = 'fb0e39bd6a564dc79efafe96384ce34d'
redirect_uri = 'http://localhost:8888/callback'
scope = 'user-modify-playback-state user-read-playback-state playlist-modify-public'

# Spotify authentication
sp_oauth = SpotifyOAuth(client_id=client_id,
                        client_secret=client_secret,
                        redirect_uri=redirect_uri,
                        scope=scope,
                        username=username)

token_info = sp_oauth.get_access_token(as_dict=False)
sp = spotipy.Spotify(auth=token_info) if token_info else None

def play_song(uri):
    if sp:
        devices = sp.devices()
        if devices['devices']:
            device_id = devices['devices'][0]['id']
            sp.start_playback(device_id=device_id, uris=[uri])
        else:
            print("No active device found")
    else:
        print("Spotify client is not authenticated")

def create_playlist(user_id, name, description):
    if sp:
        playlist = sp.user_playlist_create(user_id, name, description=description)
        return playlist['id']
    return None

def add_songs_to_playlist(playlist_id, song_uris):
    if sp:
        sp.playlist_add_items(playlist_id, song_uris)

if top_recommendation_uri:
    play_song(top_recommendation_uri)

if top_3_recommendations is not None:
    user_id = sp.current_user()['id'] if sp else None
    if user_id:
        playlist_name = "Mood-based Playlist"
        playlist_description = f"Playlist based on detected mood: {user_mood}"
        playlist_id = create_playlist(user_id, playlist_name, playlist_description)
        if playlist_id:
            song_uris = top_3_recommendations['uri'].tolist()
            add_songs_to_playlist(playlist_id, song_uris)
            print(f"Songs added to playlist: {playlist_id}")
