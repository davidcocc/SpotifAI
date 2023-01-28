import sys
import spotipy
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy
import re
import matplotlib.pyplot as plt
import csv
import os
from spotipy import SpotifyOAuth
from pprint import pprint

SPOTIPY_CLIENT_ID = 'a05452f38db9485088e5744dc2b756e3'
SPOTIPY_CLIENT_SECRET = '0bf3ed7f0de144ce849350913a88dd65'
SPOTIPY_REDIRECT_URI = 'http://localhost:8080'
SCOPE = "user-read-playback-state,user-modify-playback-state,user-library-read,playlist-modify-private,playlist-modify-public"
CACHE = '.spotipyoauthcache'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = SPOTIPY_CLIENT_ID, client_secret = SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE, requests_timeout=10))
print("Login effettuato!")
mms = MinMaxScaler(feature_range=(0, 1))

def getPlaylistTracksID(pl_id):
    offset = 0
    tracksInPlaylist = []
    while True:
        response = sp.playlist_items(pl_id,
                                 offset=offset,
                                 fields='items.track.id')

        if len(response['items']) == 0:
            break

        list = response['items']
        for item in list:
            tracksInPlaylist.append(item["track"]["id"])
        offset = offset + len(response['items'])

    strTracks = ""
    for item in tracksInPlaylist:
        strTracks += item + ","
    strTracks = strTracks[:-1]

    return tracksInPlaylist

def normalizeDataset():
    df = pd.read_csv("musicData.csv")
    print("Rimuovo i duplicati.")
    df = df[(df.Playlist != 'u fucking nerd') & (df.Playlist != 'ðŸ†ðŸ…´ðŸ…°ðŸ…» ðŸ†ƒðŸ†ðŸ…°ðŸ…¿ðŸ…¿ðŸ…¸ðŸ…½') & (df['Artist Name'] != 'Brunori Sas') & (df['Artist Name'] != 'Giovanni Truppi') & (df['Artist Name'] != 'Rosa Chemical') & (df['Artist Name'] != 'Brunori Sas') & (df['Artist Name'] != 'bdrmm') & (df['Artist Name'] != 'Tauro Boys') & (df['Artist Name'] != 'Radical') & (df['Artist Name'] != 'cmqmartina') & (df['Artist Name'] != 'ç‰©èªžã') & (df['Artist Name'] != 'C418') & (df['Artist Name'] != 'ï¼’ï¼˜ï¼‘ï¼\”') & (df['Artist Name'] != 'FSK SATELLITE')]
    df.drop_duplicates('id', inplace=True)
    df.reset_index(inplace=True)
    print("Normalizzo il dataset.")
    df[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']] = mms.fit_transform(df[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']])
    df.to_csv('C:\\Users\\david\\Desktop\\progettoFIA\\script_noMode\\normalizedMusicData.csv', encoding='utf-8', index=False)
    print("Dataset normalizzato.")
    return df

def normalizeSong():
    df = pd.read_csv("musicData.csv")
    df[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']] = mms.fit_transform(df[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']])
    df2 = pd.read_csv("C:\\Users\\david\\Desktop\\progettoFIA\\script_noMode\\songFeatures.csv")
    print("Normalizzo il brano.")
    df2[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']] = mms.transform(df2[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']])
    df2.to_csv('C:\\Users\\david\\Desktop\\progettoFIA\\script_noMode\\normalizedSong.csv', encoding='utf-8', index=False)
    print("Brano normalizzato.")
    return df2

def getPlaylistURI():
    playlistLink = input("Inserisci il link di una playlist Spotify: ")

    if match := re.match(r"https://open.spotify.com/playlist/(.*)\?", playlistLink):
        playlist_uri = match.groups()[0]
        return playlist_uri
    else:
        raise ValueError("Formato: https://open.spotify.com/playlist/...")

def getTrackURI():
    track = input("Inserisci il link di un brano: ")

    if match := re.match(r"https://open.spotify.com/track/(.*)\?", track):
        track_uri = match.groups()[0]
        return track_uri
    else:
        raise ValueError("Formato: https://open.spotify.com/track/...")

def createDataset(playlists, sp,k):
    """
    Helper function
    :param playlists- playlists to draw tracks from
    :param sp - spotipy object to manage stuff
    :k - constant to determine how many songs I want to scrape from each playlist.
    100 songs per k
    :returns a song list along with their audio features
    """
    track_art_names = []
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            print("%4d %s" % (i + 1 + playlists['offset'], playlist['name']))
            for j in range(k):
                playlist_tracks = sp.playlist_items(playlist['id'], offset=j * 100)
                items = playlist_tracks['items']
                for item in items:
                    track = item['track']
                    try:
                        artist_name = track['artists'][0]['name']
                        artist_id = track['artists'][0]['id']
                        track_name = track['name']
                        orig_playlist = playlist['name']
                        track_art_names.append((artist_name, track_name, artist_id,orig_playlist))
                    except:
                        print("Playlist Error")
                        continue
        if playlists['next']:
            playlists = sp.next(playlists)['playlists']
        else:
            playlists = None

    ids = []
    song_list = []
    cols = []
    unique_tracks = list(set(track_art_names))
    for j, data_tup in enumerate(unique_tracks):
        try:  # Avoid stoppin
            track_feats = sp.search(q='artist:' + data_tup[0] + ' track:' + data_tup[1], type='track')
            track_id = track_feats['tracks']['items'][0]['id']
            track_pop = sp.track(track_id)['popularity']
            artist = sp.artist(data_tup[2])
            features = sp.audio_features(track_id)
            del features[0]['type']
        except:  # The track_feats search query returns an empty statement sometimes
            print("err")
            continue
        ids.append(track_id)
        artist_genre = artist['genres']
        if not cols:  # First iteration
            cols = ['Artist Name', 'Track Name', 'Popularity', 'Genres','Playlist']
            cols.extend(features[0].keys())
        song_row = [data_tup[0], data_tup[1], track_pop, artist_genre,data_tup[3]]
        try:  # somehow, some songs have no features
            song_row.extend(features[0].values())
        except:
            continue
        print("Artist = ", data_tup[0], " Song = ", data_tup[1], " Id = ", track_id, " Iter = ", j)
        song_list.append(song_row)

    return song_list, cols

def getSongsFeaturesDataset():
    df = pd.read_csv('C:\\Users\\david\\Desktop\\progettoFIA\\script_noMode\\normalizedMusicData.csv')
    features_df = df[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']]
    return features_df

def addCSVtoPlaylist(playlistID, filePath):
    user = sp.me()
    playlistDF = pd.read_csv(filePath)
    for i in range(len(playlistDF)):
        sp.playlist_add_items(playlistID, [playlistDF['uri'].loc[i]], position=None)

def getPlaylistFeatures(filePath, title):
    playlistDF = pd.read_csv(filePath)
    print("Danceability:")
    print(playlistDF['danceability'].mean())
    print("energy:")
    print(playlistDF['energy'].mean())
    print("valence:")
    print(playlistDF['valence'].mean())
    print("acousticness:")
    print(playlistDF['acousticness'].mean())
    print("speechiness:")
    print(playlistDF['speechiness'].mean())
    print("instrumentalness:")
    print(playlistDF['instrumentalness'].mean())
    print("liveness:")
    print(playlistDF['liveness'].mean())
    print("tempo:")
    print(playlistDF['tempo'].mean())
    features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']
    avgs = [playlistDF['danceability'].mean(), playlistDF['energy'].mean(), playlistDF['valence'].mean(), playlistDF['acousticness'].mean(), playlistDF['speechiness'].mean(), playlistDF['instrumentalness'].mean(), playlistDF['liveness'].mean(), playlistDF['tempo'].mean()]
    colors = ['red', 'limegreen', 'fuchsia', 'blue', 'orange', 'green', 'brown', 'yellow', 'pink']
    plt.bar(features, avgs, color = colors)
    plt.title("Features della " + title)
    plt.show()

def plotSong(song, trackName):
    features = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo']
    values = [song['danceability'].mean(), song['energy'].mean(), song['valence'].mean(), song['acousticness'].mean(), song['speechiness'].mean(), song['instrumentalness'].mean(), song['liveness'].mean(), song['tempo'].mean()]
    colors = ['red', 'limegreen', 'fuchsia', 'blue', 'orange', 'green', 'brown', 'yellow', 'pink']
    plt.bar(features, values, color = colors)
    plt.title("Features del brano " + trackName)
    plt.show()