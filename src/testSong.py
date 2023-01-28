import spotipy
from spotipy import SpotifyOAuth
from utils import getTrackURI, normalizeSong, plotSong
import numpy as np
import pandas as pd

SPOTIPY_CLIENT_ID = 'il_tuo_client_ID'
SPOTIPY_CLIENT_SECRET = 'il_tuo_codice_segreto'
SPOTIPY_REDIRECT_URI = 'http://localhost:8080'
SCOPE = "user-read-playback-state,user-modify-playback-state,ugc-image-upload,playlist-modify-private,playlist-modify-public"
CACHE = '.spotipyoauthcache'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = SPOTIPY_CLIENT_ID, client_secret = SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE, requests_timeout=10))

##Stampa e plot delle features di un brano
def testTrack():
    track_URI = getTrackURI()
    track_id = "spotify:track:" + track_URI
    featureDictionary = sp.audio_features(track_id)
    ##Salvataggio delle features di un brano in .csv
    trackID = [featureDictionary[0]['id']]
    track = sp.track(trackID[0])
    trackName = track["name"]
    songFeatures = [featureDictionary[0]['danceability'], featureDictionary[0]['energy'], featureDictionary[0]['valence'], featureDictionary[0]['acousticness'], featureDictionary[0]['speechiness'], featureDictionary[0]['instrumentalness'], featureDictionary[0]['liveness'], featureDictionary[0]['tempo']]
    df = pd.DataFrame([songFeatures], columns=['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness', 'tempo'])
    df.to_csv('C:\\Users\\david\\Desktop\\progettoFIA\\script_noMode\\songFeatures.csv', encoding='utf-8', index=False)
    normalizedSong = normalizeSong()
    plotSong(normalizedSong, trackName)
    return