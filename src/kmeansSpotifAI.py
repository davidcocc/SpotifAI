import spotipy
from spotipy import SpotifyOAuth
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from utils import normalizeDataset, getSongsFeaturesDataset, addCSVtoPlaylist, getPlaylistFeatures
from testSong import testTrack
import numpy as np
import pandas as pd

##richiede Spotify Premium!
SPOTIPY_CLIENT_ID = 'il_tuo_client_ID'
SPOTIPY_CLIENT_SECRET = 'il_tuo_codice_segreto'
SPOTIPY_REDIRECT_URI = 'http://localhost:8080'
SCOPE = "user-read-playback-state,user-modify-playback-state,ugc-image-upload,playlist-modify-private,playlist-modify-public"
CACHE = '.spotipyoauthcache'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id = SPOTIPY_CLIENT_ID, client_secret = SPOTIPY_CLIENT_SECRET, redirect_uri=SPOTIPY_REDIRECT_URI, scope=SCOPE, requests_timeout=10))

##genero un file .csv a partire da brani presenti nel mio Spotify personale. Il risultato è il file musicData.csv, contenente circa 2000 brani diversi e variegati. (richiede un po' di tempo!)
playlists = sp.current_user_playlists()
print(type(playlists))
song_list, cols = createDataset(playlists, sp, 2)
df = pd.DataFrame(song_list, columns=cols)
df.to_csv("musicData.csv", header=cols)

##normalizzo il dataset
normalized_df = normalizeDataset()
print(normalized_df.columns)


##alcuni grafici che rappresentano i brani con le features più alte e più basse nel dataset.
orderedDf = normalized_df.sort_values(by = ['danceability'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Danceability', x = 'Track Name', y = 'danceability', legend = False, kind = "barh", color='red')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['energy'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Energy', x = 'Track Name', y = 'energy', legend = False, kind = "barh", color='yellow')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['valence'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Valence', x = 'Track Name', y = 'valence', legend = False, kind = "barh", color='blue')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['acousticness'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Acousticness', x = 'Track Name', y = 'acousticness', legend = False, kind = "barh", color='pink')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['mode'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Mode (maggiore o minore)', x = 'Track Name', y = 'mode', legend = False, kind = "barh", color='orange')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['speechiness'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Speechiness', x = 'Track Name', y = 'speechiness', legend = False, kind = "barh", color='orange')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['instrumentalness'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Instrumentalness', x = 'Track Name', y = 'instrumentalness', legend = False, kind = "barh", color='purple')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['liveness'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Liveness', x = 'Track Name', y = 'liveness', legend = False, kind = "barh", color='black')
ax.get_figure().tight_layout()

orderedDf = normalized_df.sort_values(by = ['tempo'], ascending = False, ignore_index=True)
dfToPlot = orderedDf.head(5)
dfToPlot = dfToPlot.append(orderedDf.tail(5))
ax = dfToPlot.plot(title = 'Tempo', x = 'Track Name', y = 'tempo', legend = False, kind = "barh", color='blue')
ax.get_figure().tight_layout()

##creo un dataframe a partire da quello normalizzato, mantenendo solo indici e features.
features_df = getSongsFeaturesDataset()

##visualizzazione dell'elbow point per il riconoscimento del numero adatto di cluster.
curva = []
K = range(2,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(features_df)
    curva.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, curva, 'bx-', marker = 'o')
plt.xlabel('k')
plt.ylabel('Inerzia')
plt.title('Il metodo Elbow Point mostra il k ottimale per il clustering.')
plt.show()

##realizzazione del PCA a 2 dimensioni
pca = PCA(2)
df_train = pca.fit_transform(features_df)

##applicazione del kMeans.
kmeans = KMeans(5, max_iter=100)
assigned_clusters = kmeans.fit_predict(df_train) ##fase di training
print("Cluster creati")
print(assigned_clusters)
print(df_train)

##plot dei risultati
filtered_label0 = df_train[assigned_clusters == 0]
filtered_label1 = df_train[assigned_clusters == 1]
filtered_label2 = df_train[assigned_clusters == 2]
filtered_label3 = df_train[assigned_clusters == 3]
filtered_label4 = df_train[assigned_clusters == 4]
centroids = kmeans.cluster_centers_
plt.xlabel('PrincipalComponent1')
plt.ylabel('PrincipalComponent2')
plt.title('Rappresentazione dei cluster (kMeans)')
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'limegreen')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'fuchsia')
plt.scatter(filtered_label3[:,0] , filtered_label3[:,1] , color = 'dodgerblue')
plt.scatter(filtered_label4[:,0] , filtered_label4[:,1] , color = 'orange')
plt.scatter(centroids[:,0], centroids[:,1], marker = "*", zorder = 10, c=['yellow', 'yellow','yellow', 'yellow', 'yellow'])
plt.show()
 
##esportazione del numero del cluster sul .csv, per ogni brano presente
normalized_df['cluster'] = assigned_clusters
normalized_df.to_csv('C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kMeansClusteredMusicData.csv', encoding='utf-8', index=True)

##creazione di un .csv diverso per ogni cluster
data_category_range = normalized_df['cluster'].unique()
data_category_range = data_category_range.tolist()
for i,value in enumerate(data_category_range):
    normalized_df[normalized_df['cluster'] == value].to_csv(r'C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_' + str(value) + r'.csv',index = False, na_rep = 'N/A')

##creazione delle playlist sulla base dei diversi .csv (richiede un po' di tempo!)
user = sp.me()
sp.user_playlist_create(user["id"], "SpotifAI: Cluster #1", public=True, collaborative=False, description='')
sp.user_playlist_create(user["id"], "SpotifAI: Cluster #2", public=True, collaborative=False, description='')
sp.user_playlist_create(user["id"], "SpotifAI: Cluster #3", public=True, collaborative=False, description='')
sp.user_playlist_create(user["id"], "SpotifAI: Cluster #4", public=True, collaborative=False, description='')
sp.user_playlist_create(user["id"], "SpotifAI: Cluster #5", public=True, collaborative=False, description='')
sp.user_playlist_create(user["id"], "SpotifAI: Cluster #6", public=True, collaborative=False, description='')
addCSVtoPlaylist("7v1DjIyITQpizvfjMNngV1", 'C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_0.csv')
addCSVtoPlaylist("219rfI6t7YTuWbXB3lsQxx", 'C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_1.csv')
addCSVtoPlaylist("4FIfBho4ftbwf2OAVBNDTy", 'C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_2.csv')
addCSVtoPlaylist("2O3kfZStAroOaFJtl8go9n", 'C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_3.csv')
addCSVtoPlaylist("6ZvSXFhiCE1jXi0AUqgvQ8", 'C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_4.csv')

##stampa delle features (e dei grafici) delle diverse playlist
print("Features della playlist #1:")
getPlaylistFeatures('C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_0.csv', "Playlist #1")
print("Features della playlist #2:")
getPlaylistFeatures('C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_1.csv', "Playlist #2")
print("Features della playlist #3:")
getPlaylistFeatures('C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_2.csv', "Playlist #3")
print("Features della playlist #4:")
getPlaylistFeatures('C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_3.csv', "Playlist #4")
print("Features della playlist #5: ")
getPlaylistFeatures('C:\\Users\\david\\Desktop\\progettoFIA\\script\\kmeans\\kmeans_cluster_4.csv', "Playlist #5")

##predizione e aggiunta di un brano a una playlist, con plot del grafico delle features.
testTrack()
song_df = pd.read_csv('C:\\Users\\david\\Desktop\\progettoFIA\\script\\normalizedSong.csv')
df_test = pca.transform(song_df)
pred = kmeans.predict(df_test)

print(pred)