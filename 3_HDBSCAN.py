# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 21:20:19 2020

1. hierachial clustering with dtw among signal components

@author: Noori Choi
"""
import pandas as pd
import os
from pathlib import Path
import numpy as np
import librosa
from scipy import spatial, stats
from scipy.signal import lfilter, butter
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn import metrics
#from tslearn import metrics

# set directories
wd = "Manual_classification_csv_directory"
csv_loc = "Destination_directory_for_hdbscan_result"
audio = "Audio_file_directory"

root_path = Path(wd)

csv_list = os.listdir(wd)
loglist = os.listdir(csv_loc)

# dt - time interval between peaks in sec
dt = 0.01
buffer = 0.05
w = 10
# frequency for bandpass filter (in Hz)
lowcut = 50
highcut = 4000
# range of peak location of triangluar membership functions (alpha < x < beta)
alpha = 0.1
beta = 0.9

def max_norm(ts, amax, amin):
    if amax == amin:
        ts_norm = ts/amax
    else:
        ts_norm = (ts-amin)/(amax - amin)
    return ts_norm

def filt(block, fs):
    f_NQ = fs / 2 # Nquist frequency
    b2, a2 = butter(5, highcut / f_NQ, btype="low")
    signal_filt = lfilter(b2, a2, block)  # bandpass filtering
    return signal_filt

def dtw(s1, s2, w):
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n+1, m+1))
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = abs(spatial.distance.euclidean(s1, s2))
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    
    dtw_matrix = dtw_matrix[1:, 1:]
    dtw_dist = dtw_matrix[-1, -1]
    return dtw_dist

def sample_gen(length, alpha, beta):
    peak_param = np.linspace(alpha, beta, num=9)
    buffer = int(0.05*length)
    
    sample_ls = []
    for var in peak_param:
        sample = []
        for i in range(length):
            peak = var*length
            if peak < 0:
                peak = 1
            
            if i <= peak - buffer:
                rms = i/(peak - buffer)
                sample.append(rms)
            elif peak - buffer < i < peak + buffer:
                rms = 1
                sample.append(rms)
            else:
                rms = (length - i)/(length - (peak + buffer))
                sample.append(rms)
        sample_ls.append(sample)
    return sample_ls

def feature_ex(sig):
    sig_rms = librosa.feature.rms(y=sig, frame_length=2048, hop_length=512)[0]
    sig_rms = max_norm(sig_rms, max(sig_rms), min(sig_rms))
    sample_sig = sample_gen(len(sig_rms), alpha, beta)
    
    features = []
    for sample in sample_sig:
        dist = dtw(sig_rms, sample, w)
        #dist = spatial.distance.euclidean(sig_rms, sample)
        features.append(dist/len(sig_rms))
    
    features = stats.zscore(features)
    #sig_len = np.log(len(sig))
    #features.extend([sig_len])
    return features

# for loop to go through every file
fin_df = pd.DataFrame()
features = []
for csv in csv_list:
    print("Analyzing the file " + csv) 
    # Data preparation
    ## load data
    df = pd.read_csv(root_path/csv)
    file = df['File'][0]
    data, fs = librosa.load(audio + '/' + file, sr=None)
    df = df[df['begin']-df['end'] != 0]
    df['ID'] = df['bout']*1000 + df['comp']
    # generate index df
    df_index = df[['File', 'ID', 'begin', 'end', 'man_knn']]
    fin_df = fin_df.append(df_index, ignore_index=True)
    # generate features
    for index, row in df.iterrows():
        start = row['begin']
        fin = row['end']
        print(start)
        
        start_sample = int(start*fs) - int(buffer*fs)
        if start_sample < 0:
            start_sample = 0
        fin_sample = int(fin*fs) + int(buffer*fs)
        if fin_sample > len(data):
            fin_sample = len(data)
        
        sig = data[start_sample:fin_sample]
        sig = filt(sig, fs)
        sig_feature = feature_ex(sig)
        features.append(sig_feature)

feature_df = pd.DataFrame(features, columns = ['peak_10', 'peak_20', 'peak_30',
                                                 'peak_40', 'peak_50', 'peak_60',
                                                 'peak_70', 'peak_80', 'peak_90'])
print('compute cosine similarity')
distance = metrics.pairwise_distances(features, metric='cosine')

eval_ls = []
min_clust_ls = np.linspace(10, 60, num=11).astype(int)
print(min_clust_ls)
for min_clust in min_clust_ls:
    print("HDBSCAN with " + str(min_clust))
    hdbscan_model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=int(min_clust), min_samples=1, 
                                    allow_single_cluster=False)
    hdbscan_model.fit(distance.astype('float64'))
    
    true_labels = fin_df['man_knn'].tolist()
    pred_labels = hdbscan_model.labels_
    fin_df['cluster'] = pred_labels
    fin_df.reset_index(drop=True, inplace=True)
    feature_df.reset_index(drop=True, inplace=True)
    fin_df2 = pd.concat([fin_df, feature_df], ignore_index=True, sort=False)
    fin_df2.to_csv(csv_loc + '/' + 'hdbscan_gensample_mc' + str(min_clust) + '.csv', index=False, sep = ",")
    # Evaluation
    silhouette = metrics.silhouette_score(features, pred_labels, metric='euclidean')
    v_measure = metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)
    adj_rand = metrics.adjusted_rand_score(true_labels, pred_labels)
    eval_ls.append([min_clust, silhouette, v_measure, adj_rand])
    #Figures
    color_palette = sns.color_palette('deep', 64)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in hdbscan_model.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, hdbscan_model.probabilities_)]
    projection = TSNE().fit_transform(features)
    plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.savefig(csv_loc + '/' + 'cluster_hdbscan_gensample_mc' + str(min_clust) + '.png')
    plt.clf()

eval_df = pd.DataFrame(eval_ls, columns = ['min_clust', 'silhouette', 'v_measure', 'rand_adj'])
eval_df.to_csv(csv_loc + '/' + 'hdbscan_model_eval.csv', index=False, sep = ",")