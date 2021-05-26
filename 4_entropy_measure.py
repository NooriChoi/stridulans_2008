# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:07:03 2020

@author: Noori Choi
"""
import pandas as pd
from scipy.stats import entropy
import os
from pathlib import Path
import numpy as np
import pyinform
from scipy.signal import lfilter, butter, stft
import librosa
import matplotlib.pyplot as plt
from hmmviz import TransGraph
from lempel_ziv_complexity import lempel_ziv_complexity

wd = "HDBSCAN_result_directory"
csv_loc = "Destination_directory_for_csv"
fig_loc = "Destination_directory_for_figure"
wd_path = Path(wd) 
wd_csv = os.listdir(wd_path)

# maximum time for analysis (seconds)
time_limit = 300
n_segment = 1

def listtostring(s):  
    str1 = ""  
    for ele in s:  
        str1 += str(ele)   
    return str1  

def filt(block, fs):
    f_NQ = fs / 2 # Nquist frequency
    b2, a2 = butter(5, [lowcut / f_NQ, highcut / f_NQ], "band")
    signal_filt = lfilter(b2, a2, block)  # bandpass filtering
    return signal_filt

def feature_extract(signal_ls, fs, frame_length, hop_length):
    signal_ls = filt(signal_ls, fs)
    # pitch
    f, t, Zxx = stft(signal_ls, fs=fs, nperseg=frame_length, noverlap=hop_length)
    pitch = f[np.argmax(Zxx, axis=0)]
    med_pitch = np.median(pitch)
    # flatness
    flat = librosa.feature.spectral_flatness(signal_ls, n_fft=frame_length, hop_length=hop_length)
    med_flat = np.median(flat)
    # specs
    features = [med_pitch, med_flat]
    return features

def transition_matrix(transitions, n_segment):
    categories = set(x for l in transitions for x in l)
    categories = [int(x) for x in categories]
    
    blank = pd.Categorical([], categories=categories)
    M = pd.crosstab(blank, blank, dropna=False)
    for sequence in transitions:
        current = pd.Categorical(sequence[:-1], categories=categories)
        subsequent = pd.Categorical(sequence[1:], categories=categories)
        sub_M = pd.crosstab(current, subsequent, dropna=False)
        M = M.add(sub_M, fill_value=0)
    #now convert to probabilities:
    sum_val = M.to_numpy().sum()
    M = M/sum_val
    M = M.reindex(index=categories, columns=categories)
    
    if n_segment == 1:
        seg_matrix = None
    else:
        # transition for segments
        seg_M = pd.crosstab(blank, blank, dropna=False)
        seg_M_ls = [seg_M for x in range(n_segment)]
        for sequence in transitions:
            sub_time = np.array_split(sequence, n_segment)
            sub_time_index = list(range(0, len(sub_time)))
            for i, j in zip(sub_time, sub_time_index):
                current = pd.Categorical(i[:-1], categories=categories)
                subsequent = pd.Categorical(i[1:], categories=categories)
                sub_M = pd.crosstab(current, subsequent, dropna=False)
                seg_M_ls[j] = seg_M_ls[j].add(sub_M, fill_value=0)
        
        seg_matrix = []
        for matrix in seg_M_ls:
            sum_val = matrix.to_numpy().sum()
            matrix = matrix/sum_val
            matrix = matrix.reindex(index=categories, columns=categories)
            seg_matrix.append(matrix)
    
    return M, seg_matrix

ent_df = pd.DataFrame()
tot_trans = []
for csv in wd_csv:
    print("Analyzing the file " + csv)
    df = pd.read_csv(wd_path/csv)
    df = df.sort_values(by=['begin'])
    df['dT'] = df['begin'].shift(-1) - df['end']
    df = df[df['begin'] <= time_limit]
    
    if len(df) == 0:
        continue
    # Entropy analysis 
    types = df['fin_knn'].values
    lz = lempel_ziv_complexity(listtostring(types))
    lz_norm = lz/(len(types)/np.log(len(types)))
    ent = entropy(types.tolist())
    ent_rate = pyinform.entropyrate.entropy_rate(types, k=1, local=False)
    
    ent_df = ent_df.append({'File': csv, 'n_comp': len(types),
                            'lz': lz, 'lz_norm': lz_norm, 
                            'ent': ent, 'entr':ent_rate
                            }, ignore_index=True)
    # variables for simulation
    tot_trans.append(types)
    
tot_trans_mat, tot_set = transition_matrix(tot_trans, n_segment)
tot_trans_graph = TransGraph(tot_trans_mat)

fig = plt.figure(figsize=(16, 8))
tot_trans_graph.draw(edgelabels=True, nodefontsize=16)
plt.savefig(fig_loc + '/' + 'transition_diagram_str.png')

ent_df.to_csv(csv_loc + '/' + 'complexity_str.csv', index=False, sep = ",")