# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 00:28:00 2019

Third part of data analysis
1. Find chunks by peak finding
2. apply amp, freq filtering

@author: Noori Choi
"""
# Library imports
import os
import numpy as np
import pandas as pd
from scipy.signal import lfilter, butter, find_peaks
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn import mixture
import librosa

# Working directory
wd = "Filtered_audio_directory"
# Listof soundfiles
sounds = os.listdir(wd)
# Directory where to save the csv file, must be different with wd
csv_loc = "Destination_directory"
loglist = os.listdir(csv_loc)

# frequency for bandpass filter (in Hz)
lowcut = 50
highcut = 8000
dt_bout = 0.03 # dt - time interval between peaks
buffer = 0.015
# frame_legnth for RMS
frame_length = 1024
hop_length = frame_length//4
min_comp_interval = 10*hop_length

def find_alpha(audio, lowest, highest, num):
    list_alpha = np.linspace(lowest, highest, num)
    
    audio_len = []
    for alpha in list_alpha:
        #cutoff = int(len(audio)*0.01)
        #ab_audio = np.sort(np.absolute(audio))[cutoff+1:-cutoff]
        ab_audio = np.absolute(audio)
        A_low = np.median(ab_audio) + (alpha*np.std(ab_audio))
        #count number of samples below amp_threshold
        length = np.count_nonzero(audio > A_low)
        audio_len.append([alpha, length])
           
    audio_len = np.array(audio_len)    
    results = pd.DataFrame({'alpha': audio_len[:, 0], 'number': audio_len[:, 1]})
    
    kn = KneeLocator(results['alpha'], results['number'], curve='convex', direction='decreasing')
    
    if kn.knee:
        knee = kn.knee #- (highest-lowest)/num
    else:
        knee = highest
    
    print(knee)
    return knee

def peak_find(env, alpha, dt):
    # sigma clipping for peak detection threshold
    A_low = np.median(env) + (alpha*np.std(env))
    peaks, properties = find_peaks(env, height = A_low, distance = dt*fs)
    time = np.ndarray.tolist(peaks/fs) # time of peaks in second
    amp = np.ndarray.tolist(properties["peak_heights"]) # amp of each peak
    peaks = np.column_stack((time, amp))
    return peaks, A_low

def filt(block):
    f_NQ = fs / 2 # Nquist frequency
    b2, a2 = butter(5, [lowcut / f_NQ, highcut / f_NQ], "band")
    signal_filt = lfilter(b2, a2, block)  # bandpass filtering
    return signal_filt

def process(data, alpha, dt):
    print("Analyzing the file " + s)
    peaks, A_low = peak_find(data, alpha, dt)
    # add file name as a separate column
    result = np.hstack((peaks, np.array([[s]] * len(peaks))))
    # convert into dataframe
    df_res = pd.DataFrame({'Time': result[:, 0], 'Amp': result[:, 1], 'File': result[:,2]})
    return df_res, A_low

def process_comp(sig, fs, begin):
    rms = librosa.feature.rms(y=sig, frame_length=frame_length, hop_length=hop_length)
    #rms = librosa.feature.spectral_flatness(y=sig, n_fft=frame_length, hop_length=hop_length)
    times = librosa.times_like(rms[0], sr=fs, hop_length=hop_length) + begin
    time_rms = np.column_stack((times, rms[0]))
    # add file name as a separate column
    result = np.hstack((time_rms, np.array([[s]] * len(time_rms))))
    # convert into dataframe
    df_res = pd.DataFrame({'File': result[:,2], 'Time': result[:, 0], 'rms': result[:, 1]})
    return df_res

# for loop to go through every file
for index, item in enumerate(loglist):
    loglist[index] = loglist[index][:-4]

loglist=[]
no_pulse = []
for s in sounds:
    if s[:-4] in loglist:
        print(s + " is already processed")
        pass
    else:
        print('peakfind in ' + s)
        data, fs = librosa.load(wd + '/' + s, sr=None)
        data = filt(data)
        #get alpha for sigma clipping
        alpha = find_alpha(data, lowest=0, highest=5, num=30)
        # get result data (df_res)
        df, A_low = process(data, alpha, dt_bout)
        
        if len(df) > 3:
            #calculate time interval
            df['Time'] = pd.to_numeric(df['Time'])
            df['dT'] = df['Time'].diff().fillna(0)
            df = df[1:]
            df['dT'] = np.log(df['dT'])
            
            df_index = df.reset_index(drop = True)
            df_dt = df[['dT']]
            X = df_dt.to_numpy().reshape(-1, 1)
            # Gaussian Mixture Model
            high_thr = np.log(3) # set outlier max limit
            low_thr = np.log(0.03)
            interval = df[['dT']].loc[(df.dT > df.dT.quantile(.00)) & (df.dT < df.dT.quantile(1.00))]
            
            # find best_fit
            N = np.arange(3, 11)
            models = [None for i in range(len(N))]
            
            for i in range(len(N)):
                models[i] = mixture.GaussianMixture(N[i]).fit(interval)
        
            # compute the AIC and the BIC
            best_fit = []
            for j in range(100):
                AIC = [m.aic(X) for m in models]
                BIC = [m.bic(X) for m in models]
                bout_gen = models[np.argmin(AIC)]
                best_fit.append(bout_gen)
            
            bout_gen = max(set(best_fit), key=best_fit.count)
            bout_gen.fit(X) #GMM model fit
            probs = bout_gen.predict_proba(X) #Soft clustering
        
            probs_df = pd.DataFrame(data=probs, columns=[str(i) for i in range(probs.shape[1])])
            probs_df['group'] = probs_df.idxmax(axis=1) #assign dT into the highest probability
            fin_df = pd.concat([df_index, probs_df], axis=1)
            
            bout_df = fin_df.groupby('group')['dT'].median()#.nsmallest(2)
            criteria_b = bout_df.idxmax()
            fin_df['group'].loc[fin_df['dT'] > high_thr] = criteria_b
            
            # Bout grouping by the largest group of dT
            fin_df['bout'] = (fin_df['group'] == criteria_b).groupby(fin_df['File']).cumsum() + 1
            
            # Pulse grouping by GMM
            fin_df = fin_df.groupby(['File', 'bout']).agg(
                begin = ('Time', min),
                end = ('Time', max)).reset_index()
            
            fin_df = fin_df.loc[fin_df['end']-fin_df['begin'] > frame_length//fs]
            bout_list = fin_df['bout'].unique().tolist()
            comp_df_fin = pd.DataFrame()
            for bout in bout_list:
                print(bout)
                start = fin_df.loc[fin_df['bout']==bout, 'begin'].values[0]
                fin = fin_df.loc[fin_df['bout']==bout, 'end'].values[0]
                start_sample = int(start*fs)
                if start_sample < 0:
                    start_sample = 0
                
                fin_sample = int(fin*fs)
                if fin_sample > len(data):
                    fin_sample = int(len(data))
                
                sig = data[start_sample:fin_sample]
                bout_df = process_comp(sig, fs, start)
                bout_df['bout'] = bout
                df_index = bout_df.reset_index(drop = True)
                ## get the best number of clusters for GMM
                bout_df['rms'] = np.log(pd.to_numeric(bout_df['rms']))
                #bout_df['rms'] = pd.to_numeric(bout_df['rms'])
                bout_df_rms = bout_df[['rms']]
                rms_fit = bout_df[['rms']].loc[
                        (bout_df.rms > bout_df.rms.quantile(.01)) & 
                        (bout_df.rms < bout_df.rms.quantile(.99))]
            
                if len(rms_fit) < 11:
                    print("too short!")
                    probs_df = pd.DataFrame({'group':[1]*len(bout_df), 
                                             'comp':[1]*len(bout_df)})
                    comp_df = pd.concat([df_index, probs_df], axis=1)
                    ## append comp_df_fin
                    comp_df = comp_df.groupby(['File', 'bout', 'comp']).agg(
                            begin = pd.NamedAgg(column='Time', aggfunc=min),
                            end = pd.NamedAgg(column='Time', aggfunc=max)).reset_index()
                    print(comp_df)
                    comp_df_fin = comp_df_fin.append(comp_df, sort=False)
                else:
                    N = np.arange(2, 5)
                    models = [None for i in range(len(N))]
                    X = bout_df_rms.to_numpy().reshape(-1, 1)
                        
                    for i in range(len(N)):
                        models[i] = mixture.GaussianMixture(N[i]).fit(rms_fit)
                    
                    # compute the AIC and the BIC
                    best_fit = []
                    for j in range(100):
                        AIC = [m.aic(X) for m in models]
                        BIC = [m.bic(X) for m in models]
                        bout_gen = models[np.argmin(AIC)]
                        best_fit.append(bout_gen)
                        
                    ## Final GMM using best_k
                    comp_gen = max(set(best_fit), key=best_fit.count)
                    comp_gen.fit(X) #GMM model fit
                    probs = bout_gen.predict_proba(X) #Soft clustering
                    probs_df = pd.DataFrame(data=probs, columns=[str(i) for i in range(probs.shape[1])])
                    probs_df['group'] = probs_df.idxmax(axis=1) #assign dT into the highest probability
                    comp_df = pd.concat([df_index, probs_df['group']], axis=1)
                    comp_df['rms'] = pd.to_numeric(comp_df['rms'])
                    ## assign comp
                    comp_threshold = comp_df.groupby('group')['rms'].median()
                    print(comp_threshold)
                    criteria_c = comp_threshold.idxmin()
                    comp_df['group'][comp_df['rms'] < A_low] = criteria_c
                    ## Find comp
                    comp_df = comp_df[comp_df['group']!=criteria_c]
                    comp_df['Time'] = pd.to_numeric(comp_df['Time'])
                    comp_df.sort_values("Time", inplace=True)
                    comp_df['dT'] = comp_df['Time'].diff().fillna(0)
                    comp_df['comp'] = ((comp_df['dT'] > min_comp_interval/fs)).cumsum() + 1
                    ## append comp_df_fin
                    comp_df = comp_df.groupby(['File', 'bout', 'comp']).agg(
                            begin = pd.NamedAgg(column='Time', aggfunc=min),
                            end = pd.NamedAgg(column='Time', aggfunc=max)).reset_index()
                    print(comp_df)
                    comp_df_fin = comp_df_fin.append(comp_df, sort=False)
            
            # Save as a csv file
            comp_df_fin.to_csv(csv_loc + '/' + s[:-4] + '.csv', index=False, sep = ",")
        else:
            print(s + " does not contain a pulse.")
            no_pulse.append(s)
        
print(no_pulse)
no_pulse_df = pd.DataFrame({'file':no_pulse})
no_pulse_df.to_csv(csv_loc + '/' + 'no_pulse.csv', index=False, sep= ",")
    
    
    