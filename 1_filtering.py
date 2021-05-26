# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:57:04 2020

Second part of data analysis
1. find the longest silence in an audio file
 - modified "pydub silenc.py" 
 - https://github.com/jiaaro/pydub/tree/master/pydub
2. adapt noise reduction function with the detected silence

@author: Noori Choi
"""
# Library imports
import os
import numpy as np
import soundfile as sf
import itertools
import noisereduce as nr
import librosa
import pandas as pd
from kneed import KneeLocator

# Working directory
wd = "Raw_audio_file_directory"
# Listof soundfiles
sounds = os.listdir(wd)
# Directory where to save the csv file, must be different with wd
filt_loc = "Destination_directory"
# Log directory
loglist = os.listdir(filt_loc)

def find_alpha(audio, lowest, highest, num):
    list_alpha = np.linspace(lowest, highest, num)
    
    audio_len = []
    for alpha in list_alpha:
        #set amp_threshold
        ab_audio = np.absolute(audio)
        A_low = np.median(ab_audio) + (alpha*np.std(ab_audio))
        #count number of samples below amp_threshold
        length = np.count_nonzero(audio > A_low)
        audio_len.append([alpha, length])
           
    audio_len = np.array(audio_len)    
    results = pd.DataFrame({'alpha': audio_len[:, 0], 'number': audio_len[:, 1]})
    
    kn = KneeLocator(results['alpha'], results['number'], curve='convex', direction='decreasing')
     
    if kn.knee:
        knee = kn.knee + 2*((highest-lowest)/num)
    else:
        knee = highest
    
    print(knee)
    return knee
    
def find_noise(s, min_noise_sec=2, dt=0.03):
    audio, samplerate = sf.read(wd + '/' + s)
    seg_len = len(audio)
    # Amplitude threshold
    ab_audio = np.absolute(audio)
    # set the seek_step and minimum noise length
    seek_step = int(samplerate * dt) # dt = minimum time interval for peak finding
    min_noise_len = samplerate*min_noise_sec # min_silence_len in Hz
    
    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_noise_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    silence_thresh = np.median(ab_audio) + (alpha*np.std(ab_audio))
    #silence_thresh = db_to_float(silence_thresh) * audio.max_possible_amplitude

    # find silence and add start and end indicies to the to_cut list
    noise_starts = []
    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_noise_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        audio_slice = audio[i:i + min_noise_len]
        if np.max(audio_slice) <= silence_thresh:
            noise_starts.append(i)

    # short circuit when there is no silence
    if not noise_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    noise_ranges = []

    prev_i = noise_starts.pop(0)
    current_range_start = prev_i

    for noise_start_i in noise_starts:
        continuous = (noise_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = noise_start_i > (prev_i + min_noise_len)

        if not continuous and silence_has_gap:
            noise_ranges.append([current_range_start,
                                  prev_i + min_noise_len])
            current_range_start = noise_start_i
        prev_i = noise_start_i

    noise_ranges.append([current_range_start,
                          prev_i + min_noise_len])
    
    noise_ranges = np.array(noise_ranges)
    
    num_noise = np.arange(0, noise_ranges.shape[0], 1).tolist() 
    long_noise = []
    for j in num_noise:
        if np.diff(noise_ranges[j]) == max(np.diff(noise_ranges)):
            long_noise.append(noise_ranges[j])
        
    return long_noise

def noise_extract(s, alpha):
    amb_noise = find_noise(s)
    
    if len(amb_noise) > 0:
        noise_sample = amb_noise[0]
        noise_begin = noise_sample[0]
        noise_end = noise_sample[-1]
    else:
        noise_begin = 0
        noise_end = 0
        
    return noise_begin, noise_end

#makeing a log of pre-existing files
for index, item in enumerate(loglist):
    loglist[index] = loglist[index][0:-8]
print(loglist)
for s in sounds:
    if s[:-4] in loglist:
        print(s + " is already processed")
        pass
    else:
        print("Filtering " + s)
        audio, rate = librosa.load(wd + '/' + s, sr=None)
        #set the values for finding alpha
        alpha = find_alpha(audio, lowest=3, highest=10, num=10)
        #noise filtering with calculated alpha
        noise_begin, noise_end = noise_extract(s, alpha)
        noise_length = noise_end - noise_begin
        
        if noise_length > 0:
            noise = audio[noise_begin:noise_end]
            reduced_noise = nr.reduce_noise(audio_clip=audio, noise_clip=noise)
            sf.write(filt_loc + '/' + s[:-4] + "_f.wav", reduced_noise, rate)
        elif noise_length == 0:
            sf.write(filt_loc + '/' + s[:-4] + "_f.wav", audio, rate)    