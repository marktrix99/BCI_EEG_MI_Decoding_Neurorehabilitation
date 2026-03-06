#BCI COMPETITION DATA PROCESSING

import numpy as np
import pandas as pd
import os
from scipy.io import loadmat  # For loading MAT files
import mne  # For EDF format of the EEG data


def process_single_subject(subj_num, test_type="full", id_trial_start=20, id_trial_end=30):
# # Filter Bank Common Spatial Pattern and Common Spatial Patern Performanse Comparison
# Attempt at implementing filter-bank common spatial filter (FBCSP) and CSP on BCI Competition IV 2a Dataset
#   
# **References:**   
# 
# [1] Kai Keng Ang, Zheng Yang Chin, Haihong Zhang and Cuntai Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface," 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), Hong Kong, 2008, pp. 2390-2397, doi: 10.1109/IJCNN.2008.4634130.    
# [2] Ang, K. K., Chin, Z. Y., Wang, C., Guan, C., & Zhang, H. (2012). Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b. Frontiers in Neuroscience, 6. doi: 10.3389/fnins.2012.00039

# # BCI Competition IV Dataset 2a (.npz data)
# Information Given in Documentation
# 
# From the documentation it is known that:
# * 25 electrodes are used, first 22 are EEG, last 3 are EOG
# * Sampling frequency (fs) is 250Hz
# * 9 subjects
# * 9 run (run 1-3 are for eye movement, run 4-9 is MI)  
#   
# 
# **-- Time Duration--**  
# 1 trials                          = 7-8s  
# 1 run              = 48 trials    = 336-384s  
# 1 session = 6 runs = 288 trials   = 2016-2304s
# 
# About the recording of eye movement
# * run 1 => 2 mins with eyes open
# * run 2 => 1 min with eyes closed
# * run 3 => 1 min with eye movements
# 
# ![timing-scheme-paradigm.png](./img/timing-scheme-paradigm.png) 


# %% [markdown]
# # Dataset (.npz)
# 
# Clinical data of 50 acute stroke patients. The .mat format is being used in the algorithms
# 


    # Seed to fix randomization
    np.random.seed(42)

    # %%
    # Define subjects for analysis
    all_results = []
    all_results_csp = []

    subj = f"sub-{subj_num:02d}"
    sub_start = subj_num
    sub_end = sub_start  # ending subject number
    sub_end = sub_end + 1 # for iteration purpose

    # Define window start and end - start and end time in seconds after cue onset of each class
    start = 0.5 # 0.5s after cue onset is common practice in EEG analysis, to avoid pre-stimulus noise
    end = 2
    # Select number of filters, 2*m is the number of channels to be used for the analysis
    m = 6

    test_type = "full" # can be full or trial
    id_trial_start = 20
    id_trial_end = 30


    # %% [markdown]
    # # Selecting Events for Classification 
    # 
    # Event codes (from the paper)
    # In the raw data, labels are:
    # * 1 = left hand MI
    # * 2 = right hand MI
    # 
    # The trial structure is:
    # * 1 = instruction (2s)
    # * 2 = movement/MI (4s)
    # * 3 = break (2s)
    # 

    # %%
    # Classification combinations
    rest_code = 768 # new trial
    left_code = 769
    right_code = 770
    foot_code = 771
    tongue_code = 772

    movement_code = [left_code, right_code, foot_code, tongue_code]

    classification_combinations = [
        ([left_code], [right_code], 'Left', 'Right'),
        ([left_code], [rest_code], 'Left', 'Rest'),
        ([right_code], [rest_code], 'Right', 'Rest')
    ]
    # %% [markdown]
    # The following classifications are expected to be analysed:
    # 
    #     left / right  
    #     movement / rest  
    #     left / rest  
    #     right / rest  
    # 

    # %% [markdown]
    # # Loading dataset

    # %%
    # Number of subject, n + 1 for iteration purpose (there are 50 subjects)
    ns = 1

    # Creating dict to store original data and modified data
    # ori_data will serve as initial loaded data that will remain unchanged
    # mod_data will contain modified original data
    ori_data = dict()
    mod_data = dict() 

    # Function to count subject
    def subject_counter(i):
        return 'subject0{}'.format(i)

    # Access directory where the data is stored (change if needed)
    base_dir = r"C:\Users\LENOVO\Documents\Uni\Master Thesis"



    # %%
    def load_subject_data(subj, base_dir):
        data_path = os.path.join(base_dir, 'FBCSP_BCI_IV2a', 'datasets', f'A{subj[-2:]}T.npz')
        try:
            npz_data = np.load(data_path, allow_pickle=True)
            keys = npz_data.files
            print(f"Keys in {data_path}:", keys)

            # Try both possible key names
            if 'etype' in keys:
                etyp = npz_data['etype']
            elif 'etyp' in keys:
                etyp = npz_data['etyp']
            else:
                raise KeyError("'etype' or 'etyp' not found in the archive")

            s = npz_data['s']
            epos = npz_data['epos']
            edur = npz_data['edur']
            artifacts = npz_data['artifacts']

            print(f"Successfully loaded {subj}:")
            print(f"- s shape: {s.shape}")
            print(f"- etyp shape: {etyp.shape}")
            print(f"- epos shape: {epos.shape}")
            print(f"- edur shape: {edur.shape}")
            print(f"- artifacts shape: {artifacts.shape}")

            return {
                's': s,
                'etype': etyp,
                'epos': epos,
                'edur': edur,
                'artifacts': artifacts
            }
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            return None
        except Exception as e:
            print(f"Error processing {data_path}: {str(e)}")
            return None

    # Load all subject data from .npz files
    for i in range(sub_start, sub_end):
        subj = subject_counter(i)
        loaded_data = load_subject_data(subj, base_dir)
        if loaded_data is None:
            continue

        # Store in ori_data
        ori_data[subj] = {
            'rawdata': loaded_data['s'],      # EEG data (samples × channels)
            'etype': loaded_data['etype'].flatten(),   # Event types (flatten for 1D)
            'epos': loaded_data['epos'].flatten(),    # Event positions (flatten for 1D)
            'edur': loaded_data['edur'].flatten(),    # Event durations
            'artifacts': loaded_data['artifacts'].flatten() # Artifact info
        }

        print(f"Loaded subject {subj}:")
        print(f"  rawdata shape: {ori_data[subj]['rawdata'].shape}")
        print(f"  etyp shape: {ori_data[subj]['etype'].shape}")
        print(f"  epos shape: {ori_data[subj]['epos'].shape}")
        print(f"  edur shape: {ori_data[subj]['edur'].shape}")
        print(f"  artifacts shape: {ori_data[subj]['artifacts'].shape}")


    # %% [markdown]
    # # Preprocessing 
    # 
    # Remove the last 3 channels - 2 EOG and 1 marker channel, raw data only with EEG channels

    # %%
    for i in range(sub_start, sub_end):
        subj = subject_counter(i)
        if subj not in ori_data:
            continue
            
        # Get raw data (40 trials × 33 channels × 4000 samples)
        rawdata = ori_data[subj]['rawdata']
        print(f"rawdata shape for {subj}: {rawdata.shape}")

        if rawdata.ndim == 3:
            eeg_data = rawdata[:, :22, :]  # 3D: trials × channels × samples
            # Reshape to continuous data for filtering (combine all trials)
            continuous_eeg = eeg_data.transpose(1, 0, 2).reshape(22, -1)  # 22×(trials*samples)
            mod_data[subj] = {
                'raw_EEG': continuous_eeg.T,  # samples×channels
                'fs': 500,
                'etype': ori_data[subj]['etype'],
                'epos': ori_data[subj]['epos']
            }
        elif rawdata.ndim == 2:
            eeg_data = rawdata[:, :22]     # 2D: samples × channels
            # No need to reshape, just transpose to samples×channels
            mod_data[subj] = {
                'raw_EEG': eeg_data,  # samples×channels
                'fs': 500,
                'etype': ori_data[subj]['etype'],
                'epos': ori_data[subj]['epos']
            }
        else:
            raise ValueError(f"Unexpected rawdata shape for {subj}: {rawdata.shape}")

    # Print the keys of mod_data to verify
    print("Keys in mod_data:")
    for key in mod_data.keys():
        print(key)

    # And also the shape
    print("\nShapes of raw_EEG in mod_data:")
    for key in mod_data.keys():
        print(f"{key}: {mod_data[key]['raw_EEG'].shape}")


    # %%
    #downsample_factor = 4  # Reduces fs from 500Hz to 125Hz (still good for EEG)
    #for subj in mod_data.keys():
    #    # Downsample by taking every nth sample
    #    mod_data[subj]['raw_EEG'] = mod_data[subj]['raw_EEG'][::downsample_factor, :]
    #    mod_data[subj]['fs'] = mod_data[subj]['fs'] // downsample_factor  # Update sampling rate
        
    #     Update event positions accordingly
    #    mod_data[subj]['epos'] = [pos // downsample_factor for pos in mod_data[subj]['epos']]

    # Print the updated shapes and fs
    #print("\nUpdated shapes and fs after downsampling:")
    #for key in mod_data.keys():
    #    print(f"{key}: {mod_data[key]['raw_EEG'].shape}, fs: {mod_data[key]['fs']}")

    # %% [markdown]
    # ## Bandpass Filtering
    # The first stage employing a filter bank is to decompose EEG into multiple frequency pass band, using causal Chebysev Type II filter/ Butterworth Filter.  
    # A total of 9 band-pass filters are used, namely, 4-8, 8-12, ... 36-40 Hz  
    # These frequency ranges are used because they yielf a stable frequency response and cover range of 4-40 Hz

    # %%
    # Band pass filter with butterworth filter
    from scipy.signal import butter, lfilter
    from scipy.signal import freqz

    # %%
    def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, signal, axis=-1)
        
        return y

    # %% [markdown]
    # ### Filtering EEG signal with Butterworth Band-pass - FBCSP
    # Following the steps mentioned in [1], there will be 8 band-pass filter with bandwidth of:
    # 1. 4-8Hz
    # 2. 8-12Hz
    # 3. 12-16Hz
    # 4. 16-20Hz
    # 5. 20-24Hz
    # 6. 24-28Hz
    # 7. 28-32Hz
    # 8. 32-36Hz
    # 9. 36-40Hz
    # 
    # *Note*   
    # Apply filter to the time-series axis, thus set 'raw_EEG' inside each subject to shape of N x T (i.e. electrodes x samples)

    # %%
    # 2. Bandpass Filtering (FBCSP approach)
    print("\n=== Applying Bandpass Filtering ===")
    lowcut = 4
    highcut = 40
    fs = 250  # New Sampling rate after downsampling 125

    for i in range(sub_start, sub_end):
        subj = subject_counter(i)
        if subj not in mod_data:
            continue
            
        print(f'Processing {subj}')
        mod_data[subj]['EEG_filtered'] = {}
        
        # Create frequency bands (4-8, 8-12, ..., 36-40 Hz)
        for start_band in range(lowcut, highcut, 4):
            band = f"{start_band:02d}_{start_band+4:02d}"
            print(f'  Filtering {band} Hz band')
            
            # Apply bandpass filter
            mod_data[subj]['EEG_filtered'][band] = {
                'EEG_all': butter_bandpass_filter(
                    mod_data[subj]['raw_EEG'], 
                    start_band, start_band+4, fs
                )
            }
            

    # %% [markdown]
    # # Extracting Epochs

    # %%
    def get_valid_event_positions(epos, etyp, event_codes, fs, window_start, window_end, total_length):
        valid_pos = []
        for i in range(min(len(epos), len(etyp))):
            if etyp[i] in event_codes:
                pos = epos[i]
                start = pos + int(window_start * fs)
                end = pos + int(window_end * fs)
                if end < total_length:
                    valid_pos.append(pos)
        return np.array(valid_pos)


    # %%
    # # print("\n=== Extracting Epochs ===")

    # for i in range(sub_start, sub_end):
    #     subj = subject_counter(i)
    #     if subj not in mod_data:
    #         continue

    #     print(f'Extracting epochs for {subj}')
    #     fs = mod_data[subj]['fs']

    #     # Check the actual data length
    #     data_length = mod_data[subj]['raw_EEG'].shape[0]
    #     print(f"  Data length: {data_length} samples ({data_length/fs:.2f} seconds)")

    #     # Print some event markers to debug
    #     print(f"  First few event markers:")
    #     for j in range(min(12, len(mod_data[subj]['etyp']))): # There is a pattern of events
    #         print(f"    Event {j}: Type {mod_data[subj]['etyp'][j]}, Position {mod_data[subj]['epos'][j]}")

    #     window_start = int(start * fs)  # start time in samples
    #     window_end = int(end * fs)      # end time in samples
    #     window_length = window_end - window_start
    #     print(f"  Epoch window: {window_start}-{window_end} samples ({start}-{end} seconds)")

    #     # For each frequency band
    #     for band in mod_data[subj]['EEG_filtered']:
    #         print(f'  Processing band {band} Hz')

    #     # Initialize empty arrays for epochs
    #     left_epochs = []
    #     right_epochs = []
    #     rest_epochs = []

    #     # Iterate through all events
    #     num_events = min(len(mod_data[subj]['etyp']), len(mod_data[subj]['epos']))
    #     for event_idx in range(num_events):
    #         event_type = mod_data[subj]['etyp'][event_idx]
    #         event_pos = mod_data[subj]['epos'][event_idx]
    #         epoch_start = event_pos + window_start
    #         epoch_end = event_pos + window_end

    #         if epoch_end > data_length:
    #             print(f"    Skipping Event {event_idx} (out of bounds)")
    #             continue

    #         epoch = mod_data[subj]['EEG_filtered'][band]['EEG_all'][epoch_start:epoch_end, :]

    #         if event_type == 1:  # Left-hand MI
    #             print(f"    Left MI Event {event_idx}")
    #             left_epochs.append(epoch)
    #         elif event_type == 2:  # Right-hand MI
    #             print(f"    Right MI Event {event_idx}")
    #             right_epochs.append(epoch)
    #         elif event_type == 3:  # Rest
    #             print(f"    Rest Event {event_idx}")
    #             rest_epochs.append(epoch)
    #         else:
    #             # Skip instruction (type 4 or anything else unexpected)
    #             continue

    #     # Convert to numpy arrays
    #     left_epochs = np.array(left_epochs) if left_epochs else np.empty((0, window_length, raw.shape[1]))
    #     right_epochs = np.array(right_epochs) if right_epochs else np.empty((0, window_length, raw.shape[1]))
    #     rest_epochs = np.array(rest_epochs) if rest_epochs else np.empty((0, window_length, raw.shape[1]))

    #     # Store
    #     mod_data[subj]['EEG_filtered'][band]['epochs'] = {
    #         'left': left_epochs,
    #         'right': right_epochs,
    #         'rest': rest_epochs
    #     }

    #     # Print counts
    #     print(f"    Left epochs: {left_epochs.shape[0]}")
    #     print(f"    Right epochs: {right_epochs.shape[0]}")
    #     print(f"    Rest epochs: {rest_epochs.shape[0]}")


    # %% [markdown]
    # # Bandpass filter application to each subject 

    # %%
    # Apply bandpass filter to each subject's EEG data
    for subj in mod_data.keys():
        for band in mod_data[subj]['EEG_filtered'].keys():
            # Get the filtered EEG data
            filtered_eeg = mod_data[subj]['EEG_filtered'][band]['EEG_all']
            
            # Apply bandpass filter
            filtered_eeg = butter_bandpass_filter(filtered_eeg, lowcut, highcut, fs)
            
            # Store the filtered data back in the dictionary
            mod_data[subj]['EEG_filtered'][band]['EEG_all'] = filtered_eeg

    # %%
    # Create function that could bandpass filtered one subject
    def butter_bandpass_one_subject(data, subj, lowcut, highcut, fs, interval=None):
        print('Processing ', subj)
        
        # Create new key 'EEG_filtered' to store filtered EEG of each subject
        data[subj]['EEG_filtered'] = {}
        
        # Current raw EEG
        temp_raw_EEG = data[subj]['raw_EEG']
        
        if interval is not None:
            startband = np.arange(lowcut, highcut, step = interval)
            
            for start in startband:
                # This will be new key inside the EEG_filtered
                band = "{:02d}_{:02d}".format(start, start+interval)
                
                print('Filtering through {} Hz band'.format(band))
                # Bandpass filtering
                data[subj]['EEG_filtered'][band] = {}
                data[subj]['EEG_filtered'][band]['EEG_all'] = butter_bandpass_filter(temp_raw_EEG, start, start+interval, fs)
                
        else:
            # This will be new key inside the EEG_filtered
            band = "{:02d}_{:02d}".format(lowcut, highcut)
            
            data[subj]['EEG_filtered'][band]['EEG_all'] = butter_bandpass_filter(temp_raw_EEG, lowcut, highcut, fs)

    # %% [markdown]
    # # Automated FBCSP implementation and processing for all classifications

    # %%
    all_results = []

    for first_class_code, second_class_code, label1, label2 in classification_combinations:
        print(f"\n========== Running classification: {label1} vs {label2} ==========\n")

        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
        
            # total_length = mod_data[subj]['raw_EEG'].shape[0]

            # # Get all possible event positions
            # first_pos_all = get_valid_event_positions(
            #     ori_data[subj]['epos'], ori_data[subj]['etyp'],
            #     first_class_code, fs, start, end, total_length
            # )
            # second_pos_all = get_valid_event_positions(
            #     ori_data[subj]['epos'], ori_data[subj]['etyp'],
            #     second_class_code, fs, start, end, total_length
            # )
            
            first_pos_all = ori_data[subj]['epos'][np.isin(ori_data[subj]['etype'], first_class_code)]
            second_pos_all = ori_data[subj]['epos'][np.isin(ori_data[subj]['etype'], second_class_code)]
            
            # Select trials depending on test_type
            if test_type == "full":
                mod_data[subj]['first_pos'] = first_pos_all[0:]
                mod_data[subj]['second_pos'] = second_pos_all[0:]
            elif test_type == "trial":
                mod_data[subj]['first_pos'] = first_pos_all[id_trial_start:id_trial_end]
                mod_data[subj]['second_pos'] = second_pos_all[id_trial_start:id_trial_end]
            else:
                raise ValueError(f"Unknown test_type: {test_type}")

    # %%
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            print('Processing ', subj)

            temp_pos_first = mod_data[subj]['first_pos']
            temp_pos_second = mod_data[subj]['second_pos']

            for band in mod_data[subj]['EEG_filtered'].keys():
                temp_EEG_all = mod_data[subj]['EEG_filtered'][band]['EEG_all']
                temp_EEG_first = []
                temp_EEG_second = []

                # First class epochs
                for pos in temp_pos_first:
                    start_sample = pos + int(start * fs)
                    end_sample = pos + int(end * fs)
                    if end_sample > temp_EEG_all.shape[0]:
                        continue
                    epoch = temp_EEG_all[start_sample:end_sample, :]
                    temp_EEG_first.append(epoch.T)

                # Second class epochs
                for pos in temp_pos_second:
                    start_sample = pos + int(start * fs)
                    end_sample = pos + int(end * fs)
                    if end_sample > temp_EEG_all.shape[0]:
                        continue
                    epoch = temp_EEG_all[start_sample:end_sample, :]
                    temp_EEG_second.append(epoch.T)

                # Convert to arrays
                temp_EEG_first = np.array(temp_EEG_first)
                temp_EEG_second = np.array(temp_EEG_second)

                # Balance classes: use minimum number of trials, no rounding
                min_trials = min(len(temp_EEG_first), len(temp_EEG_second))
                idx_first = np.random.choice(len(temp_EEG_first), min_trials, replace=False)
                idx_second = np.random.choice(len(temp_EEG_second), min_trials, replace=False)

                temp_EEG_first_bal = temp_EEG_first[idx_first]
                temp_EEG_second_bal = temp_EEG_second[idx_second]

                # Store the balanced epochs
                mod_data[subj]['EEG_filtered'][band]['EEG_first'] = temp_EEG_first_bal
                mod_data[subj]['EEG_filtered'][band]['EEG_second'] = temp_EEG_second_bal

                # # Debug print
                # print(f'Band {band}:')
                # print(f'Balanced: {temp_EEG_first_bal.shape[0]} trials each')


        # %%
        # def split_EEG_one_class(EEG_one_class, percent_train=0.8):
        #     '''
        #     split_EEG_one_class will receive EEG data of one class, with size of T x N x M, where
        #     T = number of trial
        #     N = number of electrodes
        #     M = sample number
            
        #     INPUT:
        #     EEG_data_one_class: the data of one class of EEG data
            
        #     percent_train: allocation percentage of training data, default is 0.8
            
        #     OUTPUT:
        #     EEG_train: EEG data for training
            
        #     EEG_test: EEG data for test
            
        #     Both have type of np.arrray dimension of T x M x N
        #     '''

        #     # Number of all trials
        #     n = EEG_one_class.shape[0]
            
        #     n_tr = round(n*percent_train)
        #     n_te = n - n_tr
            
        #     EEG_train = EEG_one_class[:n_tr]
        #     EEG_test = EEG_one_class[n_tr:n_tr+n_te]
                
        #     return EEG_train, EEG_test

        # %%
        # Iterate over all subjects
        for i in range(sub_start, sub_end):
            
            subj = subject_counter(i)
            
            # Iterate over all bands
            for band in mod_data[subj]['EEG_filtered'].keys():
                
                # Temporary variable for each of each band
                temp_EEG_first = mod_data[subj]['EEG_filtered'][band]['EEG_first']
                temp_EEG_second = mod_data[subj]['EEG_filtered'][band]['EEG_second']

                # Temporary variable to access each band
                temp_filt = mod_data[subj]['EEG_filtered'][band]
                
                temp_filt['EEG_first_train'] = temp_EEG_first
                temp_filt['EEG_second_train'] = temp_EEG_second

        
        #%%%%%%%%% CSP %%%%%%%%%%%
        # This step will perform CSP on each band of each subject

        # For all subject create new keys to store all result in CSP step
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            mod_data[subj]['CSP'] = {}

    
        #%% Covariance and Composite Covariance %%

        def compute_cov(EEG_data):
            cov = []
            for i in range(EEG_data.shape[0]):
                cov.append(EEG_data[i] @ EEG_data[i].T / np.trace(EEG_data[i] @ EEG_data[i].T))
            cov = np.mean(np.array(cov), 0)

            return cov

        # Iterate over all subjects and bands
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            for band in mod_data[subj]['EEG_filtered'].keys():
                temp_band = mod_data[subj]['CSP'][band] = {}

                # Get EEG data for each class
                EEG_first = mod_data[subj]['EEG_filtered'][band].get('EEG_first_train')
                EEG_second = mod_data[subj]['EEG_filtered'][band].get('EEG_second_train')

                # Skip if EEG data is empty or invalid
                if EEG_first is None or EEG_first.size == 0 or EEG_first.ndim != 3:
                    print(f"[SKIP] {subj} - {band}: EEG_first_train is empty or invalid. Shape: {getattr(EEG_first, 'shape', None)}")
                    continue
                if EEG_second is None or EEG_second.size == 0 or EEG_second.ndim != 3:
                    print(f"[SKIP] {subj} - {band}: EEG_second_train is empty or invalid. Shape: {getattr(EEG_second, 'shape', None)}")
                    continue

                # Compute covariances for both classes
                temp_band['cov_first'] = compute_cov(EEG_first)
                temp_band['cov_second'] = compute_cov(EEG_second)
                # print(f"[DEBUG] Keys in {subj} - {band}: {mod_data[subj]['CSP'][band].keys()}")
                # Compute the composite covariance
                temp_band['cov_comp'] = temp_band['cov_first'] + temp_band['cov_second']

    
        #%%% White matrix %%%
        # Create new keys for result in whitening step
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                mod_data[subj]['CSP'][band]['whitening'] = {}

        
        from scipy.linalg import sqrtm
        from scipy.linalg import inv

        def decompose_cov(avg_cov):
            '''
            This function will decompose average covariance matrix of one class of each subject into 
            eigenvalues denoted by lambda and eigenvector denoted by V
            Both will be in descending order
            
            Parameter:
            avgCov = the averaged covariance of one class
            
            Return:
            λ_dsc and V_dsc, i.e. eigenvalues and eigenvector in descending order
            
            '''
            λ, V = np.linalg.eig(avg_cov)
            λ_dsc = np.sort(λ)[::-1] # Sort eigenvalue descending order, default is ascending order sort
            idx_dsc = np.argsort(λ)[::-1] # Find index in descending order
            V_dsc = V[:, idx_dsc] # Sort eigenvectors descending order
            λ_dsc = np.diag(λ_dsc) # Diagonalize λ_dsc
            
            return λ_dsc, V_dsc

        def white_matrix(λ_dsc, V_dsc):
            '''
            '''
            λ_dsc_sqr = sqrtm(inv(λ_dsc))
            P = (λ_dsc_sqr)@(V_dsc.T)
            
            return P


        # Iterate over all subject compute whitening matrix
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                
                temp_whitening = mod_data[subj]['CSP'][band]['whitening']
                temp_cov = mod_data[subj]['CSP'][band]['cov_comp']

                # Decomposing composite covariance into eigenvector and eigenvalue
                temp_whitening['eigval'], temp_whitening['eigvec'] = decompose_cov(temp_cov)

                # White matrix
                temp_whitening['P'] = white_matrix(temp_whitening['eigval'], temp_whitening['eigvec'])


        #%%%%%%%% Common Eigenvec from Sl and Sr %%%%%%%%%%%
        # In this step the Sl and Sr will not be stored, will only be used to compute each eigenvector


        # Create new keys for result in whitening step
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                mod_data[subj]['CSP'][band]['S_first'] = {}
                mod_data[subj]['CSP'][band]['S_second'] = {}  


        def compute_S(avg_Cov, white):
            '''      '''
            # Use float32 for memory efficiency
            S = np.dot(np.dot(white, avg_Cov), white.T)
            
            return S
        def decompose_S(S_one_class, order='descending'):
            '''
            This function will decompose the S matrix of one class to get the eigen vector
            Both eigenvector will be the same but in opposite order
            
            i.e the highest eigenvector in S first will be equal to lowest eigenvector in S second matrix 
            '''
            # Decompose S
            λ, B = np.linalg.eig(S_one_class)
            
            # Sort eigenvalues either descending or ascending
            if order == 'ascending':
                idx = λ.argsort() # Use this index to sort eigenvector smallest -> largest
            elif order == 'descending':
                idx = λ.argsort()[::-1] # Use this index to sort eigenvector largest -> smallest
            else:
                print('Wrong order input')
            
            λ = λ[idx]
            B = B[:, idx]
            
            return B, λ 


        # Iterate over all subjects
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                # Where to access data
                temp_P = mod_data[subj]['CSP'][band]['whitening']['P']
                Cl = mod_data[subj]['CSP'][band]['cov_first']
                Cr = mod_data[subj]['CSP'][band]['cov_second']

                # Where to store result
                temp_Sl = mod_data[subj]['CSP'][band]['S_first']
                temp_Sr = mod_data[subj]['CSP'][band]['S_second']

                # first
                Sl = compute_S(Cl, temp_P)
                temp_Sl['eigvec'], temp_Sl['eigval'] = decompose_S(Sl, 'descending')

                # second
                Sr = compute_S(Cr, temp_P)
                temp_Sr['eigvec'], temp_Sr['eigval'] = decompose_S(Sr, 'ascending')   


        #%%%%%%% Spatial Filter (W) %%%%%%%%%%
        # Will compute the spatial filter of each subject of each band

        
        def spatial_filter(B, P):
            '''
            Will compute projection matrix using the following equation:
            W = B' @ P
            
            INPUT:
            B: the eigenvector either first or second class, choose one !, size N x N, N is number of electrodes
            P: white matrix in size of N x N 
            
            OUTPUT:
            W spatial filter to filter EEG
            '''
            
            return (B.T@P)


        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                eigvec_first = mod_data[subj]['CSP'][band]['S_first']['eigvec']
                eigvec_second = mod_data[subj]['CSP'][band]['S_second']['eigvec']
                combined_eigvec = np.hstack((eigvec_first, eigvec_second))
                temp_P = mod_data[subj]['CSP'][band]['whitening']['P']

                mod_data[subj]['CSP'][band]['W'] = spatial_filter(combined_eigvec, temp_P)

        
        #%%%%% Feature Vector Train %%%%%%

        # Create new keys for trainning and test feature vector
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            mod_data[subj]['train'] = {}
            mod_data[subj]['test'] = {}
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                mod_data[subj]['train'][band] = {}
                mod_data[subj]['test'][band] = {}       


        def compute_Z(W, E, m):
            '''
            Will compute the Z
            Z = W @ E, 
            
            E is in the shape of N x M, N is number of electrodes, M is sample
            In application, E has nth trial, so there will be n numbers of Z
                # Use float32 for projection to reduce memory usage
                Z.append(np.dot(W.astype(np.float32), E[i].astype(np.float32)))
            Z, in each trial will have dimension of m x M, 
            where m is the first and last m rows of W, corresponds to smallest and largest eigenvalues
            '''
            Z = []
            
            W = np.delete(W, np.s_[m:-m:], 0)
            
            for i in range(E.shape[0]):
                Z.append(np.dot(W, E[i]))
            
            return np.array(Z)


        def feat_vector(Z):
            feat = []
            for i in range(Z.shape[0]):
                var = np.var(Z[i], ddof=1, axis=1)
                varsum = np.sum(var)
                
                # Add small constant to prevent log(0)
                feat.append(np.log10((var + 1e-10)/(varsum + 1e-10)))
                
            return np.array(feat)

        
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                temp_W = mod_data[subj]['CSP'][band]['W']
                temp_EEG_first = mod_data[subj]['EEG_filtered'][band]['EEG_first_train']
                temp_EEG_second = mod_data[subj]['EEG_filtered'][band]['EEG_second_train']

                # first
                mod_data[subj]['train'][band]['Z_first'] = compute_Z(temp_W, temp_EEG_first, m)
                mod_data[subj]['train'][band]['feat_first'] = feat_vector(mod_data[subj]['train'][band]['Z_first'])

                first_label = np.zeros([len(mod_data[subj]['train'][band]['feat_first']), 1])
                
                # second
                mod_data[subj]['train'][band]['Z_second'] = compute_Z(temp_W, temp_EEG_second, m)
                mod_data[subj]['train'][band]['feat_second'] = feat_vector(mod_data[subj]['train'][band]['Z_second'])
                
                second_label = np.ones([len(mod_data[subj]['train'][band]['feat_second']), 1])
                
                first  = np.c_[mod_data[subj]['train'][band]['feat_first'], first_label]
                second  = np.c_[mod_data[subj]['train'][band]['feat_second'], second_label] 
                
                mod_data[subj]['train'][band]['feat_train'] = np.vstack([first, second])
                
                np.random.shuffle(mod_data[subj]['train'][band]['feat_train'])

        #%%%%%%%%%%% Merging Train Feature of All Bandwidth %%%%%%%%%%
        # Merging all feature of each bandpass horizontally  
        # This will result in an array with shape of T x (9 * 2m) without true label  
        # where m is the number of filter, and T is number of trials

        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            # Check if subject exists in mod_data
            if subj not in mod_data:
                print(f"Subject {subj} not found in mod_data")
                continue  # Skip this subject if it doesn't exist in mod_data
            
            # Initialize train key if not present
            if 'train' not in mod_data[subj]:
                mod_data[subj]['train'] = {}
            
            feat_first_all = []
            feat_second_all = []
            
            for band in mod_data[subj]['EEG_filtered'].keys():
                # Access features for each band of the first class
                feat_first = mod_data[subj]['train'][band]['feat_first']
                feat_first_all.append(feat_first)
                
                # Access features for each band of the second class
                feat_second = mod_data[subj]['train'][band]['feat_second']
                feat_second_all.append(feat_second)
            
            # MERGING FEATURES FIRST CLASS
            merge_first = np.hstack(feat_first_all)  # Stack all arrays horizontally in one go
            merge_first = merge_first[:, 2*m:]  # Remove initial zeros (slice out the first 2*m columns)

            # MERGING FEATURES SECOND CLASS
            merge_second = np.hstack(feat_second_all)  # Stack all arrays horizontally in one go
            merge_second = merge_second[:, 2*m:]  # Remove initial zeros (slice out the first 2*m columns)

            # TRUE LABELS
            true_first = np.zeros([merge_first.shape[0], 1])
            true_second = np.ones([merge_second.shape[0], 1])

            # FEATURE + TRUE LABEL
            first = np.hstack([merge_first, true_first])
            second = np.hstack([merge_second, true_second])

            # MERGE CLASSES
            train_feat = np.vstack([first, second])

            # Shuffle the final training data
            np.random.shuffle(train_feat)

            # Check if 'all_band' exists in the 'train' key and store the result
            if 'all_band' not in mod_data[subj]['train']:
                print(f"Storing 'all_band' for subject {subj}")
                mod_data[subj]['train']['all_band'] = train_feat
            else:
                print(f"'all_band' already exists for subject {subj}. Overwriting data.")
                mod_data[subj]['train']['all_band'] = train_feat



        #%%%%%%%% Mutual Based Information (MI) to Select Most Informative Band %%%%%%%%


        from sklearn.feature_selection import mutual_info_classif
        from sklearn.feature_selection import SelectKBest


        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            X_train = mod_data[subj]['train']['all_band'][:, :-1]
            y_train = mod_data[subj]['train']['all_band'][:, -1]
            
            # New dictionary to store result
            mod_data[subj]['train']['mutual'] = {}
            
            # Use mutual information to find 4 most informative feature
            select = SelectKBest(mutual_info_classif, k = 12).fit(X_train, y_train)
            mod_data[subj]['train']['mutual']['X'] = X_train[:, select.get_support()]
            mod_data[subj]['train']['mutual']['y'] = y_train    
    
            

        #%%%%%%% Classification %%%%%%%

    
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score

        
        # Define model
        model = SVC(gamma='scale')

    
        #%%%%% Evaluate Model Performance on Train Data %%%%%%


        # Iterate over each subject
        eval_acc = []
        eval_std = []

        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            print('Processing for {}'.format(subj))
            X_train = mod_data[subj]['train']['mutual']['X']
            y_train = mod_data[subj]['train']['mutual']['y']
            
            
            eval_acc.append(cross_val_score(model, X_train, y_train, cv=5).mean()*100)
            eval_std.append(cross_val_score(model, X_train, y_train, cv=5).std()*100)

        
        # Necessary variable to label x axis
        subject=[]
        for i in range(1, ns):
            subject.append(subject_counter(i))

        
        # Print each subject accuracy
        print("====== Accuracy for all subjects ======")
        for i in range(sub_end - sub_start):
            print("Subject{:02d} : {:.2f} % +/- {:.2f}".format(i+sub_start, eval_acc[i], eval_std[i]))

        #%%%%% Result on Test Data %%%%%%%%
        # Based on initial observation, svc model performed well on train data. Thus we will proceed to use the current model to evaluate on test data 

        
        # Blank list to store accuracy values
        train_score = []
        test_score = []

        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            # Prepare train and test data
            data_train = mod_data[subj]['train']['mutual']
            X_train = data_train['X']
            y_train = data_train['y']
            
            # Training the model + train accuracy
            model.fit(X_train, y_train)
            tr_score = model.score(X_train, y_train)*100
            
            train_score.append(tr_score)

        # Conducting cross-validation on the entire dataset of a single subject, utilizing all frequency bands, for all the selected subjects

        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline

        # Perform cross-validation for each subject
        cv_results = {}

        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            # Combine train and test data
            #X = np.vstack([mod_data[subj]['train']['all_band'][:, :-1], mod_data[subj]['test']['all_band'][:, :-1]])
            #y = np.hstack([mod_data[subj]['train']['all_band'][:, -1], mod_data[subj]['test']['all_band'][:, -1]])
                    
            data_train = mod_data[subj]['train']['mutual']
            X = data_train['X']
            y = data_train['y']

            pipe = Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=12)),
                ('clf', SVC(gamma='scale'))
            ])
            scores = cross_val_score(pipe, X, y, cv=5)
            cv_results[subj] = {
                'mean_accuracy': scores.mean() * 100,
                'std_accuracy': scores.std() * 100
            }

            all_results.append({
                "Subject": subj,
                "Classification": f"{label1} vs {label2}",
                "Accuracy": scores.mean() * 100,
                "Std": scores.std() * 100
            })

        

        # Display results
        print(f"\n====== Cross-Validation Results - {label1} : {label2} ======")

        for subj, result in cv_results.items():
            print(f"{subj}: Mean Accuracy = {result['mean_accuracy']:.2f}%, Std = {result['std_accuracy']:.2f}%")


    # %%
    # Comparison Table FBCSP
    cv_results_table = {}

    for result in all_results:
        subject = result["Subject"]
        classification = result["Classification"]
        accuracy = result["Accuracy"]

        if subject not in cv_results_table:
            cv_results_table[subject] = {}
        cv_results_table[subject][classification] = accuracy

    # Convert to DataFrame
    cv_results_df = pd.DataFrame(cv_results_table).T

    # Print nicely
    print("\n========== FBCSP Cross-Validation Results ==========")
    print(cv_results_df)

    # Save to Excel
    fbcsp_lr_acc = cv_results_df.loc[subj, 'Left vs Right'] if 'Left vs Right' in cv_results_df.columns else None
    fbcsp_lrest_acc = cv_results_df.loc[subj, 'Left vs Rest'] if 'Left vs Rest' in cv_results_df.columns else None
    fbcsp_rrest_acc = cv_results_df.loc[subj, 'Right vs Rest'] if 'Right vs Rest' in cv_results_df.columns else None

    # Calculate and display overall mean accuracy and standard deviation

    if test_type == "full":
        filename = f"fbcsp_results_stroke_full_{sub_start}_{sub_end-1}.xlsx"
    elif test_type == "trial":
        filename = f"fbcsp_results_stroke_trial_{id_trial_start}_{id_trial_end}.xlsx"
    else:
        raise ValueError("Invalid test_type. It must be 'full' or 'trial'.")




    # %% [markdown]
    # # CSP - Only Processing
    # Selected band is 8-30 Hz, since it covers both μ and β frequency band range, essential for the analysis

    # %%
    print("\n========== Running CSP-Only Processing ==========\n")

    # Create a copy of mod_data for CSP-only processing
    csp_data = dict()
    for i in range(sub_start, sub_end):
        subj = subject_counter(i)
        csp_data[subj] = mod_data[subj].copy()
        csp_data[subj]['raw_EEG'] = mod_data[subj]['raw_EEG'].copy()

    # Bandpass filter all subjects for CSP-only (8-30 Hz)
    lowcut = 8
    highcut = 30

    # Clear existing filtered data
    for i in range(sub_start, sub_end):
        subj = subject_counter(i)
        csp_data[subj]['EEG_filtered'] = {}

    # Apply single bandpass filter
    for i in range(sub_start, sub_end):
        subj = subject_counter(i)
        print(f'Processing CSP-only for {subj}')
        
        # Create new key 'EEG_filtered' to store filtered EEG
        csp_data[subj]['EEG_filtered']['08_30'] = {}
        
        # Bandpass filtering (8-30 Hz)
        csp_data[subj]['EEG_filtered']['08_30']['EEG_all'] = butter_bandpass_filter(
            csp_data[subj]['raw_EEG'], lowcut, highcut, fs
        )


    for first_class_code, second_class_code, label1, label2 in classification_combinations:
        print(f"\n========== Running CSP-only classification: {label1} vs {label2} ==========\n")

        # Update event positions for current classification
        for i in range(sub_start, sub_end):
            #subj = subject_counter(i)
            #csp_data[subj]['first_pos'] = ori_data[subj]['epos'][np.isin(ori_data[subj]['etyp'], first_class_code)]
            #csp_data[subj]['second_pos'] = ori_data[subj]['epos'][np.isin(ori_data[subj]['etyp'], second_class_code)]

                    
            first_pos_all = ori_data[subj]['epos'][np.isin(ori_data[subj]['etype'], first_class_code)]
            second_pos_all = ori_data[subj]['epos'][np.isin(ori_data[subj]['etype'], second_class_code)]
            
            if test_type == "full":
                csp_data[subj]['first_pos'] = first_pos_all[0:]  
                csp_data[subj]['second_pos'] = second_pos_all[0:]
            elif test_type == "trial":
                csp_data[subj]['first_pos'] = first_pos_all[id_trial_start:id_trial_end]
                csp_data[subj]['second_pos'] = second_pos_all[id_trial_start:id_trial_end]
            else:
                raise ValueError(f"Unknown test_type: {test_type}")


        # Extract EEG segments for selected classes
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            print('Processing ', subj)

            temp_pos_first = csp_data[subj]['first_pos']
            temp_pos_second = csp_data[subj]['second_pos']
        
            band = '08_30'  # Only one band for CSP-only, based on mi and beta bands
            temp_EEG_all = csp_data[subj]['EEG_filtered'][band]['EEG_all']
            temp_EEG_first = []
            temp_EEG_second = []
            
            # first class
            window_samples = int((end - start) * fs)  # Calculate window length in samples

            for pos in temp_pos_first:
                start_sample = pos + int(start * fs)
                end_sample = pos + int(end * fs)
                
                # Extract window (samples × channels)
                epoch = temp_EEG_all[start_sample:end_sample, :]  
                
                # Transpose to (channels × samples) for CSP
                temp_EEG_first.append(epoch.T)  

            for pos in temp_pos_second:
                start_sample = pos + int(start * fs)
                end_sample = pos + int(end * fs)
                epoch = temp_EEG_all[start_sample:end_sample, :]
                temp_EEG_second.append(epoch.T)

            # Keep only epochs with correct shape (channels × samples)
            expected_shape = (temp_EEG_first[0].shape[0], window_samples)

            temp_EEG_first = [e for e in temp_EEG_first if e.shape == expected_shape]
            temp_EEG_second = [e for e in temp_EEG_second if e.shape == expected_shape]

            # Convert to NumPy arrays
            temp_EEG_first = np.array(temp_EEG_first)
            temp_EEG_second = np.array(temp_EEG_second)

            # Balance the classes
            min_trials = min(len(temp_EEG_first), len(temp_EEG_second))
            if min_trials == 0:
                print(f" Skipping {subj} due to zero-length class.")
                continue

            idx_first = np.random.choice(len(temp_EEG_first), min_trials, replace=False)
            idx_second = np.random.choice(len(temp_EEG_second), min_trials, replace=False)

            temp_EEG_first_bal = temp_EEG_first[idx_first]
            temp_EEG_second_bal = temp_EEG_second[idx_second]

            # Store balanced epochs
            csp_data[subj]['EEG_filtered'][band]['EEG_first'] = temp_EEG_first_bal
            csp_data[subj]['EEG_filtered'][band]['EEG_second'] = temp_EEG_second_bal
                
        # Train/Test split
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            band = '08_30'
            
            temp_EEG_first = csp_data[subj]['EEG_filtered'][band]['EEG_first']
            temp_EEG_second = csp_data[subj]['EEG_filtered'][band]['EEG_second']
            
            csp_data[subj]['EEG_filtered'][band]['EEG_first_train'] = temp_EEG_first
            csp_data[subj]['EEG_filtered'][band]['EEG_second_train'] = temp_EEG_second

        # Initialize CSP storage
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            csp_data[subj]['CSP'] = {}
            csp_data[subj]['CSP'][band] = {}

        # Compute covariance matrices
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            band = '08_30'
            
            # Compute classes covariance
            csp_data[subj]['CSP'][band]['cov_first'] = compute_cov(csp_data[subj]['EEG_filtered'][band]['EEG_first_train'])
            csp_data[subj]['CSP'][band]['cov_second'] = compute_cov(csp_data[subj]['EEG_filtered'][band]['EEG_second_train'])
            csp_data[subj]['CSP'][band]['cov_comp'] = csp_data[subj]['CSP'][band]['cov_first'] + csp_data[subj]['CSP'][band]['cov_second']

        # Whitening
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            band = '08_30'
            
            csp_data[subj]['CSP'][band]['whitening'] = {}
            temp_cov = csp_data[subj]['CSP'][band]['cov_comp']
            
            # Decomposing composite covariance
            csp_data[subj]['CSP'][band]['whitening']['eigval'], csp_data[subj]['CSP'][band]['whitening']['eigvec'] = decompose_cov(temp_cov)
            
            # White matrix
            csp_data[subj]['CSP'][band]['whitening']['P'] = white_matrix(
                csp_data[subj]['CSP'][band]['whitening']['eigval'], 
                csp_data[subj]['CSP'][band]['whitening']['eigvec']
            )

        # Compute S matrices
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            band = '08_30'
            
            csp_data[subj]['CSP'][band]['S_first'] = {}
            csp_data[subj]['CSP'][band]['S_second'] = {}  
            
            temp_P = csp_data[subj]['CSP'][band]['whitening']['P']
            Cl = csp_data[subj]['CSP'][band]['cov_first']
            Cr = csp_data[subj]['CSP'][band]['cov_second']

            # first class
            Sl = compute_S(Cl, temp_P)
            csp_data[subj]['CSP'][band]['S_first']['eigvec'], csp_data[subj]['CSP'][band]['S_first']['eigval'] = decompose_S(Sl, 'descending')

            # second class
            Sr = compute_S(Cr, temp_P)
            csp_data[subj]['CSP'][band]['S_second']['eigvec'], csp_data[subj]['CSP'][band]['S_second']['eigval'] = decompose_S(Sr, 'ascending')

        # Spatial filters
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            band = '08_30'
            
            temp_eigvec = csp_data[subj]['CSP'][band]['S_first']['eigvec']
            temp_P = csp_data[subj]['CSP'][band]['whitening']['P']
            
            csp_data[subj]['CSP'][band]['W'] = spatial_filter(temp_eigvec, temp_P)

        # Feature extraction for training data
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            csp_data[subj]['train'] = {}
            band = '08_30'
            
            temp_W = csp_data[subj]['CSP'][band]['W']
            temp_EEG_first = csp_data[subj]['EEG_filtered'][band]['EEG_first_train']
            temp_EEG_second = csp_data[subj]['EEG_filtered'][band]['EEG_second_train']

            # first class
            csp_data[subj]['train']['Z_first'] = compute_Z(temp_W, temp_EEG_first, m)
            csp_data[subj]['train']['feat_first'] = feat_vector(csp_data[subj]['train']['Z_first'])
            first_label = np.zeros([len(csp_data[subj]['train']['feat_first']), 1])
            
            # second class
            csp_data[subj]['train']['Z_second'] = compute_Z(temp_W, temp_EEG_second, m)
            csp_data[subj]['train']['feat_second'] = feat_vector(csp_data[subj]['train']['Z_second'])
            second_label = np.ones([len(csp_data[subj]['train']['feat_second']), 1])
            
            # Combine features and labels
            first = np.c_[csp_data[subj]['train']['feat_first'], first_label]
            second = np.c_[csp_data[subj]['train']['feat_second'], second_label]
            csp_data[subj]['train']['feat_train'] = np.vstack([first, second])
            np.random.shuffle(csp_data[subj]['train']['feat_train'])

        from sklearn.pipeline import Pipeline

        # Cross-validation for CSP-only
        cv_results_csp = {}
        
        for i in range(sub_start, sub_end):
            subj = subject_counter(i)
            
            # Combine train and test data for cross-validation
            X = np.vstack([
                csp_data[subj]['train']['feat_train'][:, :-1],
            ])
            y = np.hstack([
                csp_data[subj]['train']['feat_train'][:, -1],
            ])
            
            pipe = Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=12)),
                ('clf', SVC(gamma='scale'))
            ])
            # Perform cross-validation
            scores = cross_val_score(pipe, X, y, cv=5)
            cv_results_csp[subj] = {
                'mean_accuracy': scores.mean() * 100,
                'std_accuracy': scores.std() * 100
            }
            
            # Store results for comparison
            all_results_csp.append({
                "Subject": subj,
                "Classification": f"{label1} vs {label2}",
                "Accuracy": scores.mean() * 100,
                "Std": scores.std() * 100
            })

        # Display results for current classification
        print(f"\n====== CSP-Only Cross-Validation Results - {label1} vs {label2} ======")
        for subj, result in cv_results_csp.items():
            print(f"{subj}: Mean Accuracy = {result['mean_accuracy']:.2f}%, Std = {result['std_accuracy']:.2f}%")

        # Pivot CSP results to get a table of accuracy per subject per classification
        csp_results_df = pd.DataFrame(all_results_csp)
        csp_results_pivot = csp_results_df.pivot(index='Subject', columns='Classification', values='Accuracy')
        print("\n========== CSP Cross-Validation Results ==========")
        print(csp_results_pivot)
        csp_lr_acc = csp_results_pivot.loc[subj, 'Left vs Right'] if 'Left vs Right' in csp_results_pivot.columns else None
        csp_lrest_acc = csp_results_pivot.loc[subj, 'Left vs Rest'] if 'Left vs Rest' in csp_results_pivot.columns else None
        csp_rrest_acc = csp_results_pivot.loc[subj, 'Right vs Rest'] if 'Right vs Rest' in csp_results_pivot.columns else None

    return {
        'FBCSP_Left_vs_Right': fbcsp_lr_acc,
        'FBCSP_Left_vs_Rest': fbcsp_lrest_acc,
        'FBCSP_Right_vs_Rest': fbcsp_rrest_acc,
        'CSP_Left_vs_Right': csp_lr_acc,
        'CSP_Left_vs_Rest': csp_lrest_acc,
        'CSP_Right_vs_Rest': csp_rrest_acc
    }


# %%  Collect all results into one table
all_results = []

for subj_num in range(1, 10):  # For all 50 subjects
    print(f"\n=== Processing subject {subj_num:02d} ===")
    try:
        results = process_single_subject(
            subj_num=subj_num,
            test_type="full",
            id_trial_start=20,
            id_trial_end=30
        )
        results['Subject'] = f'sub-{subj_num:02d}'
        all_results.append(results)
    except Exception as e:
        print(f"Error processing subject {subj_num}: {str(e)}")
        continue

# Convert to DataFrame and save
if all_results:
    results_df = pd.DataFrame(all_results)
    cols = ['Subject'] + [c for c in results_df.columns if c != 'Subject']
    results_df = results_df[cols]

    # Build MultiIndex for columns
    new_columns = []
    for col in results_df.columns:
        if col == 'Subject':
            new_columns.append(('','Subject'))
        elif col.startswith('FBCSP_'):
            new_columns.append(('FBCSP', col.replace('FBCSP_', '')))
        elif col.startswith('CSP_'):
            new_columns.append(('CSP', col.replace('CSP_', '')))
        else:
            new_columns.append(('', col))
    results_df.columns = pd.MultiIndex.from_tuples(new_columns)

    results_df.to_excel("subject_results_BCI_IV2a_data.xlsx")
    print("\nSuccessfully saved results for all subjects")
else:
    print("No results were generated")