# Optimized EEGNet Pipeline with Reduced Epochs and Enhanced Statistics

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import os

np.random.seed(13)

# Configuration parameters

sub_start, sub_end = 1, 51 # Add more if training with GPU 
start, end = 0.5, 2  # Time window in seconds
fs = 500  # Sampling rate
epochs = 60  # Reduced from 100 to 60
batch_size = 16

# Classification combinations
classification_combinations = [
    ([1], [2], 'Left', 'Right'),
    ([1], [3], 'Left', 'Rest'),
    ([2], [3], 'Right', 'Rest'),
    ([1, 2], [3], 'Movement', 'Rest')
]

def subject_counter(i):
    return f'sub-{i:02d}'

def load_subject_data(subj, base_dir):
    data_path = os.path.join(base_dir, 'sourcedata', subj, f'{subj}_task-motor-imagery_eeg.mat')
    try:
        mat_data = loadmat(data_path)
        eeg_struct = mat_data['eeg']
        return {
            'rawdata': eeg_struct['rawdata'][0,0],
            'labels': eeg_struct['label'][0,0]
        }
    except Exception as e:
        print(f"Error loading {subj}: {str(e)}")
        return None

# # Load and preprocess data
# ori_data = {}
# for i in range(sub_start, sub_end + 1):
#     subj = subject_counter(i)
#     loaded_data = load_subject_data(subj, os.getcwd())
#     if loaded_data:
#         ori_data[subj] = {
#             'rawdata': loaded_data['rawdata'],
#             'labels': loaded_data['labels'],
#             'etyp': None,
#             'epos': None
#         }
#         # Create event markers
#         trial_length = 8 * fs
#         events = []
#         for trial_idx in range(40):
#             events.extend([
#                 [trial_idx * trial_length, 0, 1],
#                 [trial_idx * trial_length + 2*fs, 0, ori_data[subj]['labels'][trial_idx][0]],
#                 [trial_idx * trial_length + 6*fs, 0, 3]
#             ])
#         events = np.array(events)
#         ori_data[subj]['etyp'] = events[:, 2]
#         ori_data[subj]['epos'] = events[:, 0]
        
                
# Load and preprocess data
ori_data = {}
for i in range(sub_start, sub_end):
    subj = subject_counter(i)
    loaded_data = load_subject_data(subj, os.getcwd())
    
    if loaded_data is None:
        continue
        
    # Store in our data structure
    ori_data[subj] = {
        'rawdata': loaded_data['rawdata'],
        'labels': loaded_data['labels'],
        's': loaded_data['rawdata'],  # EEG data (trials×channels×samples)
        'etyp': None,                 # Will be filled below
        'epos': None                  # Will be filled below
    }
    
    # Extract events from channel 33 (index 32 in 0-based Python)
    raw = ori_data[subj]['rawdata']
    labels = ori_data[subj]['labels']
    
    epos_list = []
    etyp_list = []
    
    for trial_idx in range(raw.shape[0]):
        event_channel = raw[trial_idx, 32, :]  # Channel 33
        
        # Get the movement imagination label for this trial (1 or 2)
        mi_label = labels[trial_idx][0]
        
        for val in [1, 2, 3]:
            event_positions = np.where(event_channel == val)[0]
            for pos in event_positions:
                global_pos = trial_idx * 8 * fs + pos  # Convert to global sample index
                epos_list.append(global_pos)
                
                # Set event type based on the event value and label
                if val == 2:  # Movement cue
                    if mi_label == 1:
                        etyp_list.append(1)  # Left-hand movement
                    elif mi_label == 2:
                        etyp_list.append(2)  # Right-hand movement
                elif val == 3:
                    etyp_list.append(3)  # Break
                elif val == 1:
                    etyp_list.append(4)  # Instruction

    ori_data[subj]['epos'] = np.array(epos_list)
    ori_data[subj]['etyp'] = np.array(etyp_list)        
        
        

# Bandpass filter (4-40 Hz)
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut/nyq, highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal, axis=0)

mod_data = {}
for i in range(sub_start, sub_end + 1):
    subj = subject_counter(i)
    if subj in ori_data:
        eeg_data = ori_data[subj]['rawdata'][:, :30, :]  # Keep only EEG channels
        continuous_eeg = eeg_data.transpose(1, 0, 2).reshape(30, -1)
        mod_data[subj] = {
            'raw_EEG': continuous_eeg.T,
            'fs': fs,
            'etyp': ori_data[subj]['etyp'],
            'epos': ori_data[subj]['epos'],
            'EEG_filtered': butter_bandpass_filter(continuous_eeg.T, 4, 40, fs)
        }

def extract_epochs_for_eegnet(data_dict, subj, class_codes, start, end, fs):
    window_samples = int((end - start) * fs)
    positions = data_dict[subj]['epos'][np.isin(data_dict[subj]['etyp'], class_codes)]
    epochs = []
    for pos in positions:
        start_sample = pos + int(start * fs)
        end_sample = pos + int(end * fs)
        if end_sample <= data_dict[subj]['EEG_filtered'].shape[0]:
            epoch = data_dict[subj]['EEG_filtered'][start_sample:end_sample, :].T
            epochs.append(epoch)
    if epochs:
        return np.expand_dims(np.array(epochs), axis=-1)  # trials × channels × samples × 1
    return None

# Main processing loop
results = []
all_accuracies = []

for class1_code, class2_code, label1, label2 in classification_combinations:
    print(f"\n=== Processing {label1} vs {label2} ===")
    class_accuracies = []
    
    for i in range(sub_start, sub_end + 1):
        subj = subject_counter(i)
        if subj not in mod_data:
            continue
            
        X1 = extract_epochs_for_eegnet(mod_data, subj, class1_code, start, end, fs)
        X2 = extract_epochs_for_eegnet(mod_data, subj, class2_code, start, end, fs)
        
        if X1 is None or X2 is None:
            print(f"Skipping {subj} - insufficient trials")
            continue
            
        # Prepare data
        X = np.concatenate((X1, X2))
        y = np.concatenate((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))
        
        # Split and shuffle
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_train_cat, y_test_cat = to_categorical(y_train), to_categorical(y_test)
        
        # Initialize and train EEGNet
        model = EEGNet(
            nb_classes=2, Chans=X_train.shape[1], Samples=X_train.shape[2],
            dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        checkpoint_path = f'checkpoints/{subj}_{label1}_vs_{label2}.h5'
        history = model.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test_cat),
            callbacks=[ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)],
            verbose=0
        )
        
        # Evaluate
        model.load_weights(checkpoint_path)
        acc = accuracy_score(y_test, model.predict(X_test).argmax(axis=-1))
        class_accuracies.append(acc)
        all_accuracies.append(acc)
        print(f"{subj}: Accuracy = {acc:.3f}")
    
    # Calculate statistics for this classification
    mean_acc = np.mean(class_accuracies)
    std_acc = np.std(class_accuracies)
    results.append({
        'Classification': f"{label1} vs {label2}",
        'Mean Accuracy': mean_acc,
        'Std Accuracy': std_acc,
        'Number of Subjects': len(class_accuracies),
        'Subject Accuracies': class_accuracies
    })

# Calculate overall statistics
overall_mean = np.mean(all_accuracies)
overall_std = np.std(all_accuracies)

# Print results
print("\n=== Classification Results ===")
results_df = pd.DataFrame([{
    'Classification': r['Classification'],
    'Mean Accuracy': r['Mean Accuracy'],
    'Std Accuracy': r['Std Accuracy'],
    'Number of Subjects': r['Number of Subjects']
} for r in results])

print(results_df.to_string(index=False))
print(f"\nOverall Mean Accuracy: {overall_mean:.3f} ± {overall_std:.3f}")

# Save detailed results
detailed_results = []
for r in results:
    for i, acc in enumerate(r['Subject Accuracies']):
        detailed_results.append({
            'Classification': r['Classification'],
            'Subject': f"sub-{i+1:02d}",
            'Accuracy': acc
        })

pd.DataFrame(detailed_results).to_csv('subject_level_results_clinical.csv', index=False)
results_df.to_csv('classification_summary_clinical.csv', index=False)
print("\nResults saved to 'subject_level_results.csv' and 'classification_summary.csv'")

subject_results = {}

for r in results:
    classification = r['Classification']
    for i, acc in enumerate(r['Subject Accuracies']):
        subject_id = subject_counter(i + sub_start)

        # Initialize subject row if it doesn't exist
        if subject_id not in subject_results:
            subject_results[subject_id] = {'Subject': subject_id}

        # Assign the accuracy to the corresponding classification column
        subject_results[subject_id][classification] = acc * 100

# Convert to DataFrame
df = pd.DataFrame.from_dict(subject_results, orient='index')

# Save to Excel
df.to_excel('subject_level_results_clinical.xlsx', index=False)