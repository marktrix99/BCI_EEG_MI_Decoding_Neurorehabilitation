import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from EEGModels import EEGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
import os

np.random.seed(13)

# Configuration parameters
sub_start, sub_end = 1, 9  # BCI IV 2a has 9 subjects
start, end = 0.5, 2  # Time window in seconds
fs = 250  # Sampling rate for BCI IV 2a
epochs = 60  # Reduced from 100 to 60
batch_size = 16

# Event codes for BCI IV 2a (from npz files)
event_codes = {
    769: 'left',
    770: 'right',
    771: 'foot',
    772: 'tongue',
    768: 'rest'
}

# Classification combinations
classification_combinations = [
    ([769], [770], 'Left', 'Right'),
    ([769], [768], 'Left', 'Rest'),
    ([770], [768], 'Right', 'Rest'),
    ([769, 770, 771, 772], [768], 'Movement', 'Rest')
]

def subject_counter(i):
    return f'subject{i:02d}'

def load_subject_data(subj, base_dir):
    """Load BCI IV 2a dataset from npz files"""
    try:
        data_path = os.path.join(base_dir, 'BCI_IVa_datasets', f'A{subj:02d}T.npz')
        data = np.load(data_path)
        
        # Extract EEG data (first 22 channels)
        eeg_data = data['s'][:, :22]  # Shape: (samples, channels)
        
        # Get events
        events = data['etyp']
        event_positions = data['epos']
        
        return {
            'eeg': eeg_data,
            'events': events,
            'event_positions': event_positions
        }
    except Exception as e:
        print(f"Error loading {subj}: {str(e)}")
        return None

# Bandpass filter (4-40 Hz)
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut/nyq, highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal, axis=0)

# Load and preprocess data
subjects_data = {}
for i in range(sub_start, sub_end + 1):
    subj = subject_counter(i)
    data = load_subject_data(i, os.getcwd())
    if data:
        # Apply bandpass filter
        filtered_eeg = butter_bandpass_filter(data['eeg'], 4, 40, fs)
        subjects_data[subj] = {
            'eeg': filtered_eeg,
            'events': data['events'],
            'event_positions': data['event_positions']
        }

def extract_epochs(eeg_data, events, event_positions, class_codes, start, end, fs):
    """Extract epochs from npz data for EEGNet"""
    # Find matching events
    matching_indices = np.where(np.isin(events, class_codes))[0]
    
    if len(matching_indices) == 0:
        return None
    
    # Calculate sample positions
    start_sample = int(start * fs)
    end_sample = int(end * fs)
    window_length = end_sample - start_sample
    
    # Extract epochs
    epochs = []
    for idx in matching_indices:
        pos = event_positions[idx]
        epoch_start = pos + start_sample
        epoch_end = pos + end_sample
        
        # Check if epoch is within bounds
        if epoch_end <= eeg_data.shape[0]:
            #epoch = eeg_data[epoch_start:epoch_end, :].T  # Transpose to (channels, samples)
            epoch = eeg_data[int(epoch_start):int(epoch_end), :].T
            epochs.append(epoch)
    
    if len(epochs) == 0:
        return None
    
    # Convert to numpy array and reshape for EEGNet (trials × channels × samples × 1)
    return np.expand_dims(np.array(epochs), axis=-1)

# Main processing loop
results = []
all_accuracies = []

for class1_code, class2_code, label1, label2 in classification_combinations:
    print(f"\n=== Processing {label1} vs {label2} ===")
    class_accuracies = []
    
    for i in range(sub_start, sub_end + 1):
        subj = subject_counter(i)
        if subj not in subjects_data:
            continue
            
        data = subjects_data[subj]
        
        X1 = extract_epochs(data['eeg'], data['events'], data['event_positions'], 
                          class1_code, start, end, fs)
        X2 = extract_epochs(data['eeg'], data['events'], data['event_positions'],
                          class2_code, start, end, fs)
        
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
            dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        checkpoint_path = f'checkpoints/{subj}_{label1}_vs_{label2}.h5'
        os.makedirs('checkpoints', exist_ok=True)
        
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
    if class_accuracies:
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
if all_accuracies:
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
    df.to_excel('subject_level_results.xlsx', index=False)

    # Save detailed results
    detailed_results = []
    for r in results:
        for i, acc in enumerate(r['Subject Accuracies']):
            detailed_results.append({
                'Classification': r['Classification'],
                'Subject': subject_counter(i+sub_start),
                'Accuracy': acc
            })

    pd.DataFrame(detailed_results).to_excel('subject_level_results_M.xlsx', index=True)
    results_df.to_excel('classification_summary_M.xlsx', index=True)
    print("\nResults saved to 'subject_level_results.csv' and 'classification_summary.csv'")
    