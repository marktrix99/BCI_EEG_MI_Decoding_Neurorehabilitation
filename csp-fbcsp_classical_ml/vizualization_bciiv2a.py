import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Extract features for all 9 subjects ===
band = '08_30'
all_subjects_movement = []
all_subjects_rest = []
subject_names = []

for i in range(1, 10):  # 9 subjects
    subj = f'subject0{i}'
    
    # Check if subject exists in csp_data
    if subj in csp_data and 'train' in csp_data[subj]:
        movement_feat = csp_data[subj]['train']['feat_first']
        rest_feat = csp_data[subj]['train']['feat_second']
        
        # Calculate mean across trials for each filter
        movement_mean = np.mean(movement_feat, axis=0)
        rest_mean = np.mean(rest_feat, axis=0)
        
        all_subjects_movement.append(movement_mean)
        all_subjects_rest.append(rest_mean)
        subject_names.append(subj)
        
        print(f"{subj}: Movement shape {movement_feat.shape}, Rest shape {rest_feat.shape}")

# Convert to arrays
all_subjects_movement = np.array(all_subjects_movement)  # shape: (n_subjects, n_filters)
all_subjects_rest = np.array(all_subjects_rest)

print(f"\nTotal subjects processed: {len(subject_names)}")
print(f"Feature matrix shape: {all_subjects_movement.shape}")

# === 1. Bar plot: Mean across all subjects for each filter ===
grand_mean_movement = np.mean(all_subjects_movement, axis=0)
grand_mean_rest = np.mean(all_subjects_rest, axis=0)
grand_std_movement = np.std(all_subjects_movement, axis=0)
grand_std_rest = np.std(all_subjects_rest, axis=0)

plt.figure(figsize=(12, 6))
x_pos = np.arange(len(grand_mean_movement))

plt.bar(x_pos - 0.2, grand_mean_movement, width=0.4,
        yerr=grand_std_movement, label="Movement", 
        color="steelblue", alpha=0.8, edgecolor='black', 
        linewidth=1.5, capsize=5)
plt.bar(x_pos + 0.2, grand_mean_rest, width=0.4,
        yerr=grand_std_rest, label="Rest", 
        color="coral", alpha=0.8, edgecolor='black', 
        linewidth=1.5, capsize=5)

plt.xlabel("Filter Index", fontsize=13, fontweight='bold')
plt.ylabel("Mean Log-normalized Variance", fontsize=13, fontweight='bold')
plt.title("CSP Features Across All 9 Subjects (8-30 Hz)", 
          fontsize=15, fontweight='bold')
plt.xticks(x_pos, fontsize=11)
plt.legend(fontsize=12, frameon=True, shadow=True)
plt.grid(axis='y', alpha=0.4, linestyle='--')
plt.tight_layout()
plt.show()

# === 2. Heatmap: All subjects × all filters ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Movement class
im1 = ax1.imshow(all_subjects_movement, aspect='auto', cmap='RdBu_r', 
                 interpolation='nearest')
ax1.set_xlabel("Filter Index", fontsize=12, fontweight='bold')
ax1.set_ylabel("Subject", fontsize=12, fontweight='bold')
ax1.set_title("Movement Class - Log-variance Features", 
              fontsize=13, fontweight='bold')
ax1.set_yticks(range(len(subject_names)))
ax1.set_yticklabels([s.replace('subject0', 'S') for s in subject_names])
ax1.set_xticks(range(all_subjects_movement.shape[1]))
plt.colorbar(im1, ax=ax1, label='Log-variance')

# Rest class
im2 = ax2.imshow(all_subjects_rest, aspect='auto', cmap='RdBu_r', 
                 interpolation='nearest')
ax2.set_xlabel("Filter Index", fontsize=12, fontweight='bold')
ax2.set_ylabel("Subject", fontsize=12, fontweight='bold')
ax2.set_title("Rest Class - Log-variance Features", 
              fontsize=13, fontweight='bold')
ax2.set_yticks(range(len(subject_names)))
ax2.set_yticklabels([s.replace('subject0', 'S') for s in subject_names])
ax2.set_xticks(range(all_subjects_rest.shape[1]))
plt.colorbar(im2, ax=ax2, label='Log-variance')

plt.tight_layout()
plt.show()

# === 3. Difference plot: Discriminative power ===
difference_per_subject = all_subjects_movement - all_subjects_rest
mean_difference = np.mean(difference_per_subject, axis=0)
std_difference = np.std(difference_per_subject, axis=0)

plt.figure(figsize=(12, 6))
colors = ['red' if d > 0 else 'blue' for d in mean_difference]
plt.bar(x_pos, mean_difference, width=0.6, yerr=std_difference,
        color=colors, alpha=0.7, edgecolor='black', 
        linewidth=1.5, capsize=5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=2)
plt.xlabel("Filter Index", fontsize=13, fontweight='bold')
plt.ylabel("Difference (Movement - Rest)", fontsize=13, fontweight='bold')
plt.title("Discriminative Power of Each Filter (All 9 Subjects)", 
          fontsize=15, fontweight='bold')
plt.xticks(x_pos, fontsize=11)
plt.grid(axis='y', alpha=0.4, linestyle='--')
plt.tight_layout()
plt.show()

# === 4. Subject-wise comparison (grouped bar plot) ===
n_filters = all_subjects_movement.shape[1]
n_subjects = len(subject_names)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, subj in enumerate(subject_names):
    ax = axes[idx]
    movement_vals = all_subjects_movement[idx]
    rest_vals = all_subjects_rest[idx]
    
    x = np.arange(n_filters)
    ax.bar(x - 0.2, movement_vals, width=0.4, label='Movement', 
           color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, rest_vals, width=0.4, label='Rest', 
           color='coral', alpha=0.8)
    
    ax.set_title(subj.replace('subject0', 'Subject '), 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Filter Index', fontsize=9)
    ax.set_ylabel('Log-variance', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(x)

plt.suptitle('CSP Features per Subject (8-30 Hz)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# === 5. Statistical summary ===
print("\n" + "="*60)
print("STATISTICAL SUMMARY ACROSS ALL SUBJECTS")
print("="*60)
print(f"\nMovement Class:")
print(f"  Mean (across subjects): {grand_mean_movement}")
print(f"  Std (across subjects):  {grand_std_movement}")
print(f"\nRest Class:")
print(f"  Mean (across subjects): {grand_mean_rest}")
print(f"  Std (across subjects):  {grand_std_rest}")
print(f"\nDifference (Movement - Rest):")
print(f"  Mean: {mean_difference}")
print(f"  Std:  {std_difference}")
print(f"\nMost discriminative filters (by absolute difference):")
most_discriminative = np.argsort(np.abs(mean_difference))[::-1]
for i, filter_idx in enumerate(most_discriminative[:5]):
    print(f"  {i+1}. Filter {filter_idx}: {mean_difference[filter_idx]:.4f}")