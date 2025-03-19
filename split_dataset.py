import os
import shutil

# Define dataset paths
DATASET_PATH = "/content/casia-b/output"
TRAIN_PATH = "/content/casia-b/train/output"
TEST_PATH = "/content/casia-b/test/output"

# Create train and test directories
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

# Define subjects (001-124)
subjects = sorted(os.listdir(DATASET_PATH))

# Split: First 74 for training, last 50 for testing
train_subjects = subjects[:74]
test_subjects = subjects[74:]

def copy_subject_folders(subject_list, dest_path):
    for subject in subject_list:
        subject_source = os.path.join(DATASET_PATH, subject)
        subject_dest = os.path.join(dest_path, subject)

        # Ensure the subject folder exists
        os.makedirs(subject_dest, exist_ok=True)

        # Copy subfolders (bg-01, cl-01, nm-01, etc.) and their images
        for condition in os.listdir(subject_source):
            condition_source = os.path.join(subject_source, condition)
            condition_dest = os.path.join(dest_path, subject, condition)

            if os.path.isdir(condition_source):  # Ensure it's a folder
                shutil.copytree(condition_source, condition_dest, dirs_exist_ok=True)

# Copy subjects while keeping walking conditions (`bg-01`, `nm-01`, etc.)
copy_subject_folders(train_subjects, TRAIN_PATH)
copy_subject_folders(test_subjects, TEST_PATH)

print("✅ Dataset split completed! Train/Test folders now contain images in the correct structure.")
