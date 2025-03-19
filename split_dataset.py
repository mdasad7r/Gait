import os
import shutil
import random

# Define dataset paths
DATASET_PATH = "/content/casia-b/output"
TRAIN_PATH = "/content/casia-b/train"
TEST_PATH = "/content/casia-b/test"

# Create train and test directories
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

# Define subjects (001-124)
subjects = sorted(os.listdir(DATASET_PATH))

# Split: First 74 for training, last 50 for testing
train_subjects = subjects[:74]
test_subjects = subjects[74:]

def move_files(subject_list, dest_path):
    for subject in subject_list:
        subject_path = os.path.join(DATASET_PATH, subject)
        dest_subject_path = os.path.join(dest_path, subject)
        shutil.move(subject_path, dest_subject_path)

# Move subjects to respective folders
move_files(train_subjects, TRAIN_PATH)
move_files(test_subjects, TEST_PATH)

print("✅ Dataset split completed! Train/Test folders created.")
