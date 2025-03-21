import os
import shutil

# Define dataset paths
DATASET_PATH = "/content/casia-b/output"
TRAIN_PATH = "/content/casia-b/train/output"
TEST_PATH = "/content/casia-b/test/output"

# Create train and test directories
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

# Walking conditions for train and test sets
TRAIN_CONDITIONS = {"nm": ["nm-01", "nm-02", "nm-03", "nm-04"], "bg": ["bg-01"], "cl": ["cl-01"]}
TEST_CONDITIONS = {"nm": ["nm-05", "nm-06"], "bg": ["bg-02"], "cl": ["cl-02"]}

def copy_condition_data(subject, conditions, source_path, dest_path):
    subject_source = os.path.join(source_path, subject)
    subject_dest = os.path.join(dest_path, subject)
    os.makedirs(subject_dest, exist_ok=True)
    
    for category, condition_list in conditions.items():
        for condition in condition_list:
            condition_source = os.path.join(subject_source, condition)
            condition_dest = os.path.join(subject_dest, condition)
            
            if os.path.isdir(condition_source):  # Ensure it's a valid folder
                shutil.move(condition_source, condition_dest)

# Get all subject IDs
subjects = sorted(os.listdir(DATASET_PATH))

# Split dataset based on walking conditions
for subject in subjects:
    copy_condition_data(subject, TRAIN_CONDITIONS, DATASET_PATH, TRAIN_PATH)
    copy_condition_data(subject, TEST_CONDITIONS, DATASET_PATH, TEST_PATH)

print("✅ Dataset split completed based on walking conditions! Train/Test folders now contain images structured correctly.")
