import os

# List of files to delete
files_to_remove = [
    "checkpoint_experiment_df.pkl", 
    "checkpoint_message_log.pkl", 
    "checkpoint_weights_log.pkl"
]

print("Cleaning up old Mock Data...")

for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted: {file}")
    else:
        print(f"Not found (already clean): {file}")

print("\nReady for Real Run! Restart the kernel and run the experiment cell.")