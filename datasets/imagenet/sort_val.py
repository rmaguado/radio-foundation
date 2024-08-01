import os
import shutil

# Path to the validation images folder
val_images_dir = './val'

# Path to the synset labels file
synset_labels_file = './val_synset_labels.txt'

# Read the synset to label ID mapping
with open(synset_labels_file, 'r') as f:
    synset_labels = [line.strip() for line in f]

# Ensure the output directories exist
for synset in list(set(synset_labels)):
    synset_dir = os.path.join(val_images_dir, synset)
    os.makedirs(synset_dir, exist_ok=True)

# Move each image to its corresponding synset folder
for i in range(len(synset_labels)):
    image_name = f'ILSVRC2012_val_{i+1:08d}.JPEG'
    src_path = os.path.join(val_images_dir, image_name)
    dst_path = os.path.join(val_images_dir, synset_labels[i], image_name)
    shutil.move(src_path, dst_path)

print("Validation images sorted successfully.")