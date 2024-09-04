import os 
import h5py 
import re

def replace_key(path, fname):

    def get_idx_from_key(key, fname):
        key = key.replace(f"{fname}_", "")
        key = key.replace("_text_label", "")
        return key
    
    def process_tags(string):
        m = re.search('>([^<>]*)<', string)
        if m:
            return m.group(1)
        else:
            return string

    h5f = h5py.File(path, "a")
    data = h5f["valid"]
    data.keys()
    text_labels = data["text_labels"]
    keys = list(text_labels.keys())

    for key in keys:
        idx = get_idx_from_key(key, fname)
        prompt = text_labels[key].asstr()[()]
        text_labels[idx] = process_tags(prompt)
        del text_labels[key]
    h5f.close()

data_path = "/home/ayz2/data/NavierStokes-2D-conditoned"

files = os.listdir(data_path)
filtered_files = []

split = "valid"

for file in files:
    if file.endswith(".h5"):
        if split in file:
            filtered_files.append(file)

print(len(filtered_files))

for fname in filtered_files:
    break
    replace_key(os.path.join(data_path, fname), fname)