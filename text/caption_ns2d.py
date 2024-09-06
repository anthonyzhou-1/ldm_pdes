import os 
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage
import base64
from anthropic import Anthropic
from tqdm import tqdm 
from skimage import feature
from skimage import segmentation
import re

'''
This script is used to label the Navier-Stokes dataset using the Anthropic API. 
Also contains code to generate a segmented image using the Canny edge detection algorithm.
'''

DATA_KEY = "labels"
IMG_DIR = "/home/ayz2/ldm_fluids/logs/imgs_new"
num_tries = 0

def do_everything():
    data_path = "/home/ayz2/data/NavierStokes-2D-conditoned"

    files = os.listdir(data_path)
    filtered_files = []

    split = "valid"

    for file in files:
        if file.endswith(".h5"):
            if split in file:
                filtered_files.append(file)

    def get_files_labeled(fname):
        img_dir = IMG_DIR
        labeled_files = os.listdir(img_dir)

        filtered_files = []
        for file in labeled_files:
            if fname in file:
                if ".txt" in file:
                    filtered_files.append(file)

        return filtered_files, len(filtered_files)

    def make_dict(txt):
        return {
            "type": "text",
            "text": txt
        }

    def make_img(img_path):

        with open(img_path, "rb") as image_file:
            binary_data = image_file.read()
            base_64_encoded_data = base64.b64encode(binary_data)
            img_data = base_64_encoded_data.decode('utf-8')
        return {
            "type": "image",
            "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_data,
                        },
        }

    preface = "You will be analyzing an image of a simulation of the Navier-Stokes equations to identify smoke plumes, count them, and describe their characteristics. You will also be provided an image with the boundaries of the smoke plumes drawn over the simulation in green. Follow these instructions carefully:\n"
    instruction = "First, examine the provided image:\n"
    img_tag = "<image>\n"
    instruction_2 = "Second, examine the provided segmented image:\n"
    #<image>
    identify = "To identify plumes:\n"
    step1_i = "1. Look for regions of bright color, which contrast against the darker background.\n"
    step2_i = "2. Pay attention to color variations and changes in the shape of the plumes.\n"
    step3_i = "3. Use the green lines in the segmented image to help identify the boundaries of the plumes.\n"

    count = "To count the number of plumes:\n"
    step1_c = "1. Scan the entire image systematically.\n"
    step2_c = "2. Use the green lines in the segmented image to help separate different plumes.\n"
    step3_c = "3. Pay attention to darker regions that separate different plumes.\n"

    describe = "To describe each plume's shape, size, and location:\n"
    step1_d = "1. Shape: Characterize the overall form (e.g., column-like, mushroom-shaped, dispersed cloud, triangular, curved) and changes in shape of the plume.\n"
    step2_d = "2. Size: Estimate the relative size compared to other elements in the image (e.g., small, medium, large) and describe how the plume changes in size.\n"
    step3_d = "3. Location: Describe the position using general terms (e.g., upper left corner, center, lower right) or in relation to other smoke plumes.\n"

    present = "Remember to use the green lines in the segmented image to help describe each plume's shape, size, and location. Present your findings in the following format:\n"
    answer_tag = "<answer>\n"
    format1 = "Number of plumes identified: [Insert number]\n"

    format2 = "Plume descriptions:\n"
    step1_f = "1. [Shape], [Size], [Location]\n"
    step2_f = "2. [Shape], [Size], [Location]\n"
    step3_f = "(Continue for each plume identified)\n"

    additional = "Additional observations: [Only include any relevant details about the overall patterns or shapes observed. Pay attention to symmetry and how the plumes interact with each other.]\n"
    # <answer>
    remember = "Remember to be as descriptive and accurate as possible in your analysis."

    def get_content(img_path, segmented_path):
        content = [
            make_dict(preface),
            make_dict(instruction),
            make_dict(img_tag),
            make_img(img_path),
            make_dict(img_tag),
            make_dict(instruction_2),
            make_dict(img_tag),
            make_img(segmented_path),
            make_dict(img_tag),
            make_dict(identify),
            make_dict(step1_i),
            make_dict(step2_i),
            make_dict(step3_i),
            make_dict(count),
            make_dict(step1_c),
            make_dict(step2_c),
            make_dict(step3_c),
            make_dict(describe),
            make_dict(step1_d),
            make_dict(step2_d),
            make_dict(step3_d),
            make_dict(present),
            make_dict(answer_tag),
            make_dict(format1),
            make_dict(format2),
            make_dict(step1_f),
            make_dict(step2_f),
            make_dict(step3_f),
            make_dict(additional),
            make_dict(answer_tag),
            make_dict(remember)
        ]
        return content

    api_key = "APIKEY"

    client = Anthropic(
        api_key=api_key,
    )

    def get_claude(img_path, segmented_path):
        content = get_content(img_path, segmented_path)
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        return message
    
    def process_tags(string):
        m = re.search('>([^<>]*)<', string)
        if m:
            return m.group(1)
        else:
            return string

    num_files = len(filtered_files)
    print(f"Labeling {split} with {num_files} files")

    for fname in tqdm(filtered_files):
        # open file and get data
        h5f = h5py.File(os.path.join(data_path, fname), "a")
        data = h5f[split]
        u = data['u'][:]
        num_data = u.shape[0]

        text_labels = {}

        print("querying file: ", fname)

        if DATA_KEY in data:
            print(f"text labels already exist: {fname}")
            continue

        labeled_files, num_labeled = get_files_labeled(fname)
        print(f"labeled files: {num_labeled}")

        for i in range(num_labeled):
            print("restoring text label for ", i)
            f = open(IMG_DIR + f"/{fname}_{i}.txt", "r")
            text_label = f.read()
            f.close()
            text_key = str(i)
            text_labels[text_key] = text_label
        
        # get text_labels for each data point
        for i in range(num_labeled, num_data):
            u_i = u[i]
            img = u_i[0] # get first frame of solution

            # get image
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(img, cmap='inferno')
            ax.set_axis_off()
            img_path = IMG_DIR + f"/{fname}_{i}.png"
            fig.savefig(img_path, bbox_inches='tight')
            plt.close(fig)

            # get segmented image
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            edges = feature.canny(img, sigma=8, low_threshold=0.1, high_threshold=0.2)
            # Get the color map by name:
            cm = plt.get_cmap('inferno')
            img = img/img.max()
            # Apply the colormap like a function to any array:
            colored_image = cm(img)
            colored_image = colored_image[:, :, :3]
            result_img = segmentation.mark_boundaries(colored_image, edges, mode='thick', color=(117/255, 255/255, 51/255))
            ax.imshow(result_img, cmap='inferno')
            ax.set_axis_off()
            segmented_path = IMG_DIR + f"/{fname}_{i}_segmented.png"
            fig.savefig(segmented_path, bbox_inches='tight')
            plt.close(fig)

            message = get_claude(img_path, segmented_path)
            text_label = message.content[0].text
            text_label = process_tags(text_label)

            print(f"writing text label for {i}")

            # save text backup
            f = open(IMG_DIR + f"/{fname}_{i}.txt", "a")
            f.write(text_label)
            f.close()

            text_key = str(i)
            text_labels[text_key] = text_label

        text_labels_group = data.create_group(DATA_KEY)

        for key, value in text_labels.items():
            text_labels_group.create_dataset(key, data=value)

        h5f.close()

from time import sleep

# Sometimes Anthropic API is overloaded or has too many requests, will restart the program in this case
def restart_on_crash():
    try:
        # Create infinite loop to simulate whatever is running
        # in your program
        while True:
            print("Starting Program")
            do_everything()
    except Exception:
        print("Something crashed your program. Let's restart it")
        handle_exception()

def handle_exception():
    global num_tries
    sleep(2)  # Restarts the script after 2 seconds
    num_tries += 1
    print(f"Number of tries: {num_tries}")
    if num_tries > 10:
        print("Too many tries. Exiting program.")
        return
    restart_on_crash()

restart_on_crash()
