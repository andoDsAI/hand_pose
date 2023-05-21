import os
import json
import random


root_dir = os.path.join("data", "DEX_YCB", "data")
annotations_path = os.path.join(root_dir, "annotations")
data_path = os.path.join(annotations_path, "DEX_YCB_s0_{}_data.json")
subset_data_path = os.path.join(annotations_path, "DEX_YCB_s0_{}_subset_data.json")

def get_subset_data(data_type='train', split_ratio=0.1):
    input_path = data_path.format(data_type)
    output_path = subset_data_path.format(data_type)

    with open(input_path, 'r') as jsonfile:
        data = json.load(jsonfile)
        jsonfile.close()
    
    print(f"Len original data: {len(data['annotations'])}")
    
    left_hand_data = []
    right_hand_data = []
    for annotation in data['annotations']:
      if annotation['hand_type'] == 'left':
          left_hand_data.append(annotation)
      elif annotation['hand_type'] == 'right':
          right_hand_data.append(annotation)

    # Calculate the number of samples to extract from each hand type
    total_samples = int(len(data['annotations']) * split_ratio)
    left_samples = int(total_samples * len(left_hand_data) / len(data['annotations']))
    right_samples = total_samples - left_samples
    
    # Shuffle the data to randomize the selection
    random.shuffle(left_hand_data)
    random.shuffle(right_hand_data)
    
    # Select the samples
    selected_annos = left_hand_data[:left_samples] + right_hand_data[:right_samples]
    
    # Get the corresponding 'images' data for the selected annotations
    selected_image_ids = set(annotation['id'] for annotation in selected_annos)
    selected_images = [image for image in data['images'] if image['id'] in selected_image_ids]

    # Sort selected_images and selected_annos based on the 'id' key
    selected_annos = sorted(selected_annos, key=lambda x: x['id'])
    selected_images = sorted(selected_images, key=lambda x: x['id'])
  
    # Create a new data dictionary with the selected samples and corresponding images
    subset_data = {
        'images': selected_images,
        'annotations': selected_annos
    }

    print(f"Len subset data: {len(subset_data['annotations'])}")

    # Save the subset data to a new JSON file
    with open(output_path, 'w') as jsonfile:
        json.dump(subset_data, jsonfile, indent=4)
        jsonfile.close()


if __name__ == "__main__":
	get_subset_data(data_type='train', split_ratio=0.1)
	get_subset_data(data_type='test', split_ratio=0.1)
