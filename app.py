import os
from flask import Flask, jsonify, request
import torch
from torchvision.transforms import functional as F
from PIL import Image
from scripts.SupCon.model import *

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backup = "/backup01"
setup = "SupCon"
exp = "D1D2"
im_size = 128
num_channels = 1

dirname = 'outputs{}/{}/saved_models/'.format(backup, setup)
# Assert that the path exists
assert os.path.exists(dirname), "Path does not exist: {}".format(dirname)

filename = '{}-{}.pt'.format(setup, exp)
# Assert that the file exists within the directory
assert os.path.isfile(os.path.join(dirname, filename)), "File does not exist: {}".format(filename)

# Build the model architecture
model = VGG16().to(device)
model.eval()

print('loading the model ...\n') 
checkpoint = torch.load(dirname + filename)

model.load_state_dict(checkpoint['model_state_dict']) 
normal_vec = checkpoint['normal_vec_enc']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((im_size,im_size)),
    transforms.Grayscale(num_output_channels=num_channels),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if an image was provided in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    # Load the image from the request
    image = Image.open(request.files['image'])

    # Preprocess the image
    preprocessed_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        _, output = model(preprocessed_image)
        similarity_score = torch.mm(output, normal_vec.t())
                
    # Convert the predictions to a human-readable format
    predicted_class = 'Hooray! I see beluga whales :)' if similarity_score[0] >= 0.5 else 'No beluga whales :('

    # Return the predictions
    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run()