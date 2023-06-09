import requests

# Specify the API endpoint URL
url = 'http://localhost:5000/classify'

# Load the image file
image_path = 'sample2.png'
image_file = open(image_path, 'rb')

# Create the POST request with the image file attached
files = {'image': image_file}
response = requests.post(url, files=files)

# Process the response
if response.status_code == 200:
    data = response.json()
    prediction = data['prediction']
    print(f"Predicted class: {prediction}")
else:
    print("Request failed.")