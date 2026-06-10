# Cross-Dataset Generalization in Underwater Beluga Whale Image Classification

This repository contains code and supporting files for a deep learning project focused on improving **cross-dataset generalization** in underwater beluga whale image classification. The project investigates how dataset-specific differences, such as water clarity, camera position, and image acquisition conditions, affect model performance when detecting beluga whales across different underwater image datasets.

## Project Overview

Monitoring marine wildlife is important for understanding ecosystem changes and supporting conservation decisions. However, manually reviewing large underwater image collections is time-consuming. Deep learning can help identify images that contain beluga whales and reduce the amount of manual image review required.

A major challenge is that underwater images collected across different years and conditions may look very different. For example, some images may have clear water, others may have muddy water, and some may include both underwater and above-water views. These dataset-specific differences can create **dataset bias**, causing models trained on one dataset to perform poorly on another.

This project explores the use of **supervised contrastive learning** to learn more generalizable image representations for beluga whale detection.

## Research Goal

The main goal of this project is to improve the ability of an image classification model to detect beluga whales across datasets with different acquisition conditions.

The project focuses on three major questions:

1. Can we detect dataset bias in underwater beluga whale image datasets?
2. How much does traditional supervised deep learning performance drop in cross-dataset evaluation?
3. Can supervised contrastive learning improve generalization to unseen datasets?

## Datasets

The study uses three underwater image datasets collected as part of beluga whale monitoring in the Churchill River Estuary near Churchill, Manitoba, Canada.

| Dataset | Description |
|---|---|
| **D1** | Clear water images with regular underwater viewing conditions |
| **D2** | Murky water images caused by poor water clarity |
| **D3** | Half-in/half-out images with both underwater and above-water views, caused by changes in camera position and boat movement |

These datasets represent different image acquisition profiles while targeting the same classification task: identifying whether a beluga whale is present in an image.

## Methodology

### 1. Dataset Bias Detection

A dataset membership classifier was trained to identify which dataset an image came from. If a model can accurately predict dataset membership, it suggests that the datasets contain unique visual footprints unrelated to the target object.

This step helps demonstrate the existence of dataset bias.

### 2. Traditional Supervised Deep Learning Baseline

A VGG-16 convolutional neural network was used as a baseline classifier for beluga whale detection. The model was evaluated in both:

- **Within-dataset settings**, where training and testing images come from the same dataset.
- **Cross-dataset settings**, where the model is trained on some datasets and tested on a held-out dataset.

The cross-dataset setup used **leave-one-dataset-out cross-validation**.

### 3. Supervised Contrastive Learning

A supervised contrastive learning framework was used to learn representations that bring beluga images closer together in the embedding space while separating them from non-beluga images.

The goal was to focus the model on the object of interest rather than dataset-specific visual artifacts.

## Key Results

The supervised contrastive learning approach improved cross-dataset performance compared with the traditional supervised deep learning baseline.

The largest improvement was observed on the most challenging dataset, **D3**, which contained half-in/half-out images and more irregular acquisition conditions.

Example reported results:

| Test Dataset | Baseline AUC | SCL AUC |
|---|---:|---:|
| D1a | 0.9314 | 0.9627 |
| D1b | 0.9390 | 0.9669 |
| D2a | 0.9339 | 0.9823 |
| D2b | 0.9322 | 0.9832 |
| D3a | 0.7509 | 0.9922 |
| D3b | 0.7496 | 0.9918 |

These results suggest that supervised contrastive learning can help reduce the negative impact of dataset bias and improve model robustness across different underwater image conditions.

## Repository Structure

```text
.
├── scripts/
│   └── Project scripts and helper files
├── templates/
│   └── Template files used by the application
├── Dockerfile
│   └── Docker configuration for running the project
├── app.py
│   └── Main application file
├── app2.py
│   └── Alternative or secondary application file
├── image_classification_request.py
│   └── Script for sending image classification requests
├── requirements.txt
│   └── Python package dependencies
├── run_script.sh
│   └── Shell script for running the project
├── sample1.png
│   └── Sample input image
├── sample2.png
│   └── Sample input image
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/NajmehSaffar/Cross-dataset-Generalization-in--Image-Classification.git
cd Cross-dataset-Generalization-in--Image-Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the main application:

```bash
python app.py
```

Or run the shell script:

```bash
bash run_script.sh
```

To test image classification requests, use:

```bash
python image_classification_request.py
```

## Docker Usage

Build the Docker image:

```bash
docker build -t beluga-image-classification .
```

Run the container:

```bash
docker run -p 5000:5000 beluga-image-classification
```

## Example Use Case

This project can be used to support wildlife monitoring workflows by automatically identifying underwater images that contain beluga whales. This can help reduce manual review time and allow researchers or citizen scientists to focus more on images that are relevant for ecological analysis.

## Technologies Used

- Python
- Deep Learning
- VGG-16
- Supervised Contrastive Learning
- Image Classification
- Flask or Python-based application structure
- Docker

## Research Contributions

This project demonstrates:

- The presence of dataset bias in underwater beluga whale image datasets.
- The performance drop of traditional deep learning models in cross-dataset testing.
- The benefit of supervised contrastive learning for improving cross-dataset generalization.
- The potential of AI-based tools for scalable marine wildlife monitoring.

## Notes

The image datasets used in the research may not be included in this repository due to data access, privacy, or storage limitations. If you use this repository, update the dataset path and configuration according to your local setup.

## Citation

If you use this project or build upon it, please cite or acknowledge the related research work:

**Supervised Contrastive Learning for Improving Cross-dataset Generalization in Underwater Beluga Whale Image Classification**  
Najmeh Saffar, Shehroz S. Khan, Ashleigh M. Westphal, Sarah Falconer, C-Jae C. Breiter, Stephen D. Petersen, Ahmed Ashraf

## Author

**Najmeh Saffar**

## License

Add your preferred license here, such as MIT License, Apache 2.0, or another license depending on how you want others to use the project.
