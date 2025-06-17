# AgriSight: Leaf Disease Detection System

**AgriSight** is a lightweight, open‑source tool for real‑time plant leaf disease detection, designed for easily deployable offline use in rural farming environments.

## 🚀 Purpose

* Enable farmers and agronomists to quickly identify common leaf diseases in the field without internet connectivity.
* Provide a simple web‑based interface for capturing and analyzing leaf images on resource‑constrained hardware.
* Improve crop health monitoring and reduce manual inspection workload.

## 🔍 How It Works

1. **Image Capture**: Users upload or snap a photo of a plant leaf through the web UI.
2. **Preprocessing**: The system applies OpenCV routines (cropping, color normalization) to prepare the image.
3. **Inference**: A pre‑trained Convolutional Neural Network (CNN) model classifies the leaf into healthy vs. diseased categories.
4. **Results**: Predictions and confidence scores are displayed instantly in the browser.

## 🛠️ Features

* **Offline Flask App**: Zero external dependencies beyond Python and OpenCV—runs entirely locally.
* **Transfer Learning**: Leverages pre‑trained backbones (e.g. MobileNetV2) for fast, accurate inference.
* **Modular Codebase**: Easily swap in new models or preprocessing pipelines.
* **Exportable Reports**: Automatically save classification results and timestamps for later review.

## 📦 Installation

1. Clone the repo:

   ```bash
   git clone git@github.com:<your‑org>/agrisight.git
   cd agrisight
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Download or train your leaf‑disease model and place it under `models/`.

## ⚙️ Usage

1. Launch the Flask app:

   ```bash
   python app.py
   ```
2. Open your browser to `http://localhost:5000`.
3. Upload a leaf image or capture via webcam.
4. View prediction and download the report if desired.

## 📂 Project Structure

```
agrisight/            # project root
├── app.py            # Flask web service
├── models/           # trained CNN models
├── static/           # CSS, JS, images
├── templates/        # HTML views
├── src/              # data preprocessing & inference modules
├── requirements.txt  # pip dependencies
└── README.md         # this file
```

## 🧪 Testing

```bash
pytest tests/ --cov=src
```

## 🔗 Extending

* **New Diseases**: Add more classes by re‑training the model on additional labeled datasets.
* **Batch Mode**: Integrate a CLI entrypoint to classify entire folders of images.
* **Mobile Deployment**: Convert the model to TensorFlow Lite or ONNX for on‑device inference.

## 🤝 Contributing

1. Fork the repo and create a feature branch.
2. Commit your changes with clear messages.
3. Open a pull request and reference any related issues.

## 📄 License

[MIT](LICENSE) © 2025 Harshit Rai
