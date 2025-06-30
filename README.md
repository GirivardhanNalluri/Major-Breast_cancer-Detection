# Major-Breast_cancer-Detection
# Breast Cancer Detection Using Deep Learning

This project aims to detect and classify breast cancer stages using 2D mammography images. It combines Convolutional Neural Networks (CNNs) for feature extraction and classification, and includes a web-based interface for interaction.

## 🧠 Project Highlights

- **Binary and Multi-class Classification**: Differentiates between benign and malignant cases and identifies stages of cancer.
- **Deep Learning Models**: Uses CNNs trained on preprocessed image datasets.
- **Visualization**: Training history and performance graphs included.
- **Web Interface**: Flask-based application with custom dashboard, login, registration, and result display pages.

## 📁 Project Structure

```
bc/
├── Breast_Cancer.ipynb             # Notebook for model training and experimentation
├── main.py                         # Entry point for the web app
├── train.py                        # Script to train the CNN model
├── test.py                         # Script to evaluate the model
├── requirements.txt                # Project dependencies
├── static/
│   ├── css/                        # Web stylesheets
│   └── images/                     # UI and output images
├── templates/                      # HTML templates for the Flask app
├── binary_classification_history.png     # Training curve (binary)
├── stage_classification_history.png      # Training curve (stages)
└── dataset/
    ├── train/
    │   ├── benign/
    │   └── malignant/
    ├── test/
    │   ├── benign/
    │   └── malignant/
    └── val/
        ├── benign/
        └── malignant/
```

- Total images: **7745**
- Data is split into `train` (**60%**), `test` (**20%**), and `val` (**20%**) directories, each containing `benign` and `malignant` subfolders.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection/bc
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Web Application

```bash
python main.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## 🖼 Sample Results

- `binary_classification_history.png`: Shows training/validation accuracy for binary classification.
- `stage_classification_history.png`: Shows training/validation for stage classification.

## 📚 Technologies Used

- Python
- TensorFlow , Keras , Pandas
- OpenCV
- Flask (for web interface)
- HTML, CSS, Js
- MySQL (Database)

## 🧪 Dataset

This project uses a curated dataset of **7745 mammogram images** organized into three main folders:
- `train`: for training the model
- `test`: for evaluating model performance
- `val`: for validation during training

Each of these contains two subfolders:
- `benign`: non-cancerous cases
- `malignant`: cancerous cases

See the `Breast_Cancer.ipynb` notebook for preprocessing and augmentation steps.

## 📬 Contact

For queries or suggestions, reach out to: 


📧 **Email:** girivardhan2301@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/girivardhan-nalluri-215341267/)
