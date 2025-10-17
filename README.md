# 🐄 Image-Based Breed Recognition System using TensorFlow & CNN

## 📌 Overview
This project — developed by me and my team for **Smart India Hackathon (SIH) 2025** — focuses on building an **AI-powered image-based breed recognition system** for Indian bovine species (cows and buffaloes).  
Using **Convolutional Neural Networks (CNN)** and **TensorFlow**, our model accurately classifies bovine breeds based on images, helping in livestock identification, tracking, and management.


---

## 📂 Dataset
We have used the **Indian Bovine Breeds Dataset** available on Kaggle:  
🔗 [Indian Bovine Breeds Dataset – Kaggle](https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds)

This dataset contains images of various Indian cattle and buffalo breeds for training and testing the CNN model.

---

## 🧠 Key Features
- ✅ Built using **TensorFlow**, **Keras**, and **CNN architecture**.  
- 🖼️ Classifies multiple Indian cattle and buffalo breeds based on input images.  
- ⚙️ Customizable for different datasets or animal recognition projects.  
- 📊 Includes model training, validation, and testing modules.  
- 💾 Option to save and load trained models (`.h5` format).  

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn  
- **Tools:** Jupyter Notebook / Google Colab  

---

## 🚀 How to Run the Project

### Option 1: Run in **Google Colab**
1. Open [Google Colab](https://colab.research.google.com/).
2. Click on **File → Upload Notebook** and upload the `.ipynb` file from this repository.
3. Mount your Google Drive:
   ```python
   Download the dataset from Kaggle and extract it to your Drive (or Colab workspace).

4.Update dataset paths in the notebook accordingly.(take care of folder and file names )

5.Run each cell sequentially to train and test the CNN model.

Option 2: Run in Jupyter Notebook

1.Clone this repository:

2.git clone https://github.com/YOUR_GITHUB_USERNAME/Image-Based-Breed-Recognition.git


3.Navigate to the project directory:



4.Install the required dependencies:

5.pip install -r requirements.txt


6.Download and extract the dataset to your local directory.

7.Open the notebook:

8.jupyter notebook


Open the .ipynb file and run all cells in sequence.

🧩 Model Architecture

Our CNN model includes:

Convolutional Layers with ReLU activation

MaxPooling Layers

Dropout for regularization

Dense Layers for classification

Softmax output layer

The architecture was tuned for high accuracy and minimal overfitting.

📈 Results

Training Accuracy: ~70-95%

Validation Accuracy: ~80-90%

Testing Accuracy: ~80%

Model saved as: breed_recognition_model.h5

(Replace XX% with your actual results once finalized.)

📜 Future Scope

1.Deploying the model as a web application for farmers and veterinarians.

2.Integration with mobile camera-based recognition.

3.Expanding dataset for improved generalization.
   
