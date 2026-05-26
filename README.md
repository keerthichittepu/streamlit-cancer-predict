🩺 streamlit-cancer-predict — AI Powered Cancer Prediction System



Predict whether a tumor is Benign or Malignant using Machine Learning and an interactive Streamlit web application.

Built using:

🧠 Machine Learning
📊 Scikit-learn
🌐 Streamlit
🐍 Python

This project demonstrates a complete end-to-end ML workflow including:

data preprocessing
feature scaling
model training
prediction
deployment with Streamlit


🌟 Features

✨ Real-time cancer prediction
🧠 Machine Learning based diagnosis system
📊 Interactive Streamlit web interface
⚡ Fast and accurate predictions
📈 Feature scaling for improved performance
💾 Pretrained model loading using Pickle
🌐 Easy-to-use web application
🎯 Predicts whether cancer is Benign or Malignant

🧠 Project Workflow
Medical Dataset
        ↓
Data Preprocessing
        ↓
Feature Scaling
        ↓
Model Training
        ↓
Model Serialization (.pkl)
        ↓
Streamlit Web Application
        ↓
User Inputs Medical Features
        ↓
ML Model Prediction
        ↓
Prediction Displayed

🛠️ Technologies Used
| Technology     | Purpose                |
| -------------- | ---------------------- |
| Python         | Core Programming       |
| Streamlit      | Web Application        |
| Scikit-learn   | Machine Learning       |
| Pandas         | Data Processing        |
| NumPy          | Numerical Operations   |
| Pickle         | Model Saving & Loading |
| StandardScaler | Feature Scaling        |


📂 Project Structure
streamlit-cancer-predict/
│
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── model.pkl              # Trained machine learning model
├── scaler.pkl             # Saved StandardScaler object
├── dataset.csv            # Dataset used for training
├── requirements.txt       # Required dependencies
├── README.md              # Project documentation

🧠 Machine Learning Concepts Used
📌 Binary Classification
📌 Feature Scaling
📌 Model Serialization
📌 Training & Prediction Pipeline


📊 Prediction Process
  User Inputs Medical Parameters
             ↓
  Convert Input to Numerical Array
             ↓
  Apply Feature Scaling
             ↓
   Pass Data to ML Model
             ↓
    Generate Prediction
             ↓
   Display Result on Screen


   ⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/keerthichittepu/streamlit-cancer-predict.git

2️⃣ Navigate to Project Folder
cd streamlit-cancer-predict

3️⃣ Create Virtual Environment
python -m venv venv

4️⃣ Activate Environment
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate

5️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py

The Streamlit server will start and automatically open the application in your browser.


📈 Model Evaluation Concepts

The project may include:

✅ Accuracy Score
✅ Confusion Matrix
✅ Precision & Recall
✅ Train-Test Split
✅ Performance Evaluation

🎯 Use Cases

🏥 Healthcare prediction systems
📚 Educational ML projects
🧠 AI-powered diagnosis support
💻 Streamlit deployment practice
📊 Machine Learning demonstrations

🚀 Future Improvements
Deep Learning-based prediction model
Add probability/confidence score
SHAP explainability dashboard
Multiple model comparison
Cloud deployment
Better UI design
Patient history tracking
Data visualization dashboard




👨‍💻 Author
Chittepu Sree Keerthi Reddy

🎓 IIIT Bhubaneswar
🌍 India

GitHub Profile:
keerthichittepu GitHub

Project Repository:
streamlit-cancer-predict Repository
