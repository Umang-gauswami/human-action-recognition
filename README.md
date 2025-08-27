# Human Action Recognition App 🕺🤖

## 🌐 Live Demo
## [Click here to open the live app](https://human-action-recognition-system.streamlit.app/)

This is a **Human Action Recognition** web app built with **Streamlit** and **TensorFlow**.  
The app allows you to upload an image of a human performing an action, and it predicts the action from 15 supported classes.

---

## ✅ Features

- Predicts human actions from images.
- Supported actions: `calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop`.
- Clean, interactive Streamlit interface.
- Runs locally in a virtual environment.
- Also you can use Lived Deployed App

---


---

## ⚙️ Requirements

- Python 3.13 (recommended)
- Packages used in this project:

```txt
streamlit==1.49.0
tensorflow==2.20.0
numpy==2.3.2
pillow==11.3.0
```
- Note: All other dependencies will be installed automatically

## Steps to run the Human Action Recognition App Locally
- 1. Create a new virtual environment and activate it ( Windows )
     ```text
     python -m venv venv
     .\venv\Scripts\activate
     ```
- 2. Create a new virtual environment and activate it ( Linux / Mac )
     ```text
     python -m venv venv
     source venv/bin/activate
     ```
- 3. Install all required packages
     ```text
     pip install -r requirements.txt
      ```
     Note:Install one by one package to avoid conflict
     ```text
     pip install numpy==2.3.2
     pip install pillow==11.3.0
     pip install tensorflow==2.20.0
     pip install streamlit==1.49.0
     ```
- 3. Run App Locally
     ```text
     streamlit run app.py
      ```
     
## 🖼While using the App
     1. Click on Browse files and select image from your system.

     2. The app will display the uploaded image on the left.

     3. The predicted action will appear on the right with the class index and name of the action.










