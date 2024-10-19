# Dynamic Machine Learning Model Selector

![{1E7A2E88-B2BF-4F86-B7C6-4623E3C3CC0C}](https://github.com/user-attachments/assets/14c6cc48-b852-4e38-b97a-4b26a48bcef5)

![{C918A573-83F2-4553-AB03-D8E6D3395FE6}](https://github.com/user-attachments/assets/18fa3930-8b5f-4e30-9d87-f247ed54b234)

![{EB8C9722-BA0A-4770-8CEA-4BEFA46A3BA8}](https://github.com/user-attachments/assets/d8cd4b48-4788-4d66-9ca1-3478d2471b90)


![{8EE47649-EFA6-44D0-B0C2-36C4681F878F}](https://github.com/user-attachments/assets/50287d7e-a29d-4e51-88f6-dfa70979ccf2)

![{64086CCD-DDEA-4DDD-8247-47B0675BDF14}](https://github.com/user-attachments/assets/3cb76739-fc91-4b08-9f5a-9cc0d87465c9)


## Overview

The **Dynamic Machine Learning Model Selector** is an interactive web application designed to facilitate the selection and evaluation of various machine learning models. This tool allows users to easily upload datasets, select features and target variables, and dynamically apply regression and classification models. It aims to streamline the model selection process by providing essential performance metrics and visualizations.

## Features

- **Dataset Upload**: Users can upload their own CSV datasets for analysis.
- **Model Selection**: Choose from multiple machine learning models, including:
  - Linear Regression
  - Logistic Regression
  - Decision Tree Classifier/Regressor
  - Random Forest Classifier/Regressor (with hyperparameter tuning)
- **Visualization**: Visualize model performance through confusion matrices, bar graphs of Mean Squared Error, and other metrics.
- **User-Friendly Interface**: Built with Streamlit for an intuitive user experience.

## Getting Started

### Prerequisites

- Python 3.x installed on your system. Download it from [python.org](https://www.python.org/downloads/).

### Installation

Follow the steps below to set up the project:

1. Open a terminal and navigate to your project folder:
    ```bash
    cd myproject
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - **Windows PowerShell**:
      ```bash
      .\venv\Scripts\Activate.ps1
      ```
    - **macOS/Linux**:
      ```bash
      source venv/bin/activate
      ```

4. Install the required libraries:
    ```bash
    pip install streamlit pandas scikit-learn matplotlib seaborn plotly
    ```

5. Launch the application:
    ```bash
    streamlit run app.py
    ```

## Requirements

The following Python libraries are required for this application:

- `streamlit`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `plotly`

These can be installed using:
```bash
pip install -r requirements.txt
