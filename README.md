# Fed_DR

Federated Learning for Diabetic Retinopathy Using Retinal Image Datasets

## 📌 Introduction

This project implements federated learning for diabetic retinopathy (DR) classification using retinal image datasets. It simulates multiple clients (e.g., hospitals) collaboratively training a global model without sharing local data.

## 🧠 Methods

- **Federated Learning (FedAvg)**
- **Client Simulation with Different Data Volumes**
- **Diabetic Retinopathy Dataset (e.g., APTOS 2019)**

## 📁 Project Structure
Fed_DR/
├── client.py # Client-side training logic
├── server.py # Federated aggregation and coordination
├── datasets.py # Data loading and preprocessing
├── README.md # Project description


## 🚀 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Zhangkangjing/Fed_DR.git
    cd Fed_DR
    ```

2. (Optional) Create and activate a virtual environment.

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the server to start federated training:
    ```bash
    python server.py
    ```

## 📊 Results

The experiment evaluates classification accuracy under different client data volume settings. The global model's performance improves as the number of clients increases.

## 📚 Dataset

The dataset used is [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection).

## 🔑 License

This project is licensed under the MIT License.

## 👩‍💻 Author

Zhang Kangjing (zhangkangjing3@gmail.com)
