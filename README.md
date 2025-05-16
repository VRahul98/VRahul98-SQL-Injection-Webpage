# SQL Injection Prediction 🚀

This is a machine learning project that predicts whether an input is a **SQL Injection attack** or **Benign**. The goal is to identify malicious queries using trained ML models.

## 📌 Project Overview

- **Type**: Script-based ML project (not a web app)
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Joblib
- **Main file**: `sql injection.ipynb`

## 🛠 Features

- Reads SQL query strings
- Uses NLP and machine learning techniques to classify input
- Trained using labeled dataset
- Saves model using Joblib for reuse

## 📁 Folder Contents

```
sql-injection-prediction/
├── sql injection.ipynb       # Main ML notebook
├── Dataset.csv               # Input dataset
├── model.pkl                 # Saved trained model
├── requirements.txt          # Python libraries
└── README.md                 # This file
```

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/VRahul98/sql-injection-prediction.git
   cd sql-injection-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook "sql injection.ipynb"
   ```

## 👨‍💻 Author

**Vasanthapuram Rahul**  
GitHub: [VRahul98](https://github.com/VRahul98)
