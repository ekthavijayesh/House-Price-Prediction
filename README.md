# 🏡 NestEstimate — House Price Prediction

**A Machine Learning-Based Bengaluru House Price Prediction Web Application**

NestEstimate is a machine learning project that estimates house prices in Bengaluru, Karnataka, based on property location, total area, number of bedrooms, and number of bathrooms. Built using Python, Scikit-learn, and Streamlit, the application provides an interactive interface for generating instant property price estimates.

---

## 📌 Project Overview

House prices vary depending on location, property size, and configuration. NestEstimate uses historical housing data and a trained machine learning model to estimate property prices based on user-provided details.

The application uses **Random Forest Regression** and provides a clean, user-friendly web interface with a property input form and an estimated price result.

This project was developed as an academic project to explore machine learning, data preprocessing, exploratory data analysis, and web application development.

## ✨ Features

- 🏠 **House Price Prediction** — Estimate property prices using a trained machine learning model.
- 📍 **Location Selection** — Choose from available Bengaluru locations in the dataset.
- 📐 **Area Input** — Enter the property area in square feet.
- 🛏️ **Property Configuration** — Select the number of bedrooms and bathrooms.
- ⚡ **Instant Results** — Generate an estimate through an interactive web application.
- 🎨 **Modern UI** — Responsive layout with custom styling and a clean, nature-inspired color palette.
- 📊 **Data Analysis** — Explore housing data and visualize price distributions.
- 🤖 **Model Comparison** — Compare Linear Regression and Random Forest Regression performance.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model training |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Saving and loading trained models |

## 📂 Project Structure

```text
housepriceprediction/
│
├── app.py
├── inspect_data.py
├── clean_data.py
├── eda.py
├── train_model.py
├── train_random_forest.py
│
├── archive (5)/
│   └── bengaluru_house_prices.csv
│
├── data/
│   └── cleaned_house_prices.csv
│
├── models/
│   ├── linear_regression_model.joblib
│   └── random_forest_model.joblib
│
├── plots/
│   └── price_distribution.png
│
├── .gitignore
└── README.md
```

*The listed structure reflects the main project files; your actual folders and generated plots may differ.*

## 📊 Dataset

The project uses the Bengaluru house price dataset, containing historical property information.

The original dataset includes fields such as:

- Area type
- Availability
- Location
- Property size
- Total square feet
- Number of bathrooms
- Number of balconies
- Price

### Data preprocessing

The dataset is cleaned before model training through the following steps:

1. Remove unnecessary columns.
2. Handle missing values.
3. Extract the number of bedrooms (BHK) from the property size.
4. Convert square-foot ranges into numerical values.
5. Convert relevant columns to numeric data types.
6. Remove duplicate and invalid records.
7. Filter extreme or unsuitable property configurations during model training.

After cleaning, the dataset contains **12,319 property records**.

## 🤖 Machine Learning Models

Two regression models were trained and evaluated:

1. **Linear Regression**
2. **Random Forest Regression**

The models use the following input features:

- `location`
- `total_sqft`
- `bhk`
- `bath`

The target variable is `price`, measured in lakh Indian rupees.

Categorical location values are transformed using One-Hot Encoding, while the numerical features are passed through to the model.

### Model Performance

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| Mean Absolute Error (MAE) | 42.89 lakh | 35.14 lakh |
| Root Mean Squared Error (RMSE) | 97.63 lakh | 90.23 lakh |
| R² Score | 0.539 | 0.606 |

**Best-performing model:** Random Forest Regression.

The Random Forest model achieved lower MAE and RMSE and a higher R² score on the test dataset than Linear Regression. However, the errors remain substantial, so predictions should be treated as educational estimates rather than reliable market valuations.

## 🚀 Installation and Setup

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/house-price-prediction.git
```

### 2. Navigate to the project folder

```bash
cd house-price-prediction
```

If your GitHub repository has a different name, use that name in the command.

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

## ▶️ Run the Application

Start the Streamlit application from the project directory:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use NestEstimate.

### How to use

1. Open the NestEstimate web application.
2. Select the property location.
3. Enter the total area in square feet.
4. Select the number of bedrooms.
5. Select the number of bathrooms.
6. Click **Calculate estimate**.
7. View the predicted property price and the entered property details.

## 📈 Exploratory Data Analysis

Exploratory Data Analysis (EDA) is used to understand the dataset before model training.

The project includes analysis of:

- Property price distribution
- Descriptive statistics
- Property area and configuration
- Missing values and data quality
- Potential outliers in housing data

Generated plots are stored in the `plots/` directory.

## 🔮 Future Enhancements

- Add more machine learning algorithms for comparison.
- Improve handling of unusual property records and outliers.
- Include additional features such as property age and amenities when suitable data is available.
- Add interactive visualizations and location-based insights.
- Deploy the application online.
- Improve model accuracy through hyperparameter tuning and further evaluation.

## ⚠️ Limitations

- The model is trained on historical Bengaluru housing data.
- Actual prices may differ because of market conditions, property age, amenities, construction quality, and other factors.
- Model performance is limited by the available data and selected features.
- Predictions are not professional property valuations or financial advice.

## 🎓 Project Purpose

NestEstimate was developed as an academic machine learning project to demonstrate data cleaning, exploratory data analysis, regression model training, model evaluation, and deployment through a Streamlit interface.

---

**Project:** NestEstimate — House Price Prediction  
**Domain:** Machine Learning · Regression · Real Estate Analytics  
**Framework:** Streamlit  
**Model:** Random Forest Regression

*Built for learning and educational purposes.*
