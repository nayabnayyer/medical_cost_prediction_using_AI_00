# 🏥 MedPredict Pro

## AI-Powered Medical Cost Prediction Platform

MedPredict Pro is a complete end-to-end machine learning web application that predicts medical costs based on patient demographics and health factors. Built for the Pakistani healthcare landscape, it combines **data preprocessing**, **model optimization**, and **interactive deployment** into a seamless user experience.


🔗 **Live Demo:** [https://medicalpredictpro-hnzfulxbrqbtp47c3lehwu.streamlit.app/](https://medicalpredictpro-hnzfulxbrqbtp47c3lehwu.streamlit.app/)  
📁 **GitHub Repo:** [https://github.com/nayabnayyer/medical_cost_prediction_using_AI_00](https://github.com/nayabnayyer/medical_cost_prediction_using_AI_00)


---

## ✨ Features

- 🤖 **Random Forest Regression** with hyperparameter tuning (RandomizedSearchCV)
- 📊 **Interactive data visualizations** (distributions, residuals, actual vs predicted)
- 📁 **CSV upload support** for batch predictions
- 🧮 **BMI calculator** (manual entry or height/weight)
- 🎨 **Custom dark/light mode** responsive UI
- 📱 **Mobile-friendly navigation**
- 🔍 **Real-time predictions** with confidence metrics

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | Streamlit, Custom CSS, HTML |
| **Backend** | Python |
| **ML/AI** | Scikit-learn (RandomForestRegressor, RandomizedSearchCV) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Deployment** | Streamlit Cloud |

---

## 📊 Model Performance

After hyperparameter tuning, the model achieves:

| Metric | Score |
|--------|-------|
| **R² Score** | 0.89 |
| Mean Absolute Error (MAE) | ~2,500 Rs. |
| Cross-Validation Score | 5-fold CV stable |

---

## 🧠 How It Works

### 1. Data Processing Pipeline
- Handles missing values
- Outlier capping using IQR method
- One-hot encoding for categorical variables
- Standard scaling for numerical features

### 2. Model Training
- Random Forest Regressor with RandomizedSearchCV
- Hyperparameters tuned:
  - `n_estimators`: 100–500
  - `max_depth`: None, 10, 20, 30, 40
  - `min_samples_split`: 2, 5, 10, 15
  - `min_samples_leaf`: 1, 2, 4, 6

### 3. Prediction Pipeline
- User inputs via interactive form OR CSV upload
- BMI calculation option
- Real-time prediction with confidence metrics

---

## 📁 Dataset Structure

The model expects a CSV with the following columns (or similar):
age, sex, bmi, children, smoker, region, charges


- `charges` is the target variable
- Other columns are used as features

---

## 🚀 Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/medpredict-pro.git
cd medpredict-pro

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🧪 Example Prediction
## Input:

Age: 30

BMI: 25.6

Children: 2

Smoker: Yes

Region: Lahore

## Output:

Estimated Medical Cost: Rs. 45,230.00

## 📈 Visualizations Included
Distribution of Charges — Understanding the target variable

Residuals Distribution — Checking prediction errors

Actual vs Predicted — Model accuracy visualization

## 👩‍💻 What I Learned
Building a complete ML pipeline from raw data to deployment

Hyperparameter tuning with RandomizedSearchCV

Handling categorical variables with OneHotEncoder

Creating custom CSS for Streamlit (navigation bar, cards, buttons)

Integrating BMI calculator into a prediction form

Deploying a model with Streamlit Cloud

Writing production-level code structure

## 🔮 Future Improvements
Add more ML models (XGBoost, LightGBM)

Implement user authentication

Add cost breakdown by procedure

Integrate real hospital pricing APIs

Add insurance coverage estimator

Multi-language support (Urdu/English)

## 📄 License
This project is for educational and portfolio purposes.

## 🤝 Connect With Me
LinkedIn: linkedin.com/in/nayab-nayyer
GitHub: github.com/nayabnayyer
Email: nayabnayyer882@gmail.com

"Built with ❤️ for the Pakistani healthcare system"

— Nayab Nayyer






