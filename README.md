# 🏥 MedPredict Pro  
## AI-Powered Healthcare Cost Prediction and Decision Support System

MedPredict Pro is an end-to-end machine learning-based healthcare decision support system designed to estimate medical costs using patient demographic and health-related features.

The project explores how machine learning models can assist individuals in understanding potential healthcare expenses before medical procedures, supporting better financial planning and informed decision-making.

The system was developed as a Final Year Project and focuses on building a complete machine learning workflow, including data preprocessing, feature engineering, model optimization, evaluation, and interactive deployment.

> Built with the goal of exploring practical applications of Artificial Intelligence in healthcare.

---

## 🔗 Links

🚀 **Live Demo:**  
https://medicalpredictpro-hnzfulxbrqbtp47c3lehwu.streamlit.app/

📁 **Repository:**  
https://github.com/nayabnayyer/medical_cost_prediction_using_AI_00

---

# 🎯 Project Motivation

Healthcare expenses can be difficult to estimate before receiving medical services. Lack of cost awareness may create financial uncertainty for patients and families.

MedPredict Pro investigates whether machine learning techniques can learn patterns from patient-related features and provide estimated healthcare cost predictions.

The project demonstrates the potential of AI-driven decision support systems in healthcare-related applications.

---

# ✨ Key Features

- 🤖 Random Forest Regression model optimized using RandomizedSearchCV
- 📊 Interactive visualization of data distributions and prediction results
- 📁 Batch prediction support through CSV upload
- 🧮 BMI calculation through user inputs
- 🎨 Interactive Streamlit interface with custom styling
- 📱 User-friendly prediction workflow
- 🔄 Complete ML pipeline from data processing to deployment

---

# 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Regression Model | RandomForestRegressor |
| Hyperparameter Optimization | RandomizedSearchCV |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit Cloud |
| Model Persistence | Joblib |

---

# 🔬 Methodology

The project follows a supervised machine learning workflow:


Raw Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Hyperparameter Optimization
↓
Performance Evaluation
↓
Interactive Deployment


---

# 📊 Data Processing Pipeline

The preprocessing pipeline includes:

### Handling Data Quality Issues
- Missing value handling
- Outlier detection and capping using the IQR method

### Feature Transformation
- One-hot encoding for categorical variables
- Standard scaling for numerical features

### Feature Engineering
- BMI-related features
- Interaction-based features

---

# 🤖 Model Development

Several machine learning approaches were explored, with Random Forest Regression selected due to its ability to model complex relationships between patient characteristics and healthcare costs.

## Hyperparameter Optimization

RandomizedSearchCV was used to optimize model parameters.

Parameters explored included:


n_estimators:
100 - 500

max_depth:
None, 10, 20, 30, 40

min_samples_split:
2, 5, 10, 15

min_samples_leaf:
1, 2, 4, 6


---

# 📈 Model Evaluation

The final model was evaluated using regression performance metrics:

| Metric | Result |
|---|---|
| R² Score | 0.89 |
| Mean Absolute Error (MAE) | ~2500 PKR |
| Validation | 5-fold cross-validation |

Evaluation metrics were selected to measure both predictive accuracy and average prediction error.

---

# 🖥️ Application Workflow

Users can:

1. Enter patient information manually  
or  
2. Upload a CSV file containing multiple patient records

The system processes the input through the trained machine learning pipeline and generates predicted healthcare cost estimates.

---

# 📊 Visualizations

The application includes:

- Distribution analysis of healthcare charges
- Residual error visualization
- Actual vs predicted comparison plots
- Feature-based data exploration

These visualizations help analyze model behavior and prediction performance.

---

# 🧪 Example Prediction

Example Input:


Age: 30
BMI: 25.6
Children: 2
Smoker: Yes
Region: Lahore


Example Output:


Estimated Medical Cost:
Rs. 45,230


---

# ⚠️ Limitations

- The model is intended for educational and research exploration and should not replace professional medical or financial advice.
- The dataset may not fully represent the complexity of real healthcare pricing systems.
- Regional healthcare cost variations require further validation using larger real-world datasets.
- Additional explainability techniques would be valuable for understanding individual predictions.

---

# 🔮 Future Research Directions

Potential improvements include:

- Implementing Explainable AI methods such as SHAP or LIME for feature contribution analysis
- Comparing additional models including XGBoost and LightGBM
- Exploring uncertainty estimation for prediction reliability
- Incorporating larger healthcare datasets
- Developing multilingual support for broader accessibility

---

# 💡 Technical Skills Demonstrated

Through this project, I developed experience in:

- Building complete machine learning pipelines
- Data preprocessing and feature engineering
- Regression modeling
- Hyperparameter optimization
- Model evaluation
- ML application deployment
- Designing AI systems for healthcare-related decision support

---

# 🚀 Local Installation

Clone the repository:

```bash
git clone https://github.com/nayabnayyer/medical_cost_prediction_using_AI_00.git

cd medical_cost_prediction_using_AI_00
```
Install dependencies:
```
pip install -r requirements.txt
```
Run the application:
```
streamlit run app.py
```

# 📄 License

This project is developed for educational and research portfolio purposes.

# 👩‍💻 Author

Nayab Nayyer

Computer Science Graduate | Artificial Intelligence & Machine Learning Research Applicant

**GitHub**: https://github.com/nayabnayyer
**LinkedIn**: https://www.linkedin.com/in/nayab-nayyer-2b6803321  
