## Student Mental Stress Analysis: Insights into Factors and Coping Mechanisms

# Project Overview
This repository contains a comprehensive data analysis project on student mental stress levels and coping mechanisms, using a dataset of ~760 student records from "Student_Mental_Stress_and_Coping_Mechanisms.csv”. The analysis explores key factors influencing mental stress, such as academic performance, sleep, social media usage, family support, and more. It employs exploratory data analysis (EDA), clustering, and predictive modeling to uncover patterns and predict high-stress cases.

Key highlights:
- EDA & Visualization: Histograms, bar charts, and correlation heatmaps to identify distributions and relationships.
- Unsupervised Learning: KMeans clustering with PCA for student segmentation into stress profiles.
- Predictive Modeling: Logistic Regression to predict "high stress" with 95%+ accuracy, highlighting top predictors like peer pressure and sleep duration.
- Real-World Impact: Provides actionable insights for educators, counselors, and policymakers to support student well-being, e.g., promoting exercise and meditation as effective coping strategies.
This project demonstrates my skills in data wrangling, visualization, machine learning, and deriving business insights—ideal for roles in data analysis, ML engineering, or health analytics. Developed as a personal portfolio project, inspired by growing concerns in student mental health post-pandemic.

# Motivation
Student mental health is a global crisis, with factors like academic pressure, social media, and lack of support contributing to high stress levels. This analysis aims to:
- Identify correlations (e.g., low sleep linked to higher stress).
- Cluster students into groups (e.g., "High Achievers with Low Support" vs. "Balanced with Strong Coping").
- Predict high-stress risks for early intervention.
- Recommend coping mechanisms based on data (e.g., exercise reduces cognitive distortions).

# Key Features & Analyses
1. Data Cleaning: Handled duplicates, nulls, and grouped genders (e.g., non-binary categories).
2. Descriptive Statistics: Value counts, distributions, and unique checks for all columns.
3. Visualization:
  - Histograms for numeric features.
  - Bar charts for categoricals.
  - Correlation heatmap revealing strong ties between stress and factors like peer pressure.
  - Clustering: KMeans (3 clusters) on scaled numeric data, visualized via PCA—e.g., Cluster 0: High stress, low sleep; Cluster 1: Balanced; Cluster 2: Low stress, high support.
  - Predictive Modeling: Binary classification for "high stress" using Logistic Regression. Achieved ~0.95 accuracy; top features: peer pressure, relationship stress, sleep duration.
4. Insights: 
  - Reading is the most common coping mechanism.
  - High GPA correlates with moderate stress, but low family support amplifies it.
  - Non-binary genders show slightly higher stress variance, warranting targeted support.
 
# Technologies Used
- Core Libraries: Pandas (data manipulation), NumPy (arrays), Seaborn & Matplotlib (visualization).
- Machine Learning: Scikit-learn (StandardScaler, PCA, KMeans, LogisticRegression, train_test_split, ColumnTransformer, Pipeline).
- Environment: Python 3.10+; Jupyter Notebook for interactive development.
- Data Source: CSV dataset with features like Age, GPA, Sleep Duration, Stress Levels, Coping Mechanisms.

# Setup & Installation
1. Clone the Repository: git clone https://github.com/komal202220/student-mental-stress-analysis.git
                         cd Student-Mental-Stress-Analysis-Insights-into-Factors-and-Coping-Mechanisms

2. Install Dependencies: pip install pandas
                                     Numpy
                                     Seaborn      
                                     Matplotlib
                                     scikit-learn
   pip install -r requirements.txt
3. Run the Analysis:
   - Open `Student_Mental_Stress_Analysis.py` in Jupyter
   - Execute cells sequentially to reproduce EDA, visualizations, clustering, and modeling.
   - Outputs: Printed reports, saved PNGs (e.g., "cluster_pca.png").
  
# Results & Visualizations
1. Cluster Analysis (PCA Projection)
- Interpretation : Three distinct student groups—purple (high-stress, low-support), green (moderate), yellow (low-stress, balanced lifestyle).

2. Coping Mechanisms Distribution
- Insight : Reading and Travelling are top choices, whereas talking to friends is not considered by most of the students.

3. Predictive Model Performance
- Accuracy: ~95%
- Classification Report (sample):
- Accuracy: 0.9539473684210527
                                 
                        precision    recall  f1-score   support
  0                        0.98      0.96      0.97       123
  1                        0.84      0.93      0.89        29

  accuracy                                    0.95       152
  macro avg               0.91      0.95      0.93       152
weighted avg              0.96      0.95      0.95       152

4. Top Predictors (Logistic Regression Coefficients)
diet_quality: -1.462810 - Better diet quality lowers stress, a key protective factor.
study_hours_per_week: 1.455477 - More study hours increase stress, reflecting workload impact.
sleep_duration_(hours_per_night): -1.441904 - More sleep reduces stress, a critical health factor.

# Contributor
Komal - komal202220@gmail.com




