import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loads a CSV file into a pandas DataFrame.
df = pd.read_csv("Student_Mental_Stress_and_Coping_Mechanisms.csv") 
print(df.shape) 

# Sets "Student ID" as the index column (so each row is uniquely identified by ID).
df = df.set_index("Student ID")
df.index.is_unique

# Cleaning Column Names
df.columns = df.columns.str.lower().str.replace(" ", "_")
print(df.columns)
df.head()

# Counts how many missing values exist per column.
print(df.isnull().sum())

# Checks how many duplicate rows are in the dataset.
print("Duplicates:", df.duplicated().sum())

# counts how many times each gender label appears.
print(df.gender)
df['gender'].value_counts()

# Replaces multiple gender identities with grouped categories
df['gender_grouped'] = df['gender'].replace({
    'Male': 'Male',
    'Female': 'Female',
    'Agender': 'Non-binary/Other',
    'Bigender': 'Non-binary/Other',
    'Genderfluid': 'Non-binary/Other',
    'Polygender': 'Non-binary/Other',
    'Genderqueer': 'Non-binary/Other',
    'Non-binary': 'Non-binary/Other'
})


# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
# Plot histograms
df[numeric_cols].hist(figsize=(19, 10), bins=30, edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

# Select categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
# Plot bar charts
for col in categorical_cols:
    plt.figure(figsize=(5,4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Correlation Heatmap of Numeric Features & identify relationships between numeric variables
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


# K-Means Clustering with PCA Visualization
numeric_df = df.select_dtypes(include=['int64', 'float64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels back to the dataframe
df['cluster'] = clusters
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['pca1'] = pca_result[:,0]
df['pca2'] = pca_result[:,1]
plt.figure(figsize=(8,6))
plt.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('KMeans Clusters (PCA Projection)')
plt.colorbar(label='Cluster')
plt.show()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
cluster_summary = df.groupby('cluster')[numeric_cols].mean()
print(cluster_summary)



# Creating the Target Variable
threshold = df['mental_stress_level'].quantile(0.75)
df['high_stress'] = (df['mental_stress_level'] > threshold).astype(int)

# Splitting into Features and Target
X = df.drop(columns=['mental_stress_level', 'high_stress'])  # predictors
y = df['high_stress']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing Pipelines
numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(exclude=np.number).columns

# Separates numeric and categorical columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Building & Training Logistic Regression
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
clf.fit(X_train, y_train)

# Evaluating the Model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Get feature names after one-hot encoding
feature_names = (numeric_features.tolist() + 
                 list(clf.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

coef = clf.named_steps['classifier'].coef_[0]

# Plot top features
feat_importance = pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)[:10]

# Plotting Feature Importance
feat_importance.plot(kind='barh')
plt.title("Top Predictors of High Stress (LogReg Coefficients)")
plt.show()

# 1. PCA Clustering Visualization
numeric_cols = df.select_dtypes(include="number").dropna(axis=1).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

# Run KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# PCA projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["pca1"], df["pca2"] = X_pca[:,0], X_pca[:,1]

# Plot clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=df, palette="Set2", s=70, alpha=0.8)
plt.title("Student Clusters (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("cluster_pca.png", dpi=300)
plt.show()


# 2. Coping Mechanism Bar Chart
plt.figure(figsize=(8,6))
sns.countplot(x="stress_coping_mechanisms", data=df, order=df["stress_coping_mechanisms"].value_counts().index, palette="viridis")
plt.title("Distribution of Coping Mechanisms")
plt.xlabel("Coping Mechanism")
plt.ylabel("Number of Students")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("coping_mechanism_bar.png", dpi=300)
plt.show()
