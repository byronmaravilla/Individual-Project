# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('25.csv')

# Data overview
print(data.info())
print(data.describe())

# Feature engineering: Creating new features
data['Exercise_Intensity'] = data['step_count'] / data['calories_burned']
data['Sleep_Category'] = pd.cut(data['hours_of_sleep'], bins=[0, 4, 7, 12], labels=['Poor', 'Average', 'Good'])
print(data[['Exercise_Intensity', 'Sleep_Category']].head())

# Scatter plot for mood vs step count
plt.scatter(data['step_count'], data['mood'], alpha=0.5)
plt.title('Step Count vs. Mood')
plt.xlabel('Step Count')
plt.ylabel('Mood Score')
plt.show()

# Linear Regression: Predict mood from step count and sleep
X = data[['step_count', 'hours_of_sleep']]
y = data['mood']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"Linear Regression RMSE (Mood): {rmse}")

# Visualize regression
plt.scatter(data['step_count'], data['mood'], alpha=0.5, label='Actual')
plt.scatter(data['step_count'], y_pred, color='red', alpha=0.5, label='Predicted')
plt.title('Mood Prediction')
plt.xlabel('Step Count')
plt.ylabel('Mood Score')
plt.legend()
plt.show()

# Check for any invalid or infinite values in clustering data
if np.any(np.isinf(data[['Exercise_Intensity', 'hours_of_sleep']])) or np.any(np.isnan(data[['Exercise_Intensity', 'hours_of_sleep']])):
    print("Data contains invalid values (inf or NaN). Cleaning data...")
    data = data.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    data = data.dropna(subset=['Exercise_Intensity', 'hours_of_sleep'])  # Drop rows with NaN

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Exercise_Intensity', 'hours_of_sleep']])


# Visualize clusters
sns.scatterplot(data=data, x='step_count', y='hours_of_sleep', hue='Cluster', palette='viridis')
plt.title('Clustering by Step Count and Sleep')
plt.xlabel('Step Count')
plt.ylabel('Hours of Sleep')
plt.show()

# Save cleaned data
data.to_csv('/mnt/data/cleaned_fitness_trends.csv', index=False)
print("Cleaned data saved to 'cleaned_fitness_trends.csv'.")
