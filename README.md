# python-week-7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully!")

    # Display the first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Explore the structure
    print("\nDataset information:")
    df.info()

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Clean the dataset (handling missing values if any)
    # In this case, the Iris dataset is clean, but for demonstration:
    if df.isnull().sum().sum() > 0:
        # Option 1: Fill missing values with the mean (for numerical columns)
        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].mean(), inplace=True)
        # Option 2: Drop rows with any missing values
        # df.dropna(inplace=True)
        print("\nMissing values handled.")
    else:
        print("\nNo missing values found.")

except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
try:
    # Compute basic statistics for numerical columns
    print("\nBasic statistics of numerical columns:")
    print(df.describe())

    # Group by the categorical column 'target' (representing species)
    grouped_data = df.groupby('target').mean()
    print("\nMean of numerical columns per species:")
    print(grouped_data)

    # Identify patterns or interesting findings
    print("\nInteresting findings:")
    print("Based on the grouped means:")
    print("- Setosa (target 0) generally has smaller sepal and petal measurements compared to the other two species.")
    print("- Versicolor (target 1) has intermediate measurements.")
    print("- Virginica (target 2) tends to have the largest sepal and petal lengths and widths.")

except Exception as e:
    print(f"An error occurred during basic data analysis: {e}")

# Task 3: Data Visualization
try:
    # Line chart (not directly applicable to this dataset in a meaningful way without time series data)
    # For demonstration, let's create a simple line plot of the mean of each feature per species index
    plt.figure(figsize=(10, 6))
    for col in grouped_data.columns:
        plt.plot(grouped_data.index, grouped_data[col], marker='o', label=col)
    plt.title('Mean Feature Values per Iris Species')
    plt.xlabel('Species (Index)')
    plt.ylabel('Mean Value (cm)')
    plt.xticks(grouped_data.index, iris.target_names)
    plt.legend(title='Feature')
    plt.grid(True)
    plt.show()

    # Bar chart: Average petal length per species
    plt.figure(figsize=(8, 6))
    sns.barplot(x='target', y='petal length (cm)', data=df, palette='viridis')
    plt.title('Average Petal Length per Iris Species')
    plt.xlabel('Species')
    plt.ylabel('Average Petal Length (cm)')
    plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
    plt.show()

    # Histogram: Distribution of sepal width
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sepal width (cm)'], kde=True, color='skyblue')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter plot: Sepal length vs. petal length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='Set2')
    plt.title('Sepal Length vs. Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species', labels=iris.target_names)
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"An error occurred during data visualization: {e}")
