import data as videogamesdata
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


# Load the dataset
videogamesdata = pd.read_csv('videogamesdata.csv')

# 1. Data Exploration and Analysis

# Display the first few rows of the dataset
print(videogamesdata.head())

# Check summary statistics
print(videogamesdata.describe())

# Check for missing values
print(videogamesdata.isnull().sum())

# Check data types of columns
print(videogamesdata.dtypes)

# Summarize sales data by region
sales_summary = videogamesdata[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()

plt.figure(figsize=(10, 6))
sales_summary.plot(kind='bar', rot=0)
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Sales (millions)')
plt.show()

# 2. Data Cleaning and Preprocessing

# Check for missing values
print(videogamesdata.isnull().sum())

# Drop rows with missing values
data_clean = videogamesdata.dropna()

# Convert column names to lowercase
videogamesdata.columns = videogamesdata.columns.str.lower()

# Convert year column to datetime format if needed
videogamesdata['year'] = pd.to_datetime(videogamesdata['year'], format='%Y')

# 3. Advanced Data Analysis

#Task1: Time Series Analysis
#Objective: Analyze sales trends over the years.

# Group data by year and sum global sales
yearly_sales = videogamesdata.groupby('Year')['Global_Sales'].sum()

# Plot time series of global sales
plt.figure(figsize=(12, 6))
yearly_sales.plot(marker='o')
plt.title('Global Sales Over Time')
plt.xlabel('year')
plt.ylabel('Global Sales (millions)')
plt.grid(True)
plt.show()


# Task2: Genre and Platform Analysis
#Objective: Identify top-selling genres and platforms.

# Top genres by global sales
top_genres = videogamesdata.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).head(10)

# Top platforms by global sales
top_platforms = videogamesdata.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)

# Visualize top genres and platforms
plt.figure(figsize=(14, 5))

# Subplot for top genres
plt.subplot(1, 2, 1)
top_genres.plot(kind='bar')
plt.title('Top 10 Genres by Global Sales')
plt.xlabel('Genre')
plt.ylabel('Global Sales (millions)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Subplot for top platforms
plt.subplot(1, 2, 2)
top_platforms.plot(kind='bar')
plt.title('Top 10 Platforms by Global Sales')
plt.xlabel('Platform')
plt.ylabel('Global Sales (millions)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Adjust layout to prevent overlapping of subplots
plt.tight_layout()

# Display the plot
plt.show()

#Hypothesis Testing
genre_groups = [videogamesdata[videogamesdata['Genre'] == genre]['Global_Sales'] for genre in videogamesdata['Genre'].unique()]
f_statistic, p_value = f_oneway(*genre_groups)
print(f"F-statistic: {f_statistic:.2f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: There is a significant difference in global sales between different genres.")
else:
    print("Fail to reject null hypothesis: There is no significant difference in global sales between different genres.")
