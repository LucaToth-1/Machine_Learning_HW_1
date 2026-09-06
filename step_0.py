import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
    
# load data (as pandas dataframes) 
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# seperating the colors of our wine
red_wine["is_red"] = 1
white_wine["is_red"] = 0

#comibine our dataframes into one
wine = pd.concat([red_wine, white_wine], ignore_index=True)

# Inspect the dataset

# Print the shape of the dataset
print("Wine dataset shape:", wine.shape)

# Print the dataset information
print("\nWine dataset info:", wine.info())

# Print the summary statistics of the dataset
print("\nWine dataset summary statistics:", wine.describe())

# Print the missing values in the dataset
print("\nMissing values in the dataset:", wine.isnull().sum())

# Print the duplicate rows in the dataset
print("\nDuplicate rows in the dataset:", wine.duplicated().sum())

# Finally, print the quality distribution of the dataset
print("\nQuality distribution of the dataset:", wine["quality"].value_counts().sort_index())

