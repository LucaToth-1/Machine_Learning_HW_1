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

# Create a histogram of the wine dataset so we can visualize the distribution of the features
wine.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()


# Seperate features and target

X = wine.drop(columns=["quality"])
Y = wine["quality"].to_numpy() # converting this to a numpy array

#split the data into training (70%), validation (15%), and testing (15%) sets
# create a random seed
np.random.seed(42)

indices = np.random.permutation(len(wine))

n = len(wine)

train_end = int(0.7 * n)
valid_end = int(0.85 * n)

train_indices = indices[:train_end]
valid_indices = indices[train_end:valid_end]
test_indices = indices[valid_end:]

train = wine.iloc[train_indices]
valid = wine.iloc[valid_indices]
test = wine.iloc[test_indices]

# Seperate X and Y for each dataset
X_train = train.drop(columns=["quality"]).to_numpy()
Y_train = train["quality"].to_numpy()

X_valid = valid.drop(columns=["quality"]).to_numpy()
Y_valid = valid["quality"].to_numpy()

X_test = test.drop(columns=["quality"]).to_numpy()
Y_test = test["quality"].to_numpy()

# Add bias/intercept column to the features

X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_valid = np.column_stack((np.ones(X_valid.shape[0]), X_valid))
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

# Verify the shapes of the final datasets

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_valid shape:", X_valid.shape)
print("Y_valid shape:", Y_valid.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Save the datasets

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)

np.save("X_valid.npy", X_valid)
np.save("Y_valid.npy", Y_valid)

np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)
