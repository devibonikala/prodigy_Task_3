# Decision Tree Classification on Bank Dataset

## Overview

This script performs classification on a bank dataset using a Decision Tree classifier. It includes the following steps:

1. **Data Loading**: Reads data from a CSV file.
2. **Data Preparation**: Prepares features and target variables, performs one-hot encoding, and splits the data into training and testing sets.
3. **Model Training**: Trains a Decision Tree classifier on the training data.
4. **Model Evaluation**: Evaluates the model using accuracy and classification report, and visualizes the decision tree.

## Requirements

To run this script, you need Python 3.x and the following libraries:

- `pandas` for data manipulation.
- `matplotlib` for plotting.
- `scikit-learn` for machine learning functions.

You can install these libraries using `pip`:

```bash
pip install pandas matplotlib scikit-learn
```
## File Path
Ensure your CSV file is located at /home/rguktong/Desktop/bank.csv. Update the file path in the script if your CSV file is located elsewhere.

## Script Details
**1. Import Libraries**
Import the necessary libraries for data manipulation, model training, and evaluation:

```bash
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
```
**2. Load Data**
Load the dataset from the specified CSV file and display the DataFrame and its columns:

```bash
data = pd.read_csv('/home/rguktong/Desktop/bank.csv')
print(data)
print(data.columns)
```
**3. Prepare Data**
Target Variable: Define the target variable (y) and features (X):

```bash
y = data['balance']   #### Target variable
X = data.drop(['age'], axis=1)  #### Features
``` 
One-Hot Encoding: Convert categorical variables into dummy/indicator variables:
```bash
X = pd.get_dummies(X, drop_first=True)
```
Split Data: Split the data into training and testing sets:

```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.value_counts())
print(y_test.value_counts())
```
**4. Train Model**
Train a Decision Tree classifier using the training data:

```bash
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```
**5. Evaluate Model**
Make Predictions: Predict on the test data:

```bash
y_pred = model.predict(X_test)
```
Accuracy and Classification Report: Print accuracy and detailed classification report:

```bash 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, zero_division=0))
```
**6. Visualize Decision Tree**
Plot and display the trained Decision Tree:
```bash
plt.figure(figsize=(30, 10))
plot_tree(model, filled=True, feature_names=X.columns)
plt.show()
```
## Troubleshooting
**File Not Found Error:** Ensure the CSV file exists at the specified path and that the path is correctly referenced in the script.

**Column Not Found:** Verify that the column names used in the script ('balance', 'age') match those in your dataset. Adjust the script accordingly if column names differ.

**Model Performance:** If the model's performance is not as expected, consider reviewing the feature selection and preprocessing steps. Adjust model parameters if necessary.
## License
This script is licensed under the MIT License. See the LICENSE file for details.



### Explanation of the `README.md` Structure

1. **Overview**: Describes the purpose and steps of the script.
2. **Requirements**: Lists necessary Python libraries and provides installation instructions.
3. **File Path**: Notes the importance of updating the file path to the dataset.
4. **Script Details**: Provides a detailed breakdown of the script, including:
   - Importing libraries.
   - Loading and inspecting data.
   - Preparing data for modeling.
   - Training and evaluating the Decision Tree classifier.
   - Visualizing the decision tree.
5. **Troubleshooting**: Offers guidance on common issues related to file paths, column names, and model performance.
6. **License**: Specifies the license under which the script is distributed.

This `README.md` file should guide users in setting up and understanding the script, as well as help troubleshoot any issues that arise.
