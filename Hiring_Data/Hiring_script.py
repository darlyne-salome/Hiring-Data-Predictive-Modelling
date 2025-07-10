import pandas as pd

# Loading 
file_path = 'C:/Users/PC/OneDrive/Desktop/Hiring_Data/Combined_hiring_data.xlsx'
train_df = pd.read_excel(file_path, sheet_name='Training_Data')
predict_df = pd.read_excel(file_path, sheet_name='Prediction_Data')

# Convert interview dates to datetime
train_df['Interview start'] = pd.to_datetime(train_df['Interview start'])
train_df['Interview end'] = pd.to_datetime(train_df['Interview end'])
predict_df['Interview start'] = pd.to_datetime(predict_df['Interview start'])
predict_df['Interview end'] = pd.to_datetime(predict_df['Interview end'])

# New columns: Interview Duration (in days)
train_df['Interview_Duration'] = (train_df['Interview end'] - train_df['Interview start']).dt.days
predict_df['Interview_Duration'] = (predict_df['Interview end'] - predict_df['Interview start']).dt.days

# Droping columns (including the original date columns)
cols_to_drop = ['Source.Name','ID','Approved','Sourcing start','Interview start','Interview end','Offered','Filled','Status']
train_encoded = train_df.drop(columns = cols_to_drop)
predict_encoded = predict_df.drop(columns = cols_to_drop)

# Encoding categorical columns
categorical_drop = ['BU Region','FP']
train_encoded = pd.get_dummies(train_encoded, columns = categorical_drop)
predict_encoded = pd.get_dummies(predict_encoded, columns = categorical_drop)

# Aligning columns in prediction set with those in training set
predict_encoded = predict_encoded.reindex(columns = train_encoded.columns.drop('Time_to_fill'), fill_value = 0)

# Results
print("Data loaded successfully.")
print("Training data shape:", train_df.shape)
print("Prediction data shape:", predict_df.shape)
print("Columns aligned for prediction:", list(predict_encoded.columns))

# regression tools needed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Features and Targets
X = train_encoded.drop(columns=['Time_to_fill'])  
y = train_encoded['Time_to_fill']

# Splitting for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=26)

# Training
model = RandomForestRegressor(n_estimators=100, random_state=26)
model.fit(X_train, y_train)

# Model Accuracy
val_preds = model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print(f"Validation MAE: {round(mae, 2)} days")

# Predicting open jobs
final_predictions = model.predict(predict_encoded)

# Rounding the predictions to whole numbers
rounded_predictions = [round(x) for x in final_predictions]

# Adding predictions
predict_df['Predicted_Time_to_Fill'] = rounded_predictions

# Exporting to a new Excel file (does not overwrite original)
output_path = 'C:/Users/PC/OneDrive/Desktop/Hiring_Data/Final_Prediction_Report.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    train_df.to_excel(writer, sheet_name='Training_Data', index=False)
    predict_df.to_excel(writer, sheet_name='Prediction_Data', index=False)

print("Prediction saved to Excel successfully.")

import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(y_val, val_preds, color='skyblue', edgecolor='black')

# Diagonal line for reference
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')

# Labels and title
plt.xlabel("Actual Time to Fill")
plt.ylabel("Predicted Time to Fill")
plt.title("Actual vs. Predicted Time to Fill")
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()

# Clear the previous figure
plt.clf()

# histogram 
plt.figure(figsize=(8,5))
plt.hist(rounded_predictions, bins=15, color='seagreen', edgecolor='black')

# Labels and title
plt.xlabel("Predicted Days to Fill")
plt.ylabel("Number of Job Roles")
plt.title("Distribution of Predicted Time to Fill")
plt.grid(axis='y')
plt.tight_layout()

# Show the plot
plt.show()

# Clear the previous figure
plt.clf()







