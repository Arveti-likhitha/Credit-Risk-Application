# Credit-Risk-Application

The goal of this project is to classify loan applicants into two categories: Good credit risk and Bad credit risk. Alongside building an accurate model, I focused on interpretability and user experience by developing a clean, responsive, and theme-adaptive Streamlit web application.

The German Credit dataset contains 1,000 rows and 20 features including personal, financial, and employment-related information. The target variable is a binary label: 1 for good credit risk and 0 for bad credit risk.

The main objectives were:
- Clean and preprocess the dataset.
- Build a robust machine learning model.
- Interpret the model and identify key influencing factors.
- Develop a user-friendly and theme-aware app for predictions.
- Generate actionable business insights.


Key preprocessing steps included:
- Handling missing values in 'Saving accounts' and 'Checking account' by imputing them as 'unknown'.
- Encoding categorical variables using one-hot encoding.
- Normalizing continuous features like Age, Credit Amount, and Duration.
- Ensuring feature alignment between training and prediction pipelines using a shared reference file.

I used a Random Forest Classifier because it performs well with tabular data and supports feature importance calculations.

Key training steps included:
- 80/20 train-test split.
- Hyperparameter tuning using GridSearchCV.
- Evaluation based on accuracy, precision, recall, F1-score, and ROC AUC.

The model was serialized using joblib, and the associated feature column structure was stored to maintain consistent input formatting.

On the test dataset, the model achieved:
- Accuracy: XX%
- Precision: XX%
- Recall: XX%
- F1 Score: XX%
- ROC AUC: XX%

These metrics are saved in a JSON file and are dynamically loaded in the app. A confusion matrix heatmap is also displayed for better understanding of true vs. false predictions.

The Streamlit app includes:
- A sidebar form for entering customer details for single prediction.
- Light/Dark mode toggle that changes background, text, and component colors.
- Prediction result card showing the risk category and a confidence bar.
- A bar chart showing top features impacting the prediction.
- LIME-based explanation embedded in a scrollable and styled iframe.
- CSV upload option for batch predictions.
- Prediction history for the current session.
- A collapsible model evaluation section.

The submission includes:
- Complete source code for training and the web app.
- Pre-trained model and evaluation metrics.
