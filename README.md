### README for NBA Player Performance Prediction Model

#### Overview

The goal of this mini project is to predict NBA players who are worth it for inverstor, since they are going to survive for more than 5 years. So the plan is to construct an ML model capable of making accurate predictions. So we followed some structured steps, starting by data exploration, feature selection; then model training, evaluation and fine-tuning.

#### Data exploration and training scripts

- **`nba_analyze.ipynb`**: The file containing the initial data exploration and some visualization to understandour data set
- **`nba_invest.ipynb`**: The equivalent of test.py, the file containing the whole script for training and resulting into the final model.

#### Data Exploration and Feature Selection
Exploring the dataset started with analyzing the data distributions and correlations between the features and the target variable. And that showed us some interesting insights:
1. **Missing Data**: Although there was some missing data, it was not extreme. on top of that it was only the 3P% feature which we later discovered was not an important feature in the correlation matrix. so we discard the whole feature. However, we did add imputer to the pipelines. This also ensures that when getting a request with some missing data we still can handle that request.
2. **Class Imbalance**: We identified a mild class imbalance, having around 500 negatives vs 800 positives. While this imbalance is not severe, was handled in the pipeline. it directed us to using some specific models like balanced random forrest classifier, which handles imbalanced data internally.
3. **Correlation Matrix**: Upon generating a correlation matrix, we examined the relationships between various features. This helped us identify multicollinearity issues and confirm which features were the most relevant. We prioritized features that had strong relationships with the target variable and discarded those with little predictive value.

The features we selected were based on the statistical significance as one can see when plotting the correlation matrix. the list of key features included:
features_to_keep = ['GP', 'MIN', 'PTS', 'FTM', 'REB', 'TOV', 'STL', 'BLK', 'FG%']

#### Model Selection and Initial Testing

We began by testing five different models to assess which ones would handle our data the best. The models were chosen based on their ability to manage imbalanced data, their flexibility in handling missing values, and their performance with various types of input data:
1. **Logistic Regression** – a strong baseline for binary classification tasks.
2. **Random Forest** – known for its flexibility and good performance on tabular data.
3. **Balanced Random Forest** – specifically designed for handling imbalanced data.
4. **Gradient Boosting** – known for its accuracy and handling of complex relationships.
5. **Support Vector Machine (SVM)** – effective in high-dimensional spaces.

The first stage of testing these models involved using default parameters and evaluating them through cross-validation. We paid special attention to how each model handled the slight imbalance and missing values in the dataset.

#### Scoring and Model Evaluation

We implemented a custom scoring function that prioritized **recall** as requested by the client. However, we also took into account **specificity**, making sure it is larger than a minimum value we judged should be 0.49, as a way to ensure balanced performance, and avoid a model learning a heurisi of giving positive output all the time.

#### Pipeline Design and Preprocessing

Our philosophy for the pipeline design was based on handling the variability in the input data:
- **Imputation**: For some models (like Random Forest and Logistic Regression), imputing missing values was crucial. While for others like XGBoost designed to take care of missing values, we didn't add any imputation. We used a `SimpleImputer` in the pipeline to replace missing values based on the mean of the feature.
- **Normalization**: For models sensitive to feature scaling, such as Logistic Regression, we added a `StandardScaler` to normalize the data. For tree-based models like Random Forest and all other models, scaling was not required.
- **Class Imbalance Handling**: For models that were not inherently designed to handle class imbalance, we used techniques like class weighting. For models like the Balanced Random Forest, this wasn’t necessary as the algorithm inherently handles imbalanced data.

#### Hyperparameter Tuning and Final Model Selection

After the initial phase of testing, we performed **GridSearchCV** on the top three models to fine-tune their hyperparameters. Based on the results of cross-validation, we narrowed down our focus to **Logistic Regression** and **Balanced Random Forest**. These two models provided the best weighted recall values.

We further validated these models on a test dataset. In this phase, we considered both the **cross-validation results** and the **inference time**. While **Logistic Regression** was significantly faster, the **Balanced Random Forest** provided better overall results and was still fast enough for real-time predictions. Given its superior handling of class imbalance and its better performance in recall and precision metrics, we chose the Balanced Random Forest model for our final deployment.

### Project API Structure

- **`app.py`**: The Flask app that serves the model for predictions.
- **`model/balancedrandomforest_final_model.joblib`**: The trained model saved as a joblib file.
- **`requirements.txt`**: List of Python packages needed to run the project.

### Setup Instructions

To run the NBA performance prediction model locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AyoubTHABIT/nba_model.git
   cd nba_model
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv nba_model_venv
   source nba_model_venv/bin/activate  # For MacOS/Linux
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**
   ```bash
   python app.py
   ```
   This will start the Flask server, and the model will be ready to accept requests.

### Example API Usage

You can send a POST request to the Flask API for predictions. Here's an example using `curl`:

#### Example Request:
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"GP": 80, "MIN": 35, "PTS": 22, "FG%": 50, "3P Made": 3, "3P%": 42, "FT%": 85, "REB": 5, "AST": 7, "STL": 1, "BLK": 0.5, "TOV": 2}'
```

This API request sends player statistics, and the model will predict the outcome. If a feature is missing, the model can handle it by imputing missing values, so you can omit features in the request.

For example, omitting `PTS` (Points):
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"GP": 80, "MIN": 35, "FG%": 50, "3P Made": 3, "3P%": 42, "FT%": 85, "REB": 5, "AST": 7, "STL": 1, "BLK": 0.5, "TOV": 2}'
```
The model will still generate a prediction, filling in missing values where necessary.

### Final Notes

- **Missing Data**: The pipeline uses imputation for handling missing data automatically.
- **Model Robustness**: The final Balanced Random Forest model is robust to imbalanced classes and missing data, making it reliable in real-world scenarios.
- **Performance Consideration**: While Balanced Random Forest has a slightly longer inference time compared to Logistic Regression, it offers better predictive performance, which is why it was chosen as the final model.

This project represents a robust, real-world approach to model building, handling missing data, class imbalances, and making trade-offs between speed and accuracy.