from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import Booster
import os
import itertools

app = Flask(__name__)

# Load models
catboost_model = CatBoostRegressor()
catboost_model.load_model('models/CatBoost_model.cbm')

xgboost_model = XGBRegressor()
xgboost_model.load_model('models/XGBoost_model.json')

# Load LightGBM model as Booster
lightgbm_booster = Booster(model_file='models/LightGBM_model.txt')

# Optimized weights from the notebook
weights = {
    'CatBoost': 0.3333,
    'XGBoost': 0.3333,
    'LightGBM': 0.3333
}

# Feature engineering functions from the notebook
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new

def add_statistical_features(df, features):
    df_new = df.copy()
    # Ensure numerical features are float
    for feature in features:
        df_new[feature] = pd.to_numeric(df_new[feature], errors='coerce').astype(float)
    # Compute statistical features as scalars
    df_new["row_mean"] = df_new[features].mean(axis=1).astype(float)
    df_new["row_std"] = df_new[features].std(axis=1).astype(float)
    df_new["row_max"] = df_new[features].max(axis=1).astype(float)
    df_new["row_min"] = df_new[features].min(axis=1).astype(float)
    df_new["row_median"] = df_new[features].median(axis=1).astype(float)
    # Replace NaN with 0
    df_new[["row_mean", "row_std", "row_max", "row_min", "row_median"]] = \
        df_new[["row_mean", "row_std", "row_max", "row_min", "row_median"]].fillna(0.0)
    return df_new

# Label encoder for Sex
le = LabelEncoder()
le.fit(['male', 'female'])  # Assuming male/female from the dataset

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Fit polynomial features on sample data to get feature names
sample_data = pd.DataFrame({
    'Age': [30], 'Height': [170], 'Weight': [70], 'Duration': [30],
    'Heart_Rate': [120], 'Body_Temp': [37]
})
poly.fit(sample_data[numerical_features])
poly_feature_names = poly.get_feature_names_out(numerical_features)

# Load feature names from the CatBoost model
FEATURES = catboost_model.feature_names_

# Verify FEATURES and categorical features
def verify_features():
    expected_initial_features = numerical_features + ['Sex']
    if not all(f in FEATURES for f in expected_initial_features):
        raise ValueError("CatBoost model features do not include all expected initial features")
    cat_features = ['Sex']
    # Ensure cat_features are column names, not indices
    if cat_features and isinstance(cat_features[0], int):
        cat_features = [FEATURES[i] for i in cat_features]
    print("Features verified:", FEATURES)
    print("Categorical features:", cat_features)
    return cat_features

CAT_FEATURES = verify_features()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'Age': float(request.form['Age']),
            'Height': float(request.form['Height']),
            'Weight': float(request.form['Weight']),
            'Duration': float(request.form['Duration']),
            'Heart_Rate': float(request.form['Heart_Rate']),
            'Body_Temp': float(request.form['Body_Temp']),
            'Sex': request.form['Sex'].lower()
        }

        if data['Sex'] not in ['male', 'female']:
            raise ValueError("Sex must be 'male' or 'female'")

        # Create DataFrame
        df = pd.DataFrame([data], index=[0])

        # Encode 'Sex' as int for CatBoost and XGBoost
        le = LabelEncoder()
        le.fit(['male', 'female'])
        df['Sex'] = le.transform(df['Sex'].str.lower()).astype(int)

        # Feature engineering
        df = add_feature_cross_terms(df, numerical_features)
        df = add_interaction_features(df, numerical_features)
        df = add_statistical_features(df, numerical_features)

        # Polynomial features
        poly_df = pd.DataFrame(poly.transform(df[numerical_features]), columns=poly_feature_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

        # Build final input DataFrame with expected features
        df_final = pd.DataFrame(0.0, index=[0], columns=FEATURES)
        for feature in FEATURES:
            if feature in df.columns:
                value = df[feature].iloc[0]
                if feature == 'Sex':
                    value = int(value) if not pd.isna(value) else 0
                else:
                    try:
                        value = float(value) if not pd.isna(value) else 0.0
                    except (ValueError, TypeError):
                        value = 0.0
                df_final.at[0, feature] = value
            else:
                df_final.at[0, feature] = 0.0

        # For LightGBM: convert 'Sex' back to string and to categorical
        df_final['Sex'] = df_final['Sex'].map({0: 'male', 1: 'female'})
        df_final['Sex'] = df_final['Sex'].astype('category')

        # Ensure categorical features for CatBoost
        catboost_pool = Pool(df_final, cat_features=CAT_FEATURES)

        # Predictions
        catboost_pred = np.expm1(catboost_model.predict(catboost_pool))
        xgboost_pred = np.expm1(xgboost_model.predict(df_final))
        lightgbm_pred = np.expm1(lightgbm_booster.predict(df_final))

        # Blended prediction
        blended_pred = (
            weights['CatBoost'] * catboost_pred +
            weights['XGBoost'] * xgboost_pred +
            weights['LightGBM'] * lightgbm_pred
        )
        blended_pred = np.clip(blended_pred, 1, 314)[0]

        return render_template('index.html', prediction=f'Predicted Calories Burned: {blended_pred:.2f}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)