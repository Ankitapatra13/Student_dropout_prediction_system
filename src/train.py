import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib

from src.features import create_features
from src.preprocess import get_preprocessor


def train_model(data_path: str):

    # Load data
    df = pd.read_csv(data_path)
    data = df.sample(10000, random_state=42)

    # Target creation
    data["dropout_labels"] = pd.qcut(
        data["dropout_risk"],
        q=3,
        labels=["Low", "Medium", "High"]
    )

    data = data.drop("dropout_risk", axis=1)

    # Feature engineering
    data = create_features(data)

    # Split
    X = data.drop("dropout_labels", axis=1)
    y = data["dropout_labels"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Preprocessing
    preprocessor = get_preprocessor(X_train)

    # Class weights
    class_weight = {0: 1, 1: 1, 2: 1.4}
    sample_weights = [class_weight[val] for val in y_train]

    # Model
    model = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=3,
        learning_rate=0.03,
        random_state=42,
        n_jobs=1
    )

    # Pipeline
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Train
    pipe.fit(X_train, y_train, model__sample_weight=sample_weights)

    # Evaluate
    y_pred = pipe.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(pipe, "models/model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("Model saved successfully!")

    return pipe