from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def get_preprocessor(X):
    nominal_cat = ["gender"]
    ordinal_cat = ["risk_level"]
    num_cols = X.select_dtypes(include="number").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cat),
            ("ordinal", OrdinalEncoder(
                categories=[["Low", "Medium", "High"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ), ordinal_cat),
            ("numerical", "passthrough", num_cols),
        ],
        remainder="passthrough"
    )

    return preprocessor