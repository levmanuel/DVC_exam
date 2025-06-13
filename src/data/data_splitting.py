import pandas as pd
import ssl
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context
df = pd.read_csv('https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv')

y = df["silica_concentrate"]
X = df.drop(columns=["silica_concentrate", "date"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('./data/processed_data/X_train.csv', index=False)
X_test.to_csv('./data/processed_data/X_test.csv', index=False)
y_train.to_csv('./data/processed_data/y_train.csv', index=False)
y_test.to_csv('./data/processed_data/y_test.csv', index=False)

print(df.info())