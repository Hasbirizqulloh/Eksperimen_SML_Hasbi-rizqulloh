import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

def load_data(path='./obesity_data.csv'):
    
    df = pd.read_csv(path)
    return df

def encode_categorical(df):
    le_gender = LabelEncoder()
    le_obesity = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])  # Male=1, Female=0
    df['ObesityCategory'] = le_obesity.fit_transform(df['ObesityCategory'])  # Label to numeric

    return df, le_gender, le_obesity

def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def split_data(df, target='ObesityCategory', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir='preprocessing/obesity_preprocessing'):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

def save_encoder_scaler(le_gender, le_obesity, scaler, output_dir='preprocessing/obesity_preprocessing'):
    with open(f'{output_dir}/label_encoder_gender.pkl', 'wb') as f:
        pickle.dump(le_gender, f)
    with open(f'{output_dir}/label_encoder_obesity.pkl', 'wb') as f:
        pickle.dump(le_obesity, f)
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def main():
    df = load_data()
    df, le_gender, le_obesity = encode_categorical(df)
    df, scaler = scale_features(df, features=['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel'])
    X_train, X_test, y_train, y_test = split_data(df)

    save_processed_data(X_train, X_test, y_train, y_test)
    save_encoder_scaler(le_gender, le_obesity, scaler)

    print("✅ Preprocessing selesai. Data tersimpan di folder 'obesity_preprocessing'.")

if __name__ == "__main__":
    main()
