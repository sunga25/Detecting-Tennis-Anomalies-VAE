import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(start_year=2000, end_year=2017):
    csv_files = [f'atp_matches_{year}.csv' for year in range(start_year, end_year + 1)]
    dataframes = []

    for file in csv_files:
        try:
            data = pd.read_csv(file)
            logging.info(f"Loaded {file}: {len(data)} rows")
            dataframes.append(data)
        except FileNotFoundError:
            logging.warning(f"File {file} not found.")
            continue

    if not dataframes:
        raise FileNotFoundError("No CSV files found. Ensure data files are present.")

    combined_df = pd.concat(dataframes, ignore_index=True)
    logging.info(f"Total rows after combining all dataframes: {len(combined_df)}")
    return combined_df

def preprocess_data(df):
    logging.info(f"Before preprocessing: {len(df)} rows")
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.drop_duplicates().reset_index(drop=True)

    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df['tourney_date_ordinal'] = df['tourney_date'].apply(lambda x: x.toordinal() if pd.notnull(x) else None)

    df['winner_id'] = df['winner_id'].fillna(df['winner_id'].mode().iloc[0])
    df['loser_id'] = df['loser_id'].fillna(df['loser_id'].mode().iloc[0])
    df['winner_name'] = df['winner_name'].fillna('Unknown')
    df['loser_name'] = df['loser_name'].fillna('Unknown')

    logging.info(f"After preprocessing: {len(df)} rows")
    logging.info(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    logging.info(f"Years present in the data: {sorted(df['tourney_date'].dt.year.unique())}")
    return df

def engineer_features(df):
    numeric_columns = ['winner_rank', 'loser_rank', 'winner_seed', 'loser_seed', 
                       'winner_age', 'loser_age', 'w_svpt', 'l_svpt', 'w_ace', 
                       'l_ace', 'w_df', 'l_df', 'w_bpSaved', 'l_bpSaved',
                       'tourney_date_ordinal']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info(f"Column {col}: {df[col].isnull().sum()} null values")
        else:
            logging.warning(f"Column '{col}' not found in dataframe.")

    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    df['age_diff'] = df['winner_age'] - df['loser_age']
    df['service_diff'] = df['w_svpt'] - df['l_svpt']
    df['ace_diff'] = df['w_ace'] - df['l_ace']
    df['df_diff'] = df['w_df'] - df['l_df']
    df['bp_saved_diff'] = df['w_bpSaved'] - df['l_bpSaved']

    numeric_columns.extend(['age_diff', 'service_diff', 'ace_diff', 'df_diff', 'bp_saved_diff'])

    logging.info(f"After feature engineering: {len(df)} rows")
    return df, numeric_columns

def create_vae_model(input_dim, latent_dim=2):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(latent_dim)
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(latent_dim,)),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])

    class VAEModel(tf.keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAEModel, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        def call(self, inputs):
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded

    vae = VAEModel(encoder, decoder)
    vae.compile(optimizer='adam', loss='mse')

    return vae

def detect_anomalies(df, threshold=None):
    if threshold is None:
        threshold = df['rank_diff'].abs().quantile(0.85)  # 85th percentile
    logging.info(f"Using anomaly threshold: {threshold}")
    logging.info(f"Years in the data before anomaly detection: {sorted(df['tourney_date'].dt.year.unique())}")

    anomalies = []
    for i, row in df.iterrows():
        rank_diff = row['winner_rank'] - row['loser_rank']
        if abs(rank_diff) > threshold:
            anomalies.append(row)

    anomalies_df = pd.DataFrame(anomalies)

    if anomalies_df.empty:
        logging.warning("No anomalies detected!")
        return anomalies_df

    yearly_counts = anomalies_df['tourney_date'].dt.year.value_counts().sort_index()
    logging.info(f"Anomalies per year:\n{yearly_counts}")

    logging.info(f"Years in the anomalies: {sorted(anomalies_df['tourney_date'].dt.year.unique())}")
    logging.info(f"Total anomalies: {len(anomalies_df)}")

    return anomalies_df

def analyze_anomalies(anomalies):
    anomalies['tourney_name'] = anomalies['tourney_name'].fillna('Unknown')

    anomalies_per_year = anomalies.groupby(anomalies['tourney_date'].dt.year).size()
    anomalies_per_player = pd.concat([anomalies['winner_name'], anomalies['loser_name']]).value_counts()
    anomalies_per_tournament = anomalies['tourney_name'].value_counts()

    grand_slams = anomalies[anomalies['tourney_name'].str.contains('Grand Slam', case=False, na=False)]['tourney_name'].value_counts()
    masters_1000 = anomalies[anomalies['tourney_name'].str.contains('Masters 1000', case=False, na=False)]['tourney_name'].value_counts()

    return anomalies_per_year, anomalies_per_player, anomalies_per_tournament, grand_slams, masters_1000

def save_results(anomalies, anomalies_per_year, anomalies_per_player, anomalies_per_tournament, grand_slams, masters_1000):
    anomalies.to_csv('anomalies.csv', index=False)
    anomalies_per_year.to_csv('anomalies_per_year.csv')
    anomalies_per_player.to_csv('anomalies_per_player.csv')
    anomalies_per_tournament.to_csv('anomalies_per_tournament.csv')
    grand_slams.to_csv('anomalies_per_grand_slam.csv')
    masters_1000.to_csv('anomalies_per_masters_1000.csv')

    pd.DataFrame(anomalies_per_player.index.tolist(), columns=['Player']).to_csv('most_anomalies_players.csv', index=False)
    pd.DataFrame(anomalies_per_tournament.index.tolist(), columns=['Tournament']).to_csv('most_anomalies_tournaments.csv', index=False)
    pd.DataFrame(grand_slams.index.tolist(), columns=['Grand Slam']).to_csv('most_anomalies_grand_slams.csv', index=False)
    pd.DataFrame(masters_1000.index.tolist(), columns=['Masters 1000']).to_csv('most_anomalies_masters_1000.csv', index=False)

def main():
    logging.info("Starting script...")

    df = load_data()
    logging.info(f"Total rows after loading: {len(df)}")

    df = preprocess_data(df)
    df, numeric_columns = engineer_features(df)
    logging.info(f"Total rows after preprocessing and feature engineering: {len(df)}")

    # Calculate rank difference
    df['rank_diff'] = df['winner_rank'] - df['loser_rank']

    # Log rank difference statistics
    logging.info(f"Rank difference stats:\n{df['rank_diff'].describe()}")
    logging.info(f"Rank difference percentiles:")
    for percentile in [50, 75, 90, 95, 99]:
        logging.info(f"{percentile}th percentile: {df['rank_diff'].abs().quantile(percentile/100)}")

    # Log some statistics about the 'winner_rank' and 'loser_rank' columns
    logging.info(f"Winner rank stats:\n{df['winner_rank'].describe()}")
    logging.info(f"Loser rank stats:\n{df['loser_rank'].describe()}")

    X = df[numeric_columns].copy()
    y = df['winner_rank'] - df['loser_rank']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train_scaled = X_train_scaled.astype('float32')
    X_test_scaled = X_test_scaled.astype('float32')

    vae = create_vae_model(X_train_scaled.shape[1])

    logging.info("Model compiled. Starting training...")
    history = vae.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=32, 
                      validation_data=(X_test_scaled, X_test_scaled), verbose=1)
    logging.info("Model training complete.")

    anomalies = detect_anomalies(df)
    if not anomalies.empty:
        logging.info(f"Number of anomalies: {len(anomalies)}")
        logging.info(f"Anomalies date range: {anomalies['tourney_date'].min()} to {anomalies['tourney_date'].max()}")

        anomalies_per_year, anomalies_per_player, anomalies_per_tournament, grand_slams, masters_1000 = analyze_anomalies(anomalies)

        logging.info("Saving results...")
        save_results(anomalies, anomalies_per_year, anomalies_per_player, anomalies_per_tournament, grand_slams, masters_1000)
    else:
        logging.warning("No anomalies detected. Skipping analysis and result saving.")

    logging.info("Script execution completed.")

if __name__ == "__main__":
    main()
