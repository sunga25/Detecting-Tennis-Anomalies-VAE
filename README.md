## Detecting Anomalies in Professional Men's Tennis Tournament Draws
This project investigates potential manipulation in professional men’s tennis tournament draws using statistical analysis and artificial intelligence (AI). By analyzing ATP match data from 2000 to 2017, this study applies a Variational Autoencoder (VAE) model to detect anomalies that may indicate non-randomness or biases in the draw process.

## Table of Contents

Overview

Features

Data

Installation

Usage

Results

Project Structure

Contributing

License

Acknowledgements


## Overview
The integrity of tournament draws is crucial for maintaining fairness in professional tennis. This project leverages machine learning, specifically Variational Autoencoders, to analyze historical match data and identify patterns or anomalies that suggest potential manipulation in the draw process.

## Features
Load and preprocess ATP match data from 2000 to 2017.

Engineer features relevant to player rankings, age differences, and match statistics.

Train a Variational Autoencoder (VAE) model to detect anomalies in match outcomes.

Analyze anomalies by year, player, and tournament to assess potential biases.

Save analysis results to CSV files for further review.

## Data
Match data used in this project is sourced from publicly available ATP records. The data includes information such as player rankings, match outcomes, tournament dates, and player statistics.

Data Files:
atp_matches_2000.csv to atp_matches_2017.csv

Ensure these data files are placed in the root directory of the project before running the scripts.


## Installation
Clone the repository and install the required dependencies.

git clone https://github.com/your-username/tennis-draw-anomalies.git

cd tennis-draw-anomalies

pip install -r requirements.txt

## Usage:

Prepare Data: 

Ensure all ATP match data CSV files are in the root directory.

Run the Main Script: 

Execute the main script to load data, preprocess it, train the model, and analyze anomalies.

python main.py

Review Results: 

Anomaly detection results will be saved as CSV files in the output directory.

## Results
The results of the analysis include:

Detected Anomalies:

A CSV file listing all detected anomalies in matchups.

Anomalies per Year: 

A summary of anomalies detected per year.

Anomalies by Player and Tournament: 

Analysis of which players and tournaments had the most anomalies.

## Project Structure

tennis-draw-anomalies/
│
├── data/                           # Directory for raw data files
│   ├── atp_matches_2000.csv
│   ├── atp_matches_2001.csv
│   └── ...
│
├── outputs/                        # Directory for saving analysis results
│   ├── anomalies.csv
│   ├── anomalies_per_year.csv
│   └── ...
│
├── scripts/                        # Scripts for data processing and analysis
│   ├── load_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── analyze_anomalies.py
│
├── main.py                         # Main script to run the entire pipeline
├── requirements.txt                # List of required Python packages
└── README.md                       # Project README file


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Thanks to the ATP Tour for making match data publicly available.
Special thanks to all contributors and the open-source community for their tools and resources.
