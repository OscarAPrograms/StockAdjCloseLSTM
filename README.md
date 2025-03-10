# StockAdjCloseLSTM
**Disclaimer:** This project was merely an exercise for me to build familiarity with Python and TensorFlow. This neural network's predictions are limited by stock statistics and could be improved with sentiment analysis of financial news.

**StockAdjCloseLSTM** is an LTSM neural network model used to predict a stock's next adjusted close based on statistics (open price, high price, low price, and adjusted close price) from the last 7 complete trading days. Given a user-inputted stock ticker, this program downloads up to 10 years of stock data (from Yahoo Finance), divides this data into 7-day batches, and uses this parsed data to train the neural network.

## Imports
After downloading "LSTMNeuralNetwork.py" and "requirements.txt", run "pip3 install -r requirements.txt" inside the project's directory to install its dependencies: yfinance, tensorflow, scikit-learn.
