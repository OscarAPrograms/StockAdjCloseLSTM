# Adj-Close-Neural-Network
Disclaimer: Because this neural network relies purely on stock statistics (without sentiment analysis), it can not predict large shifts in a stock's adjusted closing price due to financial news. This project was merely an exercise for me to build skills in Python and TensorFlow.

"Adj-Close-Neural-Network" is an LTSM neural network model used to predict a stock's adjusted close after its next trading day based on statistics (open price, high price, low price, and adjusted close price) from the last 7 days. Given a user-inputted stock ticker, this program downloads up to 10 years of stock data (from Yahoo Finance), divides this data into 7-day batches, and uses this parsed data to train the LSTM neural network. If the program is run while the market is still open, the model includes current stock statistics in its prediction.

Run "pip3 install -r requirements.txt" in the command line to install this project’s dependencies: yfinance, tensorflow, scikit-learn.

