# Stock-Market-Neural-Network
"Stock-Market-Neural-Network" is an LTSM neural network model used to predict a stock's next adjusted closing price based on statistics (open price, high price, low price, and adjusted close price) from the last 7 days.

Given a user-inputted stock ticker, this program downloads up to 10 years of stock data (from Yahoo Finance), divides this data into 7-day batches, and uses this parsed data to train an LSTM neural network. Finally, this neural network predicts the next trading day's adjusted closing price based on statistics from the last 7 days.
