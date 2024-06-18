import numpy as np
import tensorflow as tf
import yfinance as yf
from datetime import date
from datetime import datetime

from sklearn.model_selection import train_test_split

# Number of iterations the NN takes to train on the data.
EPOCHS = 50
# Fraction of the data used for evaluating the NN.
TEST_SIZE = 0.3
# Number of consecutive days inputted in the NN to predict the next
# trading day's Adj Close.
NUM_DAYS = 7

def main():
    """
    Main method of the program. Used to parse data and fit/test the NN.

    Author: Oscar Afraymovich 
    """

    # Iterates until a valid stock ticker is inputted.
    while(True): 
        # Check input for a stock ticker.
        ticker = input("\nInput a valid stock ticker to analyze. ")
        # Download up to 10 years of history on the stock ticker.
        data = yf.download(ticker, period = "10y", progress = False)

        # If the stock ticker is not found: print an error message.
        if (len(data) == 0): 
            print("\nStock ticker not found, please try again.", end = "\r")
        # If the stock ticker is found: terminate the loop.
        else:
            break

    # Drop the Close and Volume columns from data.
    data.drop(['Close', 'Volume'], axis = 1, inplace = True)
    # Add a Next Close (next trading day's Adj Close) column to data.
    data["Next Close"] = data["Adj Close"].shift(-1)

    # Iterates until market_open equals either "Yes" or "No"
    market_open = ""
    while ((market_open != "Yes") and (market_open != "No")):
         # Ask user if the stock is in its regular market hours
         market_open = input("\nIs the inputted stock currently in its "
                            "regular market hours? Make sure to enter either"
                            " \"Yes\" or \"No\". ")
         
    # When in regular market hours, remove data on the latest trading 
    # day (its stock stats are not yet fixed).
    if (market_open == "Yes"):
             # Drop the last row in data 
             data.drop(data.index[-1], inplace = True)

    # Convert the pandas DataFrame to a NumPy array.
    data_set = data.to_numpy()

    # Scale the data_set so all values are between 0 and 1
    data_set = data_set/1000

    # y contains Next Close values with at least NUM_DAYS-1 past days.
    y = list(data_set[NUM_DAYS-1:, -1]) 
    
    x = []
    # For every day in data_set with at least NUM_DAYS-1 previous days:
    for i in range(NUM_DAYS-1, len(data_set)):
            past_stock_stats = [] 
            # Append the past NUM_DAYS stock stats to past_stock_stats.
            for j in range(0, NUM_DAYS): 
                 """"
                 For days i-(NUM_DAYS-1) to i, add the day's stats (row
                 from data_set minus the Next Close column) to
                 past_stock_stats.
                 """
                 past_stock_stats.append(data_set[j+(i-(NUM_DAYS-1)), :-1]) 
            # Append these stock stats to x.
            x.append(past_stock_stats)

    """
    Remove data on the latest COMPLETE trading day from x and y (it has
    no Next Close value if the market is closed or no fixed Next Close
    value if the market is open)
    """
    x.pop()
    y.pop() 

    # Split the data into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(
         np.array(x), np.array(y), test_size = TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)
    

    # Stock statistics from the last NUM_DAYS complete trading days.
    currentData = []
    past_stock_stats = [] # past_stock_stats should be empty.

    # From day len(data)-NUM_DAYS to day len(data)-1:
    for k in range(0, NUM_DAYS): 
        """"
        For the last NUM_DAYS complete trading days, add the day's
        stats (row from data_set minus the Next Close column) to
        past_stock_stats.
        """
        past_stock_stats.append(data_set[k+(len(data)-NUM_DAYS), :-1]) 
        # Append these stock stats to currentData.
    currentData.append(past_stock_stats)

    # Resize currentData so it can be passed as input to the NN.
    currentData = np.array(currentData).reshape(1,7,4)
    # Predict the next Adj Close (scale price back by 1000)
    prediction = model.predict(currentData)*1000 
    print("Predicted next adjusted close: $", prediction[0][0])



def get_model():
    """
    Returns a compiled LTSM neural network model. 

    Author: Oscar Afraymovich
    """
    model = tf.keras.models.Sequential([
         
         tf.keras.layers.LSTM(256, input_shape = (7, 4)),
         
         tf.keras.layers.Dense(256, activation = "tanh"),
         
         tf.keras.layers.Dropout(0.5),

         tf.keras.layers.Dense(256, activation = "tanh"),

         tf.keras.layers.Dropout(0.5),

         tf.keras.layers.Dense(256, activation = "tanh"),

         tf.keras.layers.Dropout(0.5),

         tf.keras.layers.Dense(1, activation = "linear"),
    ])

    # Compile model
    model.compile(
        optimizer = "adam",
        loss = "mse"
    )     
    return model


if __name__ == "__main__":
    main()
