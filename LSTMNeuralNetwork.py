import numpy as np
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Number of iterations the NN takes to train on the data.
EPOCHS = 50
# Fraction of the data used for evaluating the NN.
TEST_SIZE = 0.2
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
        ticker = input("\nInput a valid stock ticker to analyze. ")

        # Download up to 10 years of history on the stock ticker.
        print("Downloading data ...")
        data = yf.download(ticker, period = "10y", progress = False, auto_adjust=False)
        
        # If the stock ticker is not found: 
        if (len(data) == 0): 
            print("\nStock ticker not found, please try again.", end = "\r")
        # If the stock ticker is found:
        else:
            break

    # Drop the Close and Volume columns from data.
    data.drop(['Close', 'Volume'], axis = 1, inplace = True)
    # Add a Next Close (next trading day's Adj Close) column to data.
    data["Next Close"] = data["Adj Close"].shift(-1)

    market_open = ""
    while (market_open.lower() not in ["yes", "no"]):
         # Ask user if the stock is in its regular market hours:
         market_open = input("\nIs the market currently open? Enter either \"Yes\" or \"No\". ")
         
    # When in regular market hours, remove data from the latest trading day (incomplete).
    if (market_open == "Yes" or market_open == "yes"):
             data.drop(data.index[-1], inplace = True)

    # Convert the pandas DataFrame to a NumPy array.
    data_set = data.to_numpy()

    # y contains Next Close values (for days with at least NUM_DAYS-1 previous days).
    y = list(data_set[NUM_DAYS-1:, -1]) 
    
    x = []
    # For every day with at least NUM_DAYS-1 previous days:
    for i in range(NUM_DAYS-1, len(data_set)):
            past_stock_stats = [] 
            # Append the past NUM_DAYS stock stats to past_stock_stats.
            for j in range(0, NUM_DAYS): 
                 # Add market statistics from day i-(NUM_DAYS-1) to day i to past_stock_stats.
                 past_stock_stats.append(data_set[j + (i - (NUM_DAYS - 1)), :-1]) 
            # Append these stock stats to x.
            x.append(past_stock_stats)

    # Remove data from latest COMPLETE trading day from x and y (it has no fixed Next Close value)
    x.pop()
    y.pop() 

    # Split the data into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(
         np.array(x), np.array(y), test_size = TEST_SIZE
    )

    # Number of samples in x_train and x_test:
    train_samples = x_train.shape[0]
    test_samples = x_test.shape[0]

    # Reshape the input data to (samples * days, features):
    x_train_flat = x_train.reshape(-1, 4)
    x_test_flat = x_test.reshape(-1, 4)

    # Normalize features and return it to its original shape
    x_scaler = MinMaxScaler()
    x_train_flat = x_scaler.fit_transform(x_train_flat)
    x_train = x_train_flat.reshape(train_samples, NUM_DAYS, 4)

    x_test_flat =  x_scaler.transform(x_test_flat)
    x_test = x_test_flat.reshape(test_samples, NUM_DAYS, 4)

    # Turn outputs into a 2D array, normalize features, then flatten back into a 1D array:
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test =  y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data:
    model.fit(
         x_train, 
         y_train, 
         epochs=EPOCHS, 
         validation_data=(x_test, y_test)
         )

    # Evaluate neural network performance
    test_loss = model.evaluate(x_test, y_test, verbose=2)
    train_loss = model.evaluate(x_train, y_train, verbose=2)

    # Check for overfitting:
    print("MSE test loss of normalized predictions: ", test_loss)
    print("MSE train loss of normalized predictions: ", train_loss)

    # Stock statistics from the last NUM_DAYS complete trading days.
    currentData = []
    past_stock_stats = [] # past_stock_stats should be empty.

    # Store stock statistics from the last NUM_DAYS complete trading days in  past_stock_stats.
    for k in range(0, NUM_DAYS): 
        past_stock_stats.append(data_set[k + (len(data) - NUM_DAYS), :-1]) 
        # Append these stock stats to currentData.
    currentData.append(past_stock_stats)

    # Normalize currentData and resive it so it can be passed as input to the NN.
    currentData = np.array(currentData).reshape(NUM_DAYS,4)
    currentData = x_scaler.transform(currentData).reshape(1,NUM_DAYS,4)
    # Predict the next Adj Close:
    prediction = model.predict(currentData)
    print("Predicted next adjusted close: $", y_scaler.inverse_transform(prediction)[0][0])

    #model.save('StockLSTM.h5')

def get_model():
    """
    Returns a compiled LTSM neural network model. 

    Author: Oscar Afraymovich
    """
    model = tf.keras.models.Sequential([
         
         tf.keras.layers.LSTM(256, input_shape = (NUM_DAYS, 4)),
         tf.keras.layers.Dropout(0.2),
                  
         tf.keras.layers.Dense(128, activation = "tanh"),
         tf.keras.layers.Dropout(0.2),

         tf.keras.layers.Dense(64, activation = "tanh"),
         tf.keras.layers.Dropout(0.2),

         tf.keras.layers.Dense(32, activation = "tanh"),
         tf.keras.layers.Dropout(0.2),

         tf.keras.layers.Dense(1, activation = "linear"),
    ])

    # Compile model
    model.compile(
        optimizer = "adam",
        loss = "mse",
    )     
    return model


if __name__ == "__main__":
    main()