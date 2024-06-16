import numpy as np
import tensorflow as tf
import yfinance as yf

from sklearn.model_selection import train_test_split

# Number of iterations the NN takes to train on the data.
EPOCHS = 50
# Fraction of the data used for evaluating the NN.
TEST_SIZE = 0.3
# Number of consecutive days inputted in the NN to predict the next Adj
# Close.
NUM_DAYS = 7

def main():
    """
    Main method of the program. Used to parse data and fit/test the NN.

    Author: Oscar Afraymovich 
    """

    # Iterates until a valid stock ticker is inputted.
    while(True): 
        # Check input for a stock ticker.
        ticker = input("Input a valid stock ticker to analyze.")
        # Download entire history of the stock ticker.
        data = yf.download(ticker, period = "10y")

        # If the stock ticker is not found: print an error message.
        if (len(data) == 0): 
            print("Stock ticker not found, please try again.")
        # If the stock ticker is found: terminate the loop.
        else:
            break

    # Drop the Close column from data.
    data.drop(['Close'], axis = 1, inplace = True)
    # Add a Next Close (next trading day's Adj Close) column to data.
    data["Next Close"] = data["Adj Close"].shift(-1)

    print(data)

    # Convert the pandas DataFrame to a NumPy array.
    data_set = data.to_numpy()

    data_set

    # y contains Next Close values with at least NUM_DAYS-1 past days.
    y = list(data_set[NUM_DAYS-1:, -1]) 
    
    x = []
    # For every day in data_set with at least NUM_DAYS-1 previous days:
    for i in range(NUM_DAYS-1, len(data_set)):
            past_stock_stats = [] 
            # Append the past NUM_DAYS stock stats to past_stock_stats.
            for j in range(0, NUM_DAYS):
                 past_stock_stats.append(data_set[j+(i-(NUM_DAYS-1)), :]) 
            # Append these stock stats to x.
            x.append(past_stock_stats)

    # Remove data from most recent trading day (no Next Close value).
    x.pop()
    y.pop()

    x_train, x_test, y_train, y_test = train_test_split(
         np.array(x), np.array(y), test_size = TEST_SIZE
    )

    # Get a compiled neural network
    ##model = get_model()

    print(y_train.shape)
    print(x_train.shape)


    # Fit model on training data
    ##model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    ##model.evaluate(x_test,  y_test, verbose=2)

'''
def get_model():
    """
    Returns a compiled LTSM neural network model. 

    Author: Oscar Afraymovich
    """
    model = tf.keras.models.Sequential()

    # Compile model
    model.compile(
        optimizer = "adam",
        loss = "mse",
        metrics = ["accuracy"]
    )     
    return model
    '''


if __name__ == "__main__":
    main()