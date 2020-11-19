import requests
import pandas 
import scipy 
import numpy 
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    data = pandas.read_csv(TRAIN_DATA_URL,header=None)
    data = data.iloc[:,1:]
    arr = data.values
    x = arr[0]
    y = arr[1]
    num = numpy.sum((x-numpy.mean(x))*(y-numpy.mean(y)))
    den = numpy.sum((x-numpy.mean(x))**2)
    beta_1 = num/den
    beta_0=numpy.mean(y)-beta_1*numpy.mean(x)
    return beta_0 + beta_1*area
    # YOUR IMPLEMENTATION HERE
    
    


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
