import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import datetime

def fetch_stock_data(symbol, start_date):
    data = yf.download(symbol, start=start_date)
    return data

def plot_stock_data(data, symbol, predicted_data=None, display_by_month=True):
    if display_by_month:
        data = data.resample('M').mean()

    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Closing Prices')
    if predicted_data is not None:
        plt.plot(predicted_data['Predicted'], label='Predicted Prices', linestyle='dashed')

    if display_by_month:
        plt.xlabel('Month')
    else:
        plt.xlabel('Year')

    plt.ylabel('Price')
    plt.title(f'{symbol} Stock Market Analysis')
    plt.legend()
    plt.show()

def generate_predictions(data):
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(changepoint_prior_scale=0.05)  
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  
    model.fit(df)
    future = model.make_future_dataframe(periods=365)  
    forecast = model.predict(future)
    predicted_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted'}).set_index('Date')
    return predicted_data

current_date = datetime.date.today()

choice = input("Enter '1' to check the market trends or '2' to analyze a specific stock: ")

if choice == '1':
    data = fetch_stock_data('^GSPC', start_date=current_date - datetime.timedelta(days=365*5))  
    predicted_data = generate_predictions(data)
    display_by_month = input("Display data by month? (y/n): ").lower() == 'y'
    plot_stock_data(data, 'Market Trends', predicted_data, display_by_month)
elif choice == '2':
    symbol = input("Enter the stock symbol (e.g., AAPL): ")
    data = fetch_stock_data(symbol, start_date=current_date - datetime.timedelta(days=365*5))  
    predicted_data = generate_predictions(data)
    display_by_month = input("Display data by month? (y/n): ").lower() == 'y'
    plot_stock_data(data, symbol, predicted_data, display_by_month)
else:
    print("Invalid choice. Please try again.")
