# Import libraries
from tensorflow.keras.optimizers import Adam
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# setting the side bar to collapsed taa k footer jo ha wo sahi dikhay
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
# Add an image from an online resource
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take input from the user of the app about the start and end date

# Sidebar
st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2021, 1, 1))
# end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
end_date = st.sidebar.date_input('End date', date.today())

# Add ticker symbol list
ticker_list = ["BTC-USD","ETH-USD","AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)
# Add Date as a column to the dataframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)



# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
# Add a select box to choose the column for forecasting
column_1 = st.selectbox('Choose variable to see the trends', data.columns[1:])

st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data[column_1], title=column_1 + ' price of the stock', width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to choose the column for forecasting
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])
# Subsetting the data
data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

# ADF test to check stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
# Make same plot in Plotly
st.write("## Plotting the decomposition in Plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

# Model selection
models = ['SARIMA', 'Random Forest', 'LSTM', 'Prophet']
selected_model = st.sidebar.selectbox('Select the model for forecasting', models)

if selected_model == 'SARIMA':
    # SARIMA Model
    # User input for SARIMA parameters
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)
    seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model = model.fit()

    # Print model summary
    st.header('Model Summary')
    st.write(model.summary())
    st.write("---")

    # Forecasting using SARIMA
    st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
            unsafe_allow_html=True)

    forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
    # Predict the future values
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
    predictions = predictions.predicted_mean
    # Add index to the predictions
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, "Date", predictions.index, True)
    predictions.reset_index(drop=True, inplace=True)
    st.write("Predictions", predictions)
    st.write("Actual Data", data)
    st.write("---")

    # Plot the data
    fig = go.Figure()
    # Add actual data to the plot
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    # Add predicted data to the plot
    fig.add_trace(
        go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted',
                   line=dict(color='red')))
    # Set the title and axis labels
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    # Display the plot
    st.plotly_chart(fig)

elif selected_model == 'Random Forest':
    # Random Forest Model
    st.header('Random Forest Regression')

    # Splitting data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Feature engineering
    train_X, train_y = train_data['Date'], train_data[column]
    test_X, test_y = test_data['Date'], test_data[column]

    # Initialize and fit the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(train_X.values.reshape(-1, 1), train_y.values)

    # Predict the future values
    predictions = rf_model.predict(test_X.values.reshape(-1, 1))

    # Calculate mean squared error
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)

    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Combine training and testing data for plotting
    combined_data = pd.concat([train_data, test_data])

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[column], mode='lines', name='Actual',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (Random Forest)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)

elif selected_model == 'LSTM':
    # LSTM Model
    st.header('Long Short-Term Memory (LSTM)')
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    # Sort the data by date
    data = data.sort_values(by='Date')
    df= data.copy() 
    
    #Extract the column values
    data= data[column].values.reshape(-1,1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    # Function to create dataset with lookback
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)


    # Create train and test datasets with lookback
    look_back = st.slider('Select the sequence length look_back', 5, 100, 5)

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

   # Reshape input data to [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=100)

    # Define the number of days to predict
    days_to_predict = st.slider('Select days to forecast', 10, 1000, 5)
    
    # Make predictions for the test dataset
    test_predictions = model.predict(X_test)
    test_predictions = scaler.inverse_transform(test_predictions)

    # Make predictions for the specified additional number of days
    predictions = []
    current_batch = X_test[-1]

    for i in range(days_to_predict):
        current_pred = model.predict(current_batch.reshape(1, look_back, 1))
        predictions.append(current_pred[0][0])  # Extract the scalar value from the prediction
        current_batch = np.append(current_batch[1:], current_pred[0][0])  # Append scalar prediction to current batch

    extended_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    def plot_stock_prediction(df, train_size, data, test_predictions, extended_predictions, days_to_predict, look_back):
        fig = go.Figure()
        # Plot training data
        fig.add_trace(go.Scatter(x=df['Date'][:train_size], y=data[:train_size].flatten(),
                            mode='lines', name='Training Data', line=dict(color='blue')))

        # Plot testing data
        fig.add_trace(go.Scatter(x=df['Date'][train_size+look_back:], y=data[train_size+look_back:].flatten(),
                            mode='lines', name='Testing Data', line=dict(color='green')))

        # Plot predicted data for test dataset
        fig.add_trace(go.Scatter(x=df['Date'][train_size+look_back:train_size+look_back+len(test_predictions)],
                            y=test_predictions.flatten(), mode='lines',
                            name='Predicted Data (Test Dataset)', line=dict(color='orange')))

        # Plot extended predicted data
        fig.add_trace(go.Scatter(x=pd.date_range(start=df['Date'].iloc[-1], periods=days_to_predict+1)[1:],
                            y=extended_predictions.flatten(),
                            mode='lines', name=f'Extended Predicted Data ({days_to_predict} days)', line=dict(color='red')))

        fig.update_layout(title='Stock Price Prediction using LSTM',
                        xaxis_title='Date',
                        yaxis_title='Stock Price', width=1000, height=400)
        # fig.show()
        return fig

    # Call the function with appropriate arguments
    fig1= plot_stock_prediction(df, train_size, data, test_predictions, extended_predictions, days_to_predict, look_back)
    st.plotly_chart(fig1)
    
elif selected_model == 'Prophet':
    # Prophet Model
    st.header('Facebook Prophet')

    # Prepare the data for Prophet
    prophet_data = data[['Date', column]]
    prophet_data = prophet_data.rename(columns={'Date': 'ds', column: 'y'})

    # Create and fit the Prophet model
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    # Forecast the future values
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)

    # Plot the forecast
    fig = prophet_model.plot(forecast)
    plt.title('Forecast with Facebook Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

st.write("Model selected:", selected_model)

