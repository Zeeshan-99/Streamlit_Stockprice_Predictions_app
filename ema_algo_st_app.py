import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Function to calculate the Close_Price_Diff based on the window of 'Sell' or 'Buy' signals
def calculate_close_price_diff(df):
    start_idx = None
    current_signal = None
    
    for idx, row in df.iterrows():
        signal = row['Signal']
        
        if signal != current_signal:
            if start_idx is not None:
                end_idx = idx - 1
                df.loc[end_idx, 'Close_Price_Diff'] = df.loc[start_idx, 'Close'] - df.loc[end_idx, 'Close']
            start_idx = idx
            current_signal = signal
    
    # Handle the last window
    if start_idx is not None and current_signal is not None:
        end_idx = df.index[-1]
        df.loc[end_idx, 'Close_Price_Diff'] = df.loc[start_idx, 'Close'] - df.loc[end_idx, 'Close']
        
# Streamlit app
st.title('AAPL Stock Analysis with Buy/Sell Signals')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    # Parse the Date column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the data by date (if not already sorted)
    data.sort_values('Date', inplace=True)

    # Calculate the 20-day, 50-day, and 100-day EMAs
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()

    # Initialize Signal column
    data['Signal'] = ''

    # Define tolerance for EMA convergence (5%)
    tolerance = 0.02

    # Find points where all three EMAs converge (within tolerance)
    data['EMAs_Converge'] = ((abs(data['EMA_20'] / data['EMA_50'] - 1) <= tolerance) &
                             (abs(data['EMA_20'] / data['EMA_100'] - 1) <= tolerance) &
                             (abs(data['EMA_50'] / data['EMA_100'] - 1) <= tolerance))

    # Determine buy or sell signals
    in_signal = False  # Flag to track if we're currently in a buy/sell signal
    last_signal = None  # Variable to store the last generated signal

    for i in range(2, len(data)-2):  # Adjust range to accommodate checking two candles ahead and behind
        if data['EMAs_Converge'][i]:
            # Check for "Buy" condition: EMAs are increasing and closing prices are also increasing
            if (data['Close'][i-2] < data['Close'][i-1] < data['Close'][i]) and (data['EMA_20'][i-1] > data['EMA_20'][i-2]) and (data['EMA_50'][i-1] > data['EMA_50'][i-2]) and (data['EMA_100'][i-1] > data['EMA_100'][i-2]):
                data.loc[i, 'Signal'] = 'Buy'
                in_signal = True
                last_signal = 'Buy'
            # Check for "Sell" condition: EMAs are decreasing and closing prices are also decreasing
            elif (data['Close'][i-2] > data['Close'][i-1] > data['Close'][i]) and (data['EMA_20'][i-1] < data['EMA_20'][i-2]) and (data['EMA_50'][i-1] < data['EMA_50'][i-2]) and (data['EMA_100'][i-1] < data['EMA_100'][i-2]):
                data.loc[i, 'Signal'] = 'Sell'
                in_signal = True
                last_signal = 'Sell'
            else:
                if in_signal:  # Continue with the same signal until conditions change
                    data.loc[i, 'Signal'] = last_signal

        else:
            in_signal = False  # Reset flag when EMAs do not converge

    # Filter data for points where signals are generated
    signals = data[data['Signal'].isin(['Buy', 'Sell'])]
    signals.reset_index(drop=True, inplace=True)

    # Calling the function to calculate Close_Price_Diff
    calculate_close_price_diff(signals)

    # Create a candlestick chart using Plotly
    fig = go.Figure()

    # Candlestick trace
    fig.add_trace(go.Candlestick(x=data['Date'],
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick',
                                 yaxis='y'))  # Assign to primary y-axis

    # EMAs traces
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_20'], mode='lines', name='EMA 20', yaxis='y'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_50'], mode='lines', name='EMA 50', yaxis='y'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['EMA_100'], mode='lines', name='EMA 100', yaxis='y'))

    # Buy and Sell signals
    fig.add_trace(go.Scatter(x=signals[signals['Signal'] == 'Buy']['Date'], y=signals[signals['Signal'] == 'Buy']['Low'],
                             mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
                             name='Buy Signal', yaxis='y'))
    fig.add_trace(go.Scatter(x=signals[signals['Signal'] == 'Sell']['Date'], y=signals[signals['Signal'] == 'Sell']['High'],
                             mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'),
                             name='Sell Signal', yaxis='y'))

    # Update layout for better visualization
    fig.update_layout(
        title='AAPL Stock Prices with Persistent Buy/Sell Signals at EMA Convergence (5% Tolerance)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date',
            fixedrange=False  # Allows zooming on the x-axis
        ),
        yaxis=dict(
            title='Price',
            side='left',
            showgrid=True,
            zeroline=False,
            fixedrange=False  # Allows zooming on the y-axis
        ),
        yaxis2=dict(
            title='Price',
            overlaying='y',
            side='right',
            showgrid=True,
            zeroline=False
        ),
        template='plotly_dark',
        width=1400,  # Set the width of the plot
        height=800   # Set the height of the plot
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please upload a CSV file to proceed.")
