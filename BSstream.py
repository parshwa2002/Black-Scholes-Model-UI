#!/usr/bin/env python
# coding: utf-8

# In[12]:





# In[10]:


import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import requests
import yfinance as yf

#BLACK SCHOLES MODEL: USED FOR PRICING EUROPEAN OPTIONS.
#ASSUMPTIONS:1)EUROPEAN OPTIONS
            #2)NO DIVIDEND
            #3)NO TRANSACTION COST
            #4)INTEREST RATES & VOLATILITY ARE KNOWN AND CONSTANT.
            #5)LOGNORMAL DISTRIBUTION OF STOCKS
            #6)MARKETS ARE RANDOM
#DEFINING VARIABLES:
#define the variables
#S =    current stock price
#K =    strike price
#V =    volatility
#R =    interest rate
#T =    time to expiry

def black_scholes_model(S, K, T, R, V, option_type='call'): #DEFINING THE BLACK SCHOLES MODEL
    """Calculate Black-Scholes option price."""
    d1 = (np.log(S / K) + (R + 0.5 * V**2) * T) / (V * np.sqrt(T))
    d2 = d1 - (V * np.sqrt(T))
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2) #HERE N(d1)->PROB THAT PRICE OF STOCK WOULD BE HIGHER THAN STRIKE PRICE
    elif option_type == 'put':
        price = K * np.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)# N(d2)-> PROB THAT THE STOCK WILL "ACTUALLY" END UP BEING ABOVE STRIKE PRICE
    return price
#PAYOFF GRAPHS FOR TRADER,HEDGERS.
def payoff_long_call(S, K, premium):
    return np.maximum(S - K, 0) - premium

def payoff_long_put(S, K, premium):
    return np.maximum(K - S, 0) - premium

def payoff_short_call(S, K, premium):
    return premium - np.maximum(S - K, 0)

def payoff_short_put(S, K, premium):
    return premium - np.maximum(K - S, 0)

# Function to fetch options data from Yahoo Finance(As it would help students & traders to compare their theoretical values with the market values in real-time).
def fetch_options_data(symbol):
    ticker = yf.Ticker(symbol)
    options_dates = ticker.options
    if not options_dates:
        return None
    options_data = {}
    for date in options_dates:
        calls = ticker.option_chain(date).calls
        puts = ticker.option_chain(date).puts
        options_data[date] = {'calls': calls, 'puts': puts}
    return options_data


# Title
st.title('Black-Scholes Pricing Model')

# Subtitle with the creators
st.markdown(
    """
    <div style='text-align: center;'>
        <b>CREATED BY PARSHWA SHAH</b><br>
        <a href='http://www.linkedin.com/in/parshwa-shah-3a0015252' target='_blank' style='text-decoration: none; color: gray;'>LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar inputs
st.sidebar.header('Black-Scholes Inputs')
S = st.sidebar.number_input('Current Market Price', value=120.0)
K = st.sidebar.number_input('Strike Price', value=100.0)
T = st.sidebar.number_input('Time to Expiry (annualized)', value=1.0)
V = st.sidebar.number_input('Vol (σ)', value=0.1)
R = st.sidebar.number_input('Interest Rate', value=0.08)

# Calculate option prices
call_price = black_scholes_model(S, K, T, R, V, option_type='call')
put_price = black_scholes_model(S, K, T, R, V, option_type='put')

# Display option prices
st.subheader('Option Prices')
st.write(f'**Call Value:** ${call_price:.2f}')
st.write(f'**Put Value:** ${put_price:.2f}')

# Heatmap Parameters
st.sidebar.header('Heatmap Inputs')
min_spot_price = st.sidebar.slider('Min Spot Price', 50, 5000, 80) #assigning min,max,default values.
max_spot_price = st.sidebar.slider('Max Spot Price', 50, 5000, 120)
min_volatility = st.sidebar.slider('Min Volatility for Heatmap', 0.01, 1.0, 0.1)
max_volatility = st.sidebar.slider('Max Volatility for Heatmap', 0.01, 1.0, 0.3)

#create arrays
spot_prices = np.linspace(min_spot_price, max_spot_price, 20)
volatilities = np.linspace(min_volatility, max_volatility, 20)
heatmap_call = np.zeros((len(spot_prices), len(volatilities)))
heatmap_put = np.zeros((len(spot_prices), len(volatilities)))



# Compute heatmap values
for i, S in enumerate(spot_prices):
    for j, V in enumerate(volatilities):
        heatmap_call[i, j] = black_scholes_model(S, K, T, R, V, option_type='call')
        heatmap_put[i, j] = black_scholes_model(S, K, T, R, V, option_type='put')

# Display heatmaps
st.subheader('Options Price - Interactive Heatmap')
st.write('Price variation of the Option prices with changes based on Volatility, Interest Rate, and Time to Expiry.')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Call Heatmap
c1 = ax1.contourf(spot_prices, volatilities, heatmap_call.T, cmap='RdYlGn', levels=20)
fig.colorbar(c1, ax=ax1)
ax1.set_title('Call Price Heatmap')
ax1.set_xlabel('Underlying Price')
ax1.set_ylabel('Volatility')

# Put Heatmap
c2 = ax2.contourf(spot_prices, volatilities, heatmap_put.T, cmap='RdYlGn', levels=20)
fig.colorbar(c2, ax=ax2)
ax2.set_title('Put Price Heatmap')
ax2.set_xlabel('Underlying Price')
ax2.set_ylabel('Volatility')

st.pyplot(fig)


st.subheader('Payoff Graphs')

spot_prices = np.linspace(0.5 * S, 1.5 * S, 100)
call_payoff = payoff_long_call(spot_prices, K, call_price)
put_payoff = payoff_long_put(spot_prices, K, put_price)
short_call_payoff = payoff_short_call(spot_prices, K, call_price)
short_put_payoff = payoff_short_put(spot_prices, K, put_price)

fig, ax = plt.subplots()
ax.plot(spot_prices, call_payoff, label='Long Call')
ax.plot(spot_prices, put_payoff, label='Long Put')
ax.plot(spot_prices, short_call_payoff, label='Short Call')
ax.plot(spot_prices, short_put_payoff, label='Short Put')
ax.axhline(0, color='black', lw=1)
ax.axvline(S, color='gray', linestyle='--')
ax.set_xlabel('Spot Price')
ax.set_ylabel('Payoff')
ax.legend()
ax.set_title('Option Payoff Diagrams')

st.pyplot(fig)

# Fetching options data from Yahoo Finance
st.sidebar.header('Yahoo Finance Options Data')
symbol = st.sidebar.text_input('Enter Stock Symbol', value='AAPL')

if st.sidebar.button('Fetch Options Data'):
    data = fetch_options_data(symbol)
    if data is None:
        st.error("Failed to fetch data. Please check the stock symbol and try again.")
    else:
        st.subheader(f'Options Data for {symbol}')
        
        # Display calls and puts for each expiration date
        for date in data:
            st.write(f"**Expiration Date: {date}**")
            st.write("Calls:")
            st.write(data[date]['calls'])
            st.write("Puts:")
            st.write(data[date]['puts'])





# In[ ]:




