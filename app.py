import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Option Pricing Models", layout="wide")

st.title("📈 Option Pricing Models")
st.markdown("**Binomial Model & Black–Scholes Model (Actual vs Theoretical)**")

# ======================
# Sidebar Inputs
# ======================
st.sidebar.header("Input Parameters")

symbol = '^SPX'
r_rate = st.sidebar.number_input("Risk Free Rate", value=0.0375, step=0.0005)
N = st.sidebar.slider("Binomial Steps", 10, 300, 100)

# ======================
# Fetch Data
# ======================
@st.cache_data
def load_price_data(symbol):
    end = date.today()
    start = end.replace(year=end.year - 2)
    ticker = yf.Ticker(symbol)
    return ticker.history(start=start, end=end), ticker

df, ticker = load_price_data(symbol)

st.subheader("Underlying Price Data")
st.dataframe(df.tail())

# ======================
# Annual Volatility
# ======================
def annual_vol(df):
    log_return = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    return log_return.std() * np.sqrt(252)

vol = annual_vol(df)
st.metric("Annualized Volatility", f"{vol:.2%}")

# ======================
# Options Chain
# ======================
expiries = ticker.options
expiry = st.sidebar.selectbox("Select Expiry", expiries)
opt_chain = ticker.option_chain(expiry)

expiry_date = pd.to_datetime(expiry).date()
T = (expiry_date - date.today()).days / 365

S_now = df["Close"].iloc[-1]

# ======================
# Binomial Model
# ======================
u = np.exp(vol * np.sqrt(T / N))
d = 1 / u

def binomial_call(S, K, T, r, u, d, N):
    dt = T / N
    p = (np.exp(r * dt) - d) / (u - d)
    C = {}

    for m in range(N + 1):
        ST = S * (u ** m) * (d ** (N - m))
        C[(N, m)] = max(ST - K, 0)

    for k in range(N - 1, -1, -1):
        for m in range(k + 1):
            C[(k, m)] = np.exp(-r * dt) * (
                p * C[(k + 1, m + 1)] + (1 - p) * C[(k + 1, m)]
            )
    return C[(0, 0)]

binomial_prices = {
    K: binomial_call(S_now, K, T, r_rate, u, d, N)
    for K in opt_chain.calls["strike"]
}

bin_df = pd.DataFrame.from_dict(binomial_prices, orient="index", columns=["Theoretical"])
bin_df["Actual"] = opt_chain.calls.set_index("strike")["lastPrice"]

# ======================
# Black–Scholes Model
# ======================
def black_scholes(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

bs_call = {
    K: black_scholes(S_now, K, T, r_rate, vol, "call")
    for K in opt_chain.calls["strike"]
}

bs_put = {
    K: black_scholes(S_now, K, T, r_rate, vol, "put")
    for K in opt_chain.puts["strike"]
}

bs_call_df = pd.DataFrame.from_dict(bs_call, orient="index", columns=["Theoretical"])
bs_call_df["Actual"] = opt_chain.calls.set_index("strike")["lastPrice"]

bs_put_df = pd.DataFrame.from_dict(bs_put, orient="index", columns=["Theoretical"])
bs_put_df["Actual"] = opt_chain.puts.set_index("strike")["lastPrice"]

# ======================
# Display Results
# ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Binomial Call Pricing")
    st.dataframe(bin_df)

with col2:
    st.subheader("📊 Black–Scholes Call Pricing")
    st.dataframe(bs_call_df)

st.subheader("📊 Black–Scholes Put Pricing")
st.dataframe(bs_put_df)

# ======================
# Plots
# ======================
def plot_comparison(df, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Price")
    ax.grid(True)
    st.pyplot(fig)

plot_comparison(bin_df, "Binomial Model: Actual vs Theoretical")
plot_comparison(bs_call_df, "Black–Scholes Call: Actual vs Theoretical")
plot_comparison(bs_put_df, "Black–Scholes Put: Actual vs Theoretical")
