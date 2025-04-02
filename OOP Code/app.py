import streamlit as st
from OOP_BSM import BSM

st.title("Black-Scholes-Merton Option Pricing")

S0 = st.number_input("Stock Price (S0)", value=100.0)
K = st.number_input("Strike Price (K)", value=105.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
q = st.number_input("Dividend Yield (q)", value=0.02)
T = st.number_input("Time to Maturity (years)", value=1.0)
v = st.number_input("Volatility (σ)", value=0.2)

if st.button("Calculate Prices"):
    model = BSM(S0, K, r, q, T, v)
    st.success(f"European Call Price: {model.EuropeanCall():.4f}")
    st.success(f"European Put Price: {model.EuropeanPut():.4f}")
