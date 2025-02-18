# Code which generates an estimation of American options price assuming
# inputs of strike, underlying spot, maturity, interest rate and volatility

import numpy as np
import QuantLib as ql

# Prices American options using grid method
def grid_method(spot, strike, mat_time, int_rate, vol):
    # Define Black-Scholes-Merton process
    today = ql.Date().todaysDate()
    risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, int_rate, ql.Actual365Fixed()))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    initial_val = ql.QuoteHandle(ql.SimpleQuote(spot))
    process = ql.BlackScholesMertonProcess(initial_val, dividend_ts, risk_free_ts, volatility)

    #Define the option
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    end_date = today + int(365*mat_time)
    am_exercise = ql.AmericanExercise(today, end_date)
    american_option = ql.VanillaOption(payoff, am_exercise)

    #Define the pricing engine
    xGrid = 200
    tGrid = 2000
    engine = ql.FdBlackScholesVanillaEngine(process, tGrid, xGrid)
    american_option.setPricingEngine(engine)

    return np.float64(american_option.NPV())