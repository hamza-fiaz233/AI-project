import yfinance as yf
import yfinance as yf
from datetime import datetime, timezone
from langchain_core.tools import tool

@tool
def get_quote(ticker: str) -> dict:
    """Return the latest price, daily change, volume, and timestamp for a US equity ticker."""
    try:
        t = yf.Ticker(ticker.upper())
        # fast_info provides quick access without heavy historical downloads
        info = t.fast_info
        
        return {
            "ticker": ticker.upper(),
            "price": getattr(info, "last_price", None),
            "previous_close": getattr(info, "previous_close", None),
            "volume": getattr(info, "last_volume", None),
            "as_of_utc": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to fetch quote for {ticker}: {str(e)}"}

@tool
def get_history(ticker: str, period: str = "1mo", interval: str = "1d") -> dict:
    """Return historical OHLCV price series summary statistics for a given ticker and time period."""
    try:
        t = yf.Ticker(ticker.upper())
        df = t.history(period=period, interval=interval)
        if df.empty:
            return {"error": f"No historical data found for {ticker} with period {period}."}
        
        summary = {
            "ticker": ticker.upper(),
            "period": period,
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "data_points": len(df),
            "recent_close": float(df["Close"].iloc[-1]),
            "max_high": float(df["High"].max()),
            "min_low": float(df["Low"].min())
        }
        return summary
    except Exception as e:
        return {"error": f"Failed to fetch history for {ticker}: {str(e)}"}

@tool
def get_ratios(ticker: str) -> dict:
    """Return fundamental valuation ratios such as P/E, EPS, market cap, and dividend yield."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        
        return {
            "ticker": ticker.upper(),
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow")
        }
    except Exception as e:
        return {"error": f"Failed to fetch ratios for {ticker}: {str(e)}"}

if __name__ == "__main__":
    # Quick sanity test
    print("Testing get_quote for AAPL:")
    print(get_quote.invoke({"ticker": "AAPL"}))
    
    print("\nTesting get_ratios for AAPL:")
    print(get_ratios.invoke({"ticker": "AAPL"}))