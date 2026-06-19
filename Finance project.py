# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         STATISTICAL PAIRS TRADING ENGINE — COINTEGRATION-BASED             ║
# ║         Project 4 | Quantitative Trading | Intermediate → Advanced         ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Strategy  : Cointegration-based long-short mean reversion                 ║
# ║  Universe  : S&P 500 (configurable subset)                                 ║
# ║  Tests     : ADF · Engle-Granger · Johansen                                ║
# ║  Signal    : Ornstein-Uhlenbeck z-score                                    ║
# ║  Firms     : D.E. Shaw · Citadel · Two Sigma · Goldman Sachs Stat-Arb     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ─── PROJECT OVERVIEW ──────────────────────────────────────────────────────────
#
#  This notebook implements a production-style statistical pairs trading engine.
#  The strategy identifies stock pairs that move together in the long run
#  (cointegrated) but periodically diverge. When the spread between two stocks
#  deviates significantly from its historical mean, we bet on reversion.
#
#  Mathematical Foundation:
#    Spread_t = log(P_A_t) - β · log(P_B_t)
#    Z_t      = (Spread_t - μ_rolling) / σ_rolling
#    Signal   = SHORT spread if Z > +2.0 (overvalued relative to B)
#               LONG  spread if Z < -2.0 (undervalued relative to B)
#               EXIT         if |Z| < 0.5
#               STOP-LOSS    if |Z| > 3.0
#
#  Cointegration: Two non-stationary I(1) series X and Y are cointegrated if
#    a linear combination (Y - β·X) is stationary — i.e., mean-reverting.
#
# ─── SYSTEM ARCHITECTURE ───────────────────────────────────────────────────────
#
#  [Data Layer]   →  [Pair Screener]  →  [Statistical Tests]  →  [Signal Engine]
#      ↓                   ↓                    ↓                      ↓
#  yfinance           Sector filter          ADF + EG             Z-score calc
#  Wikipedia         Within-sector         Johansen              Entry/exit logic
#                    pair combos           OU half-life           Stop-loss
#
#  [Signal Engine]  →  [Backtest Engine]  →  [Risk Analytics]  →  [Portfolio]
#          ↓                   ↓                    ↓                  ↓
#     Position array      P&L per day          Sharpe, DD         Aggregate
#     Entry/exit dates    TC subtraction       Beta to SPY        Equity curve
#
# ─── REAL-WORLD FINANCE USE CASE ───────────────────────────────────────────────
#
#  D.E. Shaw pioneered this in the late 1980s. Today:
#  • Goldman Sachs Quantitative Strategies desk runs hundreds of pairs
#  • Millennium Management's equity L/S pods use similar models
#  • Two Sigma's systematic equity strategies extend to baskets of 10-50 stocks
#  • Entry-level quant researcher interviews often include this as a case study
#
# ══════════════════════════════════════════════════════════════════════════════


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — INSTALLATION                                                      ║
# ║  Run this cell first. Runtime → Restart after installation.                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# !pip install -q yfinance statsmodels scipy plotly kaleido tqdm tabulate
# !pip install -q pandas-datareader


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — IMPORTS & GLOBAL SETTINGS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import warnings
warnings.filterwarnings('ignore')

# ── Core Data Science ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ── Financial Data ─────────────────────────────────────────────────────────────
import yfinance as yf

# ── Statistical Tests ──────────────────────────────────────────────────────────
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS

# ── Visualization ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# ── Utilities ──────────────────────────────────────────────────────────────────
import time
import datetime
from tqdm import tqdm
from tabulate import tabulate
from typing import Dict, List, Tuple, Optional

# ── Style Configuration ────────────────────────────────────────────────────────
plt.style.use('dark_background')
sns.set_palette("husl")
pio.templates.default = "plotly_dark"

# Color palette (institutional dark theme)
COLORS = {
    'primary':    '#00D4FF',   # Cyan — main line
    'secondary':  '#FF6B35',   # Orange — secondary
    'positive':   '#00FF88',   # Green — long trades
    'negative':   '#FF4444',   # Red — short trades
    'neutral':    '#888888',   # Grey — inactive
    'background': '#0A0E1A',   # Dark navy — background
    'grid':       '#1A2035',   # Lighter navy — grid lines
    'text':       '#E0E0E0',   # Light — text
    'gold':       '#FFD700',   # Gold — highlights
    'purple':     '#9B59B6',   # Purple — tertiary
}

print("✅ All libraries imported successfully.")
print(f"   pandas {pd.__version__} | numpy {np.__version__} | statsmodels {sm.__version__}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — CONFIGURATION PARAMETERS                                          ║
# ║  All strategy parameters in one place for easy tuning                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # ── Date Range ─────────────────────────────────────────────────────────────
    'START_DATE':            '2019-01-01',   # 5 years of data
    'END_DATE':              '2024-12-31',
    'TRAIN_RATIO':           0.60,           # 60% training, 40% out-of-sample

    # ── Universe ───────────────────────────────────────────────────────────────
    'MAX_STOCKS':            120,            # Limit for Colab speed (increase for full scan)
    'MIN_PRICE':             5.0,            # Exclude penny stocks
    'MIN_VOLUME':            500_000,        # Min avg daily volume

    # ── Cointegration Filters ──────────────────────────────────────────────────
    'EG_PVALUE_THRESHOLD':   0.05,           # Engle-Granger p-value cutoff
    'ADF_STAT_THRESHOLD':   -2.9,            # ADF test statistic cutoff
    'JOHANSEN_CONFIRM':      True,           # Require Johansen to confirm EG

    # ── OU Process Filters ─────────────────────────────────────────────────────
    'HALF_LIFE_MIN':         5,              # Min half-life in trading days
    'HALF_LIFE_MAX':         60,             # Max half-life in trading days

    # ── Signal Parameters ──────────────────────────────────────────────────────
    'ZSCORE_WINDOW':         60,             # Rolling z-score lookback (days)
    'ENTRY_ZSCORE':          2.0,            # Enter trade at |z| > this
    'EXIT_ZSCORE':           0.5,            # Exit trade at |z| < this
    'STOP_ZSCORE':           3.0,            # Stop-loss at |z| > this

    # ── Risk & Costs ───────────────────────────────────────────────────────────
    'TRANSACTION_COST':      0.001,          # 10bps per side (0.20% round trip)
    'MAX_PAIRS':             25,             # Max pairs to trade simultaneously
    'ANNUAL_RISK_FREE':      0.05,           # 5% annual risk-free rate

    # ── Visualization ──────────────────────────────────────────────────────────
    'TOP_N_PAIRS_DISPLAY':   5,              # Number of pairs to show in detail
}

# ── Derived Parameters ─────────────────────────────────────────────────────────
RISK_FREE_DAILY = CONFIG['ANNUAL_RISK_FREE'] / 252

print("📋 Configuration loaded:")
print(f"   Date Range    : {CONFIG['START_DATE']} → {CONFIG['END_DATE']}")
print(f"   Universe Size : Up to {CONFIG['MAX_STOCKS']} stocks")
print(f"   Signal        : Entry |z|>{CONFIG['ENTRY_ZSCORE']} | Exit |z|<{CONFIG['EXIT_ZSCORE']} | Stop |z|>{CONFIG['STOP_ZSCORE']}")
print(f"   Transaction   : {CONFIG['TRANSACTION_COST']*100:.1f}bps per side ({CONFIG['TRANSACTION_COST']*200:.1f}bps round trip)")
print(f"   OU Half-Life  : {CONFIG['HALF_LIFE_MIN']}–{CONFIG['HALF_LIFE_MAX']} trading days")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — UTILITY FUNCTIONS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def print_header(title: str, width: int = 70) -> None:
    """Print a formatted section header."""
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

def print_subheader(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n  ┌─ {title} {'─' * max(0, 55 - len(title))}┐")

def format_pct(x: float, decimals: int = 2) -> str:
    """Format a decimal as a percentage string."""
    return f"{x * 100:.{decimals}f}%"

def format_ratio(x: float, decimals: int = 3) -> str:
    """Format a ratio with sign coloring indicator."""
    return f"{x:+.{decimals}f}"

def annualized_sharpe(returns: pd.Series, rf: float = RISK_FREE_DAILY) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    excess = returns - rf
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)

def max_drawdown(cum_returns: pd.Series) -> float:
    """Compute maximum drawdown from a cumulative return series."""
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max.replace(0, np.nan)
    return drawdown.min()

def calmar_ratio(returns: pd.Series) -> float:
    """Compute Calmar ratio: CAGR / |Max Drawdown|."""
    cum = (1 + returns).cumprod()
    dd = abs(max_drawdown(cum))
    cagr = (cum.iloc[-1]) ** (252 / len(returns)) - 1
    return cagr / dd if dd != 0 else 0.0

def compute_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute market beta of strategy returns relative to benchmark."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else np.nan

print("✅ Utility functions defined.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — DATA COLLECTION: S&P 500 UNIVERSE                                 ║
# ║  Scrapes Wikipedia for S&P 500 constituents with GICS sector info           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 1: BUILDING S&P 500 UNIVERSE")

def get_sp500_constituents() -> pd.DataFrame:
    """
    Scrape S&P 500 constituent list from Wikipedia.
    Returns DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry.
    
    Wikipedia is the most reliable free source for S&P 500 membership and
    GICS sector classification — identical to what Bloomberg provides.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        print("  📡 Fetching S&P 500 constituent list from Wikipedia...")
        tables = pd.read_html(url)
        sp500 = tables[0]  # First table has current constituents

        # Standardize column names
        sp500.columns = [c.strip() for c in sp500.columns]

        # Fix common column name variations
        col_map = {
            'Symbol':           'ticker',
            'Security':         'company',
            'GICS Sector':      'sector',
            'GICS Sub-Industry':'sub_industry',
        }
        sp500 = sp500.rename(columns={k: v for k, v in col_map.items() if k in sp500.columns})

        # Clean ticker symbols — some have dots (BRK.B → BRK-B for yfinance)
        sp500['ticker'] = sp500['ticker'].str.replace('.', '-', regex=False)

        print(f"  ✅ Retrieved {len(sp500)} S&P 500 constituents across {sp500['sector'].nunique()} sectors")
        return sp500[['ticker', 'company', 'sector', 'sub_industry']].copy()

    except Exception as e:
        print(f"  ⚠️  Wikipedia scrape failed: {e}")
        print("  🔄 Using fallback hardcoded sector mapping...")
        return _fallback_sp500()

def _fallback_sp500() -> pd.DataFrame:
    """
    Fallback sector mapping for the most liquid S&P 500 stocks.
    Used if Wikipedia scraping fails.
    """
    data = {
        'ticker': [
            # Technology
            'AAPL','MSFT','NVDA','GOOGL','META','AVGO','ORCL','AMD','QCOM','TXN',
            # Financials
            'JPM','BAC','WFC','GS','MS','BLK','SCHW','AXP','USB','PNC',
            # Healthcare
            'LLY','UNH','JNJ','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN',
            # Consumer Discretionary
            'AMZN','TSLA','HD','MCD','NKE','SBUX','TJX','BKNG','LOW','CMG',
            # Energy
            'XOM','CVX','COP','EOG','SLB','PSX','MPC','VLO','PXD','OXY',
            # Industrials
            'GE','HON','CAT','UPS','DE','MMM','LMT','RTX','BA','FDX',
            # Consumer Staples
            'PG','KO','PEP','COST','WMT','PM','MO','CL','GIS','K',
            # Utilities
            'NEE','DUK','SO','D','AEP','EXC','XEL','ED','ETR','PPL',
            # Real Estate
            'PLD','AMT','EQIX','CCI','PSA','WELL','DLR','SPG','EQR','AVB',
            # Materials
            'LIN','APD','SHW','FCX','NEM','NUE','ALB','CF','MOS','PPG',
            # Communication Services
            'NFLX','DIS','CMCSA','VZ','T','TMUS','EA','TTWO','OMC','IPG',
        ],
        'sector': (
            ['Information Technology'] * 10 +
            ['Financials'] * 10 +
            ['Health Care'] * 10 +
            ['Consumer Discretionary'] * 10 +
            ['Energy'] * 10 +
            ['Industrials'] * 10 +
            ['Consumer Staples'] * 10 +
            ['Utilities'] * 10 +
            ['Real Estate'] * 10 +
            ['Materials'] * 10 +
            ['Communication Services'] * 10
        ),
    }
    df = pd.DataFrame(data)
    df['company'] = df['ticker']
    df['sub_industry'] = df['sector']
    return df

# ── Execute Data Collection ────────────────────────────────────────────────────
sp500_df = get_sp500_constituents()

# Limit universe for Colab performance (increase MAX_STOCKS for full scan)
if len(sp500_df) > CONFIG['MAX_STOCKS']:
    # Sample proportionally from each sector to maintain diversity
    sp500_df = (sp500_df
                .groupby('sector', group_keys=False)
                .apply(lambda g: g.head(max(2, CONFIG['MAX_STOCKS'] // sp500_df['sector'].nunique())))
                .reset_index(drop=True))
    sp500_df = sp500_df.iloc[:CONFIG['MAX_STOCKS']]

print(f"\n  📊 Working universe: {len(sp500_df)} stocks across {sp500_df['sector'].nunique()} sectors")
print("\n  Sector distribution:")
sector_counts = sp500_df['sector'].value_counts()
for sector, count in sector_counts.items():
    bar = '█' * count
    print(f"    {sector:<35} {bar} ({count})")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — DATA COLLECTION: DOWNLOAD HISTORICAL PRICES                       ║
# ║  Batch download adjusted close prices via yfinance                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 2: DOWNLOADING HISTORICAL PRICE DATA")

def _extract_close_prices(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Robustly extract Close prices from yfinance output.

    Why this is needed:
    yfinance has changed its MultiIndex column structure across versions:

      v0.1.x  : flat columns ['Open','High','Low','Close','Volume'] (single ticker)
      v0.2.x  : MultiIndex (price_type, ticker) — default group_by='column'
      v0.2.x  : MultiIndex (ticker, price_type) — when group_by='ticker' is set
      v0.2.50+: MultiIndex structure changed again; 'Price' replaces 'Close' in
                some builds when auto_adjust=True

    This function detects the actual structure at runtime and handles all cases.
    Removing group_by='ticker' (use default 'column') is the primary fix, but
    we keep fallbacks for every known variant to be future-proof.
    """
    # ── Case 1: Not a MultiIndex (single ticker or already flat) ───────────────
    if not isinstance(raw.columns, pd.MultiIndex):
        for col in ['Close', 'close', 'Adj Close']:
            if col in raw.columns:
                prices = raw[[col]].rename(columns={col: tickers[0]})
                return prices
        # Nothing matched — return what we have and let caller handle it
        return raw

    level_0_vals = raw.columns.get_level_values(0).unique().tolist()
    level_1_vals = raw.columns.get_level_values(1).unique().tolist()

    # ── Case 2: (price_type, ticker) — default yfinance behaviour ──────────────
    # e.g. raw.columns = [('Close','AAPL'), ('Close','MSFT'), ('Open','AAPL')…]
    for price_key in ['Close', 'close', 'Adj Close', 'Price']:
        if price_key in level_0_vals:
            prices = raw[price_key].copy()
            # Result is a DataFrame with tickers as columns — exactly what we want
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
            return prices

    # ── Case 3: (ticker, price_type) — happens when group_by='ticker' is used ──
    # e.g. raw.columns = [('AAPL','Close'), ('AAPL','Open'), ('MSFT','Close')…]
    for price_key in ['Close', 'close', 'Adj Close', 'Price']:
        if price_key in level_1_vals:
            prices = raw.xs(price_key, axis=1, level=1).copy()
            return prices

    # ── Case 4: Last resort — print available columns so the user can debug ─────
    raise KeyError(
        f"Could not locate 'Close' prices in yfinance output.\n"
        f"  MultiIndex Level 0 values: {level_0_vals}\n"
        f"  MultiIndex Level 1 values: {level_1_vals}\n"
        f"  → Check your yfinance version: pip install --upgrade yfinance"
    )


def download_price_data(
    tickers: List[str],
    start: str,
    end: str,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers using yfinance batch download.

    Key fix vs. original:
    - Removed group_by='ticker'  ← This was inverting the MultiIndex, putting
      ticker names at level 0 and price types at level 1, so raw['Close'] raised
      KeyError. The default group_by='column' keeps price types at level 0.
    - Delegates column extraction to _extract_close_prices() which handles all
      known yfinance MultiIndex structures across every released version.

    Returns:
        DataFrame with dates as index, tickers as columns (adjusted close prices)
    """
    print(f"  📥 Downloading {len(tickers)} tickers | {start} → {end}")
    print(f"  ⏱️  Estimated time: ~{len(tickers) // 30 + 1} minutes (Colab network)")

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,   # Dividend + split adjusted
                progress=True,
                threads=True,
                # ✅ FIX: Do NOT pass group_by='ticker'.
                # Default is group_by='column', which puts price type at level 0
                # → raw['Close'] returns a DataFrame of tickers correctly.
            )

            prices = _extract_close_prices(raw, tickers)

            # Sanity-check: must be a 2-D DataFrame with at least one ticker
            if prices.empty or not isinstance(prices, pd.DataFrame):
                raise ValueError(f"Extracted price table is empty or wrong type: {type(prices)}")

            print(f"  ✅ Download complete: {prices.shape[0]} trading days × {prices.shape[1]} stocks")
            return prices

        except Exception as e:
            print(f"  ⚠️  Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"     Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"Price download failed after {max_retries} attempts.\n"
                    f"Last error: {e}\n"
                    f"Try: pip install --upgrade yfinance"
                )


def download_benchmark(
    ticker: str = 'SPY',
    start: str = CONFIG['START_DATE'],
    end: str = CONFIG['END_DATE']
) -> pd.Series:
    """Download benchmark (SPY) for comparison and beta calculation."""
    print(f"  📥 Downloading benchmark: {ticker}")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    # Use the same robust extractor; squeeze to Series since it's one ticker
    prices = _extract_close_prices(raw, [ticker])
    return prices.squeeze()

# ── Execute Downloads ──────────────────────────────────────────────────────────
tickers_to_download = sp500_df['ticker'].tolist()

price_data_raw = download_price_data(
    tickers=tickers_to_download,
    start=CONFIG['START_DATE'],
    end=CONFIG['END_DATE']
)

spy_prices = download_benchmark('SPY')
spy_returns = np.log(spy_prices / spy_prices.shift(1)).dropna()

print(f"\n  📈 SPY benchmark: {len(spy_prices)} trading days downloaded")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — DATA CLEANING & QUALITY CONTROL                                   ║
# ║  Remove low-quality stocks before cointegration analysis                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 3: DATA CLEANING & QUALITY CONTROL")

def clean_price_data(
    prices: pd.DataFrame,
    min_price: float = CONFIG['MIN_PRICE'],
    max_missing_pct: float = 0.02,    # Max 2% missing values
    min_history_pct: float = 0.95,    # Must have 95%+ of the date range
) -> pd.DataFrame:
    """
    Production-grade data cleaning for price series.

    Steps:
    1. Drop columns (tickers) with excessive missing data
    2. Forward-fill gaps (weekend quotes, brief halts)
    3. Remove stocks trading below minimum price (liquidity filter)
    4. Ensure complete date coverage

    In a real institutional system, you would also:
    - Apply CRSP adjustment factors
    - Filter by average daily dollar volume
    - Check for data errors (price jumps > 50%)
    """
    print(f"  🧹 Starting with {prices.shape[1]} tickers")
    original_count = prices.shape[1]

    # ── Step 1: Remove tickers with too many NaN values ────────────────────────
    missing_pct = prices.isna().mean()
    valid_mask = missing_pct <= max_missing_pct
    prices = prices.loc[:, valid_mask]
    dropped_missing = original_count - prices.shape[1]

    # ── Step 2: Forward-fill remaining gaps (max 3-day gap) ────────────────────
    prices = prices.ffill(limit=3)

    # ── Step 3: Drop rows where entire row is NaN (e.g., holidays) ────────────
    prices = prices.dropna(how='all')

    # ── Step 4: Remove stocks trading below minimum price ──────────────────────
    # Use the average price over the last 252 days
    recent_prices = prices.iloc[-252:] if len(prices) >= 252 else prices
    avg_price = recent_prices.mean()
    valid_price = avg_price >= min_price
    prices = prices.loc[:, valid_price]
    dropped_price = (original_count - dropped_missing) - prices.shape[1]

    # ── Step 5: Require sufficient history ─────────────────────────────────────
    total_days = len(prices)
    coverage = prices.notna().mean()
    prices = prices.loc[:, coverage >= min_history_pct]
    dropped_history = prices.shape[1]

    # ── Final cleanup ──────────────────────────────────────────────────────────
    prices = prices.dropna()

    print(f"  ✅ Cleaning complete:")
    print(f"     Dropped (>2% missing)    : {dropped_missing} tickers")
    print(f"     Dropped (price < ${min_price}) : {dropped_price} tickers")
    print(f"     Remaining clean universe  : {prices.shape[1]} tickers")
    print(f"     Date range               : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"     Total trading days        : {len(prices)}")
    return prices

# ── Execute Cleaning ───────────────────────────────────────────────────────────
prices_clean = clean_price_data(price_data_raw)

# ── Compute Log Prices ─────────────────────────────────────────────────────────
# We work in log price space for the spread construction.
# Reason: log(P_A) - β*log(P_B) gives a more stationary spread than levels.
log_prices = np.log(prices_clean)
daily_returns = log_prices.diff().dropna()

# ── Train/Test Split ───────────────────────────────────────────────────────────
# Critical: ALL cointegration testing uses ONLY training data.
# We test the strategy on the held-out test period.
n_days = len(prices_clean)
n_train = int(n_days * CONFIG['TRAIN_RATIO'])

train_idx = prices_clean.index[:n_train]
test_idx  = prices_clean.index[n_train:]

train_end = train_idx[-1]
test_start = test_idx[0]

log_prices_train = log_prices.loc[train_idx]
log_prices_test  = log_prices.loc[test_idx]

print(f"\n  📅 Train period : {train_idx[0].date()} → {train_end.date()} ({n_train} days)")
print(f"     Test period  : {test_start.date()} → {test_idx[-1].date()} ({len(test_idx)} days)")

# Update sp500_df to only include clean tickers
clean_tickers = list(prices_clean.columns)
sp500_df_clean = sp500_df[sp500_df['ticker'].isin(clean_tickers)].copy()
print(f"\n  🎯 Final universe: {len(clean_tickers)} clean tickers")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — SECTOR-BASED PAIR GENERATION                                      ║
# ║  Filter pairs to within-sector only — reduces spurious cointegration        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 4: GENERATING WITHIN-SECTOR PAIRS")

def generate_sector_pairs(
    sp500_df: pd.DataFrame,
    valid_tickers: List[str]
) -> List[Tuple[str, str, str]]:
    """
    Generate all possible stock pairs within the same GICS sector.

    Why within-sector only?
    - Stocks in the same sector share common economic drivers
    - Industry-wide shocks affect both stocks similarly
    - Reduces the probability of spurious cointegration
    - Better economic justification: two oil majors vs. an oil stock + bank

    Example: (XOM, CVX) ✅ Both Energy — share oil price exposure
             (XOM, JPM) ❌ Cross-sector — fundamentally different drivers

    Returns: List of (ticker_A, ticker_B, sector) tuples
    """
    pairs = []
    valid_set = set(valid_tickers)

    for sector in sp500_df['sector'].unique():
        sector_tickers = sp500_df[
            (sp500_df['sector'] == sector) &
            (sp500_df['ticker'].isin(valid_set))
        ]['ticker'].tolist()

        if len(sector_tickers) < 2:
            continue

        # All unique pairs within this sector (no repeats)
        sector_pairs = list(combinations(sector_tickers, 2))
        pairs.extend([(a, b, sector) for a, b in sector_pairs])

    return pairs

candidate_pairs = generate_sector_pairs(sp500_df_clean, clean_tickers)

print(f"  📊 Candidate pairs by sector:")
sector_pair_counts = {}
for _, _, sector in candidate_pairs:
    sector_pair_counts[sector] = sector_pair_counts.get(sector, 0) + 1

for sector, count in sorted(sector_pair_counts.items(), key=lambda x: -x[1]):
    bar = '▓' * min(count, 40)
    print(f"    {sector:<35} {bar} {count:>4} pairs")

print(f"\n  📋 Total candidate pairs: {len(candidate_pairs)}")
print(f"     (All pairs will be tested for cointegration on the training data)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — STATISTICAL TESTS: ADF, ENGLE-GRANGER, JOHANSEN                  ║
# ║  The core statistical machinery of the pairs trading engine                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 5: STATISTICAL TEST FUNCTIONS")

def run_adf_test(series: pd.Series, maxlag: int = None) -> Dict:
    """
    Augmented Dickey-Fuller Test for stationarity.

    H0: The series has a unit root (non-stationary, I(1))
    H1: The series is stationary (I(0))

    We want the SPREAD to be stationary (reject H0), which means the ADF
    test statistic should be MORE NEGATIVE than the critical value, and
    p-value < 0.05.

    Note: We CANNOT apply ADF to the individual stock prices (they're I(1)).
    We apply it to the estimated spread (which should be I(0) if cointegrated).
    """
    try:
        result = adfuller(series.dropna(), maxlag=maxlag, autolag='AIC')
        return {
            'adf_stat':   result[0],
            'p_value':    result[1],
            'n_lags':     result[2],
            'n_obs':      result[3],
            'cv_1pct':    result[4]['1%'],
            'cv_5pct':    result[4]['5%'],
            'cv_10pct':   result[4]['10%'],
            'is_stationary_5pct': result[1] < 0.05
        }
    except Exception as e:
        return {'adf_stat': np.nan, 'p_value': 1.0, 'is_stationary_5pct': False, 'error': str(e)}

def run_engle_granger_test(
    log_y1: pd.Series,
    log_y2: pd.Series
) -> Dict:
    """
    Engle-Granger Two-Step Cointegration Test.

    Step 1: Regress log(P_A) on log(P_B) to estimate hedge ratio β
    Step 2: Test the residuals for stationarity (ADF test)

    If residuals are stationary → the series are cointegrated.

    The statsmodels `coint` function implements this exactly:
    - Fits OLS: y1 = α + β*y2 + ε
    - Runs ADF on ε
    - Critical values are adjusted for the two-step nature of the test

    Returns hedge ratio (β) from Step 1 OLS regression.
    """
    try:
        # statsmodels coint: y1 is the dependent variable
        eg_stat, eg_pvalue, eg_cvs = coint(log_y1, log_y2, trend='c')

        # Estimate OLS hedge ratio directly for use in spread construction
        X = sm.add_constant(log_y2)
        ols_result = sm.OLS(log_y1, X).fit()
        beta_ols = ols_result.params.iloc[1]
        alpha_ols = ols_result.params.iloc[0]

        # Compute residuals (the spread)
        residuals = log_y1 - alpha_ols - beta_ols * log_y2

        # ADF on residuals (separate from coint's internal ADF for transparency)
        adf_on_spread = run_adf_test(residuals)

        return {
            'eg_stat':      eg_stat,
            'eg_pvalue':    eg_pvalue,
            'eg_cv_1pct':   eg_cvs[0],
            'eg_cv_5pct':   eg_cvs[1],
            'eg_cv_10pct':  eg_cvs[2],
            'beta_ols':     beta_ols,
            'alpha_ols':    alpha_ols,
            'eg_significant': eg_pvalue < CONFIG['EG_PVALUE_THRESHOLD'],
            'spread_adf_stat':  adf_on_spread['adf_stat'],
            'spread_adf_pval':  adf_on_spread['p_value'],
        }
    except Exception as e:
        return {
            'eg_stat': np.nan, 'eg_pvalue': 1.0,
            'eg_significant': False, 'error': str(e)
        }

def run_johansen_test(log_y1: pd.Series, log_y2: pd.Series) -> Dict:
    """
    Johansen Maximum Likelihood Cointegration Test.

    Unlike Engle-Granger (single equation), Johansen tests for cointegration
    in a VECM (Vector Error Correction Model) framework — symmetric treatment
    of both variables.

    Null hypothesis: r = 0 (no cointegrating relationships)
    We want to reject this → evidence of 1 cointegrating relationship.

    Two test statistics:
    1. Trace: tests H0: r ≤ j vs. H1: r > j
    2. Max Eigenvalue: tests H0: r = j vs. H1: r = j+1

    Also extracts the Johansen hedge ratio from the eigenvector.
    """
    try:
        df_joint = pd.concat([log_y1, log_y2], axis=1).dropna()

        if len(df_joint) < 50:  # Need sufficient observations
            return {'johansen_coint': False, 'error': 'Insufficient data'}

        result = coint_johansen(df_joint, det_order=0, k_ar_diff=1)

        # Test statistics for r=0 (first cointegrating vector)
        trace_stat   = result.lr1[0]    # Trace statistic for r=0
        trace_cv_95  = result.cvt[0, 1] # 95% critical value for trace test
        maxeig_stat  = result.lr2[0]    # Max eigenvalue stat for r=0
        maxeig_cv_95 = result.cvm[0, 1] # 95% critical value for max eig test

        is_coint = (trace_stat > trace_cv_95) and (maxeig_stat > maxeig_cv_95)

        # Johansen hedge ratio from first eigenvector
        # evec[:, 0] is the first cointegrating vector [β_1, β_2]
        # Normalized so that β_1 = 1 → spread = log_y1 - (β_2/β_1)*log_y2
        evec = result.evec[:, 0]
        johansen_beta = -evec[1] / evec[0] if evec[0] != 0 else np.nan

        return {
            'johansen_coint':   is_coint,
            'trace_stat':       trace_stat,
            'trace_cv_95':      trace_cv_95,
            'maxeig_stat':      maxeig_stat,
            'maxeig_cv_95':     maxeig_cv_95,
            'johansen_beta':    johansen_beta,
        }

    except Exception as e:
        return {'johansen_coint': False, 'error': str(e)}

print("✅ Statistical test functions defined:")
print("   • run_adf_test()           — Augmented Dickey-Fuller stationarity test")
print("   • run_engle_granger_test() — EG two-step cointegration test")
print("   • run_johansen_test()      — Johansen ML cointegration test (bivariate)")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — ORNSTEIN-UHLENBECK PROCESS FITTING                               ║
# ║  Quantify mean-reversion speed and expected holding period                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def fit_ou_process(spread: pd.Series) -> Dict:
    """
    Fit an Ornstein-Uhlenbeck (OU) process to the spread series.

    The OU process: dX = κ(μ - X)dt + σ dW
    Discrete equivalent: X_t - X_{t-1} = α + β*X_{t-1} + ε_t

    Where:
        κ     = -β         (mean reversion speed, must be > 0)
        μ     = -α/β       (long-run equilibrium mean)
        σ_OU  = std(ε)     (residual volatility)

    Half-life = ln(2) / κ
    → Number of days for the spread to decay 50% toward its mean.

    Financial Interpretation:
        Half-life = 5 days  → Very fast reversion (short-term signal)
        Half-life = 30 days → Moderate reversion (monthly signal)
        Half-life = 90 days → Slow reversion (might not be exploitable)

    We filter for 5 ≤ half-life ≤ 60 days.
    """
    try:
        spread_clean = spread.dropna()

        if len(spread_clean) < 60:
            return {'half_life': np.nan, 'kappa': np.nan, 'valid': False}

        # AR(1) regression: ΔX_t = α + β * X_{t-1} + ε
        spread_lag  = spread_clean.shift(1)
        spread_diff = spread_clean.diff()

        # Align and remove NaN
        valid_mask = ~(spread_lag.isna() | spread_diff.isna())
        X = sm.add_constant(spread_lag[valid_mask])
        y = spread_diff[valid_mask]

        model = sm.OLS(y, X).fit()

        beta  = model.params.iloc[1]   # Coefficient on X_{t-1}
        alpha = model.params.iloc[0]   # Constant

        # Mean reversion speed (κ = -β must be positive for mean reversion)
        kappa = -beta
        if kappa <= 0:
            return {'half_life': np.nan, 'kappa': kappa, 'valid': False,
                    'reason': 'No mean reversion (κ ≤ 0)'}

        # Long-run equilibrium mean
        mu = -alpha / beta if beta != 0 else spread_clean.mean()

        # Half-life in trading days
        half_life = np.log(2) / kappa

        # OU residual volatility (annualized)
        sigma_ou = model.resid.std() * np.sqrt(252)

        # Equilibrium band (±1σ around long-run mean)
        spread_std = spread_clean.std()

        return {
            'kappa':        kappa,
            'mu':           mu,
            'sigma_ou':     sigma_ou,
            'half_life':    half_life,
            'ou_r2':        model.rsquared,
            'ou_pvalue':    model.pvalues.iloc[1],
            'spread_std':   spread_std,
            'valid':        True,
            'in_hl_range':  CONFIG['HALF_LIFE_MIN'] <= half_life <= CONFIG['HALF_LIFE_MAX']
        }

    except Exception as e:
        return {'half_life': np.nan, 'kappa': np.nan, 'valid': False, 'error': str(e)}

print("✅ OU process fitting function defined.")
print(f"   Valid half-life range: {CONFIG['HALF_LIFE_MIN']}–{CONFIG['HALF_LIFE_MAX']} trading days")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11 — COINTEGRATION SCANNING ENGINE                                    ║
# ║  Runs EG + Johansen on all candidate pairs; extracts valid trading pairs    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 6: COINTEGRATION SCAN")

def scan_all_pairs(
    candidate_pairs: List[Tuple[str, str, str]],
    log_prices_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the full cointegration screening pipeline on all candidate pairs.

    Pipeline per pair:
    1. Engle-Granger test (primary filter)
    2. Johansen test (confirmation — optional based on config)
    3. OU half-life estimation (exploitability filter)

    Uses only TRAINING data to avoid look-ahead bias.
    Critical: All statistical parameters (β, μ, σ) are estimated on training
    data only. They are then applied out-of-sample for backtesting.

    Returns:
        DataFrame of ALL tested pairs with test results, sorted by significance.
    """
    results = []
    valid_tickers = set(log_prices_train.columns)

    print(f"  🔍 Scanning {len(candidate_pairs)} candidate pairs...")
    print(f"     (Using training data only: {log_prices_train.index[0].date()} → {log_prices_train.index[-1].date()})")
    print()

    for ticker_a, ticker_b, sector in tqdm(candidate_pairs, desc="  Cointegration Scan"):

        # Skip if either ticker missing from clean data
        if ticker_a not in valid_tickers or ticker_b not in valid_tickers:
            continue

        log_a = log_prices_train[ticker_a]
        log_b = log_prices_train[ticker_b]

        # ── Engle-Granger Test (primary) ──────────────────────────────────────
        eg = run_engle_granger_test(log_a, log_b)

        if not eg.get('eg_significant', False):
            # Also test in reverse direction (A~B vs. B~A)
            eg_rev = run_engle_granger_test(log_b, log_a)
            if eg_rev.get('eg_significant', False):
                eg = eg_rev
                eg['reversed'] = True
                # Swap A and B
                ticker_a, ticker_b = ticker_b, ticker_a
                log_a, log_b = log_b, log_a
            else:
                results.append({
                    'ticker_a': ticker_a, 'ticker_b': ticker_b, 'sector': sector,
                    'eg_pvalue': eg.get('eg_pvalue', 1.0), 'passed_eg': False,
                })
                continue

        # ── Johansen Test (confirmation) ──────────────────────────────────────
        johansen = run_johansen_test(log_a, log_b)

        # ── Compute Spread & OU Half-Life ─────────────────────────────────────
        beta  = eg.get('beta_ols', 1.0)
        alpha = eg.get('alpha_ols', 0.0)
        spread_train = log_a - alpha - beta * log_b

        ou = fit_ou_process(spread_train)

        # ── Compile Results ───────────────────────────────────────────────────
        results.append({
            'ticker_a':         ticker_a,
            'ticker_b':         ticker_b,
            'sector':           sector,

            # Engle-Granger
            'eg_stat':          eg.get('eg_stat', np.nan),
            'eg_pvalue':        eg.get('eg_pvalue', 1.0),
            'passed_eg':        eg.get('eg_significant', False),
            'beta_ols':         eg.get('beta_ols', np.nan),
            'alpha_ols':        eg.get('alpha_ols', np.nan),
            'spread_adf_stat':  eg.get('spread_adf_stat', np.nan),

            # Johansen
            'passed_johansen':  johansen.get('johansen_coint', False),
            'trace_stat':       johansen.get('trace_stat', np.nan),
            'johansen_beta':    johansen.get('johansen_beta', np.nan),

            # OU Process
            'half_life':        ou.get('half_life', np.nan),
            'kappa':            ou.get('kappa', np.nan),
            'sigma_ou':         ou.get('sigma_ou', np.nan),
            'ou_r2':            ou.get('ou_r2', np.nan),
            'in_hl_range':      ou.get('in_hl_range', False),

            # Overall Pass/Fail
            'passed_eg':        True,
        })

    results_df = pd.DataFrame(results)

    # Sort by significance (most significant first)
    if 'eg_pvalue' in results_df.columns:
        results_df = results_df.sort_values('eg_pvalue')

    return results_df

# ── Execute Scan ───────────────────────────────────────────────────────────────
scan_results = scan_all_pairs(candidate_pairs, log_prices_train)

# ── Filter Valid Pairs ─────────────────────────────────────────────────────────
valid_pairs = scan_results[
    scan_results['passed_eg'] &
    scan_results['in_hl_range'].fillna(False)
].copy()

if CONFIG['JOHANSEN_CONFIRM']:
    valid_pairs_confirmed = valid_pairs[valid_pairs['passed_johansen'].fillna(False)]
    if len(valid_pairs_confirmed) == 0:
        print("  ⚠️  No pairs passed both EG + Johansen. Relaxing to EG-only.")
    else:
        valid_pairs = valid_pairs_confirmed

# Limit to max pairs for portfolio
valid_pairs = valid_pairs.head(CONFIG['MAX_PAIRS'])

print(f"\n  📊 Cointegration scan summary:")
print(f"     Pairs tested            : {len(scan_results)}")
print(f"     Passed Engle-Granger    : {scan_results['passed_eg'].sum()}")
print(f"     Passed Johansen         : {scan_results['passed_johansen'].sum() if 'passed_johansen' in scan_results else 'N/A'}")
print(f"     Valid half-life range   : {scan_results['in_hl_range'].sum() if 'in_hl_range' in scan_results else 'N/A'}")
print(f"     ✅ FINAL VALID PAIRS     : {len(valid_pairs)}")

if len(valid_pairs) > 0:
    print(f"\n  Top 10 most significant cointegrated pairs:")
    display_cols = ['ticker_a', 'ticker_b', 'sector', 'eg_pvalue', 'half_life', 'beta_ols']
    display_cols = [c for c in display_cols if c in valid_pairs.columns]
    print(tabulate(
        valid_pairs[display_cols].head(10).round(4),
        headers='keys', tablefmt='rounded_grid', showindex=False
    ))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 12 — SIGNAL GENERATION ENGINE                                         ║
# ║  Z-score based entry/exit signals with stop-loss logic                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 7: SIGNAL GENERATION ENGINE")

def compute_rolling_zscore(
    spread: pd.Series,
    window: int = CONFIG['ZSCORE_WINDOW']
) -> pd.Series:
    """
    Compute rolling z-score of the spread series.

    Z_t = (Spread_t - μ_rolling) / σ_rolling

    Where μ and σ are computed over the past `window` days.

    Critical detail: We use rolling (not expanding) window z-score because:
    1. Expanding window converges too slowly in the test period
    2. Rolling window adapts to structural shifts in the spread level
    3. 60-day window balances responsiveness and stability

    In practice, sophisticated stat-arb desks use dynamic lookback windows
    calibrated to each pair's estimated OU half-life.
    """
    rolling_mean = spread.rolling(window=window, min_periods=window // 2).mean()
    rolling_std  = spread.rolling(window=window, min_periods=window // 2).std()

    z_score = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    return z_score

def generate_trading_signals(
    z_score: pd.Series,
    entry_z:  float = CONFIG['ENTRY_ZSCORE'],
    exit_z:   float = CONFIG['EXIT_ZSCORE'],
    stop_z:   float = CONFIG['STOP_ZSCORE'],
) -> pd.Series:
    """
    Generate long/short/flat signals from the z-score time series.

    Signal Convention:
        +1 = LONG spread  (long A, short B) — entered when z ≤ -entry_z
        -1 = SHORT spread (short A, long B) — entered when z ≥ +entry_z
         0 = FLAT

    Mean Reversion Logic:
        If z > +2.0: spread is ABOVE its historical mean → SHORT (expect reversion)
        If z < -2.0: spread is BELOW its historical mean → LONG  (expect reversion)
        If |z| < 0.5: spread has reverted to mean → EXIT
        If |z| > 3.0: spread has moved further against us → STOP LOSS

    Position State Machine:
        ┌──────────┬──────────────────┬───────────────────────────────────┐
        │ Current  │ Condition        │ Action                            │
        ├──────────┼──────────────────┼───────────────────────────────────┤
        │ FLAT     │ z ≥ +entry_z     │ → SHORT spread (-1)               │
        │ FLAT     │ z ≤ -entry_z     │ → LONG spread (+1)                │
        │ LONG     │ z ≥ -exit_z      │ → FLAT (0)  [profit target]       │
        │ LONG     │ z ≤ -stop_z      │ → FLAT (0)  [stop loss]           │
        │ SHORT    │ z ≤ +exit_z      │ → FLAT (0)  [profit target]       │
        │ SHORT    │ z ≥ +stop_z      │ → FLAT (0)  [stop loss]           │
        └──────────┴──────────────────┴───────────────────────────────────┘
    """
    z = z_score.fillna(0).values
    n = len(z)
    positions = np.zeros(n, dtype=np.int8)

    position = 0   # Current position: 0=flat, 1=long, -1=short

    for i in range(1, n):
        z_i = z[i]

        if position == 0:   # Currently flat — look for entry
            if z_i >= entry_z:
                position = -1   # Short spread (spread will revert downward)
            elif z_i <= -entry_z:
                position = 1    # Long spread (spread will revert upward)

        elif position == 1:  # Currently long spread
            if z_i >= -exit_z:  # Spread has reverted up → take profit
                position = 0
            elif z_i <= -stop_z:  # Spread diverged further → stop loss
                position = 0

        elif position == -1:  # Currently short spread
            if z_i <= exit_z:    # Spread has reverted down → take profit
                position = 0
            elif z_i >= stop_z:  # Spread diverged further → stop loss
                position = 0

        positions[i] = position

    return pd.Series(positions, index=z_score.index, name='signal')

def compute_trade_metadata(signals: pd.Series) -> Dict:
    """Extract trade entry/exit dates and statistics from signal series."""
    trades = []
    in_trade = False
    entry_date = None
    entry_direction = 0

    for date, sig in signals.items():
        if not in_trade and sig != 0:
            in_trade = True
            entry_date = date
            entry_direction = sig

        elif in_trade and (sig == 0 or sig != entry_direction):
            trades.append({
                'entry_date':  entry_date,
                'exit_date':   date,
                'direction':   entry_direction,
                'duration':    (date - entry_date).days,
            })
            in_trade = False
            if sig != 0:
                in_trade = True
                entry_date = date
                entry_direction = sig

    if in_trade:  # Close any open trade at end of period
        trades.append({
            'entry_date':  entry_date,
            'exit_date':   signals.index[-1],
            'direction':   entry_direction,
            'duration':    (signals.index[-1] - entry_date).days,
        })

    return {
        'trades':           trades,
        'n_trades':         len(trades),
        'avg_duration':     np.mean([t['duration'] for t in trades]) if trades else 0,
        'pct_time_in_mkt':  (signals != 0).mean(),
        'n_long':           sum(1 for t in trades if t['direction'] == 1),
        'n_short':          sum(1 for t in trades if t['direction'] == -1),
    }

print("✅ Signal generation functions defined.")
print(f"   Entry threshold  : |z| > {CONFIG['ENTRY_ZSCORE']}")
print(f"   Exit threshold   : |z| < {CONFIG['EXIT_ZSCORE']}")
print(f"   Stop-loss        : |z| > {CONFIG['STOP_ZSCORE']}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 13 — BACKTESTING ENGINE                                               ║
# ║  Production-grade event-driven backtest with realistic transaction costs    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 8: BACKTESTING ENGINE")

def backtest_pair(
    prices_a:     pd.Series,
    prices_b:     pd.Series,
    beta:         float,
    alpha:        float,
    signals:      pd.Series,
    tc:           float = CONFIG['TRANSACTION_COST'],
) -> Dict:
    """
    Backtest a single cointegrated pair with realistic transaction costs.

    P&L Model:
        Long spread  (+1): P&L_t = ΔlogA_t - β · ΔlogB_t
        Short spread (-1): P&L_t = -(ΔlogA_t - β · ΔlogB_t)

    Transaction Cost Model:
        On each position change: subtract 2 × TC
        (Pay TC on both legs: both the A trade and the B trade)
        Round-trip cost = 4 × TC = 4 × 10bps = 40bps

    Important: We use yesterday's SIGNAL to trade TODAY (avoid look-ahead).
    This is the 'signal from close, trade at next close' methodology.
    In production, you'd actually trade at the open the next morning.

    Dollar Neutrality:
        We assume $1 long in A and $β short in B (approximately dollar neutral
        assuming prices are similar). In production, you'd dollar-weight exactly.
    """
    # Align all series to common dates
    common_idx = prices_a.index.intersection(prices_b.index).intersection(signals.index)
    pa = prices_a.loc[common_idx]
    pb = prices_b.loc[common_idx]
    sig = signals.loc[common_idx]

    # Daily log returns for each leg
    ret_a = np.log(pa / pa.shift(1))
    ret_b = np.log(pb / pb.shift(1))

    # Spread return: position in this combined instrument
    spread_return = ret_a - beta * ret_b

    # Daily strategy P&L (use previous day's signal for today's return)
    # Signal at t determines position entering day t+1
    prev_signal = sig.shift(1).fillna(0)
    strategy_pnl = prev_signal * spread_return

    # ── Transaction Cost Calculation ───────────────────────────────────────────
    # Position changes (entry + exit events)
    position_changes = sig.diff().fillna(0).abs()
    tc_series = position_changes * 2 * tc  # Both legs pay TC

    # Apply transaction costs on the day of position change
    strategy_pnl = strategy_pnl - tc_series

    # Remove first row (NaN from shift)
    strategy_pnl = strategy_pnl.dropna()

    # ── Cumulative Performance ─────────────────────────────────────────────────
    cum_pnl = strategy_pnl.cumsum()
    cum_equity = np.exp(cum_pnl)  # Dollar value of $1 invested

    return {
        'daily_pnl':    strategy_pnl,
        'cum_pnl':      cum_pnl,
        'cum_equity':   cum_equity,
        'spread_return': spread_return,
    }

class PairsTradingBacktester:
    """
    Full pairs trading backtest engine.

    Orchestrates the complete pipeline for each valid pair:
    1. Extract full-period price data (train + test)
    2. Compute spread using TRAINING-period estimated parameters
    3. Generate signals on the FULL period
    4. Backtest on the OUT-OF-SAMPLE test period only

    This ensures no look-ahead bias: parameters estimated in-sample,
    performance measured out-of-sample.
    """

    def __init__(self, valid_pairs_df: pd.DataFrame, log_prices: pd.DataFrame,
                 prices: pd.DataFrame, train_end: pd.Timestamp):
        self.valid_pairs = valid_pairs_df
        self.log_prices  = log_prices
        self.prices      = prices
        self.train_end   = train_end
        self.results     = {}

    def run(self) -> Dict:
        """Execute backtest for all valid pairs."""
        print(f"  🚀 Running backtest on {len(self.valid_pairs)} pairs...")
        print(f"     Out-of-sample period: {test_start.date()} → {self.log_prices.index[-1].date()}")
        print()

        for _, row in tqdm(self.valid_pairs.iterrows(), total=len(self.valid_pairs),
                          desc="  Backtesting"):

            pair_key  = f"{row['ticker_a']}_{row['ticker_b']}"
            ticker_a  = row['ticker_a']
            ticker_b  = row['ticker_b']
            beta      = row['beta_ols']
            alpha_val = row['alpha_ols']

            try:
                # ── Full-period data ───────────────────────────────────────────
                log_a_full = self.log_prices[ticker_a]
                log_b_full = self.log_prices[ticker_b]
                px_a_full  = self.prices[ticker_a]
                px_b_full  = self.prices[ticker_b]

                # ── Spread construction using TRAINING parameters ──────────────
                # Note: β and α estimated from training period, applied full period
                spread_full = log_a_full - alpha_val - beta * log_b_full

                # ── Z-Score on FULL period ─────────────────────────────────────
                z_score_full = compute_rolling_zscore(spread_full, CONFIG['ZSCORE_WINDOW'])

                # ── Signals on FULL period ─────────────────────────────────────
                signals_full = generate_trading_signals(z_score_full)

                # ── Backtest on TEST period only ───────────────────────────────
                test_mask     = self.log_prices.index > self.train_end
                signals_test  = signals_full[test_mask]
                px_a_test     = px_a_full[test_mask]
                px_b_test     = px_b_full[test_mask]
                spread_test   = spread_full[test_mask]
                z_score_test  = z_score_full[test_mask]

                backtest_result = backtest_pair(
                    prices_a = px_a_test,
                    prices_b = px_b_test,
                    beta     = beta,
                    alpha    = alpha_val,
                    signals  = signals_test,
                    tc       = CONFIG['TRANSACTION_COST'],
                )

                # ── Trade Metadata ─────────────────────────────────────────────
                trade_meta = compute_trade_metadata(signals_test)

                self.results[pair_key] = {
                    'ticker_a':     ticker_a,
                    'ticker_b':     ticker_b,
                    'sector':       row.get('sector', 'Unknown'),
                    'beta':         beta,
                    'half_life':    row.get('half_life', np.nan),
                    'eg_pvalue':    row.get('eg_pvalue', np.nan),

                    # Full-period data for plotting
                    'spread_full':  spread_full,
                    'z_score_full': z_score_full,
                    'signals_full': signals_full,

                    # Test-period data
                    'spread_test':  spread_test,
                    'z_score_test': z_score_test,
                    'signals_test': signals_test,
                    'prices_a_test':px_a_test,
                    'prices_b_test':px_b_test,

                    # P&L results
                    **backtest_result,
                    **trade_meta,
                }

            except Exception as e:
                print(f"\n  ⚠️  Failed for {pair_key}: {e}")
                continue

        print(f"\n  ✅ Backtest complete: {len(self.results)} pairs processed")
        return self.results

# ── Execute Backtest ───────────────────────────────────────────────────────────
backtester = PairsTradingBacktester(
    valid_pairs_df = valid_pairs,
    log_prices     = log_prices,
    prices         = prices_clean,
    train_end      = train_end,
)

all_pair_results = backtester.run()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 14 — PERFORMANCE METRICS ENGINE                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 9: PERFORMANCE ANALYTICS")

def compute_pair_metrics(
    pair_result: Dict,
    benchmark_returns: pd.Series,
    rf_daily: float = RISK_FREE_DAILY
) -> Dict:
    """
    Compute comprehensive performance metrics for a single pair.

    Metrics:
        Sharpe Ratio   : (E[R] - Rf) / σ * √252  [target: >1.0]
        Sortino Ratio  : (E[R] - Rf) / σ_down * √252  [penalizes only downside vol]
        Calmar Ratio   : CAGR / |Max Drawdown|  [target: >0.5]
        Hit Rate       : % of trading days with positive P&L
        Profit Factor  : Total gains / Total losses  [target: >1.5]
        Beta to Market : Covariance(strategy, SPY) / Variance(SPY)  [target: near 0]
    """
    daily_pnl = pair_result['daily_pnl']

    if len(daily_pnl) == 0 or daily_pnl.std() == 0:
        return {k: np.nan for k in ['sharpe', 'sortino', 'calmar', 'max_dd', 'cagr',
                                     'hit_rate', 'profit_factor', 'beta', 'total_return']}

    cum_equity  = pair_result['cum_equity']
    n_days      = len(daily_pnl)

    # ── Return Metrics ─────────────────────────────────────────────────────────
    total_return = cum_equity.iloc[-1] - 1
    cagr         = cum_equity.iloc[-1] ** (252 / n_days) - 1
    vol_annual   = daily_pnl.std() * np.sqrt(252)

    # ── Risk-Adjusted Returns ──────────────────────────────────────────────────
    excess       = daily_pnl - rf_daily
    sharpe       = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

    down_dev     = daily_pnl[daily_pnl < rf_daily].std() * np.sqrt(252)
    sortino      = (cagr - CONFIG['ANNUAL_RISK_FREE']) / down_dev if down_dev > 0 else 0

    dd           = max_drawdown(cum_equity)
    calmar       = cagr / abs(dd) if dd != 0 else 0

    # ── Trade-Level Metrics ───────────────────────────────────────────────────
    hit_rate     = (daily_pnl > 0).mean()
    gains        = daily_pnl[daily_pnl > 0].sum()
    losses       = abs(daily_pnl[daily_pnl < 0].sum())
    profit_factor= gains / losses if losses > 0 else np.inf

    # ── Market Neutrality ──────────────────────────────────────────────────────
    bm_aligned   = benchmark_returns.reindex(daily_pnl.index).dropna()
    strat_aligned= daily_pnl.reindex(bm_aligned.index).dropna()
    beta_mkt     = compute_beta(strat_aligned, bm_aligned)

    return {
        'total_return':   total_return,
        'cagr':           cagr,
        'vol_annual':     vol_annual,
        'sharpe':         sharpe,
        'sortino':        sortino,
        'calmar':         calmar,
        'max_drawdown':   dd,
        'hit_rate':       hit_rate,
        'profit_factor':  profit_factor,
        'beta':           beta_mkt,
        'n_days':         n_days,
    }

# ── Compute Metrics for All Pairs ─────────────────────────────────────────────
metrics_records = []

spy_test = spy_returns.reindex(
    pd.date_range(test_start, prices_clean.index[-1], freq='B')
).dropna()

for pair_key, result in all_pair_results.items():
    m = compute_pair_metrics(result, spy_test)
    m['pair'] = pair_key
    m['ticker_a'] = result['ticker_a']
    m['ticker_b'] = result['ticker_b']
    m['sector'] = result.get('sector', '')
    m['half_life'] = result.get('half_life', np.nan)
    m['n_trades'] = result.get('n_trades', 0)
    m['pct_in_mkt'] = result.get('pct_time_in_mkt', np.nan)
    metrics_records.append(m)

metrics_df = pd.DataFrame(metrics_records)

if len(metrics_df) > 0:
    metrics_df = metrics_df.sort_values('sharpe', ascending=False)

    print("  📊 TOP PAIR PERFORMANCE SUMMARY (Out-of-Sample)")
    print()

    display_metrics = metrics_df[[
        'ticker_a', 'ticker_b', 'sector', 'sharpe', 'total_return',
        'max_drawdown', 'half_life', 'n_trades', 'beta'
    ]].head(15).copy()

    display_metrics.columns = [
        'Stock A', 'Stock B', 'Sector', 'Sharpe', 'Total Ret',
        'Max DD', 'Half-Life', 'Trades', 'Beta'
    ]

    for col in ['Total Ret', 'Max DD']:
        display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    print(tabulate(
        display_metrics.round(3),
        headers='keys', tablefmt='rounded_grid', showindex=False
    ))

    # Summary statistics
    profitable = (metrics_df['sharpe'] > 1.0).sum()
    print(f"\n  📈 Pairs with Sharpe > 1.0  : {profitable} / {len(metrics_df)}")
    print(f"     Median Sharpe Ratio      : {metrics_df['sharpe'].median():.3f}")
    print(f"     Median Beta to SPY       : {metrics_df['beta'].median():.3f}")
    print(f"     Median Half-Life (days)  : {metrics_df['half_life'].median():.1f}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 15 — PORTFOLIO AGGREGATION & MARKET NEUTRALITY                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 10: PORTFOLIO AGGREGATION")

def aggregate_portfolio(
    all_results: Dict,
    spy_returns:  pd.Series
) -> Dict:
    """
    Aggregate all pair P&Ls into a single portfolio equity curve.

    Capital Allocation:
        Equal weight across all active pairs (1/N allocation).
        Each pair is sized to contribute equally to portfolio risk.

    Market Neutrality Verification:
        A properly constructed pairs portfolio should have near-zero beta
        to the overall market. We verify this empirically.
    """
    if not all_results:
        return {}

    # Stack all daily P&L series
    all_pnl = {}
    for key, result in all_results.items():
        if 'daily_pnl' in result and len(result['daily_pnl']) > 0:
            all_pnl[key] = result['daily_pnl']

    if not all_pnl:
        print("  ⚠️  No valid pair P&L series found.")
        return {}

    pnl_df = pd.DataFrame(all_pnl)

    # Equal-weight portfolio (1/N per pair)
    n_pairs = pnl_df.shape[1]
    portfolio_pnl = pnl_df.mean(axis=1)  # Average = equal weight

    # Cumulative equity curve
    cum_equity = np.exp(portfolio_pnl.cumsum())

    # Align with benchmark
    spy_aligned = spy_returns.reindex(portfolio_pnl.index).dropna()
    port_aligned = portfolio_pnl.reindex(spy_aligned.index).dropna()

    # Rolling beta (60-day window) — should hover near zero
    rolling_beta = pd.Series(index=portfolio_pnl.index, dtype=float)
    for i in range(60, len(portfolio_pnl)):
        window_strat = portfolio_pnl.iloc[i-60:i]
        window_spy   = spy_returns.reindex(window_strat.index).dropna()
        window_strat = window_strat.reindex(window_spy.index)
        b = compute_beta(window_strat, window_spy)
        rolling_beta.iloc[i] = b

    # Compute portfolio metrics
    port_metrics = compute_pair_metrics(
        {'daily_pnl': portfolio_pnl, 'cum_equity': cum_equity},
        spy_aligned
    )

    # SPY performance for comparison
    spy_cum = np.exp(spy_aligned.cumsum())
    spy_total = spy_cum.iloc[-1] - 1
    spy_sharpe = annualized_sharpe(spy_aligned)
    spy_dd = max_drawdown(spy_cum)

    print("  📊 PORTFOLIO vs. BENCHMARK (Equal-Weight, N={} pairs)".format(n_pairs))
    print()
    comparison = pd.DataFrame({
        'Metric':          ['Total Return', 'CAGR', 'Ann. Volatility', 'Sharpe Ratio',
                            'Max Drawdown', 'Beta to SPY', 'Calmar Ratio'],
        'Pairs Portfolio': [
            format_pct(port_metrics.get('total_return', 0)),
            format_pct(port_metrics.get('cagr', 0)),
            format_pct(port_metrics.get('vol_annual', 0)),
            f"{port_metrics.get('sharpe', 0):.3f}",
            format_pct(port_metrics.get('max_drawdown', 0)),
            f"{port_metrics.get('beta', 0):.4f}",
            f"{port_metrics.get('calmar', 0):.3f}",
        ],
        'SPY Benchmark': [
            format_pct(spy_total),
            format_pct((spy_cum.iloc[-1]) ** (252/len(spy_aligned)) - 1),
            format_pct(spy_aligned.std() * np.sqrt(252)),
            f"{spy_sharpe:.3f}",
            format_pct(spy_dd),
            "1.0000",
            f"{((spy_cum.iloc[-1]) ** (252/len(spy_aligned)) - 1) / abs(spy_dd):.3f}",
        ],
    })
    print(tabulate(comparison, headers='keys', tablefmt='rounded_grid', showindex=False))

    beta_abs = abs(port_metrics.get('beta', 999))
    neutrality = "✅ MARKET NEUTRAL" if beta_abs < 0.15 else "⚠️  NOT FULLY NEUTRAL"
    print(f"\n  Market Neutrality Check: {neutrality} (|β| = {beta_abs:.4f})")

    return {
        'portfolio_pnl':   portfolio_pnl,
        'cum_equity':      cum_equity,
        'rolling_beta':    rolling_beta,
        'pnl_df':          pnl_df,
        'n_pairs':         n_pairs,
        'spy_cum':         spy_cum,
        **port_metrics,
    }

portfolio_results = aggregate_portfolio(all_pair_results, spy_returns)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 16 — VISUALIZATIONS: SINGLE PAIR DEEP DIVE                            ║
# ║  4-panel analysis chart for the best-performing pair                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 11: GENERATING VISUALIZATIONS")

def plot_pair_deep_dive(pair_result: Dict, pair_name: str) -> go.Figure:
    """
    Create a comprehensive 4-panel analysis chart for a single pair.

    Panels:
    1. Normalized Price Series   — how A and B co-move
    2. Spread with Signals       — spread + entry/exit markers
    3. Z-Score with Bands        — trading signal with thresholds
    4. Cumulative P&L            — equity curve of the strategy

    This is the type of chart you would include in a hedge fund research report
    or a portfolio manager's morning pack.
    """
    ticker_a = pair_result['ticker_a']
    ticker_b = pair_result['ticker_b']

    # Use full-period data for visualization (train + test)
    spread    = pair_result['spread_full']
    z_score   = pair_result['z_score_full']
    signals   = pair_result['signals_full']
    px_a      = prices_clean[ticker_a]
    px_b      = prices_clean[ticker_b]

    # Normalized prices (rebased to 100)
    px_a_norm = px_a / px_a.iloc[0] * 100
    px_b_norm = px_b / px_b.iloc[0] * 100

    # P&L is for test period
    cum_equity = pair_result.get('cum_equity', pd.Series(dtype=float))
    n_trades   = pair_result.get('n_trades', 0)
    half_life  = pair_result.get('half_life', 0)
    beta       = pair_result.get('beta', 0)

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f"<b>①</b> Normalized Prices — {ticker_a} vs. {ticker_b}",
            f"<b>②</b> Spread = log({ticker_a}) - {pair_result.get('beta', 1.0):.3f}·log({ticker_b})",
            f"<b>③</b> Rolling Z-Score (60-Day Window)",
            f"<b>④</b> Cumulative P&L — Out-of-Sample Strategy",
        ],
        vertical_spacing=0.07,
        shared_xaxes=True,
    )

    # ── Panel 1: Normalized Prices ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=px_a_norm.index, y=px_a_norm.values,
        name=ticker_a, line=dict(color=COLORS['primary'], width=1.5),
        mode='lines',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=px_b_norm.index, y=px_b_norm.values,
        name=ticker_b, line=dict(color=COLORS['secondary'], width=1.5),
        mode='lines',
    ), row=1, col=1)

    # Train/Test divider
    fig.add_vline(x=str(train_end)[:10], line_dash="dash",
                  line_color=COLORS['gold'], line_width=1.5, row=1, col=1)

    # ── Panel 2: Spread ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread.values,
        name="Spread", line=dict(color=COLORS['purple'], width=1.5),
        fill='tozeroy', fillcolor='rgba(155, 89, 182, 0.1)',
    ), row=2, col=1)

    # Rolling mean of spread (in-sample)
    spread_mean = spread.rolling(CONFIG['ZSCORE_WINDOW']).mean()
    fig.add_trace(go.Scatter(
        x=spread_mean.index, y=spread_mean.values,
        name="Rolling Mean", line=dict(color=COLORS['gold'], width=1, dash='dash'),
        mode='lines', showlegend=False,
    ), row=2, col=1)

    # ── Panel 3: Z-Score with Bands ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=z_score.index, y=z_score.values,
        name="Z-Score", line=dict(color=COLORS['primary'], width=1.2),
    ), row=3, col=1)

    # Threshold bands
    for y_val, label, color in [
        (CONFIG['ENTRY_ZSCORE'],   'Short Entry', 'rgba(255,68,68,0.7)'),
        (-CONFIG['ENTRY_ZSCORE'],  'Long Entry',  'rgba(0,255,136,0.7)'),
        (CONFIG['EXIT_ZSCORE'],    'Exit Band',   'rgba(255,215,0,0.5)'),
        (-CONFIG['EXIT_ZSCORE'],   'Exit Band',   'rgba(255,215,0,0.5)'),
        (CONFIG['STOP_ZSCORE'],    'Stop Loss',   'rgba(255,100,100,0.4)'),
        (-CONFIG['STOP_ZSCORE'],   'Stop Loss',   'rgba(255,100,100,0.4)'),
    ]:
        fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                      line_width=1, row=3, col=1)

    # Signal shading (long = green, short = red)
    long_mask  = signals == 1
    short_mask = signals == -1

    # ── Panel 4: Cumulative P&L ─────────────────────────────────────────────────
    if len(cum_equity) > 0:
        fig.add_trace(go.Scatter(
            x=cum_equity.index, y=(cum_equity.values - 1) * 100,
            name="Strategy P&L (%)", line=dict(color=COLORS['positive'], width=2),
            fill='tozeroy', fillcolor='rgba(0,255,136,0.1)',
        ), row=4, col=1)

        # Drawdown shading
        roll_max = cum_equity.cummax()
        dd_series = (cum_equity - roll_max) / roll_max * 100
        fig.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values,
            name="Drawdown", line=dict(color=COLORS['negative'], width=1),
            fill='tozeroy', fillcolor='rgba(255,68,68,0.15)',
        ), row=4, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>Statistical Pairs Analysis: {ticker_a} / {ticker_b}</b>"
                 f"<br><sup>Sector: {pair_result.get('sector','')} | "
                 f"EG p-value: {pair_result.get('eg_pvalue', 0):.4f} | "
                 f"Half-Life: {half_life:.1f} days | "
                 f"Trades: {n_trades} | "
                 f"Sharpe: {pair_result.get('sharpe', 0):.2f}</sup>",
            font=dict(size=14, color=COLORS['text']),
            x=0.02,
        ),
        height=900,
        showlegend=True,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=11),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(10,14,26,0.8)',
            bordercolor=COLORS['grid'],
            borderwidth=1,
        ),
    )

    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True)

    return fig

def plot_portfolio_dashboard(portfolio_results: Dict, spy_returns: pd.Series) -> go.Figure:
    """
    Create an institutional-grade portfolio dashboard with 6 panels:

    1. Equity Curve          — Portfolio vs. SPY
    2. Rolling Sharpe        — 60-day rolling Sharpe (strategy quality over time)
    3. Rolling Beta          — Market neutrality over time (should be ~0)
    4. Monthly Returns Heat  — Calendar heatmap of monthly P&L
    5. Return Distribution   — Histogram vs. normal distribution
    6. Drawdown Profile      — Underwater equity curve
    """
    port_pnl    = portfolio_results.get('portfolio_pnl', pd.Series(dtype=float))
    cum_equity  = portfolio_results.get('cum_equity', pd.Series(dtype=float))
    rolling_beta= portfolio_results.get('rolling_beta', pd.Series(dtype=float))
    spy_cum     = portfolio_results.get('spy_cum', pd.Series(dtype=float))

    if len(port_pnl) == 0:
        print("  ⚠️  No portfolio data to plot.")
        return go.Figure()

    # Rolling metrics
    rolling_sharpe = port_pnl.rolling(60).apply(
        lambda x: annualized_sharpe(x), raw=False
    )

    # Monthly returns
    monthly_pnl = port_pnl.resample('M').sum()
    monthly_df  = pd.DataFrame({
        'year':  monthly_pnl.index.year,
        'month': monthly_pnl.index.month_name().str[:3],
        'pnl':   monthly_pnl.values * 100,
    })
    pivot = monthly_df.pivot(index='year', columns='month', values='pnl')
    months_order = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])

    # Drawdown
    roll_max  = cum_equity.cummax()
    drawdown  = (cum_equity - roll_max) / roll_max * 100

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "<b>Portfolio Equity Curve vs. SPY</b>",
            "<b>Rolling 60-Day Sharpe Ratio</b>",
            "<b>Rolling Beta to SPY (Market Neutrality)</b>",
            "<b>Monthly P&L Heatmap (%)</b>",
            "<b>P&L Return Distribution</b>",
            "<b>Drawdown Profile</b>",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # ── Panel 1: Equity Curves ─────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=cum_equity.index, y=(cum_equity.values - 1) * 100,
        name="Pairs Portfolio", line=dict(color=COLORS['positive'], width=2.5),
        fill='tozeroy', fillcolor='rgba(0,255,136,0.08)',
    ), row=1, col=1)

    if len(spy_cum) > 0:
        spy_aligned = spy_cum.reindex(cum_equity.index).ffill()
        fig.add_trace(go.Scatter(
            x=spy_aligned.index, y=(spy_aligned.values - 1) * 100,
            name="SPY Benchmark", line=dict(color=COLORS['secondary'], width=2, dash='dash'),
        ), row=1, col=1)

    fig.add_hline(y=0, line_color=COLORS['neutral'], line_width=1, row=1, col=1)

    # ── Panel 2: Rolling Sharpe ─────────────────────────────────────────────────
    valid_sharpe = rolling_sharpe.dropna()
    fig.add_trace(go.Scatter(
        x=valid_sharpe.index, y=valid_sharpe.values,
        name="Rolling Sharpe", line=dict(color=COLORS['gold'], width=1.5),
        fill='tozeroy', fillcolor='rgba(255,215,0,0.1)',
    ), row=1, col=2)

    fig.add_hline(y=1.0, line_color=COLORS['positive'], line_dash='dot',
                  line_width=1, row=1, col=2)
    fig.add_hline(y=0.0, line_color=COLORS['neutral'], line_width=0.5, row=1, col=2)

    # ── Panel 3: Rolling Beta ──────────────────────────────────────────────────
    valid_beta = rolling_beta.dropna()
    fig.add_trace(go.Scatter(
        x=valid_beta.index, y=valid_beta.values,
        name="Rolling Beta", line=dict(color=COLORS['purple'], width=1.5),
    ), row=2, col=1)

    fig.add_hline(y=0.0,  line_color=COLORS['positive'], line_width=1,
                  line_dash='dash', row=2, col=1)
    fig.add_hline(y=0.2,  line_color=COLORS['negative'], line_width=1,
                  line_dash='dot',  row=2, col=1)
    fig.add_hline(y=-0.2, line_color=COLORS['negative'], line_width=1,
                  line_dash='dot',  row=2, col=1)

    # ── Panel 4: Monthly Heatmap ───────────────────────────────────────────────
    if len(pivot) > 0:
        z_vals = pivot.values
        fig.add_trace(go.Heatmap(
            z=z_vals,
            x=pivot.columns.tolist(),
            y=pivot.index.astype(str).tolist(),
            colorscale=[
                [0.0, '#8B0000'],
                [0.35, '#CC0000'],
                [0.45, '#1A2035'],
                [0.5,  '#1A2035'],
                [0.55, '#1A2035'],
                [0.65, '#006400'],
                [1.0,  '#00AA00'],
            ],
            zmid=0,
            text=np.round(z_vals, 1),
            texttemplate='%{text}%',
            colorbar=dict(len=0.3, y=0.5, x=1.01),
            showscale=True,
        ), row=2, col=2)

    # ── Panel 5: Return Distribution ──────────────────────────────────────────
    daily_pct = port_pnl * 100
    fig.add_trace(go.Histogram(
        x=daily_pct.dropna().values,
        nbinsx=60,
        name="Daily Returns",
        marker_color=COLORS['primary'],
        opacity=0.7,
    ), row=3, col=1)

    # Normal distribution overlay
    mu_ret = daily_pct.mean()
    sd_ret = daily_pct.std()
    x_norm = np.linspace(daily_pct.min(), daily_pct.max(), 200)
    y_norm = stats.norm.pdf(x_norm, mu_ret, sd_ret) * len(daily_pct) * (daily_pct.max() - daily_pct.min()) / 60
    fig.add_trace(go.Scatter(
        x=x_norm, y=y_norm, name="Normal Dist.",
        line=dict(color=COLORS['gold'], width=2),
    ), row=3, col=1)

    fig.add_vline(x=0, line_color=COLORS['neutral'], line_width=1, row=3, col=1)

    # ── Panel 6: Drawdown ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name="Drawdown", fill='tozeroy',
        fillcolor='rgba(255,68,68,0.3)',
        line=dict(color=COLORS['negative'], width=1),
    ), row=3, col=2)

    fig.add_hline(y=0, line_color=COLORS['neutral'], line_width=0.5, row=3, col=2)

    # ── Layout ─────────────────────────────────────────────────────────────────
    total_return = portfolio_results.get('total_return', 0)
    sharpe_ratio = portfolio_results.get('sharpe', 0)
    max_dd_val   = portfolio_results.get('max_drawdown', 0)
    beta_val     = portfolio_results.get('beta', 0)
    n_pairs_used = portfolio_results.get('n_pairs', 0)

    fig.update_layout(
        title=dict(
            text=(f"<b>Statistical Pairs Trading Portfolio — Institutional Dashboard</b>"
                  f"<br><sup>N={n_pairs_used} pairs | "
                  f"Total Return: {total_return:.1%} | "
                  f"Sharpe: {sharpe_ratio:.2f} | "
                  f"Max DD: {max_dd_val:.1%} | "
                  f"Beta: {beta_val:.3f}</sup>"),
            font=dict(size=14, color=COLORS['text']),
            x=0.02,
        ),
        height=1000,
        showlegend=True,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=10),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(10,14,26,0.8)', bordercolor=COLORS['grid'], borderwidth=1),
    )

    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True)

    return fig

def plot_halftime_distribution(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Half-life distribution across all valid pairs.
    Shows the distribution of mean-reversion speeds in the universe.

    Why this matters: Half-life determines your expected holding period
    and the frequency of signal turnover. Institutional traders optimize
    for pairs with consistent half-lives in the 5-30 day range.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Half-Life Distribution</b>",
            "<b>Sharpe Ratio vs. Half-Life</b>"
        ],
        horizontal_spacing=0.12,
    )

    valid_hl = metrics_df['half_life'].dropna()

    fig.add_trace(go.Histogram(
        x=valid_hl,
        nbinsx=30,
        marker_color=COLORS['primary'],
        marker_line=dict(color=COLORS['background'], width=0.5),
        opacity=0.85,
        name="Pairs",
    ), row=1, col=1)

    fig.add_vline(x=CONFIG['HALF_LIFE_MIN'], line_color=COLORS['positive'],
                  line_dash='dash', line_width=2, row=1, col=1)
    fig.add_vline(x=CONFIG['HALF_LIFE_MAX'], line_color=COLORS['negative'],
                  line_dash='dash', line_width=2, row=1, col=1)

    # Scatter: Sharpe vs. Half-Life
    valid_scatter = metrics_df[['half_life', 'sharpe', 'sector', 'ticker_a', 'ticker_b']].dropna()
    sectors = valid_scatter['sector'].unique()
    colors_scatter = px.colors.qualitative.Set2

    for i, sector in enumerate(sectors):
        mask = valid_scatter['sector'] == sector
        sub  = valid_scatter[mask]
        fig.add_trace(go.Scatter(
            x=sub['half_life'],
            y=sub['sharpe'],
            mode='markers',
            name=sector,
            text=[f"{a}/{b}" for a, b in zip(sub['ticker_a'], sub['ticker_b'])],
            hovertemplate="<b>%{text}</b><br>Half-Life: %{x:.1f} days<br>Sharpe: %{y:.3f}",
            marker=dict(
                size=8,
                color=colors_scatter[i % len(colors_scatter)],
                line=dict(width=0.5, color='white'),
                opacity=0.8,
            ),
        ), row=1, col=2)

    fig.add_hline(y=1.0, line_color=COLORS['positive'], line_dash='dot',
                  line_width=1, row=1, col=2)
    fig.add_hline(y=0.0, line_color=COLORS['neutral'],  line_width=0.5, row=1, col=2)

    fig.update_layout(
        title=dict(
            text="<b>OU Half-Life Analysis Across Cointegrated Pairs</b>",
            font=dict(size=13, color=COLORS['text']), x=0.02,
        ),
        height=420,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=11),
        legend=dict(bgcolor='rgba(10,14,26,0.8)', bordercolor=COLORS['grid'], borderwidth=1),
    )
    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True)
    fig.update_xaxes(title_text="Half-Life (Trading Days)", row=1, col=1)
    fig.update_xaxes(title_text="Half-Life (Trading Days)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Pairs", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)

    return fig

def plot_rolling_cointegration_stability(
    best_pairs: List[str],
    log_prices: pd.DataFrame,
    window: int = 252
) -> go.Figure:
    """
    Rolling cointegration p-value for top pairs over time.

    Key insight: Cointegration is NOT static — pairs can break down.
    This chart shows which pairs maintain stable cointegration throughout
    the period vs. which are unstable.

    Institutional quants monitor this in real-time. When p-value rises above
    0.10-0.15, they reduce or exit the position.
    """
    fig = go.Figure()
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['positive'],
              COLORS['gold'], COLORS['purple']]

    for i, pair_key in enumerate(best_pairs[:5]):
        if pair_key not in all_pair_results:
            continue

        result    = all_pair_results[pair_key]
        ticker_a  = result['ticker_a']
        ticker_b  = result['ticker_b']

        if ticker_a not in log_prices.columns or ticker_b not in log_prices.columns:
            continue

        log_a = log_prices[ticker_a]
        log_b = log_prices[ticker_b]

        rolling_pvals = []
        dates = []

        for end_idx in range(window, len(log_prices), 21):  # Monthly steps
            window_a = log_a.iloc[end_idx - window:end_idx]
            window_b = log_b.iloc[end_idx - window:end_idx]

            try:
                _, pval, _ = coint(window_a, window_b, trend='c')
                rolling_pvals.append(pval)
                dates.append(log_prices.index[end_idx])
            except Exception:
                rolling_pvals.append(np.nan)
                dates.append(log_prices.index[end_idx])

        if rolling_pvals:
            fig.add_trace(go.Scatter(
                x=dates, y=rolling_pvals,
                name=f"{ticker_a}/{ticker_b}",
                line=dict(color=colors[i % len(colors)], width=1.5),
                mode='lines',
            ))

    # Significance thresholds
    fig.add_hline(y=0.05, line_color=COLORS['positive'], line_dash='dash',
                  line_width=2, annotation_text="5% Threshold", annotation_position="right")
    fig.add_hline(y=0.10, line_color=COLORS['negative'], line_dash='dot',
                  line_width=1.5, annotation_text="10% Threshold", annotation_position="right")
    fig.add_vline(x=str(train_end)[:10], line_color=COLORS['gold'],
                  line_dash='dash', line_width=2)

    fig.update_layout(
        title=dict(
            text="<b>Rolling Cointegration Stability (1-Year Window)</b>"
                 "<br><sup>P-value < 0.05 (green line) = statistically cointegrated | "
                 "Gold line = train/test split</sup>",
            font=dict(size=13, color=COLORS['text']), x=0.02,
        ),
        height=420,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=11),
        xaxis_title="Date",
        yaxis_title="EG Cointegration P-Value",
        hovermode='x unified',
        legend=dict(bgcolor='rgba(10,14,26,0.8)', bordercolor=COLORS['grid'], borderwidth=1),
    )
    fig.update_xaxes(gridcolor=COLORS['grid'])
    fig.update_yaxes(gridcolor=COLORS['grid'], range=[0, 0.4])

    return fig

# ── Generate All Visualizations ────────────────────────────────────────────────
print("  📊 Generating visualizations...")

if all_pair_results:
    best_pairs_list = metrics_df['pair'].tolist() if len(metrics_df) > 0 else list(all_pair_results.keys())

    # Chart 1: Best pair deep-dive
    if best_pairs_list:
        best_pair_key = best_pairs_list[0]
        if best_pair_key in all_pair_results:
            fig_pair = plot_pair_deep_dive(all_pair_results[best_pair_key], best_pair_key)
            fig_pair.show()
            print(f"  ✅ Chart 1: Pair deep-dive for {best_pair_key}")

    # Chart 2: Portfolio dashboard
    if portfolio_results:
        fig_portfolio = plot_portfolio_dashboard(portfolio_results, spy_returns)
        fig_portfolio.show()
        print(f"  ✅ Chart 2: Portfolio dashboard")

    # Chart 3: Half-life distribution
    if len(metrics_df) > 0:
        fig_hl = plot_halftime_distribution(metrics_df)
        fig_hl.show()
        print(f"  ✅ Chart 3: Half-life distribution")

    # Chart 4: Rolling cointegration stability
    if best_pairs_list:
        fig_stability = plot_rolling_cointegration_stability(
            best_pairs_list[:5], log_prices
        )
        fig_stability.show()
        print(f"  ✅ Chart 4: Rolling cointegration stability")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 17 — KALMAN FILTER ENHANCEMENT (DYNAMIC HEDGE RATIO)                  ║
# ║  Extension: Replace static OLS β with time-varying Kalman Filter β          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 12: OPTIONAL ENHANCEMENT — KALMAN FILTER HEDGE RATIO")

def kalman_filter_hedge_ratio(
    prices_a: pd.Series,
    prices_b: pd.Series,
    delta: float = 1e-5,
    Ve:    float = 0.001
) -> Tuple[pd.Series, pd.Series]:
    """
    Dynamic hedge ratio estimation using a Kalman Filter.

    The Kalman Filter treats the hedge ratio β as a latent state that
    evolves over time, rather than being fixed (as in OLS).

    State Model: β_t = β_{t-1} + w_t    where w_t ~ N(0, W)
    Observation:  y_t = β_t * x_t + v_t  where v_t ~ N(0, V)

    Parameters:
        delta: State noise variance (higher = faster adaptation)
        Ve:    Observation noise variance (higher = smoother β)

    Why use Kalman Filter?
    - Pairs don't always maintain constant hedge ratios
    - The relationship between two stocks can drift over time
    - Dynamic β adapts to structural shifts (e.g., mergers, strategy changes)
    - Academic research: Optimal pairs trading uses time-varying hedge ratios
      (Vidyamurthy 2004, Liu & Timmermann 2013)

    Returns:
        betas:      Time-varying hedge ratio series
        intercepts: Time-varying intercept series
    """
    log_a = np.log(prices_a.values)
    log_b = np.log(prices_b.values)
    n     = len(log_a)

    # State: [intercept, beta] — 2-dimensional
    Wt = delta / (1 - delta) * np.eye(2)   # State covariance (process noise)
    Vt = Ve                                  # Observation noise variance

    # Initialize
    theta   = np.zeros((n, 2))             # State estimate
    P       = np.zeros((n, 2, 2))          # State covariance
    P[0]    = 1e4 * np.eye(2)              # Large initial uncertainty

    for t in range(1, n):
        # Observation vector: F = [1, log_b_t]
        F = np.array([1.0, log_b[t]])

        # ── Predict ─────────────────────────────────────────────────────────
        theta_pred = theta[t-1]
        P_pred     = P[t-1] + Wt

        # ── Update ──────────────────────────────────────────────────────────
        # Kalman Gain
        S       = float(F @ P_pred @ F.T) + Vt
        K       = (P_pred @ F.T) / S         # (2,) vector

        # Innovation (prediction error)
        innovation = log_a[t] - float(F @ theta_pred)

        # Update state
        theta[t] = theta_pred + K * innovation
        P[t]     = P_pred - np.outer(K, F) @ P_pred

    betas      = pd.Series(theta[:, 1], index=prices_a.index, name='kalman_beta')
    intercepts = pd.Series(theta[:, 0], index=prices_a.index, name='kalman_intercept')

    return betas, intercepts

def plot_kalman_vs_ols(
    pair_result: Dict,
    prices: pd.DataFrame
) -> go.Figure:
    """Compare OLS (static) vs. Kalman Filter (dynamic) hedge ratios."""
    ticker_a = pair_result['ticker_a']
    ticker_b = pair_result['ticker_b']
    beta_ols  = pair_result['beta']

    pa = prices[ticker_a]
    pb = prices[ticker_b]

    kf_betas, kf_intercepts = kalman_filter_hedge_ratio(pa, pb)

    # Dynamic spread (using Kalman betas)
    kf_spread = np.log(pa) - kf_intercepts - kf_betas * np.log(pb)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[
                            "<b>Hedge Ratio: OLS (Static) vs. Kalman Filter (Dynamic)</b>",
                            "<b>Spread Comparison: OLS vs. Kalman</b>",
                        ],
                        vertical_spacing=0.1)

    # Kalman beta over time
    fig.add_trace(go.Scatter(
        x=kf_betas.index, y=kf_betas.values,
        name="Kalman β (Dynamic)", line=dict(color=COLORS['primary'], width=1.5),
    ), row=1, col=1)

    # OLS beta (flat line)
    fig.add_hline(y=beta_ols, line_color=COLORS['secondary'],
                  line_dash='dash', line_width=2, row=1, col=1,
                  annotation_text=f"OLS β = {beta_ols:.3f}")

    # Kalman spread
    fig.add_trace(go.Scatter(
        x=kf_spread.index, y=kf_spread.values,
        name="Kalman Spread", line=dict(color=COLORS['positive'], width=1.2),
    ), row=2, col=1)

    # OLS spread (using static beta)
    ols_spread = pair_result.get('spread_full')
    if ols_spread is not None:
        fig.add_trace(go.Scatter(
            x=ols_spread.index, y=ols_spread.values,
            name="OLS Spread", line=dict(color=COLORS['secondary'], width=1.2, dash='dot'),
        ), row=2, col=1)

    fig.add_vline(x=str(train_end)[:10], line_color=COLORS['gold'],
                  line_dash='dash', line_width=1.5)

    fig.update_layout(
        height=550,
        plot_bgcolor=COLORS['background'],
        paper_bgcolor='#0D1117',
        font=dict(color=COLORS['text'], size=11),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(10,14,26,0.8)', bordercolor=COLORS['grid'], borderwidth=1),
    )
    fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True)
    fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True)

    return fig

# ── Demo Kalman Filter on Best Pair ───────────────────────────────────────────
if all_pair_results and best_pairs_list:
    best_pair_key = best_pairs_list[0]
    best_result   = all_pair_results[best_pair_key]

    print(f"  🔬 Fitting Kalman Filter for: {best_result['ticker_a']} / {best_result['ticker_b']}")
    fig_kf = plot_kalman_vs_ols(best_result, prices_clean)
    fig_kf.show()
    print(f"  ✅ Kalman Filter comparison chart generated.")
    print(f"\n  📌 Key Insight: When the Kalman β drifts significantly from OLS β,")
    print(f"     it may indicate a structural change in the pair relationship.")
    print(f"     Dynamic hedge ratios reduce tracking error and improve profitability.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 18 — PAIR BREAKDOWN ANALYSIS                                          ║
# ║  Show which pairs failed and why — critical for understanding limitations   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("STEP 13: PAIR BREAKDOWN ANALYSIS")

def analyze_pair_failures(
    scan_results: pd.DataFrame,
    all_results:  Dict
) -> None:
    """
    Analyze and categorize WHY pairs failed cointegration or backtesting.

    This section is critically important for a portfolio-worthy project.
    It demonstrates understanding of strategy limitations — essential for
    any serious quant research presentation or fund pitch.

    Failure Modes:
    1. Statistical: EG test not significant (p > 0.05) — not cointegrated
    2. Half-life: Too short (<5d) or too long (>60d) — not exploitable
    3. Performance: Strategy lost money (negative Sharpe) in OOS
    4. Breakdown: Pair was cointegrated in training, failed in test
    """
    print("  📊 Failure Mode Analysis:")
    print()

    total_candidates = len(scan_results)

    # ── Failure Reason 1: Not Cointegrated ────────────────────────────────────
    failed_eg = scan_results[~scan_results.get('passed_eg', pd.Series([False]*len(scan_results)))].shape[0]
    passed_eg = scan_results['passed_eg'].sum() if 'passed_eg' in scan_results.columns else 0

    print(f"  Stage 1 — Engle-Granger Test:")
    print(f"    Pairs Tested      : {total_candidates}")
    print(f"    Passed (p<0.05)   : {passed_eg}  ({passed_eg/max(total_candidates,1):.1%})")
    print(f"    Failed            : {total_candidates - passed_eg}  ({(total_candidates-passed_eg)/max(total_candidates,1):.1%})")
    print(f"    ← Most pairs fail here. Markets are MORE correlated than cointegrated.")

    # ── Failure Reason 2: Half-Life Filter ────────────────────────────────────
    if 'in_hl_range' in scan_results.columns:
        passed_hl = scan_results['in_hl_range'].sum()
        failed_hl = passed_eg - passed_hl
        print(f"\n  Stage 2 — Half-Life Filter ({CONFIG['HALF_LIFE_MIN']}–{CONFIG['HALF_LIFE_MAX']} days):")
        print(f"    Passed EG         : {passed_eg}")
        print(f"    Valid Half-Life   : {passed_hl}  ({passed_hl/max(passed_eg,1):.1%})")
        print(f"    Rejected          : {failed_hl}")
        print(f"    ← Pairs with HL < {CONFIG['HALF_LIFE_MIN']}d = too noisy; HL > {CONFIG['HALF_LIFE_MAX']}d = too slow")

    # ── OOS Performance Analysis ───────────────────────────────────────────────
    if len(metrics_df) > 0:
        print(f"\n  Stage 3 — Out-of-Sample Performance:")
        profitable_pairs = (metrics_df['sharpe'] > 0.5).sum()
        losing_pairs     = (metrics_df['sharpe'] < 0.0).sum()
        total_tested     = len(metrics_df)

        print(f"    Pairs Backtested  : {total_tested}")
        print(f"    Sharpe > 0.5      : {profitable_pairs}  ({profitable_pairs/max(total_tested,1):.1%})")
        print(f"    Sharpe > 1.0      : {(metrics_df['sharpe'] > 1.0).sum()}")
        print(f"    Negative Sharpe   : {losing_pairs}  ({losing_pairs/max(total_tested,1):.1%})")

        # Most common failure: cointegration breakdown
        if 'n_trades' in metrics_df.columns:
            zero_trades = (metrics_df['n_trades'] == 0).sum()
            print(f"    Zero Trades       : {zero_trades}  (signal never triggered)")

    # ── Common Breakdown Patterns ──────────────────────────────────────────────
    print(f"""
  📌 Common Pair Breakdown Patterns:

  1. SECTOR ROTATION: When a sector rotates, previously cointegrated stocks
     decouple temporarily (e.g., growth vs. value within Tech in 2022)

  2. CORPORATE EVENTS: Mergers, spin-offs, or management changes can
     permanently shift the pair relationship (e.g., GOOGL after Alphabet split)

  3. REGIME CHANGE: Rising rate environment (2022) broke many L/S equity pairs
     as capital costs changed differentially by company quality

  4. MEAN-REVERSION SATURATION: If everyone trades the same pairs, the signal
     gets arbitraged away — this is called "statistical arbitrage crowding"

  5. COVARIANCE INSTABILITY: The OLS hedge ratio (β) drifts over time,
     making the spread non-stationary even if it once was (→ use Kalman Filter)
  """)

analyze_pair_failures(scan_results, all_pair_results)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 19 — FINAL SUMMARY REPORT                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("FINAL REPORT: STATISTICAL PAIRS TRADING ENGINE")

def print_final_report(portfolio_results: Dict, metrics_df: pd.DataFrame) -> None:
    """Print comprehensive final report suitable for a research presentation."""

    if not portfolio_results:
        print("  ⚠️  No portfolio results to report.")
        return

    total_return = portfolio_results.get('total_return', 0)
    cagr         = portfolio_results.get('cagr', 0)
    vol          = portfolio_results.get('vol_annual', 0)
    sharpe       = portfolio_results.get('sharpe', 0)
    sortino      = portfolio_results.get('sortino', 0)
    max_dd       = portfolio_results.get('max_drawdown', 0)
    calmar       = portfolio_results.get('calmar', 0)
    beta         = portfolio_results.get('beta', 0)
    n_pairs      = portfolio_results.get('n_pairs', 0)

    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║           STATISTICAL PAIRS TRADING — FINAL RESULTS         ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Strategy     : Cointegration-Based Statistical Arbitrage   ║
  ║  Period       : {CONFIG['START_DATE']} → {CONFIG['END_DATE']}                ║
  ║  OOS Window   : {test_start.date()} → {prices_clean.index[-1].date()}       ║
  ║  Universe     : {len(clean_tickers):>3d} stocks, {n_pairs:>2d} traded pairs             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  RETURN METRICS                                              ║
  ║    Total Return      : {total_return:>8.2%}                          ║
  ║    CAGR              : {cagr:>8.2%}                          ║
  ║    Annualized Vol    : {vol:>8.2%}                          ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  RISK-ADJUSTED METRICS                                       ║
  ║    Sharpe Ratio      : {sharpe:>8.3f}                          ║
  ║    Sortino Ratio     : {sortino:>8.3f}                          ║
  ║    Calmar Ratio      : {calmar:>8.3f}                          ║
  ║    Max Drawdown      : {max_dd:>8.2%}                          ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  MARKET NEUTRALITY                                           ║
  ║    Beta to SPY       : {beta:>8.4f}  {'✅ NEUTRAL' if abs(beta) < 0.2 else '⚠️  CHECK':<20}       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  PAIR STATISTICS                                             ║
  ║    Pairs with Sharpe > 1.0  : {(metrics_df['sharpe'] > 1.0).sum() if len(metrics_df) > 0 else 'N/A':>4}                   ║
  ║    Median Pair Half-Life    : {metrics_df['half_life'].median() if len(metrics_df) > 0 else 0:>4.1f} days               ║
  ║    Median Pair Sharpe       : {metrics_df['sharpe'].median() if len(metrics_df) > 0 else 0:>4.3f}                   ║
  ╚══════════════════════════════════════════════════════════════╝
  """)

    print("  📝 RESUME DESCRIPTION (Copy-Paste Ready):")
    print("""  ─────────────────────────────────────────────────────────────
  Built a production-grade statistical pairs trading engine in
  Python, scanning the S&P 500 for cointegrated stock pairs using
  Engle-Granger and Johansen cointegration tests. Implemented an
  Ornstein-Uhlenbeck mean-reversion model to estimate spread half-
  lives, generating z-score-based long/short signals with dynamic
  entry, exit, and stop-loss logic. Backtested the market-neutral
  strategy out-of-sample with realistic 10bps transaction costs,
  achieving a Sharpe ratio of {:>4.2f} and maximum drawdown of {:>4.1%}
  across {} simultaneous pairs. Implemented Kalman Filter dynamic
  hedge ratio estimation as an institutional-grade extension.
  """.format(sharpe, abs(max_dd), n_pairs))

    print("  🎯 POTENTIAL UPGRADES FOR INSTITUTIONAL LEVEL:")
    upgrades = [
        ("1", "Kalman Filter Hedge Ratio",    "Time-varying β instead of static OLS — reduces tracking error by ~15-30%"),
        ("2", "HMM Regime Detection",          "Only trade pairs when regime = 'mean-reverting'; exit in trending markets"),
        ("3", "Multi-Asset Baskets",            "Johansen multivariate: trade 3-5 asset baskets instead of just pairs"),
        ("4", "Dollar-Neutral Position Sizing", "Vol-target each pair at 10% annual vol for consistent risk allocation"),
        ("5", "Dynamic Entry Thresholds",       "Calibrate entry z-score to pair-specific OU parameters (not fixed 2.0)"),
        ("6", "Execution Optimization",         "TWAP execution for large positions to reduce market impact"),
        ("7", "Pair Stability Monitoring",      "Real-time rolling cointegration monitoring → auto-stop failing pairs"),
        ("8", "Cross-Asset Extension",          "Apply to ETF pairs, commodity pairs, FX pairs for broader universe"),
    ]
    for num, title, detail in upgrades:
        print(f"  [{num}] {title}")
        print(f"       → {detail}")

    print(f"""
  🏦 HOW HEDGE FUNDS ACTUALLY USE THIS:

  D.E. Shaw (1988): Pioneered this with dozens of pairs across all
    U.S. equities. Used proprietary execution algorithms to minimize
    market impact — the execution edge matters as much as the signal.

  Millennium Management: Multi-PM structure where individual PMs run
    sector-specific pairs strategies. 100-300 pairs simultaneously.

  Goldman Sachs Global Alpha (pre-2008): Extended to global equities
    across 30+ markets. Strategy suffered heavily in the "quant quake"
    of Aug 2007 when multiple funds de-leveraged simultaneously.

  Two Sigma: Uses ML to identify regime shifts that break pairs,
    automatically re-estimating hedge ratios in real time.

  Key Insight: The edge in institutional pairs trading is NOT the
  cointegration test — that's public knowledge. The edge is:
    (1) Speed of execution
    (2) Signal sophistication (Kalman, ML)
    (3) Universe breadth (1000+ pairs)
    (4) Transaction cost optimization
  """)

print_final_report(portfolio_results, metrics_df)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CELL 20 — GITHUB & PORTFOLIO PRESENTATION GUIDE                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print_header("GITHUB & PORTFOLIO PRESENTATION GUIDE")

print("""
  📁 RECOMMENDED GITHUB REPOSITORY STRUCTURE:

  statistical-pairs-trading/
  ├── README.md                    ← Comprehensive write-up (use the template below)
  ├── notebooks/
  │   ├── 01_pairs_trading_engine.ipynb    ← This notebook
  │   └── 02_kalman_filter_extension.ipynb ← Enhancement notebook
  ├── src/
  │   ├── __init__.py
  │   ├── data_loader.py           ← Price download utilities
  │   ├── cointegration.py         ← Statistical test functions
  │   ├── signal_generator.py      ← Z-score signal logic
  │   ├── backtester.py            ← Backtest engine class
  │   ├── kalman_filter.py         ← KF dynamic hedge ratio
  │   └── performance.py           ← Metrics computation
  ├── data/
  │   └── sp500_sectors.csv        ← Sector classification cache
  ├── outputs/
  │   ├── charts/                  ← PNG exports of visualizations
  │   └── results/                 ← CSV of pair metrics
  └── requirements.txt

  📋 README.md TEMPLATE SECTIONS:
  1. Strategy Overview (2-3 sentences)
  2. Mathematical Foundation (LaTeX equations for spread, z-score, OU)
  3. Results Table (Sharpe, DD, CAGR vs. benchmark)
  4. Key Visualizations (embed 2-3 charts as GIFs/PNGs)
  5. How to Run (3-step: clone → pip install → run notebook)
  6. References (Engle-Granger 1987, Johansen 1988, Vidyamurthy 2004)

  💼 RECRUITER-FACING HIGHLIGHTS:
  • "Implemented statistical arbitrage strategy used by D.E. Shaw and Citadel"
  • "Backtested against 5 years of S&P 500 data with realistic transaction costs"
  • "Implemented both classical (OLS) and dynamic (Kalman Filter) hedge ratios"
  • "Market-neutral portfolio with |β| < 0.15 verified empirically"
  • "Achieved Sharpe ratio of X.XX out-of-sample (2021-2024)"

  📊 EXPORT CHARTS TO PNG FOR README:
  Run these lines to save charts as high-resolution PNGs:

    import plotly.io as pio
    pio.write_image(fig_pair,      'outputs/charts/pair_deep_dive.png',      width=1400, height=900)
    pio.write_image(fig_portfolio, 'outputs/charts/portfolio_dashboard.png', width=1400, height=1000)
    pio.write_image(fig_hl,        'outputs/charts/half_life_analysis.png',  width=900,  height=450)
    pio.write_image(fig_stability, 'outputs/charts/coint_stability.png',     width=900,  height=450)
    pio.write_image(fig_kf,        'outputs/charts/kalman_vs_ols.png',       width=900,  height=550)

  📧 IN YOUR COVER LETTER OR EMAIL:
  "I built a statistical pairs trading engine that identifies cointegrated
   stock pairs using Engle-Granger and Johansen tests, constructs mean-
   reverting spreads via the Ornstein-Uhlenbeck model, and backtests
   a long-short strategy with realistic transaction costs. The portfolio
   achieved a Sharpe of X.XX with near-zero market beta. GitHub: [link]"
""")

print("═" * 70)
print("  ✅ PROJECT COMPLETE — Statistical Pairs Trading Engine")
print(f"     Pairs discovered   : {len(valid_pairs)}")
print(f"     Pairs backtested   : {len(all_pair_results)}")
print(f"     Portfolio Sharpe   : {portfolio_results.get('sharpe', 0):.3f}")
print(f"     Market Beta        : {portfolio_results.get('beta', 0):.4f}")
print("═" * 70)


 