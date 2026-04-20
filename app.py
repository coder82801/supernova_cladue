"""
NextDay Scanner Pro - Matematiksel Model
=========================================
Ertesi gun momentum devami adaylarini tamamen matematiksel bir
sinyal modeliyle bulur. Hicbir sezgi yok; her skor aciklanabilir
bir formulden cikar.

- Veri: Alpaca Market Data API
- Arayuz: Streamlit
- Backtest dahil (expectancy, win-rate, Kelly, max DD)
- Emir gondermez; sadece sinyal/oneri verir.

Calistirma:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import math
from datetime import datetime, timedelta, date, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Alpaca Market Data (alpaca-py)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_OK = True
except ImportError:
    ALPACA_OK = False


# ============================================================
# SAYFA
# ============================================================
st.set_page_config(page_title="NextDay Scanner — Matematiksel", layout="wide")
st.title("NextDay Scanner Pro — Matematiksel Model")
st.caption("Duyguya yer yok. Her sinyal skoru acik formullerden hesaplanir.")

if not ALPACA_OK:
    st.error("`alpaca-py` paketi kurulu degil. Kurulum: `pip install alpaca-py`")
    st.stop()


# ============================================================
# SIDEBAR — API ve MODEL PARAMETRELERI
# ============================================================
with st.sidebar:
    st.header("Alpaca API")
    api_key = st.text_input(
        "API Key ID", value=os.getenv("ALPACA_API_KEY", ""), type="password"
    )
    secret_key = st.text_input(
        "Secret Key", value=os.getenv("ALPACA_SECRET_KEY", ""), type="password"
    )

    st.divider()
    st.header("Sert Filtreler (Gecilmesi Zorunlu)")
    MIN_RVOL = st.number_input("Min RVOL (20g)", 1.0, 10.0, 1.5, 0.1)
    MIN_CS = st.number_input("Min Kapanis Gucu", 0.5, 1.0, 0.70, 0.05)
    MAX_DIST = st.number_input("Max Kirilim Uzakligi (%)", 0.5, 10.0, 3.0, 0.5) / 100
    MIN_PX = st.number_input("Min Fiyat ($)", 0.5, 100.0, 2.0, 0.5)
    MAX_PX = st.number_input("Max Fiyat ($)", 5.0, 500.0, 50.0, 5.0)
    MIN_VOL = st.number_input("Min Gunluk Hacim", 100_000, 20_000_000, 500_000, 100_000)
    REQ_VOLWAVG = st.checkbox("20g Hacim Agirlikli Ortalama Ustu Kapanis Zorunlu", True)
    REQ_POS_RS = st.checkbox("Pozitif RS vs SPY Zorunlu", True)
    REQ_POS_OBV = st.checkbox("Pozitif OBV Egimi Zorunlu", True)

    st.divider()
    st.header("Skor Esigi")
    MIN_SCORE = st.slider("Onerilecek min skor (0-100)", 40, 90, 60, 1)

    st.divider()
    st.header("Trade Seviyeleri")
    BREAKOUT_NEAR_PCT = st.slider("Kirilima yakin sayilacak mesafe (%)", 0.2, 3.0, 1.0, 0.1) / 100
    ENTRY_BUFFER_PCT = st.slider("Entry buffer (%)", 0.05, 1.0, 0.2, 0.05) / 100
    ATR_STOP_MULT = st.slider("ATR14 stop carpan", 0.5, 3.0, 1.2, 0.1)
    MAX_STOP_PCT = st.slider("Maksimum stop mesafesi (%)", 4.0, 20.0, 12.0, 0.5) / 100
    FALLBACK_STOP_PCT = st.slider("ATR yoksa fallback stop (%)", 2.0, 15.0, 8.0, 0.5) / 100
    TP1_R = st.slider("TP1 (R)", 0.5, 3.0, 1.5, 0.1)
    TP2_R = st.slider("TP2 (R)", 1.0, 6.0, 3.0, 0.1)

    st.divider()
    st.header("Sermaye / Risk")
    ACCOUNT = st.number_input("Hesap buyuklugu ($)", 100.0, 1_000_000.0, 2000.0, 100.0)
    RISK_PCT = st.number_input("Trade basina risk (%)", 0.5, 10.0, 2.0, 0.5) / 100
    KELLY_FRAC = st.slider("Kelly kesri (%)", 10, 100, 25, 5) / 100


# ============================================================
# ALPACA DATA CLIENT
# ============================================================
@st.cache_resource
def get_data_client(key: str, secret: str):
    if not key or not secret:
        return None
    return StockHistoricalDataClient(key, secret)


client = get_data_client(api_key, secret_key)
if client is None:
    st.warning("Sol panelden Alpaca API anahtarlarini gir.")
    st.stop()


# ============================================================
# MATEMATIKSEL GOSTERGELER
# ============================================================
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period).mean()


def closing_strength(close: float, low: float, high: float) -> float:
    rng = high - low
    if rng <= 0 or pd.isna(rng):
        return np.nan
    return (close - low) / rng


def obv_series(df: pd.DataFrame) -> pd.Series:
    diff = df["close"].diff().fillna(0)
    v = np.where(diff > 0, df["volume"], np.where(diff < 0, -df["volume"], 0))
    return pd.Series(v, index=df.index).cumsum()


def vol_weighted_price_from_bars(bars: pd.DataFrame) -> float:
    """
    Gercek intraday session VWAP DEGILDIR.
    Daily barlardan son 20 gun icin hacim agirlikli tipik fiyat ortalamasi:
        Sigma((H+L+C)/3 * V) / Sigma(V)
    """
    if bars.empty or bars["volume"].sum() == 0:
        return np.nan
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    return float((tp * bars["volume"]).sum() / bars["volume"].sum())


# ============================================================
# SINYAL MOTORU — TAMAMEN MATEMATIKSEL
# ============================================================
# Agirliklar (toplam = 1.00). Degistirmek icin sadece bu sabitleri oyna.
WEIGHTS = {
    "rvol":        0.25,
    "close_str":   0.20,
    "breakout":    0.15,
    "volwavg":     0.10,
    "obv":         0.10,
    "atr_expand":  0.10,
    "rel_str":     0.10,
}


def _clip(x, a=0.0, b=1.0):
    if pd.isna(x):
        return 0.0
    return max(a, min(b, float(x)))


def compute_features(df: pd.DataFrame, spy_df: pd.DataFrame) -> Optional[dict]:
    """
    Bir hissenin son gun bar'i icin tum ozellikleri hesaplar.
    `df` en az 220 bar icermeli. `spy_df` ayni periyoda ait SPY barlari.
    """
    if df is None or df.empty or len(df) < 60:
        return None

    df = df.copy().sort_index()
    last = df.iloc[-1]

    last_close = float(last["close"])
    last_open = float(last["open"])
    last_high = float(last["high"])
    last_low = float(last["low"])
    last_vol = float(last["volume"])

    # 1) RVOL (20 gun ort. hacme kiyasla)
    avg_vol_20 = df["volume"].rolling(20).mean().iloc[-1]
    rvol = last_vol / avg_vol_20 if avg_vol_20 and avg_vol_20 > 0 else np.nan

    # 2) Kapanis gucu
    cs = closing_strength(last_close, last_low, last_high)

    # 3) Kirilim uzakligi (bugunun kapanisina gore)
    prior_20d_high = df["high"].shift(1).rolling(20).max().iloc[-1]
    if pd.notna(prior_20d_high) and last_close < prior_20d_high:
        breakout_dist = (prior_20d_high - last_close) / last_close
    else:
        breakout_dist = 0.0

    # 4) 20g hacim agirlikli ortalama (gercek intraday VWAP DEGILDIR)
    recent_20 = df.tail(20)
    vol_wavg_20d = vol_weighted_price_from_bars(recent_20)
    above_volwavg = last_close > vol_wavg_20d if pd.notna(vol_wavg_20d) else False

    # 5) OBV egimi (son 10 gun)
    obv = obv_series(df)
    obv_slope_10 = float(obv.iloc[-1] - obv.iloc[-10]) if len(obv) >= 10 else 0.0

    # 6) ATR genislemesi (ATR5 / ATR20)
    atr5 = atr(df, 5).iloc[-1]
    atr14 = atr(df, 14).iloc[-1]
    atr20 = atr(df, 20).iloc[-1]
    atr_ratio = float(atr5 / atr20) if pd.notna(atr5) and pd.notna(atr20) and atr20 > 0 else np.nan

    # 7) Gorece guc vs SPY (10 gunluk getiri farki)
    # rs_valid: SPY verisi yoksa RS hesaplanamadi demektir; negatif sayilmamali.
    rs_valid = False
    if len(df) >= 11 and spy_df is not None and not spy_df.empty and len(spy_df) >= 11:
        stock_ret = (df["close"].iloc[-1] - df["close"].iloc[-11]) / df["close"].iloc[-11]
        spy_ret = (spy_df["close"].iloc[-1] - spy_df["close"].iloc[-11]) / spy_df["close"].iloc[-11]
        rs_10d = stock_ret - spy_ret
        rs_valid = True
    else:
        rs_10d = np.nan

    # 8) Gap (referans)
    prev_close = df["close"].iloc[-2] if len(df) >= 2 else last_close
    gap_pct = ((last_open - prev_close) / prev_close) if prev_close > 0 else 0.0

    # SMA50 / SMA200 (referans)
    sma50 = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else np.nan
    sma200 = df["close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else np.nan

    return {
        "close": last_close,
        "open": last_open,
        "high": last_high,
        "low": last_low,
        "volume": last_vol,
        "prev_close": float(prev_close),
        "rvol": float(rvol) if pd.notna(rvol) else np.nan,
        "close_strength": float(cs) if pd.notna(cs) else np.nan,
        "prior_20d_high": float(prior_20d_high) if pd.notna(prior_20d_high) else np.nan,
        "breakout_dist": float(breakout_dist),
        "vol_wavg_20d": float(vol_wavg_20d) if pd.notna(vol_wavg_20d) else np.nan,
        "above_volwavg": bool(above_volwavg),
        "obv_slope_10": float(obv_slope_10),
        "atr5": float(atr5) if pd.notna(atr5) else np.nan,
        "atr14": float(atr14) if pd.notna(atr14) else np.nan,
        "atr20": float(atr20) if pd.notna(atr20) else np.nan,
        "atr_ratio": float(atr_ratio) if pd.notna(atr_ratio) else np.nan,
        "rs_10d": float(rs_10d) if pd.notna(rs_10d) else np.nan,
        "rs_valid": rs_valid,
        "gap_pct": float(gap_pct),
        "sma50": float(sma50) if pd.notna(sma50) else np.nan,
        "sma200": float(sma200) if pd.notna(sma200) else np.nan,
    }


def signal_score(f: dict) -> tuple[float, dict]:
    """
    0-100 arasi skor ve alt-skor detaylari dondurur.
    Skor = Sum( w_i * s_i ) * 100
    """
    # Alt-skorlar: hepsi [0,1]'e normalize
    s_rvol       = _clip((f["rvol"] - 1.0) / 4.0)                 # 1.0->0, 5.0->1
    s_close      = _clip(f["close_strength"])                      # 0..1
    s_breakout   = _clip(1.0 - (f["breakout_dist"] / 0.05))        # 0% -> 1, 5% -> 0
    s_volwavg    = 1.0 if f["above_volwavg"] else 0.0
    s_obv        = 1.0 if f["obv_slope_10"] > 0 else 0.0
    s_atr        = _clip((f["atr_ratio"] - 0.80) / 0.40) if pd.notna(f["atr_ratio"]) else 0.0
    s_rel        = _clip((f["rs_10d"] + 0.05) / 0.15) if pd.notna(f["rs_10d"]) else 0.0  # -5%->0, +10%->1

    components = {
        "rvol":       s_rvol,
        "close_str":  s_close,
        "breakout":   s_breakout,
        "volwavg":    s_volwavg,
        "obv":        s_obv,
        "atr_expand": s_atr,
        "rel_str":    s_rel,
    }
    score01 = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return round(100.0 * score01, 2), components


def passes_hard_filters(f: dict) -> tuple[bool, str]:
    """Sert filtreler. Gecilmeyen hissede islem YOK."""
    if f["close"] < MIN_PX or f["close"] > MAX_PX:
        return False, f"Fiyat disi ({f['close']:.2f})"
    if f["volume"] < MIN_VOL:
        return False, "Hacim dusuk"
    if pd.isna(f["rvol"]) or f["rvol"] < MIN_RVOL:
        return False, f"RVOL < {MIN_RVOL}"
    if pd.isna(f["close_strength"]) or f["close_strength"] < MIN_CS:
        return False, f"Kapanis gucu < {MIN_CS}"
    if f["breakout_dist"] > MAX_DIST:
        return False, f"Kirilim uzakligi > {MAX_DIST*100:.1f}%"
    if REQ_VOLWAVG and not f["above_volwavg"]:
        return False, "20g hacim agirlikli ortalama altinda"
    if REQ_POS_RS:
        # rs_valid=False ise SPY verisi hic yok demektir; "negatif" sayma.
        if not f.get("rs_valid", False):
            return False, "RS verisi yok (SPY eksik)"
        if f["rs_10d"] <= 0:
            return False, "RS negatif"
    if REQ_POS_OBV and f["obv_slope_10"] <= 0:
        return False, "OBV negatif"
    return True, "OK"


def compute_trade_levels(f: dict) -> dict:
    """
    Entry/Stop/TP1/TP2 — tamamen formuller, sezgi yok.
    Tum esikler sidebar'dan parametre olarak gelir.
    """
    prior_high = f["prior_20d_high"]
    close = f["close"]
    breakout_mode = pd.notna(prior_high) and f["breakout_dist"] <= BREAKOUT_NEAR_PCT

    # Entry: kirilima yakinsa prior high uzeri buffer; degilse kapanis uzeri buffer
    if breakout_mode:
        entry = max(close, prior_high * (1 + ENTRY_BUFFER_PCT))
        entry_mode = "breakout_confirm"
    else:
        entry = close * (1 + ENTRY_BUFFER_PCT)
        entry_mode = "continuation"

    # Stop: ATR tabanli, max-stop cap ile ust sinir (en yukseklerden biri)
    atr14 = f.get("atr14", np.nan)
    if pd.notna(atr14) and atr14 > 0:
        atr_stop = entry - ATR_STOP_MULT * atr14
        max_stop_floor = entry * (1 - MAX_STOP_PCT)
        stop = max(atr_stop, max_stop_floor)
    else:
        stop = entry * (1 - FALLBACK_STOP_PCT)

    if stop >= entry or stop <= 0:
        stop = entry * (1 - FALLBACK_STOP_PCT)

    risk = max(entry - stop, 0.01)
    tp1 = entry + TP1_R * risk
    tp2 = entry + TP2_R * risk

    return {
        "entry": round(entry, 4),
        "stop": round(stop, 4),
        "tp1": round(tp1, 4),
        "tp2": round(tp2, 4),
        "risk_per_share": round(risk, 4),
        "entry_mode": entry_mode,
        "stop_pct": round((risk / entry) * 100, 2) if entry > 0 else np.nan,
        "rr_tp1": round(TP1_R, 2),
        "rr_tp2": round(TP2_R, 2),
    }


def position_size(account: float, risk_pct: float, entry: float, stop: float,
                  kelly_f: float = 1.0) -> dict:
    """Pozisyon adedi. kelly_f=1.0 ise risk_pct dogrudan uygulanir."""
    if entry <= 0 or stop <= 0 or entry <= stop:
        return {"shares": 0, "dollar_size": 0.0, "risk_dollars": 0.0}
    risk_per_share = entry - stop
    max_risk = account * risk_pct * kelly_f
    shares = max(0, math.floor(max_risk / risk_per_share))
    return {
        "shares": shares,
        "dollar_size": round(shares * entry, 2),
        "risk_dollars": round(shares * risk_per_share, 2),
    }


# ============================================================
# ALPACA VERI INDIRME
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_daily_bars(_client, symbol: str, days: int = 260) -> pd.DataFrame:
    try:
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=int(days * 1.6) + 10)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all",
            feed="iex",
        )
        bars = _client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        return df.tail(days)
    except Exception as e:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_daily_bars_batch(_client, symbols: list[str], days: int = 260) -> dict[str, pd.DataFrame]:
    out = {}
    if not symbols:
        return out
    try:
        end = datetime.now(ZoneInfo("America/New_York"))
        start = end - timedelta(days=int(days * 1.6) + 10)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment="all",
            feed="iex",
        )
        bars = _client.get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            return out
        if isinstance(df.index, pd.MultiIndex):
            for sym in df.index.get_level_values(0).unique():
                sub = df.loc[sym][["open", "high", "low", "close", "volume"]].copy()
                if sub.index.tz is not None:
                    sub.index = sub.index.tz_convert(None)
                sub.index = pd.to_datetime(sub.index).normalize()
                out[sym] = sub.tail(days)
        else:
            sub = df[["open", "high", "low", "close", "volume"]].copy()
            if sub.index.tz is not None:
                sub.index = sub.index.tz_convert(None)
            sub.index = pd.to_datetime(sub.index).normalize()
            out[symbols[0]] = sub.tail(days)
    except Exception as e:
        st.error(f"Alpaca batch hatasi: {e}")
    return out


def precompute_feature_series(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tum gunler icin tum gostergeleri VEKTORIZE sekilde hesaplar.
    compute_features'un hizli, backtest dostu versiyonu.
    Cikti: her gun icin feature satiri iceren DataFrame.
    """
    if df is None or df.empty or len(df) < 60:
        return pd.DataFrame()

    out = df.copy().sort_index()

    # Rolling hacimler, ATR, OBV, SMA --- hepsi vektorize
    out["avg_vol_20"] = out["volume"].rolling(20).mean()
    out["rvol"] = out["volume"] / out["avg_vol_20"]

    rng = out["high"] - out["low"]
    out["close_strength"] = np.where(rng > 0, (out["close"] - out["low"]) / rng, np.nan)

    out["prior_20d_high"] = out["high"].shift(1).rolling(20).max()
    out["breakout_dist"] = np.where(
        out["prior_20d_high"].notna() & (out["close"] < out["prior_20d_high"]),
        (out["prior_20d_high"] - out["close"]) / out["close"],
        0.0,
    )

    # 20g hacim agirlikli ortalama (typical*volume / volume) -- gercek intraday VWAP DEGIL
    tp = (out["high"] + out["low"] + out["close"]) / 3
    vp = tp * out["volume"]
    vol20 = out["volume"].rolling(20).sum()
    vp20 = vp.rolling(20).sum()
    out["vol_wavg_20d"] = np.where(vol20 > 0, vp20 / vol20, np.nan)
    out["above_volwavg"] = out["close"] > out["vol_wavg_20d"]

    # OBV
    diff = out["close"].diff().fillna(0)
    obv_step = np.where(diff > 0, out["volume"], np.where(diff < 0, -out["volume"], 0))
    out["obv"] = pd.Series(obv_step, index=out.index).cumsum()
    out["obv_slope_10"] = out["obv"] - out["obv"].shift(10)

    # ATR5/14/20
    out["tr"] = true_range(out)
    out["atr5"] = out["tr"].rolling(5).mean()
    out["atr14"] = out["tr"].rolling(14).mean()
    out["atr20"] = out["tr"].rolling(20).mean()
    out["atr_ratio"] = np.where(out["atr20"] > 0, out["atr5"] / out["atr20"], np.nan)

    # SMA
    out["sma50"] = out["close"].rolling(50).mean()
    out["sma200"] = out["close"].rolling(200).mean()

    # Gap
    out["prev_close"] = out["close"].shift(1)
    out["gap_pct"] = np.where(out["prev_close"] > 0, (out["open"] - out["prev_close"]) / out["prev_close"], 0.0)

    # Relative Strength vs SPY -- rs_valid True sadece SPY varsa
    if spy_df is not None and not spy_df.empty:
        spy_aligned = spy_df["close"].reindex(out.index).ffill()
        stock_ret_10 = (out["close"] - out["close"].shift(10)) / out["close"].shift(10)
        spy_ret_10 = (spy_aligned - spy_aligned.shift(10)) / spy_aligned.shift(10)
        out["rs_10d"] = stock_ret_10 - spy_ret_10
        out["rs_valid"] = spy_aligned.notna()
    else:
        out["rs_10d"] = np.nan
        out["rs_valid"] = False

    # Sadece feature kolonlarini dondur
    feat_cols = [
        "open", "high", "low", "close", "volume",
        "rvol", "close_strength", "prior_20d_high", "breakout_dist",
        "vol_wavg_20d", "above_volwavg", "obv_slope_10",
        "atr5", "atr14", "atr20", "atr_ratio",
        "sma50", "sma200", "prev_close", "gap_pct", "rs_10d", "rs_valid",
    ]
    return out[feat_cols].dropna(subset=["rvol", "close_strength", "atr14"])


def passes_filters_row(row) -> bool:
    """Vektorize kullanima uygun, tek satir filtreleme."""
    try:
        if row["close"] < MIN_PX or row["close"] > MAX_PX:
            return False
        if row["volume"] < MIN_VOL:
            return False
        if pd.isna(row["rvol"]) or row["rvol"] < MIN_RVOL:
            return False
        if pd.isna(row["close_strength"]) or row["close_strength"] < MIN_CS:
            return False
        if row["breakout_dist"] > MAX_DIST:
            return False
        if REQ_VOLWAVG and not bool(row["above_volwavg"]):
            return False
        if REQ_POS_RS:
            if not bool(row.get("rs_valid", False)):
                return False
            if pd.isna(row["rs_10d"]) or row["rs_10d"] <= 0:
                return False
        if REQ_POS_OBV and row["obv_slope_10"] <= 0:
            return False
        return True
    except Exception:
        return False


def score_row(row) -> float:
    """Vektorize kullanima uygun tek-satir skor."""
    s_rvol = _clip((row["rvol"] - 1.0) / 4.0)
    s_close = _clip(row["close_strength"])
    s_breakout = _clip(1.0 - (row["breakout_dist"] / 0.05))
    s_volwavg = 1.0 if bool(row["above_volwavg"]) else 0.0
    s_obv = 1.0 if row["obv_slope_10"] > 0 else 0.0
    s_atr = _clip((row["atr_ratio"] - 0.80) / 0.40) if pd.notna(row["atr_ratio"]) else 0.0
    s_rel = _clip((row["rs_10d"] + 0.05) / 0.15) if pd.notna(row["rs_10d"]) else 0.0
    val = (WEIGHTS["rvol"]*s_rvol + WEIGHTS["close_str"]*s_close +
           WEIGHTS["breakout"]*s_breakout + WEIGHTS["volwavg"]*s_volwavg +
           WEIGHTS["obv"]*s_obv + WEIGHTS["atr_expand"]*s_atr +
           WEIGHTS["rel_str"]*s_rel)
    return round(100.0 * val, 2)


# ============================================================
# PARABOLIK RUNNER TESPITI — MATEMATIKSEL GOSTERGELER
# ============================================================
# Hedef: ertesi gun >%20-30 INTRADAY HIGH yapma potansiyeli olan
# hisseleri, uydurma veri olmadan, sadece fiyat/hacim anomalilerinden
# tespit etmek. Yani "patlamadan bir gun onceki karakteristik".
#
# Kullanilan matematiksel olgular:
#   1) Extreme RVOL (asiri hacim gelisi; 3x ve ustu)
#   2) Bollinger Band squeeze (volatilite sikismasi, patlama habercisi)
#   3) Narrow Range N (NR4 / NR7: son N gunun en dar araliginda kapanis)
#   4) 52-hafta yuksek yakinligi veya yeni 52w high
#   5) Accumulation: OBV yukseliyor, fiyat yatay (gizli birikim)
#   6) Volume dry-up then pop: son haftalar sessiz -> bugun patlama
#   7) ATR expansion: ATR5 / ATR20 > esik (volatilite aciliyor)
#   8) Small-cap bonus: fiyat dusukse parabolic olasiligi daha yuksek
#
# Tum esikler parametreleri ASAGIDA sabit; ayar icin kodu oynatabilirsin.
# Hic birinde "fundamentals" veya "news" yok -- sadece OHLCV.


def parabolic_features(fs: pd.DataFrame) -> pd.DataFrame:
    """
    Vektorize: precompute_feature_series'in uzerine parabolik gostergeler ekler.
    Girdi: fs (precompute_feature_series sonucu; OHLCV + temel gostergeler)
    Cikti: ayni indeks + yeni kolonlar.
    """
    if fs is None or fs.empty:
        return fs

    out = fs.copy()

    # --- Bollinger Band genisligi ve squeeze yuzdesi ---
    bb_mid = out["close"].rolling(20).mean()
    bb_std = out["close"].rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    out["bb_width"] = bb_width

    # Squeeze yuzdesi: son 120 gunluk BB genisligine gore siralamada kacinci %'de
    # 0.0 = en sikismis (en dar); 1.0 = en genis
    # Standart percentile rank: kac gunun BB genisliginden kucuk veya esitim.
    def _pct_rank(x: pd.Series) -> pd.Series:
        return x.rolling(120).apply(
            lambda w: (w < w.iloc[-1]).mean() if len(w) > 0 else np.nan,
            raw=False,
        )
    out["bb_squeeze_pctile"] = _pct_rank(bb_width)

    # --- NR4 / NR7 flag ---
    daily_range = out["high"] - out["low"]
    min_range_6 = daily_range.shift(1).rolling(6).min()
    min_range_3 = daily_range.shift(1).rolling(3).min()
    out["nr7"] = (daily_range <= min_range_6).fillna(False).astype(bool)
    out["nr4"] = (daily_range <= min_range_3).fillna(False).astype(bool)

    # --- 52-hafta yuksek yakinligi / yeni high ---
    high_252 = out["high"].rolling(252, min_periods=60).max()
    out["high_252"] = high_252
    out["dist_52w_high"] = (high_252 - out["close"]) / out["close"]
    out["new_52w_high"] = (out["close"] >= high_252 * 0.999).fillna(False).astype(bool)

    # --- Accumulation: OBV yukseliyor + fiyat yatay ---
    if "obv_slope_10" in out.columns:
        # 20-gunluk normalize OBV egimi yaklasigi
        obv_diff = out["obv_slope_10"]
    else:
        obv_diff = pd.Series(0.0, index=out.index)
    px_chg_20 = (out["close"] - out["close"].shift(20)) / out["close"].shift(20)
    out["price_flat_20"] = (px_chg_20.abs() < 0.08)
    out["accumulation"] = (obv_diff > 0) & out["price_flat_20"]

    # --- Volume dry-up then pop ---
    avg_vol_20 = out["volume"].rolling(20).mean()
    avg_vol_60 = out["volume"].rolling(60).mean()
    # son 20 gun ortalamasi uzun donem ortalamanin altindaysa = dry
    dry_ratio = avg_vol_20 / avg_vol_60
    out["volume_dry_ratio"] = dry_ratio
    # bugun rvol >= 3 + son 20g dry (ratio < 1.0)
    out["dry_then_pop"] = (
        (out["rvol"] >= 3.0) & (dry_ratio < 1.0)
    ).fillna(False).astype(bool)

    # --- Small-cap proxy (fiyat dusukse penny/small-cap) ---
    # fiyat < 5: guclu small-cap; < 10: orta; >= 10: kucuk bonus
    out["small_cap_score"] = np.where(
        out["close"] < 5.0, 1.0,
        np.where(out["close"] < 10.0, 0.6,
                 np.where(out["close"] < 25.0, 0.3, 0.0)),
    )

    return out


PAR_WEIGHTS = {
    "extreme_rvol":   0.25,  # en guclu tekil gosterge
    "bb_squeeze":     0.15,  # sikisma
    "accumulation":   0.15,  # gizli birikim
    "high_proximity": 0.10,  # 52w yakinlik / breakout
    "nr_compression": 0.10,  # dar menzil (explosion setup)
    "close_strength": 0.10,  # gunluk kuvvetli kapanis
    "atr_expansion":  0.05,  # ATR5/ATR20 genisleme
    "small_cap":      0.05,  # dusuk fiyat bonusu
    "dry_then_pop":   0.05,  # hacim sessizligi sonrasi patlama
}


def parabolic_score_row(row) -> tuple[float, dict]:
    """
    0-100 arasi parabolik skor.
    Tum alt-bilesenler [0,1]'e normalize, agirlikli toplanir.
    """
    # 1) Extreme RVOL: 3x -> 0.5, 5x -> 1.0, 1x -> 0
    rvol = row.get("rvol", np.nan)
    s_rvol = _clip((rvol - 1.0) / 4.0) if pd.notna(rvol) else 0.0

    # 2) BB squeeze: pctile 0.0 (en sikisik) -> 1.0; 0.5 -> 0.0
    sq = row.get("bb_squeeze_pctile", np.nan)
    s_bb = _clip(1.0 - (sq / 0.5)) if pd.notna(sq) else 0.0

    # 3) Accumulation flag
    s_acc = 1.0 if bool(row.get("accumulation", False)) else 0.0

    # 4) 52w high proximity: 0% -> 1.0, 15% -> 0
    dist = row.get("dist_52w_high", np.nan)
    if pd.notna(dist):
        s_high = _clip(1.0 - (dist / 0.15))
        if bool(row.get("new_52w_high", False)):
            s_high = 1.0
    else:
        s_high = 0.0

    # 5) NR compression: NR4 > NR7 > yok
    if bool(row.get("nr4", False)):
        s_nr = 1.0
    elif bool(row.get("nr7", False)):
        s_nr = 0.6
    else:
        s_nr = 0.0

    # 6) Close strength
    cs = row.get("close_strength", np.nan)
    s_cs = _clip(cs) if pd.notna(cs) else 0.0

    # 7) ATR expansion: ATR5/ATR20 >= 1.5 -> 1.0; 1.0 -> 0
    ar = row.get("atr_ratio", np.nan)
    s_atr = _clip((ar - 1.0) / 0.5) if pd.notna(ar) else 0.0

    # 8) Small-cap
    s_sc = float(row.get("small_cap_score", 0.0))

    # 9) Dry-then-pop
    s_dp = 1.0 if bool(row.get("dry_then_pop", False)) else 0.0

    components = {
        "extreme_rvol":   s_rvol,
        "bb_squeeze":     s_bb,
        "accumulation":   s_acc,
        "high_proximity": s_high,
        "nr_compression": s_nr,
        "close_strength": s_cs,
        "atr_expansion":  s_atr,
        "small_cap":      s_sc,
        "dry_then_pop":   s_dp,
    }
    val = sum(PAR_WEIGHTS[k] * components[k] for k in PAR_WEIGHTS)
    return round(100.0 * val, 2), components


def parabolic_passes_filters(row, min_rvol: float, min_cs: float) -> tuple[bool, str]:
    """
    Parabolik adaylar icin sert filtreler. Mevcut sidebar fiyat/hacim
    filtrelerine EK olarak uygulanir.
    """
    if pd.isna(row.get("rvol", np.nan)) or row["rvol"] < min_rvol:
        return False, f"RVOL<{min_rvol}"
    if pd.isna(row.get("close_strength", np.nan)) or row["close_strength"] < min_cs:
        return False, f"CloseStr<{min_cs}"
    if not bool(row.get("above_volwavg", False)):
        return False, "VolWAvg alti"
    if row.get("obv_slope_10", 0) <= 0:
        return False, "OBV-"
    # En az bir "patlama hazirligi" sinyali olmali
    has_setup = (
        bool(row.get("accumulation", False))
        or bool(row.get("nr7", False))
        or bool(row.get("dry_then_pop", False))
        or (pd.notna(row.get("bb_squeeze_pctile", np.nan)) and row["bb_squeeze_pctile"] < 0.25)
        or (pd.notna(row.get("dist_52w_high", np.nan)) and row["dist_52w_high"] < 0.10)
    )
    if not has_setup:
        return False, "Setup yok"
    return True, "OK"


# ============================================================
# EVREN: TRADINGVIEW ILE HIZLI ON-FILTRE
# ============================================================
TV_URL = "https://scanner.tradingview.com/america/scan"


@st.cache_data(ttl=300, show_spinner=False)
def tv_universe(max_records: int = 500) -> list[str]:
    """
    Alpaca'da tradable, RVOL>1.5, fiyat araliginda hisseleri TV'den cek.
    Sadece SEMBOL listesi dondurur; OHLC'yi Alpaca'dan aliriz.
    """
    payload = {
        "filter": [
            {"left": "close", "operation": "in_range", "right": [MIN_PX, MAX_PX]},
            {"left": "volume", "operation": "greater", "right": MIN_VOL},
            {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
            {"left": "relative_volume_10d_calc", "operation": "greater", "right": max(1.0, MIN_RVOL - 0.3)},
        ],
        "options": {"lang": "en"},
        "markets": ["america"],
        "symbols": {"query": {"types": ["stock"]}, "tickers": []},
        "columns": ["name"],
        "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
        "range": [0, max_records],
    }
    try:
        r = requests.post(TV_URL, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
        syms = []
        for it in data:
            s = it["d"][0]
            if s and "." not in s and "-" not in s and s.isalpha():
                syms.append(s)
        return syms
    except Exception as e:
        st.error(f"TradingView evren hatasi: {e}")
        return []


# ============================================================
# SEKMELER
# ============================================================
tab1, tab2, tab4, tab3 = st.tabs([
    "Canli Tarama", "Backtest", "Patlayici Aday", "Istatistikler"
])


# ============================================================
# TAB 1 — CANLI TARAMA
# ============================================================
with tab1:
    st.subheader("Ertesi Gun Adaylari — Canli Tarama")
    st.write(
        "Algoritma: TradingView'dan RVOL>1.5 evreni cekilir, "
        "Alpaca Daily Bars ile matematiksel skor hesaplanir, "
        "skor >= esige ve sert filtrelere gecenler listelenir."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        universe_size = st.number_input("Evren buyuklugu", 50, 1500, 300, 50)
    with col2:
        batch_size = st.slider("Batch (Alpaca istegi basina sembol)", 50, 500, 200, 50)

    if st.button("Taramayi Baslat", type="primary"):
        with st.spinner("Evren indiriliyor (TradingView)..."):
            universe = tv_universe(max_records=universe_size)
            if "SPY" not in universe:
                universe.append("SPY")
        st.info(f"Evren boyutu: {len(universe)} sembol")

        if not universe:
            st.warning("Evren bos.")
        else:
            # Batch halinde daily bars indir
            all_bars: dict[str, pd.DataFrame] = {}
            progress = st.progress(0.0, text="Alpaca'dan daily bars indiriliyor...")
            total = len(universe)
            for i in range(0, total, batch_size):
                chunk = universe[i:i + batch_size]
                bars_map = fetch_daily_bars_batch(client, chunk, days=260)
                all_bars.update(bars_map)
                progress.progress(min(1.0, (i + batch_size) / total),
                                  text=f"{min(i+batch_size,total)}/{total}")
            progress.empty()

            spy_df = all_bars.get("SPY", pd.DataFrame())
            if spy_df.empty:
                st.warning("SPY verisi alinamadi. RS zorunlu aciksa tum adaylar 'RS verisi yok' sebebiyle elenir.")

            # Her sembol icin skor ve filtre
            candidates = []
            rejected = []
            for sym, df in all_bars.items():
                if sym == "SPY":
                    continue
                feats = compute_features(df, spy_df)
                if feats is None:
                    rejected.append({"symbol": sym, "reason": "veri yetersiz"})
                    continue
                passed, reason = passes_hard_filters(feats)
                score, comp = signal_score(feats)
                if not passed:
                    rejected.append({"symbol": sym, "reason": reason, "score": score})
                    continue
                if score < MIN_SCORE:
                    rejected.append({"symbol": sym, "reason": f"skor dusuk ({score})", "score": score})
                    continue
                levels = compute_trade_levels(feats)
                pos = position_size(ACCOUNT, RISK_PCT, levels["entry"], levels["stop"], KELLY_FRAC)

                candidates.append({
                    "Symbol": sym,
                    "Score": score,
                    "Close": round(feats["close"], 4),
                    "RVOL": round(feats["rvol"], 2) if pd.notna(feats["rvol"]) else None,
                    "Close_Str": round(feats["close_strength"], 2),
                    "Dist_High_%": round(feats["breakout_dist"] * 100, 2),
                    "Above_VolWAvg": feats["above_volwavg"],
                    "OBV+": feats["obv_slope_10"] > 0,
                    "RS_vs_SPY_%": round(feats["rs_10d"] * 100, 2) if pd.notna(feats["rs_10d"]) else None,
                    "Gap_%": round(feats["gap_pct"] * 100, 2),
                    "ATR14": round(feats["atr14"], 4) if pd.notna(feats["atr14"]) else None,
                    "EntryMode": levels["entry_mode"],
                    "Entry": levels["entry"],
                    "Stop": levels["stop"],
                    "Stop_%": levels["stop_pct"],
                    "TP1": levels["tp1"],
                    "TP2": levels["tp2"],
                    "RR_TP1": levels["rr_tp1"],
                    "RR_TP2": levels["rr_tp2"],
                    "Shares": pos["shares"],
                    "Risk_$": pos["risk_dollars"],
                    "Pos_$": pos["dollar_size"],
                })

            cands_df = pd.DataFrame(candidates).sort_values("Score", ascending=False) if candidates else pd.DataFrame()

            if cands_df.empty:
                st.warning("Filtreleri gecen aday yok.")
            else:
                st.success(f"{len(cands_df)} aday bulundu.")
                st.dataframe(cands_df, use_container_width=True, hide_index=True)
                csv = cands_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSV indir", csv,
                                   file_name=f"nextday_{datetime.now():%Y%m%d_%H%M}.csv",
                                   mime="text/csv")

            with st.expander(f"Reddedilenler ({len(rejected)})"):
                if rejected:
                    st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)


# ============================================================
# TAB 2 — BACKTEST
# ============================================================
with tab2:
    st.subheader("Backtest — Gecmis Veride Strateji Simulasyonu")
    st.write(
        "Algoritmayi gecmis N gunun her gunune uygular, "
        "ertesi gun limit dolumunu simule eder, "
        "expectancy / win-rate / max drawdown / Kelly hesaplar."
    )

    colA, colB, colC = st.columns(3)
    with colA:
        bt_days = st.number_input("Backtest gun sayisi", 30, 360, 120, 30)
    with colB:
        bt_exit = st.selectbox("Cikis modu",
                               ["Ertesi gun OPEN",
                                "Ertesi gun HIGH (en iyi durum)",
                                "Ertesi gun CLOSE",
                                "Stop veya TP vurursa, yoksa CLOSE"])
    with colC:
        bt_universe_size = st.number_input("Evren buyuklugu (sabit liste)", 50, 500, 150, 50)

    st.caption(
        "Not: Backtest icin sabit bir sembol listesi kullanilir "
        "(surviviorship bias uyarisi: delistelenen hisseler dahil degil)."
    )

    if st.button("Backtesti Baslat", type="primary"):
        try:
            status = st.empty()
            status.info("Asama 1/4: Evren cekiliyor (TradingView)...")
            symbols = tv_universe(max_records=bt_universe_size)
            if "SPY" not in symbols:
                symbols.append("SPY")

            st.info(f"{len(symbols)} sembol uzerinde backtest yapilacak.")

            # --- Alpaca'dan veri indir (kucuk batch'lerle) ---
            status.info("Asama 2/4: Alpaca'dan gunluk bar verisi indiriliyor...")
            bars_map: dict[str, pd.DataFrame] = {}
            step = 50  # Daha kucuk batch, hata izolasyonu icin
            progress = st.progress(0.0, text="0/0")
            total = len(symbols)
            for i in range(0, total, step):
                chunk = symbols[i:i + step]
                got = fetch_daily_bars_batch(client, chunk, days=bt_days + 260)
                bars_map.update(got)
                done = min(i + step, total)
                progress.progress(done / total, text=f"{done}/{total} sembol")
            progress.empty()

            if not bars_map:
                st.error("Alpaca'dan hic veri alinamadi. API anahtarlarini kontrol et.")
                st.stop()

            spy_df = bars_map.get("SPY", pd.DataFrame())
            if spy_df.empty:
                st.error("SPY verisi yok, backtest yapilamaz. Alpaca baglantini kontrol et.")
                st.stop()

            st.info(f"{len(bars_map)} sembol icin veri alindi.")

            # --- Asama 3: Tum sembollerin feature serisini VEKTORIZE hesapla ---
            status.info("Asama 3/4: Gostergeler vektorize hesaplaniyor...")
            feat_map: dict[str, pd.DataFrame] = {}
            prog3 = st.progress(0.0, text="0/0")
            syms_list = [s for s in bars_map.keys() if s != "SPY"]
            for i, sym in enumerate(syms_list):
                try:
                    fs = precompute_feature_series(bars_map[sym], spy_df)
                    if not fs.empty:
                        feat_map[sym] = fs
                except Exception:
                    pass
                if (i + 1) % 10 == 0 or i == len(syms_list) - 1:
                    prog3.progress((i + 1) / len(syms_list), text=f"{i+1}/{len(syms_list)}")
            prog3.empty()

            st.info(f"{len(feat_map)} sembolde gosterge hesaplandi.")

            # --- Asama 4: Tarihleri tara, sinyal uret, ertesi gun simulasyonu ---
            status.info("Asama 4/4: Tarihler taraniyor, trade'ler simule ediliyor...")
            test_dates = spy_df.index[-bt_days:]
            trades = []
            prog4 = st.progress(0.0, text=f"0/{len(test_dates)}")

            for di, dt in enumerate(test_dates):
                for sym, fs in feat_map.items():
                    # dt tarihinin feature satirini al
                    if dt not in fs.index:
                        continue
                    row = fs.loc[dt]

                    # Filtreler
                    if not passes_filters_row(row):
                        continue
                    score = score_row(row)
                    if score < MIN_SCORE:
                        continue

                    # Entry/Stop/TP hesabi (sidebar parametreleri)
                    prior_high = row["prior_20d_high"]
                    close = float(row["close"])
                    atr14 = float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
                    breakout_dist = float(row["breakout_dist"])

                    breakout_mode = pd.notna(prior_high) and breakout_dist <= BREAKOUT_NEAR_PCT
                    if breakout_mode:
                        entry = max(close, float(prior_high) * (1 + ENTRY_BUFFER_PCT))
                    else:
                        entry = close * (1 + ENTRY_BUFFER_PCT)

                    if pd.notna(atr14) and atr14 > 0:
                        atr_stop = entry - ATR_STOP_MULT * atr14
                        max_stop_floor = entry * (1 - MAX_STOP_PCT)
                        stop = max(atr_stop, max_stop_floor)
                    else:
                        stop = entry * (1 - FALLBACK_STOP_PCT)

                    if stop >= entry or stop <= 0:
                        stop = entry * (1 - FALLBACK_STOP_PCT)

                    entry = round(entry, 4)
                    stop = round(stop, 4)
                    risk = max(entry - stop, 0.01)
                    tp1 = round(entry + TP1_R * risk, 4)
                    tp2 = round(entry + TP2_R * risk, 4)

                    # Ertesi gun bar'ini bul
                    full_df = bars_map[sym]
                    next_bars = full_df.loc[full_df.index > dt]
                    if next_bars.empty:
                        continue
                    next_bar = next_bars.iloc[0]

                    filled = float(next_bar["low"]) <= entry
                    if not filled:
                        trades.append({
                            "date": dt.date(), "symbol": sym, "score": score,
                            "entry": entry, "exit": None, "stop": stop,
                            "tp1": tp1, "tp2": tp2, "filled": False,
                            "ret_pct": 0.0, "result": "NO_FILL",
                        })
                        continue

                    n_open = float(next_bar["open"])
                    n_high = float(next_bar["high"])
                    n_low = float(next_bar["low"])
                    n_close = float(next_bar["close"])

                    if bt_exit == "Ertesi gun OPEN":
                        exit_px = n_open; result = "OPEN"
                    elif bt_exit == "Ertesi gun HIGH (en iyi durum)":
                        exit_px = n_high; result = "HIGH"
                    elif bt_exit == "Ertesi gun CLOSE":
                        exit_px = n_close; result = "CLOSE"
                    else:
                        if n_low <= stop:
                            exit_px = stop; result = "STOP"
                        elif n_high >= tp2:
                            exit_px = tp2; result = "TP2"
                        elif n_high >= tp1:
                            exit_px = tp1; result = "TP1"
                        else:
                            exit_px = n_close; result = "CLOSE"

                    ret_pct = (exit_px - entry) / entry
                    trades.append({
                        "date": dt.date(), "symbol": sym, "score": score,
                        "entry": round(entry, 4), "exit": round(exit_px, 4),
                        "stop": round(stop, 4), "tp1": round(tp1, 4), "tp2": round(tp2, 4),
                        "filled": True, "ret_pct": round(ret_pct * 100, 3),
                        "result": result,
                    })

                if (di + 1) % 5 == 0 or di == len(test_dates) - 1:
                    prog4.progress((di + 1) / len(test_dates),
                                   text=f"{di+1}/{len(test_dates)} gun | {len(trades)} sinyal")

            prog4.empty()
            status.empty()
        except Exception as e:
            st.error(f"Backtest hatasi: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            st.warning("Hic islem sinyali uretilmedi.")
        else:
            filled_df = trades_df[trades_df["filled"]].copy()
            n_all = len(trades_df)
            n_fill = len(filled_df)
            st.success(f"Toplam sinyal: {n_all}, dolum sayisi: {n_fill}")

            if n_fill == 0:
                st.warning("Hic trade dolmamis (limit emirler tetiklenmemis).")
            else:
                wins = filled_df[filled_df["ret_pct"] > 0]
                losses = filled_df[filled_df["ret_pct"] <= 0]
                win_rate = len(wins) / n_fill
                avg_win = wins["ret_pct"].mean() if len(wins) else 0.0
                avg_loss = losses["ret_pct"].mean() if len(losses) else 0.0
                expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
                total_pnl = filled_df["ret_pct"].sum()

                # Kelly
                if avg_loss < 0 and abs(avg_loss) > 0:
                    b = abs(avg_win / avg_loss)
                    p = win_rate
                    q = 1 - p
                    kelly_raw = (b * p - q) / b if b > 0 else 0
                    kelly_raw = max(0, kelly_raw)
                else:
                    kelly_raw = 0

                # Max drawdown (equity egrisi)
                filled_df = filled_df.sort_values("date").reset_index(drop=True)
                equity = (1 + filled_df["ret_pct"] / 100).cumprod()
                peak = equity.cummax()
                drawdown = (equity - peak) / peak
                max_dd = drawdown.min() * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Win-rate", f"{win_rate*100:.1f}%")
                c2.metric("Ort. Kazanan", f"+{avg_win:.2f}%")
                c3.metric("Ort. Kaybeden", f"{avg_loss:.2f}%")
                c4.metric("Expectancy / trade", f"{expectancy:+.2f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Toplam trade", n_fill)
                c6.metric("Kumulatif getiri", f"{(equity.iloc[-1]-1)*100:+.1f}%")
                c7.metric("Max Drawdown", f"{max_dd:.1f}%")
                c8.metric("Kelly (tam)", f"{kelly_raw*100:.1f}%")

                st.info(
                    f"Onerilen pozisyon boyutu: Kelly x {int(KELLY_FRAC*100)}% "
                    f"= sermayenin **%{kelly_raw*KELLY_FRAC*100:.1f}**'i her trade basina."
                )

                # Equity grafik
                eq_df = pd.DataFrame({
                    "date": filled_df["date"],
                    "equity": equity,
                    "drawdown_%": drawdown * 100,
                })
                st.line_chart(eq_df.set_index("date")[["equity"]])
                st.area_chart(eq_df.set_index("date")[["drawdown_%"]])

                st.subheader("Trade log")
                st.dataframe(filled_df.sort_values("date", ascending=False),
                             use_container_width=True, hide_index=True)

                csv2 = filled_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Trade log indir", csv2,
                                   file_name=f"backtest_{datetime.now():%Y%m%d_%H%M}.csv")


# ============================================================
# TAB 4 — PATLAYICI ADAY (Parabolic Runner Detector)
# ============================================================
with tab4:
    st.subheader("Patlayici Aday — Ertesi Gun Yuksek Potansiyel Hareket")
    st.caption(
        "Amac: ertesi gun INTRADAY HIGH olarak %20+ hareket yapma olasiligi "
        "yuksek adaylari tespit etmek. Haber / fundamentals / float YOKTUR; "
        "sadece fiyat-hacim anomalisi, volatilite sikismasi ve birikim "
        "desenleri kullanilir."
    )

    with st.expander("Matematiksel model nasil calisiyor?", expanded=False):
        st.markdown(
            "- **Extreme RVOL**: bugunun hacmi / 20 gun ort. >= 3x.\n"
            "- **Bollinger Band squeeze**: BB genisligi son 120 gunun en dar %25'inde ise sikismis.\n"
            "- **NR4 / NR7**: son 4 / 7 gunun en dar menzili -> volatilite patlamasi habercisi.\n"
            "- **52-hafta yuksek yakinligi**: breakout adayi.\n"
            "- **Accumulation**: OBV yukseliyor + fiyat son 20g +-%8 icinde yatay (gizli birikim).\n"
            "- **Dry-then-pop**: son 20 gun hacmi uzun donemin altinda + bugun RVOL>=3x.\n"
            "- **ATR expansion**: ATR5/ATR20 > 1.0 -> volatilite acilmaya basladi.\n"
            "- **Small-cap bonus**: dusuk fiyat hisselerde parabolic olasiligi istatistiksel olarak yuksek.\n\n"
            "Her bilesen [0,1]'e normalize, agirlikli toplanir. Skor 0-100."
        )

    colP1, colP2, colP3 = st.columns(3)
    with colP1:
        par_universe_size = st.number_input("Evren buyuklugu", 100, 1500, 500, 100, key="par_uni")
    with colP2:
        par_min_rvol = st.slider("Min RVOL (parabolik)", 2.0, 10.0, 3.0, 0.5, key="par_rvol")
    with colP3:
        par_min_score = st.slider("Min parabolik skor", 30, 90, 55, 1, key="par_score")

    par_min_cs = st.slider("Min kapanis gucu", 0.5, 1.0, 0.75, 0.05, key="par_cs")

    if st.button("Patlayici Aday Taramasi", type="primary", key="btn_par_scan"):
        try:
            status = st.empty()
            status.info("Asama 1/3: TV evren cekiliyor...")
            # Evreni daha genis al: TV ozel bir RVOL>=2 esigi ile
            payload = {
                "filter": [
                    {"left": "close", "operation": "in_range", "right": [MIN_PX, MAX_PX]},
                    {"left": "volume", "operation": "greater", "right": MIN_VOL},
                    {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                    {"left": "relative_volume_10d_calc", "operation": "greater",
                     "right": max(1.5, par_min_rvol - 0.5)},
                ],
                "options": {"lang": "en"},
                "markets": ["america"],
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name"],
                "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
                "range": [0, int(par_universe_size)],
            }
            try:
                r = requests.post(TV_URL, json=payload,
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                r.raise_for_status()
                data = r.json().get("data", [])
                par_universe = []
                for it in data:
                    s = it["d"][0]
                    if s and "." not in s and "-" not in s and s.isalpha():
                        par_universe.append(s)
            except Exception as e:
                st.error(f"TV hatasi: {e}")
                st.stop()

            if "SPY" not in par_universe:
                par_universe.append("SPY")
            st.info(f"Evren: {len(par_universe)} sembol")

            status.info("Asama 2/3: Alpaca daily bars indiriliyor...")
            par_bars: dict[str, pd.DataFrame] = {}
            prog = st.progress(0.0, text="0/0")
            step = 50
            total = len(par_universe)
            for i in range(0, total, step):
                chunk = par_universe[i:i + step]
                got = fetch_daily_bars_batch(client, chunk, days=300)
                par_bars.update(got)
                done = min(i + step, total)
                prog.progress(done / total, text=f"{done}/{total}")
            prog.empty()

            spy_df = par_bars.get("SPY", pd.DataFrame())

            status.info("Asama 3/3: Parabolik gostergeler hesaplaniyor...")
            par_candidates = []
            par_rejected = []
            syms_list = [s for s in par_bars.keys() if s != "SPY"]
            prog2 = st.progress(0.0, text="0/0")

            for idx, sym in enumerate(syms_list):
                try:
                    fs = precompute_feature_series(par_bars[sym], spy_df)
                    if fs.empty:
                        continue
                    pf = parabolic_features(fs)
                    if pf.empty:
                        continue
                    row = pf.iloc[-1]

                    # Temel fiyat/hacim filtresi
                    if row["close"] < MIN_PX or row["close"] > MAX_PX:
                        continue
                    if row["volume"] < MIN_VOL:
                        continue

                    # Parabolik sert filtre
                    passed, reason = parabolic_passes_filters(row, par_min_rvol, par_min_cs)
                    if not passed:
                        par_rejected.append({"symbol": sym, "reason": reason})
                        continue

                    score, comp = parabolic_score_row(row)
                    if score < par_min_score:
                        par_rejected.append({"symbol": sym,
                                             "reason": f"skor dusuk ({score})"})
                        continue

                    # Trade seviyeleri (mevcut sidebar parametreleri)
                    feats_dict = {
                        "prior_20d_high": float(row["prior_20d_high"]) if pd.notna(row["prior_20d_high"]) else np.nan,
                        "close": float(row["close"]),
                        "breakout_dist": float(row["breakout_dist"]),
                        "atr14": float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan,
                    }
                    levels = compute_trade_levels(feats_dict)
                    pos = position_size(ACCOUNT, RISK_PCT, levels["entry"],
                                        levels["stop"], KELLY_FRAC)

                    par_candidates.append({
                        "Symbol": sym,
                        "ParScore": score,
                        "Close": round(float(row["close"]), 4),
                        "RVOL": round(float(row["rvol"]), 2) if pd.notna(row["rvol"]) else None,
                        "BB_Sqz_%": round(float(row["bb_squeeze_pctile"]) * 100, 1) if pd.notna(row["bb_squeeze_pctile"]) else None,
                        "NR7": bool(row["nr7"]),
                        "NR4": bool(row["nr4"]),
                        "Dist_52w_%": round(float(row["dist_52w_high"]) * 100, 2) if pd.notna(row["dist_52w_high"]) else None,
                        "Accum": bool(row["accumulation"]),
                        "DryPop": bool(row["dry_then_pop"]),
                        "Close_Str": round(float(row["close_strength"]), 2),
                        "ATR_Ratio": round(float(row["atr_ratio"]), 2) if pd.notna(row["atr_ratio"]) else None,
                        "Entry": levels["entry"],
                        "Stop": levels["stop"],
                        "TP1": levels["tp1"],
                        "TP2": levels["tp2"],
                        "Stop_%": levels["stop_pct"],
                        "Shares": pos["shares"],
                        "Risk_$": pos["risk_dollars"],
                        "Pos_$": pos["dollar_size"],
                    })
                except Exception:
                    continue

                if (idx + 1) % 25 == 0 or idx == len(syms_list) - 1:
                    prog2.progress((idx + 1) / len(syms_list),
                                   text=f"{idx+1}/{len(syms_list)}")
            prog2.empty()
            status.empty()

            par_df = pd.DataFrame(par_candidates).sort_values("ParScore", ascending=False) if par_candidates else pd.DataFrame()
            if par_df.empty:
                st.warning("Filtreleri gecen parabolik aday yok. Esikleri biraz dusurebilirsin.")
            else:
                st.success(f"{len(par_df)} parabolik aday bulundu.")
                st.dataframe(par_df, use_container_width=True, hide_index=True)

                csv = par_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Parabolik adaylari indir (CSV)", csv,
                    file_name=f"parabolic_{datetime.now():%Y%m%d_%H%M}.csv",
                    mime="text/csv",
                )

            with st.expander(f"Reddedilenler ({len(par_rejected)})"):
                if par_rejected:
                    st.dataframe(pd.DataFrame(par_rejected),
                                 use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Parabolik tarama hatasi: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.divider()
    st.subheader("Parabolik Model Backtesti")
    st.caption(
        "Bu modelin gecmiste URETECEGI sinyallerin ertesi gun "
        "INTRADAY HIGH olarak ne kadar hareket yakaladigini olcer. "
        "Asil hedef: 'next day high > entry * 1.20' olaylarinin oranini gormek."
    )

    colB1, colB2 = st.columns(2)
    with colB1:
        pb_days = st.number_input("Backtest gun sayisi", 30, 360, 120, 30, key="pb_days")
    with colB2:
        pb_uni = st.number_input("Evren buyuklugu (sabit)", 50, 500, 150, 50, key="pb_uni")

    if st.button("Parabolik Backtest Baslat", type="primary", key="btn_par_bt"):
        try:
            status = st.empty()
            status.info("Asama 1/3: Evren cekiliyor...")
            payload = {
                "filter": [
                    {"left": "close", "operation": "in_range", "right": [MIN_PX, MAX_PX]},
                    {"left": "volume", "operation": "greater", "right": MIN_VOL},
                    {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE", "AMEX"]},
                    {"left": "relative_volume_10d_calc", "operation": "greater", "right": 1.5},
                ],
                "options": {"lang": "en"},
                "markets": ["america"],
                "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                "columns": ["name"],
                "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
                "range": [0, int(pb_uni)],
            }
            try:
                r = requests.post(TV_URL, json=payload,
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                r.raise_for_status()
                data = r.json().get("data", [])
                symbols = []
                for it in data:
                    s = it["d"][0]
                    if s and "." not in s and "-" not in s and s.isalpha():
                        symbols.append(s)
            except Exception as e:
                st.error(f"TV hatasi: {e}")
                st.stop()

            if "SPY" not in symbols:
                symbols.append("SPY")

            status.info("Asama 2/3: Alpaca verisi...")
            bars_map: dict[str, pd.DataFrame] = {}
            prog = st.progress(0.0, text="0/0")
            step = 50
            total = len(symbols)
            for i in range(0, total, step):
                chunk = symbols[i:i + step]
                got = fetch_daily_bars_batch(client, chunk, days=pb_days + 300)
                bars_map.update(got)
                done = min(i + step, total)
                prog.progress(done / total, text=f"{done}/{total}")
            prog.empty()

            spy_df = bars_map.get("SPY", pd.DataFrame())
            if spy_df.empty:
                st.error("SPY verisi alinamadi.")
                st.stop()

            status.info("Asama 3/3: Parabolik sinyalleri simule ediliyor...")
            feat_map: dict[str, pd.DataFrame] = {}
            syms_list = [s for s in bars_map.keys() if s != "SPY"]
            prog2 = st.progress(0.0, text="Gostergeler...")
            for i, sym in enumerate(syms_list):
                try:
                    fs = precompute_feature_series(bars_map[sym], spy_df)
                    if fs.empty:
                        continue
                    pf = parabolic_features(fs)
                    if not pf.empty:
                        feat_map[sym] = pf
                except Exception:
                    pass
                if (i + 1) % 20 == 0 or i == len(syms_list) - 1:
                    prog2.progress((i + 1) / len(syms_list))
            prog2.empty()

            test_dates = spy_df.index[-pb_days:]
            par_trades = []
            prog3 = st.progress(0.0, text=f"0/{len(test_dates)}")

            for di, dt in enumerate(test_dates):
                for sym, pf in feat_map.items():
                    if dt not in pf.index:
                        continue
                    row = pf.loc[dt]
                    if row["close"] < MIN_PX or row["close"] > MAX_PX:
                        continue
                    passed, _ = parabolic_passes_filters(row, par_min_rvol, par_min_cs)
                    if not passed:
                        continue
                    score, _ = parabolic_score_row(row)
                    if score < par_min_score:
                        continue

                    # Ertesi gun bar'i
                    full_df = bars_map[sym]
                    next_bars = full_df.loc[full_df.index > dt]
                    if next_bars.empty:
                        continue
                    nb = next_bars.iloc[0]
                    entry_px = float(row["close"]) * (1 + ENTRY_BUFFER_PCT)
                    # Dolum: gun ici low <= entry
                    filled = float(nb["low"]) <= entry_px
                    fwd_high_ret = (float(nb["high"]) - entry_px) / entry_px
                    fwd_close_ret = (float(nb["close"]) - entry_px) / entry_px

                    par_trades.append({
                        "date": dt.date(),
                        "symbol": sym,
                        "score": score,
                        "close": round(float(row["close"]), 4),
                        "entry": round(entry_px, 4),
                        "filled": bool(filled),
                        "next_high": round(float(nb["high"]), 4),
                        "next_close": round(float(nb["close"]), 4),
                        "fwd_high_%": round(fwd_high_ret * 100, 2),
                        "fwd_close_%": round(fwd_close_ret * 100, 2),
                    })
                if (di + 1) % 5 == 0 or di == len(test_dates) - 1:
                    prog3.progress((di + 1) / len(test_dates),
                                   text=f"{di+1}/{len(test_dates)} gun | {len(par_trades)} sinyal")
            prog3.empty()
            status.empty()

            pt_df = pd.DataFrame(par_trades)
            if pt_df.empty:
                st.warning("Hic parabolik sinyal uretilmedi.")
            else:
                filled = pt_df[pt_df["filled"]]
                n_sig = len(pt_df)
                n_fill = len(filled)
                st.success(f"Toplam sinyal: {n_sig}, dolum: {n_fill}")

                if n_fill == 0:
                    st.warning("Hic sinyal dolmamis.")
                else:
                    # Kritik metrikler
                    p_20 = (filled["fwd_high_%"] >= 20).mean()
                    p_30 = (filled["fwd_high_%"] >= 30).mean()
                    p_50 = (filled["fwd_high_%"] >= 50).mean()
                    p_100 = (filled["fwd_high_%"] >= 100).mean()
                    med_high = filled["fwd_high_%"].median()
                    med_close = filled["fwd_close_%"].median()
                    avg_high = filled["fwd_high_%"].mean()

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("P(next_high >= 20%)", f"{p_20*100:.1f}%")
                    c2.metric("P(>= 30%)", f"{p_30*100:.1f}%")
                    c3.metric("P(>= 50%)", f"{p_50*100:.1f}%")
                    c4.metric("P(>= 100%)", f"{p_100*100:.1f}%")

                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Medyan fwd high", f"{med_high:+.2f}%")
                    c6.metric("Medyan fwd close", f"{med_close:+.2f}%")
                    c7.metric("Ort. fwd high", f"{avg_high:+.2f}%")
                    c8.metric("Toplam dolum", n_fill)

                    st.caption(
                        "Not: 'fwd_high_%' = ertesi gun INTRADAY HIGH / entry. "
                        "Gercek hayatta bu fiyati yakalamak icin akilli limit-sat "
                        "emirleri gerekir; %100 HIGH'da cikisi garanti ETMEZ."
                    )

                    st.subheader("Top 30 sinyal (en yuksek fwd_high_%)")
                    top = filled.sort_values("fwd_high_%", ascending=False).head(30)
                    st.dataframe(top, use_container_width=True, hide_index=True)

                    # Distribution histogram
                    hist_df = filled[["fwd_high_%"]].copy()
                    st.subheader("Dagilim: ertesi gun forward high yuzdeleri")
                    st.bar_chart(
                        hist_df["fwd_high_%"].value_counts(bins=30).sort_index()
                    )

                    csv = filled.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Parabolik backtest log", csv,
                        file_name=f"parabolic_bt_{datetime.now():%Y%m%d_%H%M}.csv",
                    )
        except Exception as e:
            st.error(f"Parabolik backtest hatasi: {e}")
            import traceback
            st.code(traceback.format_exc())


# ============================================================
# TAB 3 — ISTATISTIKLER (Kendi canli trade'lerini takip)
# ============================================================
with tab3:
    st.subheader("Kendi Canli Trade Istatistiklerim")
    st.caption(
        "Gercekten yaptigin trade'leri buraya gir. Her ay sonu kendi "
        "expectancy ve Kelly rakamini hesaplayip karsi gorursun."
    )

    LOG_PATH = "trade_log.csv"
    cols = ["date", "symbol", "entry", "exit", "shares", "pnl_pct", "pnl_usd", "notes"]

    if "trade_log" not in st.session_state:
        if os.path.exists(LOG_PATH):
            try:
                st.session_state.trade_log = pd.read_csv(LOG_PATH)
            except Exception:
                st.session_state.trade_log = pd.DataFrame(columns=cols)
        else:
            st.session_state.trade_log = pd.DataFrame(columns=cols)

    with st.form("add_trade"):
        c1, c2, c3 = st.columns(3)
        with c1:
            t_date = st.date_input("Tarih", value=date.today())
            t_sym = st.text_input("Sembol").upper().strip()
        with c2:
            t_entry = st.number_input("Giris ($)", min_value=0.01, value=1.00, step=0.01)
            t_exit = st.number_input("Cikis ($)", min_value=0.01, value=1.10, step=0.01)
        with c3:
            t_shares = st.number_input("Adet", min_value=1, value=100)
            t_notes = st.text_input("Not (opsiyonel)")

        submit = st.form_submit_button("Kaydet")
        if submit and t_sym:
            pnl_pct = (t_exit - t_entry) / t_entry * 100
            pnl_usd = (t_exit - t_entry) * t_shares
            row = {
                "date": str(t_date),
                "symbol": t_sym,
                "entry": round(t_entry, 4),
                "exit": round(t_exit, 4),
                "shares": int(t_shares),
                "pnl_pct": round(pnl_pct, 3),
                "pnl_usd": round(pnl_usd, 2),
                "notes": t_notes,
            }
            st.session_state.trade_log = pd.concat(
                [st.session_state.trade_log, pd.DataFrame([row])], ignore_index=True
            )
            try:
                st.session_state.trade_log.to_csv(LOG_PATH, index=False)
                st.success("Kaydedildi.")
            except Exception as e:
                st.warning(f"Dosyaya yazilamadi: {e}")

    log = st.session_state.trade_log
    if not log.empty:
        st.dataframe(log.sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)

        wins = log[log["pnl_pct"] > 0]
        losses = log[log["pnl_pct"] <= 0]
        n = len(log)
        wr = len(wins) / n if n else 0
        avg_w = wins["pnl_pct"].mean() if len(wins) else 0.0
        avg_l = losses["pnl_pct"].mean() if len(losses) else 0.0
        exp_ = wr * avg_w + (1 - wr) * avg_l

        if avg_l < 0 and abs(avg_l) > 0:
            b = abs(avg_w / avg_l)
            k = (b * wr - (1 - wr)) / b if b > 0 else 0
            k = max(0, k)
        else:
            k = 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam trade", n)
        c2.metric("Win-rate", f"{wr*100:.1f}%")
        c3.metric("Expectancy", f"{exp_:+.2f}%")
        c4.metric("Kelly tam", f"{k*100:.1f}%")

        st.metric("Toplam net P&L ($)", f"{log['pnl_usd'].sum():+.2f}")

        if st.button("Log'u temizle"):
            st.session_state.trade_log = pd.DataFrame(columns=cols)
            try:
                os.remove(LOG_PATH)
            except Exception:
                pass
            st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Bu arac yatirim tavsiyesi degildir. Matematiksel model sadece olasilik "
    "verir, garanti vermez. Once paper account'ta test et, ardindan kucuk "
    "sermayeli canli teste gec. Parametreleri degistirmeden once backtest'te "
    "etkilerini mutlaka gor."
)
