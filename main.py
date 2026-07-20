from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import datetime
import uuid
import os
import json
import yfinance as yf
import numpy as np
import logging
import pandas as pd
from openai import AzureOpenAI
from typing import Optional
from scipy.stats import norm


# --- ログ設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-ml-api")

app = FastAPI(title="stock-ml-api (GPT+ML hybrid, AzureOpenAI)")

# ============================================================
# 1) データモデル
# ============================================================

class MarketState(BaseModel):
    stock_price: float
    atm_iv: float
    otm_iv: float
    gamma: float
    delta: float
    days_to_expiry: int
    hv_20d: float
    market_view: Optional[str] = ""   # 自分の市場予想

# ============================================================
# 2) ユーティリティ関数（分類・HV計算・Azureチェック）
# ============================================================

def classify_month(df_month):
    try:
        if df_month is None or len(df_month) == 0:
            return "FLAT"
        open_price = float(df_month["Open"].iloc[0])
        close_price = float(df_month["Close"].iloc[-1])
        high_price = float(df_month["High"].max())
        low_price = float(df_month["Low"].min())

        mid_index = max(0, len(df_month) // 2)
        mid_price = float(df_month["Close"].iloc[mid_index])

        change_total = (close_price - open_price) / open_price if open_price != 0 else 0
        change_open_mid = (mid_price - open_price) / open_price if open_price != 0 else 0
        change_mid_close = (close_price - mid_price) / mid_price if mid_price != 0 else 0
        range_month = (high_price - low_price) / open_price if open_price != 0 else 0

        if change_total > 0.03:
            return "UP"
        if change_total < -0.03:
            return "DOWN"
        if range_month < 0.02:
            return "FLAT"
        if change_open_mid > 0.02 and change_mid_close < -0.02:
            return "UPDOWN"
        if change_open_mid < -0.02 and change_mid_close > 0.02:
            return "DOWNUP"

        return "FLAT"
    except Exception as e:
        logger.exception("classify_month error")
        return "FLAT"

def calc_hv(df):
    try:
        if df is None or len(df) < 2:
            return None
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        if len(returns) == 0:
            return None
        hv = float(returns.std() * np.sqrt(252))
        return hv
    except Exception as e:
        logger.exception("calc_hv error")
        return None

def azure_config_ok():
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_DEPLOYMENT") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        return False
    return True


def bs_call_price(S, K, sigma, T, r=0.001):
    """
    Black-Scholes コールオプション理論価格（本物）
    S: 現物価格
    K: ストライク
    sigma: ボラティリティ（HVをIVとして使用）
    T: 満期（年換算）
    r: 無リスク金利（日本なら 0.1% 程度）
    """
    if sigma <= 0 or T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call


def bs_put_price(S, K, sigma, T, r=0.001):
    """
    Black-Scholes プットオプション理論価格（本物）
    """
    if sigma <= 0 or T <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put


# ============================================================
# 3) ML推論ロジック
# ============================================================

def ml_predict(m: MarketState):
    try:
        if m.atm_iv > 0.3 and m.days_to_expiry <= 7:
            return "short_close", 0.82
        if m.atm_iv < 0.15 and m.days_to_expiry >= 20:
            return "spread_hold", 0.76
        if m.hv_20d > 0.25 and m.otm_iv > 0.3:
            return "long_only", 0.71
        return "no_trade", 0.63
    except Exception:
        logger.exception("ml_predict error")
        return "no_trade", 0.0

# ============================================================
# 4) GPT推論（AzureOpenAI）
# ============================================================

def gpt_predict(m: MarketState):
    if not azure_config_ok():
        logger.info("Azure OpenAI config missing")
        return None

    try:
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        prompt = f"""
あなたはプロのオプション戦略アナリストです。
以下の市場状態と市場予想を総合評価し、最適な戦略を1つ選び、理由を説明してください。

【市場データ】
株価: {m.stock_price}
ATM IV: {m.atm_iv}
OTM IV: {m.otm_iv}
ガンマ: {m.gamma}
デルタ: {m.delta}
残存日数: {m.days_to_expiry}
HV: {m.hv_20d}

【市場予想（自分の判断）】
{m.market_view}

【出力形式】
次の JSON のみを返す：

{{
  "strategy": "",
  "expert_reason": "",
  "beginner_explanation": "",
  "beginner_caution": "",
  "next_step": ""
}}
"""
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()

        # 安全に JSON 部分を抽出してパース
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            logger.warning("GPT did not return JSON")
            return None

        json_text = raw[json_start:json_end]
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        return json.loads(json_text)

    except Exception:
        logger.exception("gpt_predict error")
        return None


# ============================================================
# 6) 株価自動取得 API
# ============================================================

@app.get("/api/price")
def api_price(ticker: str = "^N225"):
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if hist is None or len(hist) == 0:
            return {"price": None}
        price = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
        return {"price": price, "previous_close": prev}
    except Exception:
        logger.exception("api_price error")
        return {"error": "price fetch failed"}

# ============================================================
# 7) HV自動取得 API
# ============================================================

@app.get("/api/hv")
def api_hv(ticker: str = "^N225", days: int = 20):
    try:
        hist = yf.Ticker(ticker).history(period=f"{days+1}d")
        if hist is None or len(hist) < 2:
            return {"hv": None}
        close = hist["Close"].values
        log_returns = np.log(close[1:] / close[:-1])
        hv = float(np.std(log_returns) * np.sqrt(252))
        return {"hv": hv}
    except Exception:
        logger.exception("api_hv error")
        return {"error": "hv fetch failed"}

# ============================================================
# 8) GPT 市場予想 API
# ============================================================

@app.get("/api/market_view_auto")
def api_market_view_auto(ticker: str = "^N225"):
    if not azure_config_ok():
        return {"market_view_auto": "Azure config missing", "reason": ""}

    try:
        hist = yf.Ticker(ticker).history(period="30d")
        if hist is None or len(hist) < 10:
            return {"market_view_auto": "データ不足", "reason": ""}

        closes = hist["Close"].values
        hv = float(np.std(np.log(closes[1:] / closes[:-1])) * np.sqrt(252))

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        prompt = f"""
あなたは市場アナリストです。
以下の市場データから、日経平均の短期市場予想を1つ生成してください。

【市場データ】
直近10日終値: {list(closes[-10:])}
HV: {hv}

【出力形式】
次の JSON のみを返す：

{{
  "market_view_auto": "",
  "reason": ""
}}
"""

        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()

        try:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            json_text = raw[json_start:json_end]
            json_text = json_text.replace("```json", "").replace("```", "").strip()
            return json.loads(json_text)
        except Exception:
            logger.exception("market_view_auto parse error")
            return {"market_view_auto": "GPTがJSONを返しませんでした", "reason": raw}

    except Exception:
        logger.exception("api_market_view_auto error")
        return {"error": "market view failed"}

# ============================================================
# GPT専用推論 API（MarketState を使う）
# ============================================================

class MarketState(BaseModel):
    stock_price: float
    atm_iv: float
    otm_iv: float
    gamma: float
    delta: float
    days_to_expiry: int
    hv_20d: float
    market_view: Optional[str] = ""

@app.post("/api/predict_gpt")
def api_predict_gpt(m: MarketState):
    try:
        gpt = gpt_predict(m)
        if gpt and gpt.get("strategy"):
            return {
                "source": "GPT",
                "result": gpt,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4())
            }
        else:
            raise HTTPException(status_code=500, detail="GPT推論が失敗しました")
    except Exception as e:
        logger.exception("api_predict_gpt error")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 9) MLデータ収集 API（5年間の月次データ）
# ============================================================
@app.get("/api/ml_collect_5y")
def api_ml_collect_5y():
    try:
        df_n225 = yf.Ticker("^N225").history(period="5y")
        df_n225["Month"] = df_n225.index.to_period("M")

        df_spx = yf.Ticker("^GSPC").history(period="5y")
        df_spx["Month"] = df_spx.index.to_period("M")

        raw_results = []

        for month, df_month in df_n225.groupby("Month"):

            # 月の中間日を基準にHVを計算（改善版）
            hv_n225 = calc_hv_mid20("^N225", df_month)
            hv_spx = calc_hv_mid20("^GSPC", df_month)

            pattern = classify_month(df_month)

            raw_results.append({
                "month": str(month),
                "pattern_prev": pattern,
                "hv_n225_prev": hv_n225,
                "hv_spx_prev": hv_spx
            })

        # --- 平滑化処理（ロバスト移動平均） ---
        hv_n225_list = [r["hv_n225_prev"] for r in raw_results]
        hv_spx_list = [r["hv_spx_prev"] for r in raw_results]

        hv_n225_smooth = smooth_hv(hv_n225_list)
        hv_spx_smooth = smooth_hv(hv_spx_list)

        # 平滑化結果を反映
        results = []
        for i in range(len(raw_results)):
            r = raw_results[i]
            results.append({
                "month": r["month"],
                "pattern_prev": r["pattern_prev"],
                "hv_n225_prev": hv_n225_smooth[i],
                "hv_spx_prev": hv_spx_smooth[i]
            })

        return results

    except Exception as e:
        return {"error": str(e)}


def calc_hv_last20(ticker, end_date):
    """
    end_date（その月の最終日）を基準に過去20日分の連続データでHVを計算する
    """
    try:
        df = yf.Ticker(ticker).history(start=end_date - datetime.timedelta(days=40),
                                       end=end_date + datetime.timedelta(days=1))

        if df is None or len(df) < 20:
            return None

        closes = df["Close"].values
        log_returns = np.log(closes[1:] / closes[:-1])
        hv = float(np.std(log_returns[-20:]) * np.sqrt(252))
        return hv

    except Exception as e:
        return None


def smooth_hv(values, window=3):
    """
    ロバスト移動平均（中央値ベース）
    スパイクを抑えつつトレンドを維持する
    MLに最適な平滑化方法
    """
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window)
        end = min(len(values), i + window + 1)
        window_vals = values[start:end]
        median = float(np.median(window_vals))
        smoothed.append(median)
    return smoothed


def calc_hv_mid20(ticker, df_month):
    """
    月の中間日を基準に過去20日 HV を計算する（ノイズを減らす改善版）
    """
    try:
        # 月の中間日を取得
        mid_index = len(df_month) // 2
        mid_date = df_month.index[mid_index]

        # 過去40日分を取得（20日分を確実に確保するため）
        df = yf.Ticker(ticker).history(
            start=mid_date - datetime.timedelta(days=40),
            end=mid_date + datetime.timedelta(days=1)
        )

        if df is None or len(df) < 20:
            return None

        closes = df["Close"].values
        log_returns = np.log(closes[1:] / closes[:-1])

        hv = float(np.std(log_returns[-20:]) * np.sqrt(252))
        return hv

    except Exception as e:
        print("calc_hv_mid20 error:", e)
        return None


@app.get("/api/backtest_strategies")
def api_backtest_strategies():
    """
    Black-Scholes を使った改善版バックテスト：
    - HV を IV として使用
    - BS でプレミアムを計算
    - Spread / Iron Condor の比較が正常化
    """

    try:
        # ① MLデータ（分類＋HV）を取得
        ml_data = api_ml_collect_5y()
        if "error" in ml_data:
            return ml_data

        # ② 月末株価を取得
        df_price = yf.Ticker("^N225").history(period="5y")
        df_price["Month"] = df_price.index.to_period("M")

        results = []

        # ③ 月次バックテスト
        for row in ml_data:
            month = row["month"]
            pattern = row["pattern_prev"]
            iv = row["hv_n225_prev"]  # ← HV を IV として使う（重要）

            df_month = df_price[df_price["Month"] == month]
            if len(df_month) == 0:
                continue

            S = float(df_month["Close"].iloc[-1])

            next_month = str((pd.Period(month) + 1))
            df_next = df_price[df_price["Month"] == next_month]
            if len(df_next) == 0:
                continue

            S_next = float(df_next["Close"].iloc[-1])

            T = 30 / 365  # 月次バックテストなので 30日固定

            # ④ 戦略の損益計算（Straddle は除外）
            def bull_call_spread():
                width = 300
                long = S + width
                short = S + width * 2

                premium = bs_call_price(S, long, iv, T) - bs_call_price(S, short, iv, T)
                intrinsic = max(0, S_next - long) - max(0, S_next - short)
                return intrinsic - premium

            def bear_put_spread():
                width = 300
                long = S - width
                short = S - width * 2

                premium = bs_put_price(S, long, iv, T) - bs_put_price(S, short, iv, T)
                intrinsic = max(0, long - S_next) - max(0, short - S_next)
                return intrinsic - premium

            def iron_condor():
                width = 200
                call_long = S + width * 2
                call_short = S + width
                put_long = S - width * 2
                put_short = S - width

                premium = (
                    bs_call_price(S, call_short, iv, T)
                    - bs_call_price(S, call_long, iv, T)
                    + bs_put_price(S, put_short, iv, T)
                    - bs_put_price(S, put_long, iv, T)
                )

                intrinsic = (
                    max(0, S_next - call_short) - max(0, S_next - call_long)
                    + max(0, put_short - S_next) - max(0, put_long - S_next)
                )

                return premium - intrinsic

            # ⑤ 戦略比較（Straddle は除外）
            strategies = [
                ("bull_call_spread", bull_call_spread()),
                ("bear_put_spread", bear_put_spread()),
                ("iron_condor", iron_condor()),
            ]

            best = max(strategies, key=lambda x: x[1])

            results.append({
                "month": month,
                "pattern_prev": pattern,
                "best_strategy": best[0],
                "best_pnl": best[1],
                "S": S,
                "S_next": S_next,
                "iv_used": iv
            })

        return results

    except Exception as e:
        return {"error": str(e)}


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import pickle
import os
from azure.storage.blob import BlobServiceClient

# ============================================================
# ML入力（3変数専用）
# ============================================================

class MLPredictRequest(BaseModel):
    pattern_prev: str
    hv_n225_prev: float
    hv_spx_prev: float

# ============================================================
# LightGBM 学習（3変数専用）
# ============================================================

@app.post("/api/train_lightgbm")
def train_lightgbm():
    try:

        # MLデータ（pattern_prev, hv_n225_prev, hv_spx_prev）
        ml_data = api_ml_collect_5y()
        df_ml = pd.DataFrame(ml_data)

        # MLデータは 61 行 → バックテストに合わせて最後の 1 行を削除
        df_ml = df_ml.iloc[:-1].reset_index(drop=True)

        # バックテスト結果（60行）
        bt_data = api_backtest_strategies()
        df_bt = pd.DataFrame(bt_data)

        # 月で結合（60行になる）
        df = df_ml.merge(df_bt, on="month")

        # pattern_prev をラベルエンコード（df_ml を使う）
        le = LabelEncoder()
        df["pattern_prev_enc"] = le.fit_transform(df_ml["pattern_prev"])

        # 特徴量（3変数＋エンコード）
        X = df[["hv_n225_prev", "hv_spx_prev", "pattern_prev_enc"]]

        # 目的変数（戦略別PNL）
        y_bull = df["best_pnl"].where(df["best_strategy"] == "bull_call_spread", 0)
        y_bear = df["best_pnl"].where(df["best_strategy"] == "bear_put_spread", 0)
        y_condor = df["best_pnl"].where(df["best_strategy"] == "iron_condor", 0)

        # LightGBM 学習
        model_bull = LGBMRegressor().fit(X, y_bull)
        model_bear = LGBMRegressor().fit(X, y_bear)
        model_condor = LGBMRegressor().fit(X, y_condor)

        # Blob Storage 保存
        blob_service = BlobServiceClient.from_connection_string(
            os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        container_name = os.getenv("MODEL_CONTAINER", "models")
        container = blob_service.get_container_client(container_name)

        def upload_pkl(name, obj):
            blob = container.get_blob_client(name)
            blob.upload_blob(pickle.dumps(obj), overwrite=True)

        upload_pkl("model_bull.pkl", model_bull)
        upload_pkl("model_bear.pkl", model_bear)
        upload_pkl("model_condor.pkl", model_condor)
        upload_pkl("pattern_encoder.pkl", le)

        return {"status": "学習完了（3変数ML）", "records": len(df)}

    except Exception as e:
        return {"error": str(e)}

# ============================================================
# ML推論（3変数専用）
# ============================================================
@app.post("/api/predict_strategy")
def api_predict_strategy(m: MLPredictRequest):
    try:
        # モデル読み込み
        blob_service = BlobServiceClient.from_connection_string(
            os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
        container_name = os.getenv("MODEL_CONTAINER", "models")
        container = blob_service.get_container_client(container_name)

        def load_pkl(name):
            blob = container.get_blob_client(name)
            data = blob.download_blob().readall()
            return pickle.loads(data)

        model_bull = load_pkl("model_bull.pkl")
        model_bear = load_pkl("model_bear.pkl")
        model_condor = load_pkl("model_condor.pkl")
        le = load_pkl("pattern_encoder.pkl")

        # pattern_prev をエンコード
        pattern_enc = int(le.transform([m.pattern_prev])[0])

        # 特徴量ベクトル
        X = np.array([[m.hv_n225_prev, m.hv_spx_prev, pattern_enc]])

        # 推論
        bull = float(model_bull.predict(X)[0])
        bear = float(model_bear.predict(X)[0])
        condor = float(model_condor.predict(X)[0])

        # 最良戦略
        strategies = {
            "bull_call_spread": bull,
            "bear_put_spread": bear,
            "iron_condor": condor
        }
        best = max(strategies, key=strategies.get)

        return {
            "source": "ML",
            "bull_call_spread": bull,
            "bear_put_spread": bear,
            "iron_condor": condor,
            "best_strategy": best,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 10) ログ保存 API（UI の logState() が呼ぶ）
# ============================================================

LOG_FILE = "market_logs.json"

@app.post("/api/log_market_state")
def api_log_market_state(payload: dict):
    try:
        log_id = str(uuid.uuid4())
        saved_at = datetime.datetime.utcnow().isoformat()
        entry = {
            "log_id": log_id,
            "saved_at": saved_at,
            "payload": payload
        }

        # ファイルに追記（JSON配列形式）
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        else:
            data = []

        data.append(entry)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {"log_id": log_id, "saved_at": saved_at}
    except Exception:
        logger.exception("api_log_market_state error")
        raise HTTPException(status_code=500, detail="log save failed")

# ============================================================
# 11) HTML（スマホ最適化 UI）
# ============================================================

INDEX_HTML = """

<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>stock-ml-api</title>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">

<style>
  :root{
    --bg:#ffffff;
    --panel:#f2f2f2;
    --accent:#0078ff;
    --text:#000;
  }
  body{
    margin:0;
    background:var(--bg);
    color:var(--text);
    font-family:system-ui, -apple-system, "Hiragino Kaku Gothic ProN", sans-serif;
    padding:16px;
    font-size:22px;
  }
  input, select{
    width:100%;
    font-size:24px;
    padding:16px;
    margin:10px 0;
    border-radius:10px;
    border:1px solid #ccc;
  }
  button{
    width:100%;
    font-size:26px;
    padding:18px;
    border-radius:12px;
    margin-top:16px;
    background:var(--accent);
    color:#fff;
    border:none;
  }
  #resultBox, #logBox, #hvBox, #autoMarketViewBox, #mlDataBox{
    background:var(--panel);
    padding:16px;
    border-radius:10px;
    font-size:24px;
    margin-top:16px;
  }
</style>
</head>

<body>

<h2>stock-ml-api</h2>

<h3>市場状態入力</h3>

株価 S:<br>
<input id="stock_price" type="number" placeholder="例: 39000">
<button onclick="loadPrice()">株価を自動取得する</button>

ATM IV (%):<br>
<input id="atm_iv" type="number" placeholder="例: 20">

OTM IV (%):<br>
<input id="otm_iv" type="number" placeholder="例: 25">

ガンマ:<br>
<input id="gamma" type="number" step="0.0001" placeholder="例: 0.0012">

デルタ:<br>
<input id="delta" type="number" step="0.01" placeholder="例: 0.45">

残存日数:<br>
<input id="days_to_expiry" type="number" placeholder="例: 7">

HV (%):<br>
<input id="hv_20d" type="number" placeholder="例: 18">
<button onclick="loadHV()">HVを自動取得する</button>
<div id="hvBox"></div>

<hr>

<!-- AI市場予想（参考） -->
<button onclick="loadAutoMarketView()">AI市場予想（参考）を取得する</button>
<div id="autoMarketViewBox"></div>

<hr>

市場予想（自分の判断・選択式）:<br>
<select id="market_view">
  <option value="">選択してください</option>
  <option value="上昇予想">上昇予想</option>
  <option value="下落予想">下落予想</option>
  <option value="横ばい予想">横ばい予想</option>
  <option value="荒れやすい（ボラティリティ上昇）">荒れやすい（ボラティリティ上昇）</option>
  <option value="イベント前で不安定（SQ・FOMCなど）">イベント前で不安定（SQ・FOMCなど）</option>
</select>

<button onclick="predict()">推論する</button>

<div id="resultBox"></div>

<hr>

<h3>来月の戦略を機械学習(ML)で予測</h3>

<div class="input-block">
  <label>来月の市場予想（UP / DOWN / FLAT / UPDOWN / DOWNUP）:</label>
  <input id="pred_pattern_prev" type="text" placeholder="例: UP">
  <small>大文字で入力してください（UP / DOWN / FLAT / UPDOWN / DOWNUP）</small>
</div>

<div class="input-block">
  <label>ヒストリカルボラティリティ（日経225） hv_n225_prev:</label>
  <input id="pred_hv_n225_prev" type="number" step="0.0001" placeholder="例: 0.3522">
  <small>小数表記（例: 0.3522 = 35.22%）</small>
</div>

<div class="input-block">
  <label>ヒストリカルボラティリティ（SPX） hv_spx_prev:</label>
  <input id="pred_hv_spx_prev" type="number" step="0.0001" placeholder="例: 0.1517">
  <small>小数表記（例: 0.1517 = 15.17%）</small>
</div>

<button type="button" onclick="predictStrategy()">機械学習(ML)で予測する </button>
<div id="predictResultBox" class="panel"></div>

<hr>

<h3>ログ保存</h3>
<button onclick="logState()">ログ保存する</button>
<div id="logBox"></div>

<hr>

<h3>MLデータ収集（5年間）</h3>
<button onclick="collectML()">MLデータ収集する</button>
<div id="mlDataBox"></div>

<hr>

<h3>バックテスト（5年間）</h3>
<button onclick="runBacktest()">バックテストを実行する</button>
<div id="backtestBox"></div>

<hr>

<button id="trainBtn">ML学習（LightGBM）を実行する</button>

<div id="trainResult" style="margin-top:10px; font-size:14px;"></div>

<script>
document.getElementById("trainBtn").addEventListener("click", async () => {
    document.getElementById("trainResult").innerText = "学習中です…（1〜3秒ほど）";

    try {
        const response = await fetch("/api/train_lightgbm", {
            method: "POST"
        });

        const data = await response.json();

        // ★ ここを修正：3変数ML版のステータスに対応
        if (data.status && data.status.includes("学習完了")) {
            document.getElementById("trainResult").innerText =
                "✔ LightGBMモデル（3変数版）の学習が完了しました（" + data.records + "件）";
        } else {
            document.getElementById("trainResult").innerText =
                "⚠ エラー：" + JSON.stringify(data);
        }

    } catch (err) {
        document.getElementById("trainResult").innerText =
            "⚠ 通信エラー：" + err;
    }
});
</script>



<script>
async function loadPrice(){
    const data = await fetch("/api/price").then(r => r.json());
    if(data.price){
        document.getElementById("stock_price").value = data.price;
    }
}

async function loadHV(){
    const data = await fetch("/api/hv").then(r => r.json());

    let text = "";

    if(data.hv){
        const vol = (data.hv * 100).toFixed(2);
        document.getElementById("hv_20d").value = vol;
        text = `volatility: ${vol} %`;
    }else{
        text = "volatility: データなし";
    }

    document.getElementById("hvBox").innerHTML = `
<b>【ヒストリカルボラ（20日）】</b><br>
${text}
    `;
}

async function loadAutoMarketView(){
    const data = await fetch("/api/market_view_auto").then(r => r.json());

    if(data.market_view_auto){
        document.getElementById("autoMarketViewBox").innerHTML = `
<b>【GPT市場予想（参考）】</b><br>
${data.market_view_auto}<br><br>
理由: ${data.reason}
        `;
    } else {
        document.getElementById("autoMarketViewBox").innerHTML = `
<b>【GPT市場予想（参考）】</b><br>
取得できませんでした。
        `;
    }
}

function getInputData(){
    return {
        stock_price: parseFloat(document.getElementById("stock_price").value) || 0,
        atm_iv: (parseFloat(document.getElementById("atm_iv").value) || 0) / 100,
        otm_iv: (parseFloat(document.getElementById("otm_iv").value) || 0) / 100,
        gamma: parseFloat(document.getElementById("gamma").value) || 0,
        delta: parseFloat(document.getElementById("delta").value) || 0,
        days_to_expiry: parseInt(document.getElementById("days_to_expiry").value) || 0,
        hv_20d: (parseFloat(document.getElementById("hv_20d").value) || 0) / 100,
        market_view: document.getElementById("market_view").value || ""
    };
}


function predict(){
    const data = getInputData();

    fetch("/api/predict_gpt", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {

        if(result.source === "GPT"){
            const r = result.result;

            document.getElementById("resultBox").innerHTML = `
<b>【市場予想（自分の判断）】</b><br>
${data.market_view || "（なし）"}<br><br>

<b>【GPT推論結果】</b><br><br>

strategy: ${r.strategy}<br><br>
expert_reason: ${r.expert_reason}<br><br>
beginner_explanation: ${r.beginner_explanation}<br><br>
beginner_caution: ${r.beginner_caution}<br><br>
next_step: ${r.next_step}<br><br>

timestamp: ${result.timestamp}<br>
request_id: ${result.request_id}
            `;
        } else {
            document.getElementById("resultBox").innerHTML = `
<b>GPT推論に失敗しました</b><br>
${JSON.stringify(result)}
            `;
        }
    });
}


function logState(){
    const data = getInputData();
    const chosen = prompt("選択した戦略を入力してください（例：spread_hold）", "");
    const note = prompt("任意メモ（理由など）", "");

    const payload = Object.assign({}, data, {
        chosen_strategy: chosen || "",
        note: note || ""
    });

    fetch("/api/log_market_state", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(res => {
        document.getElementById("logBox").innerHTML = `
<b>【ログ保存完了】</b><br>
log_id: ${res.log_id}<br>
保存時刻: ${res.saved_at}
        `;
    });
}

async function collectML(){
    const data = await fetch("/api/ml_collect_5y").then(r => r.json());

    if(data.error){
        document.getElementById("mlDataBox").innerHTML = `<b>エラー:</b> ${data.error}`;
        return;
    }

    let html = "<b>【MLデータ収集結果（5年間）】</b><br><br>";

    data.forEach(row => {
        html += `
month: ${row.month}<br>
pattern_prev: ${row.pattern_prev}<br>
hv_n225_prev: ${row.hv_n225_prev}<br>
hv_spx_prev: ${row.hv_spx_prev}<br><br>
        `;
    });

    document.getElementById("mlDataBox").innerHTML = html;
}

async function runBacktest(){
    const data = await fetch("/api/backtest_strategies").then(r => r.json());

    if(data.error){
        document.getElementById("backtestBox").innerHTML = `<b>エラー:</b> ${data.error}`;
        return;
    }

    let html = "<b>【バックテスト結果（5年間）】</b><br><br>";

    data.forEach(row => {
        html += `
month: ${row.month}<br>
pattern_prev: ${row.pattern_prev}<br>
best_strategy: ${row.best_strategy}<br>
best_pnl: ${row.best_pnl}<br>
S: ${row.S}<br>
S_next: ${row.S_next}<br><br>
        `;
    });

    document.getElementById("backtestBox").innerHTML = html;
}


function getMLInput3() {
    return {
        pattern_prev: (document.getElementById("pred_pattern_prev").value || "").trim().toUpperCase(),
        hv_n225_prev: parseFloat(document.getElementById("pred_hv_n225_prev").value),
        hv_spx_prev: parseFloat(document.getElementById("pred_hv_spx_prev").value)
    };
}

function predictStrategy() {
    const data = getMLInput3();

    // --- 入力チェック ---
    if (!["UP","DOWN","FLAT","UPDOWN","DOWNUP"].includes(data.pattern_prev)) {
        document.getElementById("predictResultBox").innerHTML =
            "⚠ pattern_prev は UP / DOWN / FLAT / UPDOWN / DOWNUP のいずれかを入力してください";
        return;
    }
    if (isNaN(data.hv_n225_prev)) {
        document.getElementById("predictResultBox").innerHTML =
            "⚠ hv_n225_prev が未入力または不正です";
        return;
    }
    if (isNaN(data.hv_spx_prev)) {
        document.getElementById("predictResultBox").innerHTML =
            "⚠ hv_spx_prev が未入力または不正です";
        return;
    }

    // --- 送信 ---
    fetch("/api/predict_strategy", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    })
    .then(async r => {
        const json = await r.json();
        if (!r.ok) throw json;
        return json;
    })
    .then(result => {
        // --- ML結果の表示 ---
        const bull = (result.bull_call_spread !== undefined)
            ? Number(result.bull_call_spread).toFixed(2)
            : "N/A";

        const bear = (result.bear_put_spread !== undefined)
            ? Number(result.bear_put_spread).toFixed(2)
            : "N/A";

        const condor = (result.iron_condor !== undefined)
            ? Number(result.iron_condor).toFixed(2)
            : "N/A";

        const best = result.best_strategy || "N/A";

        document.getElementById("predictResultBox").innerHTML = `
<b>【LightGBM 推論結果（3変数版）】</b><br><br>
bull_call_spread：${bull}<br>
bear_put_spread：${bear}<br>
iron_condor：${condor}<br><br>
<b>推奨戦略：${best}</b><br><br>
timestamp: ${result.timestamp}<br>
request_id: ${result.request_id}
        `;
    })
    .catch(err => {
        document.getElementById("predictResultBox").innerHTML =
            "⚠ サーバエラー：" + JSON.stringify(err);
    });
}


window.onload = async () => {
    await loadPrice();
};
</script>

</body>
</html>

"""

# ============================================================
# 12) ルート（HTML返却）
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)
