from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import datetime
import uuid
import os
import json
import yfinance as yf
import numpy as np
from openai import AzureOpenAI

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
    market_view: str | None = ""   # 自分の市場予想

# ============================================================
# 2) ML推論ロジック
# ============================================================

def ml_predict(m: MarketState):
    if m.atm_iv > 0.3 and m.days_to_expiry <= 7:
        return "short_close", 0.82
    if m.atm_iv < 0.15 and m.days_to_expiry >= 20:
        return "spread_hold", 0.76
    if m.hv_20d > 0.25 and m.otm_iv > 0.3:
        return "long_only", 0.71
    return "no_trade", 0.63

# ============================================================
# 3) GPT推論（AzureOpenAI）
# ============================================================

def gpt_predict(m: MarketState):

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

    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()

        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1

        if json_start == -1 or json_end == -1:
            return None

        json_text = raw[json_start:json_end]
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        return json.loads(json_text)

    except Exception:
        return None

# ============================================================
# 4) GPT→ML 二本立て推論 API
# ============================================================

@app.post("/api/predict_strategy")
def api_predict_strategy(m: MarketState):

    gpt = gpt_predict(m)

    if gpt and gpt.get("strategy"):
        return {
            "source": "GPT",
            "result": gpt,
            "timestamp": datetime.datetime.utcnow(),
            "request_id": str(uuid.uuid4())
        }

    strategy, confidence = ml_predict(m)

    return {
        "source": "ML",
        "strategy": strategy,
        "confidence": confidence,
        "timestamp": datetime.datetime.utcnow(),
        "request_id": str(uuid.uuid4())
    }

# ============================================================
# 5) 株価自動取得 API
# ============================================================

@app.get("/api/price")
def api_price(ticker: str = "^N225"):
    try:
        info = yf.Ticker(ticker).info
        return {
            "price": info.get("regularMarketPrice"),
            "previous_close": info.get("regularMarketPreviousClose")
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 6) HV自動取得 API
# ============================================================

@app.get("/api/hv")
def api_hv(ticker: str = "^N225", days: int = 20):
    try:
        hist = yf.Ticker(ticker).history(period=f"{days+1}d")
        if len(hist) < days + 1:
            return {"hv": None}

        close = hist["Close"].values
        log_returns = np.log(close[1:] / close[:-1])
        hv = float(np.std(log_returns) * np.sqrt(252))
        return {"hv": hv}

    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 7) HTML（スマホ最適化 UI）
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
  #resultBox, #logBox, #hvBox, #autoMarketViewBox{
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

<!-- GPT市場予想（参考） -->
<button onclick="loadAutoMarketView()">GPT市場予想（参考）を取得する</button>
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

<h3>ログ保存</h3>
<button onclick="logState()">ログ保存する</button>
<div id="logBox"></div>

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

    fetch("/api/predict_strategy", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {

        if(result.source === "GPT"){
            const r = result.result;

            document.getElementById("resultBox").innerHTML = `
<b>【GPT市場予想（参考）】</b><br>
${document.getElementById("autoMarketViewBox").innerText || "（未取得）"}<br><br>

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
<b>【ML推論結果】</b><br><br>
strategy: ${result.strategy}<br>
confidence: ${result.confidence}<br>
timestamp: ${result.timestamp}<br>
request_id: ${result.request_id}
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

window.onload = async () => {
    await loadPrice();
};
</script>

</body>
</html>

"""

# ============================================================
# 8) ルート（HTML返却）
# ============================================================

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)
