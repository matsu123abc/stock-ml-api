from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import datetime
import uuid
import os

app = FastAPI(title="stock-ml-api (UI手入力版)")

# =========================
# UIテンプレート生成
# =========================

if not os.path.exists("templates"):
    os.makedirs("templates")

if not os.path.exists("static"):
    os.makedirs("static")

html_ui = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>stock-ml-api UI</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        input { width: 200px; padding: 5px; margin: 5px; }
        button { padding: 10px 20px; margin-top: 10px; }
        #result { margin-top: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h2>構造判断型ML UI（手入力版）</h2>

    <p>市場状態を入力してください：</p>

    <div>
        株価: <input id="stock_price" type="number"><br>
        ATM IV (%): <input id="atm_iv" type="number"><br>
        OTM IV (%): <input id="otm_iv" type="number"><br>
        ガンマ: <input id="gamma" type="number" step="0.0001"><br>
        デルタ: <input id="delta" type="number" step="0.01"><br>
        残存日数: <input id="days_to_expiry" type="number"><br>
        HV (%): <input id="hv_20d" type="number"><br>
    </div>

    <button onclick="predict()">推論する</button>

    <h3>結果</h3>
    <div id="result"></div>

    <script>
        function predict() {
            const data = {
                stock_price: parseFloat(document.getElementById("stock_price").value),
                atm_iv: parseFloat(document.getElementById("atm_iv").value) / 100,
                otm_iv: parseFloat(document.getElementById("otm_iv").value) / 100,
                gamma: parseFloat(document.getElementById("gamma").value),
                delta: parseFloat(document.getElementById("delta").value),
                days_to_expiry: parseInt(document.getElementById("days_to_expiry").value),
                hv_20d: parseFloat(document.getElementById("hv_20d").value) / 100
            };

            fetch("/predict-strategy", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            })
            .then(res => res.json())
            .then(result => {
                document.getElementById("result").innerText =
                    "【推論結果】\\n" +
                    "戦略: " + result.strategy + "\\n" +
                    "信頼度: " + result.confidence + "\\n" +
                    "時刻: " + result.timestamp + "\\n" +
                    "ID: " + result.request_id + "\\n\\n" +
                    "【入力JSON】\\n" + JSON.stringify(data, null, 2);
            });
        }
    </script>

</body>
</html>
"""

with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html_ui)

# =========================
# UI設定
# =========================

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# =========================
# APIモデル
# =========================

class MarketState(BaseModel):
    stock_price: float
    atm_iv: float
    otm_iv: float
    gamma: float
    delta: float
    days_to_expiry: int
    hv_20d: float

class PredictStrategyResponse(BaseModel):
    strategy: str
    confidence: float
    timestamp: datetime.datetime
    request_id: str

# =========================
# ダミーMLロジック
# =========================

def dummy_ml_predict(m: MarketState):
    if m.atm_iv > 0.3 and m.days_to_expiry <= 7:
        return "short_close", 0.8
    if m.atm_iv < 0.15 and m.days_to_expiry >= 20:
        return "spread_hold", 0.75
    if m.hv_20d > 0.25 and m.otm_iv > 0.3:
        return "long_only", 0.7
    return "no_trade", 0.6

# =========================
# 推論API
# =========================

@app.post("/predict-strategy", response_model=PredictStrategyResponse)
def predict_strategy(market: MarketState):
    strategy, confidence = dummy_ml_predict(market)
    return PredictStrategyResponse(
        strategy=strategy,
        confidence=confidence,
        timestamp=datetime.datetime.utcnow(),
        request_id=str(uuid.uuid4())
    )
