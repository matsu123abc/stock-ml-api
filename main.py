from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import datetime
import uuid
from typing import List

app = FastAPI(title="stock-ml-api (API→HTML→JS single-file)")

# =========================
# 1) API モデル / ロジック（先頭に配置）
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

class LogMarketStateRequest(MarketState):
    chosen_strategy: str | None = None
    note: str | None = None

class LogMarketStateResponse(BaseModel):
    log_id: str
    saved_at: datetime.datetime

# 簡易メモリストレージ（デモ用）
MARKET_LOGS: List[LogMarketStateRequest] = []

def dummy_ml_predict(m: MarketState) -> tuple[str, float]:
    """
    ダミー判定。実運用ではここをクラスタリング／モデル推論に差し替える。
    戦略ラベル例: spread_hold, short_close, long_only, no_trade
    """
    if m.atm_iv > 0.3 and m.days_to_expiry <= 7:
        return "short_close", 0.8
    if m.atm_iv < 0.15 and m.days_to_expiry >= 20:
        return "spread_hold", 0.75
    if m.hv_20d > 0.25 and m.otm_iv > 0.3:
        return "long_only", 0.7
    return "no_trade", 0.6

@app.post("/predict-strategy", response_model=PredictStrategyResponse)
def predict_strategy(market: MarketState):
    """
    入力：
      - 株価, ATM IV, OTM IV, ガンマ, デルタ, 残存日数, HV
    出力：
      - strategy (ラベル), confidence (0-1), timestamp, request_id
    """
    strategy, confidence = dummy_ml_predict(market)
    return PredictStrategyResponse(
        strategy=strategy,
        confidence=confidence,
        timestamp=datetime.datetime.utcnow(),
        request_id=str(uuid.uuid4())
    )

@app.post("/log-market-state", response_model=LogMarketStateResponse)
def log_market_state(log: LogMarketStateRequest):
    """
    学習用ログを蓄積するエンドポイント。
    入力：市場状態 + chosen_strategy + note
    """
    MARKET_LOGS.append(log)
    return LogMarketStateResponse(
        log_id=str(uuid.uuid4()),
        saved_at=datetime.datetime.utcnow()
    )

# =========================
# 2) HTML（APIの後に配置）
# =========================

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>stock-ml-api UI</title>
    <style>
        body { font-family: Arial; margin: 24px; }
        .row { margin: 6px 0; }
        label { display:inline-block; width:140px; }
        input { width:180px; padding:6px; }
        button { padding:8px 14px; margin-right:8px; }
        #result { margin-top:18px; white-space:pre-wrap; background:#f7f7f7; padding:12px; border-radius:6px; }
    </style>
</head>
<body>
    <h2>構造判断型ML UI（手入力）</h2>

    <div class="row"><label>株価</label><input id="stock_price" type="number" step="0.01"></div>
    <div class="row"><label>ATM IV (%)</label><input id="atm_iv" type="number" step="0.01"></div>
    <div class="row"><label>OTM IV (%)</label><input id="otm_iv" type="number" step="0.01"></div>
    <div class="row"><label>ガンマ</label><input id="gamma" type="number" step="0.0001"></div>
    <div class="row"><label>デルタ</label><input id="delta" type="number" step="0.01"></div>
    <div class="row"><label>残存日数</label><input id="days_to_expiry" type="number"></div>
    <div class="row"><label>HV (%)</label><input id="hv_20d" type="number" step="0.01"></div>

    <div style="margin-top:12px;">
        <button onclick="predict()">推論する</button>
        <button onclick="openLogDialog()">ログ保存（選択）</button>
    </div>

    <div id="result"></div>

    <!-- JS は下に埋め込み（次セクション） -->
    <script>
"""

# =========================
# 3) JS（HTMLの直後に配置）
# =========================

INDEX_HTML += """
function getInputData() {
    return {
        stock_price: parseFloat(document.getElementById("stock_price").value) || 0,
        atm_iv: (parseFloat(document.getElementById("atm_iv").value) || 0) / 100,
        otm_iv: (parseFloat(document.getElementById("otm_iv").value) || 0) / 100,
        gamma: parseFloat(document.getElementById("gamma").value) || 0,
        delta: parseFloat(document.getElementById("delta").value) || 0,
        days_to_expiry: parseInt(document.getElementById("days_to_expiry").value) || 0,
        hv_20d: (parseFloat(document.getElementById("hv_20d").value) || 0) / 100
    };
}

function renderResult(text) {
    document.getElementById("result").innerText = text;
}

function predict() {
    const data = getInputData();
    fetch("/predict-strategy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(async res => {
        if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || "HTTP error " + res.status);
        }
        return res.json();
    })
    .then(result => {
        const out = [
            "【推論結果】",
            "戦略: " + result.strategy,
            "信頼度: " + result.confidence,
            "時刻: " + result.timestamp,
            "ID: " + result.request_id,
            "",
            "【入力JSON】",
            JSON.stringify(data, null, 2)
        ].join("\\n");
        renderResult(out);
    })
    .catch(err => {
        renderResult("エラー: " + err.message);
    });
}

function openLogDialog() {
    // 簡易プロンプトで chosen_strategy と note を取得してログ保存
    const chosen = prompt("実際に選んだ戦略ラベルを入力してください（例: spread_hold, short_close, long_only, no_trade）", "");
    if (chosen === null) return;
    const note = prompt("任意メモ（理由など）", "");
    const data = getInputData();
    const payload = Object.assign({}, data, { chosen_strategy: chosen, note: note });
    fetch("/log-market-state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(async res => {
        if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || "HTTP error " + res.status);
        }
        return res.json();
    })
    .then(r => {
        renderResult("ログ保存完了\\nlog_id: " + r.log_id + "\\n保存時刻: " + r.saved_at);
    })
    .catch(err => {
        renderResult("ログ保存エラー: " + err.message);
    });
}
</script>
</body>
</html>
"""

# =========================
# 4) ルート（HTML返却）
# =========================

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    """
    ルートは HTML を直接返す（テンプレートファイルを生成しないため Azure 安全）。
    構成順は API → HTML → JS（ユーザの慣れに合わせた順序）。
    """
    return HTMLResponse(content=INDEX_HTML, status_code=200)
