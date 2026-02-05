let chart;
let lastPrediction = null;

// Index symbols mapping
const INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "SENSEX"];

// DOM elements
const searchInput = document.getElementById("search");
const result = document.getElementById("result");
const alertBox = document.getElementById("alertBox");
let ALL = [...INDEX_SYMBOLS];

// Load additional symbols from backend
fetch("http://127.0.0.1:8001/symbols")
    .then(r => r.json())
    .then(d => {
        ALL = ALL.concat(d);
    });

// --------------------
// Predict
// --------------------
function predict() {

    let s = searchInput.value.trim();

    fetch(`http://127.0.0.1:8001/predict?symbol=${s}`)
        .then(r => r.json())
        .then(d => {

            if (d.error) {
                alert(d.error);
                return;
            }

            lastPrediction = d;

            result.innerText =
`Current: ${d.current_price}
Direction: ${d.direction}
Confidence: ${d.confidence}%
Expected: ${d.expected_price}
High: ${d.high_target}
Low: ${d.low_target}
Safe: ${d.safe_price}`;

            drawChart(d.prices, d.expected_price);
        });
}

// --------------------
// Draw Chart
// --------------------
function drawChart(prices, predicted) {

    if (chart) chart.destroy();

    let labels = prices.map((v, i) => i);
    labels.push("PREDICT");

    let combined = [...prices];
    combined.push(predicted);

    chart = new Chart(document.getElementById("priceChart"), {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                data: combined,
                borderWidth: 2,
                borderColor: "#22c55e",
                pointBackgroundColor: (ctx) =>
                    ctx.dataIndex === combined.length - 1
                        ? "#facc15"
                        : "#22c55e",
                fill: false,
                tension: 0.3
            }]
        },
        options: {
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// --------------------
// Invest
// --------------------
function invest() {

    if (!lastPrediction) {
        alert("Predict first");
        return;
    }

    fetch("http://127.0.0.1:8001/invest", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            symbol: searchInput.value,
            entry_price: lastPrediction.current_price,
            direction: lastPrediction.direction,
            safe: lastPrediction.safe_price,
            high: lastPrediction.high_target,
            low: lastPrediction.low_target
        })
    });

    alert("Position saved. Monitoring started.");
}

// --------------------
// Alert Polling
// --------------------
setInterval(() => {

    fetch("http://127.0.0.1:8001/alerts")
        .then(res => res.json())
        .then(data => {

            if (data.length > 0) {
                alertBox.style.display = "block";
                alertBox.innerText =
                    data[0].symbol + " : " + data[0].message;
            } else {
                alertBox.style.display = "none";
            }

        });

}, 10000);

// --------------------
// Auto Refresh Prediction
// --------------------
setInterval(() => {
    if (searchInput.value) {
        predict();
    }
}, 60000);
