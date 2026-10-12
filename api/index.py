
import json
import joblib
import pandas as pd
from http.server import BaseHTTPRequestHandler

MODEL_PATH = "housing_model.pkl"
FEATURES_PATH = "feature_columns.pkl"

EXPECTED_FEATURES = [
    "House_Size",
    "bedrooms",
    "bathrooms",
    "House Age",
    "Distance to City Center (miles)",
    "Floors"
]

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 10000:
                return self.send_json(400, {"error": "Invalid request body"})

            payload = json.loads(self.rfile.read(length))

            features = joblib.load(FEATURES_PATH)
            if features != EXPECTED_FEATURES:
                return self.send_json(
                    500, {"error": "Model feature names do not match"}
                )

            values = []
            for name in EXPECTED_FEATURES:
                value = float(payload[name])
                if not pd.notna(value):
                    raise ValueError("Values must be finite numbers")
                values.append(value)

            model = joblib.load(MODEL_PATH)
            input_df = pd.DataFrame([values], columns=EXPECTED_FEATURES)
            prediction = float(model.predict(input_df)[0])

            return self.send_json(200, {"prediction": prediction})

        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return self.send_json(400, {"error": "Please enter valid numeric values"})
        except Exception:
            return self.send_json(500, {"error": "Prediction service error"})

    def do_GET(self):
        return self.send_json(405, {"error": "Use POST to request a prediction"})

    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
