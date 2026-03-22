from flask import Flask, request, jsonify
import pickle, numpy as np, os

app = Flask(__name__)

with open("xgb_model.pkl",           "rb") as f: model       = pickle.load(f)
with open("scaler.pkl",              "rb") as f: scaler      = pickle.load(f)
with open("label_encoder.pkl",       "rb") as f: le          = pickle.load(f)
with open("network_type_encoder.pkl","rb") as f: net_encoder = pickle.load(f)
with open("operator_encoder.pkl",    "rb") as f: op_encoder  = pickle.load(f)
with open("feature_cols.pkl",        "rb") as f: feature_cols= pickle.load(f)

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status":"ok","classes":list(le.classes_),"features":feature_cols})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Encoder Network_Type
        try:    net_val = float(net_encoder.transform([str(data.get("Network_Type","4G LTE"))])[0])
        except: net_val = 2.0

        # Encoder Operator
        try:    op_val = float(op_encoder.transform([str(data.get("Operator","Mobilis"))])[0])
        except: op_val = 1.0

        feature_map = {
            "RSRP":                  float(data.get("RSRP",                -85)),
            "RSRQ":                  float(data.get("RSRQ",                -10)),
            "SINR":                  float(data.get("SINR",                 10)),
            "RSSI":                  float(data.get("RSSI",                -80)),
            "Latence_ms":            float(data.get("latence_ms",           50)),
            "Jitter_ms":             float(data.get("jitter_ms",             5)),
            "Packet_Loss_%":         float(data.get("perte_paquets_pct",     0)),
            "Download_Mbps":         float(data.get("download_mbps",         8)),
            "Upload_Mbps":           float(data.get("upload_mbps",           2)),
            "Signal_Strength_Level": float(data.get("Signal_Strength_Level", 3)),
            "Network_Type_enc":      net_val,
            "Operator_enc":          op_val,
        }

        features = [feature_map.get(col, 0.0) for col in feature_cols]
        x        = np.array(features).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred     = model.predict(x_scaled)[0]
        proba    = model.predict_proba(x_scaled)[0]
        etat     = le.inverse_transform([pred])[0]
        probas   = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}

        return jsonify({"etat_reseau": etat, "probabilites": probas})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)