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
    # Correction latence Render (soustrait le délai serveur ~200ms)
latence = float(data.get("latence_ms", 50))
latence_corrigee = max(20.0, latence - 200.0)

feature_map = {
    "RSRP":                  rsrp,
    "RSRQ":                  rsrq,
    "SINR":                  sinr,
    "RSSI":                  rssi,
    "Latence_ms":            latence_corrigee,  # ← corrigée
    "Jitter_ms":             float(data.get("jitter_ms", 5)),
    "Packet_Loss_%":         float(data.get("perte_paquets_pct", 0)),
    "Download_Mbps":         float(data.get("download_mbps", 8)),
    "Upload_Mbps":           float(data.get("upload_mbps", 2)),
    "Signal_Strength_Level": level,
    "Network_Type_enc":      net_val,
    "Operator_enc":          op_val,
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Encoder Network_Type
        try:
            net_val = float(net_encoder.transform(
                [str(data.get("Network_Type", "4G LTE"))])[0])
        except:
            net_map = {"5G NR": 3, "4G LTE": 2, "3G HSPA": 1, "2G": 0}
            net_val = float(net_map.get(
                str(data.get("Network_Type", "4G LTE")), 2))

        # Encoder Operator
        try:
            op_val = float(op_encoder.transform(
                [str(data.get("Operator", "Mobilis"))])[0])
        except:
            op_map = {"Djezzy": 0, "Mobilis": 1, "Ooredoo": 2}
            op_val = float(op_map.get(
                str(data.get("Operator", "Mobilis")), 1))

        # RSRP reçu — valeur réelle 4G ou 5G
        rsrp = float(data.get("RSRP", -85))
        rsrq = float(data.get("RSRQ", -10))
        sinr = float(data.get("SINR", 10))

        # Si valeurs -999 (pas de signal) → utilise valeurs moyennes
        if rsrp <= -140 or rsrp == -999:
            rsrp = -95.0
        if rsrq <= -20 or rsrq == -999:
            rsrq = -12.0
        if sinr <= -23 or sinr == -999:
            sinr = 8.0

        # RSSI estimé depuis RSRP
        rssi = rsrp + 20.0

        # Niveau signal depuis RSRP
        if rsrp > -80:    level = 4.0
        elif rsrp > -90:  level = 3.0
        elif rsrp > -100: level = 2.0
        elif rsrp > -110: level = 1.0
        else:             level = 0.0

        feature_map = {
            "RSRP":                  rsrp,
            "RSRQ":                  rsrq,
            "SINR":                  sinr,
            "RSSI":                  rssi,
            "Latence_ms":            float(data.get("latence_ms",        50)),
            "Jitter_ms":             float(data.get("jitter_ms",          5)),
            "Packet_Loss_%":         float(data.get("perte_paquets_pct",  0)),
            "Download_Mbps":         float(data.get("download_mbps",      8)),
            "Upload_Mbps":           float(data.get("upload_mbps",        2)),
            "Signal_Strength_Level": float(data.get("Signal_Strength_Level", level)),
            "Network_Type_enc":      net_val,
            "Operator_enc":          op_val,
        }

        features  = [feature_map.get(col, 0.0) for col in feature_cols]
        x         = np.array(features).reshape(1, -1)
        x_scaled  = scaler.transform(x)
        pred      = model.predict(x_scaled)[0]
        proba     = model.predict_proba(x_scaled)[0]
        etat      = le.inverse_transform([pred])[0]
        probas    = {
            le.classes_[i]: float(proba[i])
            for i in range(len(le.classes_))
        }

        return jsonify({
            "etat_reseau":   etat,
            "probabilites":  probas,
            "features_used": feature_map
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
