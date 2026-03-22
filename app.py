from flask import Flask, request, jsonify
import pickle, numpy as np, os

app = Flask(__name__)

with open("xgb_model.pkl",            "rb") as f: model        = pickle.load(f)
with open("scaler.pkl",               "rb") as f: scaler       = pickle.load(f)
with open("label_encoder.pkl",        "rb") as f: le           = pickle.load(f)
with open("network_type_encoder.pkl", "rb") as f: net_encoder  = pickle.load(f)
with open("operator_encoder.pkl",     "rb") as f: op_encoder   = pickle.load(f)
with open("feature_cols.pkl",         "rb") as f: feature_cols = pickle.load(f)

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "classes": list(le.classes_)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ── Valeurs radio ─────────────────────────────────────
        rsrp = float(data.get("RSRP", -90))
        rsrq = float(data.get("RSRQ", -12))
        sinr = float(data.get("SINR",   8))

        # Détecter si on est sur WiFi (pas de signal mobile)
        # SS-RSRP = -999 → téléphone sur WiFi ou PC
        is_wifi = (rsrp <= -140 or rsrp == -999 or rsrp == -999.0)

        if is_wifi:
            # ── Mode WiFi : évaluer uniquement avec performance réseau
            latence  = float(data.get("latence_ms",        50))
            jitter   = float(data.get("jitter_ms",          5))
            perte    = float(data.get("perte_paquets_pct",  0))
            download = float(data.get("download_mbps",      8))
            upload   = float(data.get("upload_mbps",        2))

            # Calculer score de qualité WiFi
            score = 0

            # Latence
            if latence < 50:   score += 3
            elif latence < 150: score += 2
            elif latence < 300: score += 1
            else:               score += 0

            # Download
            if download > 30:  score += 3
            elif download > 5:  score += 2
            elif download > 1:  score += 1
            else:               score += 0

            # Upload
            if upload > 10:    score += 2
            elif upload > 2:    score += 1
            else:               score += 0

            # Jitter
            if jitter < 10:    score += 2
            elif jitter < 30:   score += 1
            else:               score += 0

            # Perte paquets
            if perte < 1:      score += 2
            elif perte < 5:     score += 1
            else:               score += 0

            # Score total max = 12
            # Bon >= 9, Moyen 5-8, Mauvais < 5
            if score >= 9:
                etat = "Bon"
                probas = {"Bon": 0.85, "Mauvais": 0.05, "Moyen": 0.10}
            elif score >= 5:
                etat = "Moyen"
                probas = {"Bon": 0.15, "Mauvais": 0.15, "Moyen": 0.70}
            else:
                etat = "Mauvais"
                probas = {"Bon": 0.05, "Mauvais": 0.85, "Moyen": 0.10}

            return jsonify({
                "etat_reseau":   etat,
                "probabilites":  probas,
                "mode":          "WiFi",
                "score_wifi":    score
            })

        else:
            # ── Mode Mobile : utilise le vrai signal + XGBoost ──
            # Corriger valeurs aberrantes
            if rsrp <= -140: rsrp = -95.0
            if rsrq <= -20:  rsrq = -12.0
            if sinr <= -23:  sinr =   8.0

            rssi  = rsrp + 20.0
            level = (4 if rsrp > -80 else
                     3 if rsrp > -90 else
                     2 if rsrp > -100 else
                     1 if rsrp > -110 else 0)

            # Encoder Network_Type
            try:    net_val = float(net_encoder.transform([str(data.get("Network_Type","4G LTE"))])[0])
            except: net_val = 2.0

            # Encoder Operator
            try:    op_val = float(op_encoder.transform([str(data.get("Operator","Mobilis"))])[0])
            except: op_val = 1.0

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

            features = [feature_map.get(col, 0.0) for col in feature_cols]
            x        = np.array(features).reshape(1, -1)
            x_scaled = scaler.transform(x)
            pred     = model.predict(x_scaled)[0]
            proba    = model.predict_proba(x_scaled)[0]
            etat     = le.inverse_transform([pred])[0]
            probas   = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}

            return jsonify({
                "etat_reseau":   etat,
                "probabilites":  probas,
                "mode":          "Mobile"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
