import joblib, os
bundles = ["enhanced_ensemble_bundle.pkl", "sbert_ensemble_bundle.pkl", "model_bundle.pkl"]
for b in bundles:
    if os.path.exists(b):
        d = joblib.load(b)
        acc = d.get("accuracy", "N/A")
        auc = d.get("auc_roc", "N/A")
        print(f"{b}: accuracy={acc} ({float(acc)*100:.2f}% if numeric)  AUC-ROC={auc}")
