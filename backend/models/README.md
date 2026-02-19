# Trained Model Files

Place your trained model files here after exporting from Google Colab.

## Required Files

| File                    | Format  | Description                  | Size (approx) |
| ----------------------- | ------- | ---------------------------- | ------------- |
| `energy_predictor.pth`  | PyTorch | Energy consumption DNN       | ~8 MB         |
| `driver_classifier.pth` | PyTorch | Driver behavior LSTM         | ~3 MB         |
| `traffic_estimator.pkl` | Pickle  | Traffic impact Random Forest | ~1 MB         |
| `metrics.json`          | JSON    | Training validation results  | <1 KB         |

## How to Export from Colab

```python
# In your Colab notebook, after training:

# 1. Save PyTorch models
torch.save(model.state_dict(), 'energy_predictor.pth')
torch.save(classifier.state_dict(), 'driver_classifier.pth')

# 2. Save sklearn model
import pickle
with open('traffic_estimator.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# 3. Save metrics
import json
metrics = {
    "test_mape": 3.2,
    "test_rmse": 0.45,
    "physics_agreement": 96.5,
    "calibration_ece": 0.03
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

Then download the files and place them in this directory.

## Note

The backend works without these files — it falls back to physics-only mode.
When model files are present, the hybrid predictor uses ML for fast predictions.
