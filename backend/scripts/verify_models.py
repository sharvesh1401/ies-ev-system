"""
Verify Model Files — CLI script.

Checks whether trained model files exist in the models/ directory
and attempts to load them for validation.

Usage:
    python scripts/verify_models.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("=" * 60)
    print("  MODEL FILE VERIFICATION")
    print("=" * 60)

    from pathlib import Path

    models_dir = Path("models")
    print(f"\nModels directory: {models_dir.absolute()}")
    print(f"Directory exists: {models_dir.exists()}\n")

    # Check expected files
    expected_files = [
        ("energy_predictor.pth", "PyTorch DNN", "~8 MB"),
        ("driver_classifier.pth", "PyTorch LSTM", "~3 MB"),
        ("traffic_estimator.pkl", "Scikit-learn RF", "~1 MB"),
        ("metrics.json", "Training metrics", "<1 KB"),
    ]

    found_count = 0
    for filename, desc, size in expected_files:
        filepath = models_dir / filename
        if filepath.exists():
            actual_size = filepath.stat().st_size
            print(f"  ✓ {filename:30s} ({desc}) — {actual_size:,} bytes")
            found_count += 1
        else:
            print(f"  ✗ {filename:30s} ({desc}) — NOT FOUND")

    print(f"\nFound {found_count}/{len(expected_files)} model files.\n")

    # Try loading via ModelLoader
    print("-" * 60)
    print("Attempting to load models...\n")

    try:
        from app.ml.model_loader import ModelLoader

        loader = ModelLoader(models_dir=str(models_dir))
        load_result = loader.load_all()

        print("Load results:")
        for model_name, loaded in load_result["loaded"].items():
            status = "✓ Loaded" if loaded else "✗ Not loaded"
            print(f"  {model_name:25s} {status}")

        if load_result["errors"]:
            print("\nErrors:")
            for name, err in load_result["errors"].items():
                print(f"  {name}: {err}")

        # Verify loaded models
        if any(load_result["loaded"].values()):
            print("\n" + "-" * 60)
            print("Verifying loaded models...\n")
            verify_result = loader.verify_models()
            for name, info in verify_result.items():
                if isinstance(info, dict) and "status" in info:
                    print(f"  {name:25s} {info['status']}")
        else:
            print("\nNo models loaded — system will run in physics-only mode.")

    except ImportError as e:
        print(f"Could not import ModelLoader: {e}")
        print("This is OK — Python dependencies may not be installed.")

    print("\n" + "=" * 60)
    print("  VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
