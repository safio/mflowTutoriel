#!/usr/bin/env python3
"""Start MLflow UI server"""

import subprocess
import sys

if __name__ == "__main__":
    try:
        # Use python -m mlflow ui instead of direct mlflow command
        cmd = [sys.executable, "-m", "mlflow", "ui", "--host", "0.0.0.0", "--port", "5001"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow UI stopped")
    except Exception as e:
        print(f"Error starting MLflow UI: {e}")
        sys.exit(1)