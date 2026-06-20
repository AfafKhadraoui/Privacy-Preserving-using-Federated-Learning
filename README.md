# Privacy-Preserving Federated Learning Face Recognition System

This project implements a decoupled, privacy-preserving face recognition system using Federated Learning (FL). The architecture separates the central aggregation server from the edge client to ensure raw biometric data (images) never leaves the user's device.

## Architecture Overview
- **Central FL Server (`port 5000`)**: Orchestrates the Federated Learning pipeline using Flower (`flwr`). It distributes the global model and aggregates weights, but never processes raw images.
- **Edge Client API (`port 5001`)**: Simulates the local user device. It runs the MTCNN face detector and PyTorch model locally to generate embeddings and train local model updates.
- **React Frontend (`port 3000`)**: The Web UI where users interact with the system.
### 🛡️ Differential Privacy Modes
This project supports multiple DP implementations to balance between formal guarantees and hardware constraints:

1. **Opacus DP-SGD (Default)**: Uses the industry-standard Opacus library. 
   - **Note**: Backbone freezing is enabled to prevent OOM on 8GB-16GB RAM machines.
2. **Manual DP-SGD (Lightweight)**: A "from-scratch" implementation of gradient clipping and noise injection.
   - Use this if you want to inspect the math/logic without external library overhead.

**How to switch:**
Modify `config.py`:
```python
DP_MODE = "opacus"     # Standard mode
# OR
DP_MODE = "manual_sgd" # From-scratch mode
```

## 1. Prerequisites
- Python 3.11+ installed.
- Node.js and npm installed.
- Ensure your local Python environment named **`face-recog`** is created and has all requirements installed.

---

## 2. Start the Central FL Server
The backend manages the global model distribution and the Federated Learning simulation.

1. Open a terminal in the project root folder.
2. Activate your `face-recog` environment:
   - **Conda**: `conda activate face-recog`
   - **Venv (Windows PowerShell)**: `.\face-recog\Scripts\Activate.ps1`
   - **Venv (Windows CMD)**: `call face-recog\Scripts\activate.bat`
3. Set your desired Privacy Version:
   - **Version 1 (Server Embedding Storage)**: 
     - PowerShell: `$env:PRIVACY_VERSION="1"`
     - CMD: `set PRIVACY_VERSION=1`
   - **Version 2 (Local Embedding Storage)**: 
     - PowerShell: `$env:PRIVACY_VERSION="2"`
     - CMD: `set PRIVACY_VERSION=2`
4. Start the server:
   ```bash
   uvicorn linking.api:app --reload --port 5000
   ```

---

## 3. Start the Local Client Edge API
This acts as the local processor on the user's edge device, keeping raw photos local.

1. Open a **second** terminal in the project root folder.
2. Activate the `face-recog` environment (same as Step 2).
3. Start the client edge API:
   ```bash
   uvicorn linking.client.client_api:app --reload --port 5001
   ```

---

## 4. Start the React Frontend
This is the user interface.

1. Open a **third** terminal in the `frontend` folder.
2. Install dependencies (first time only):
   ```bash
   npm install
   ```
3. Start the React app:
   ```bash
   npm start
   ```
4. The browser will open automatically to `http://localhost:3000`.

---

## 5. How to Use the System

1. **Register**: Go to the "Register" tab in the React app, enter your name, scan your face, and click "Register". 
   - Your face tensor is saved locally on your device for FL.
   - An embedding is saved (either locally or globally based on the Privacy Version).
2. **Recognize**: Go to the "Recognize" tab, scan your face, and click "Recognize Face". The system will process your face locally and match it against the stored embeddings to identify you.
3. **Federated Update**: Over time, as more users register, you can trigger a global federated learning round. Click **Run Federated Update** in the "FL Results" tab. The system will securely launch the Flower FL engine in the background, aggregate all local updates, and save a stronger global model!
