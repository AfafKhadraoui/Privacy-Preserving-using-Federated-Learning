# How to Run the Federated Learning Face Recognition System

This guide explains how to run the full end-to-end system including the React Frontend and the FastAPI Backend.

## 1. Prerequisites
- Python 3.11 installed.
- Node.js and npm installed.

## 2. Start the Backend API (Linking Layer)
The backend manages face registration, recognition, and the Federated Learning simulation.

1. Open a terminal in the project root folder.
2. Activate the virtual environment:
   - **Windows PowerShell**: `.\.venv\Scripts\Activate.ps1`
   - **Windows CMD**: `call .venv\Scripts\activate.bat`
3. Set your desired Privacy Version:
   - **Version 1 (Server Embedding Storage)**: 
     - PowerShell: `$env:PRIVACY_VERSION="1"`
     - CMD: `set PRIVACY_VERSION=1`
   - **Version 2 (Local Embedding Storage)**: 
     - PowerShell: `$env:PRIVACY_VERSION="2"`
     - CMD: `set PRIVACY_VERSION=2`
4. Start the server on port 5000:
   `uvicorn linking.api:app --reload --port 5000`

## 3. Start the Local Client Edge API
This acts as the local processor on the user's edge device.
1. Open a *second* terminal in the project root folder.
2. Activate the virtual environment.
3. Start the client edge API on port 5001:
   `uvicorn linking.client.client_api:app --reload --port 5001`

## 4. Start the React Frontend
1. Open a *second* terminal in the `frontend` folder.
2. Install dependencies (if you haven't already):
   `npm install`
3. Start the React app:
   `npm start`
4. The browser will open automatically to `http://localhost:3000`.

## 4. How to Use the System
1. **Register**: Go to the "Register" tab in the React app, enter your name, scan your face, and click "Register". Your face tensor is saved locally for FL, and an embedding is saved (either locally or globally based on the Privacy Version).
2. **Recognize**: Go to the "Recognize" tab, scan your face, and click "Recognize Face". The system will match your face against the stored embeddings and tell you who you are!
3. **Federated Update**: Over time, as more users register, you can trigger a global federated learning round to improve the model by calling the `/api/federated-update` endpoint. You can monitor the results of these FL rounds in the "FL Results" tab.
