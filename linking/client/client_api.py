import os
import tempfile
import shutil
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from linking.client.client_service import process_and_register_locally, recognize_face

app = FastAPI(title="FL Client Edge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/client/register")
def client_register(name: str = Form(...), image: UploadFile = File(...)):
    """
    Simulates a local client registering a face.
    Takes image from the web UI, processes it locally using the global model,
    and sends the embedding to the central API.
    """
    try:
        name = name.strip()
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"{name}.jpg")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        embedding = process_and_register_locally(name, [temp_path])
        
        return {
            "status": "success",
            "message": f"Client {name} registered locally and embedding sent to FL server."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/client/recognize")
def client_recognize(image: UploadFile = File(...)):
    """
    Simulates a local client recognizing a face.
    Takes image from the web UI, gets global model locally,
    generates embedding and sends it to central /api/recognize.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "recog.jpg")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
            
        result = recognize_face(temp_path)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
