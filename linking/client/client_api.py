import os
import tempfile
import shutil
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from linking.client.client_service import process_and_register_locally, recognize_face
import config as proj_cfg

app = FastAPI(title="FL Client Edge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/client/register")
def client_register(name: str = Form(...), images: list[UploadFile] = File(...)):
    """
    Simulates a local client registering a face.
    Takes registration images from the web UI, processes them locally using
    the global model, and sends only the embedding update to the central API.
    """
    temp_dir = None
    try:
        name = name.strip()
        if not name:
            raise ValueError("Name is required.")
        if len(images) != proj_cfg.NUM_REGISTRATION_IMAGES:
            raise ValueError(
                f"Expected {proj_cfg.NUM_REGISTRATION_IMAGES} registration images, "
                f"received {len(images)}."
            )

        temp_dir = tempfile.mkdtemp()
        temp_paths = []
        for idx, image in enumerate(images):
            temp_path = os.path.join(temp_dir, f"{name}_{idx}.jpg")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            temp_paths.append(temp_path)

        process_and_register_locally(name, temp_paths)
        
        return {
            "status": "success",
            "message": f"Client {name} registered locally with {len(temp_paths)} images."
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/client/registration-config")
def registration_config():
    return {
        "num_registration_images": proj_cfg.NUM_REGISTRATION_IMAGES,
        "instructions": list(proj_cfg.REGISTRATION_IMAGE_INSTRUCTIONS),
    }

@app.post("/client/recognize")
def client_recognize(image: UploadFile = File(...)):
    """
    Simulates a local client recognizing a face.
    Takes image from the web UI, gets global model locally,
    generates embedding and sends it to central /api/recognize.
    """
    temp_dir = None
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
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
