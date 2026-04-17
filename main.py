from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.services import audio_split, lang_detection, denoise, denoise_librosa
from app.services.output import save_output
import os
import time 
import shutil
import uuid

# Configuration
AUDIO_DATA_DIR = "./data/audio"
CLEANED_DIR  = "data/cleaned" 
CHUNKS_DIR = "./data/chunks"
OUTPUT_DIR = "./data/outputs"

os.makedirs(AUDIO_DATA_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 50)  
    print("  Vocal Extraction API - Starting Up")
    print("=" * 50)
    yield  


app = FastAPI(
    title="Vocal Extraction API",
    description="Upload audio and extract vocals using Demucs",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve output files statically
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def root():
    return {"message": "Vocal Extraction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/extract-vocals")
async def extract_vocals(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Generate a unique ID for this processing job
    job_id = str(uuid.uuid4())
    temp_filename = f"{job_id}_{file.filename}"
    input_path = os.path.join(AUDIO_DATA_DIR, temp_filename)

    # Save the uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")

    try:
        # Step: Extract Vocals
        print(f"\n Processing job {job_id}: {file.filename}")
        start_time = time.time()
        
        # Call the denoise service which extracts vocals
        # The service returns the path to the cleaned file
        cleaned_path = denoise.denoise_audio(input_path, OUTPUT_DIR)
        
        latency = time.time() - start_time
        print(f"   Extraction latency : {latency:.3f}s")

        # Get the relative path for the URL
        result_filename = os.path.basename(cleaned_path)
        result_url = f"/outputs/{result_filename}"

        return {
            "job_id": job_id,
            "original_filename": file.filename,
            "result_url": result_url,
            "processing_time": f"{latency:.2f}s"
        }

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Optional: cleanup input file if needed
        # os.remove(input_path)
        pass


from fastapi.responses import FileResponse

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )