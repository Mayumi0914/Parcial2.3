from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import shutil
from pathlib import Path
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

app = FastAPI()

# Directorio para guardar archivos subidos
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Cargar el modelo y el procesador de Hugging Face
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Abrir la imagen
    image = Image.open(file_path)
    
    # Procesar la imagen y hacer la predicción
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    
    # Eliminar el archivo después de procesar
    file_path.unlink()
    
    return JSONResponse({"filename": file.filename, "predicted_class": predicted_class})

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

@app.get("/get-image/{filename}")
async def get_image(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="image/jpeg")
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
