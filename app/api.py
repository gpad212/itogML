# app/api.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.model import predict_from_csv

app = FastAPI(title="URL Classifier")

templates = Jinja2Templates(directory="app/templates")
os.makedirs("/app/tmp", exist_ok=True)
os.makedirs("/app/output", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    # Проверка расширения
    filename = file.filename
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Пожалуйста, загрузите CSV файл.")
    tmp_path = os.path.join("/app/tmp", filename)
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        out = predict_from_csv(tmp_path)
    except Exception as e:
        # вернуть полезную ошибку
        raise HTTPException(status_code=500, detail=str(e))
    return FileResponse(out, filename=f"predictions_{filename}", media_type="text/csv")
