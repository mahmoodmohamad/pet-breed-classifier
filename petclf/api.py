import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

from petclf.inference import Predictor

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()
    yield
    state.clear()


app = FastAPI(title="Pet Breed Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        return state["predictor"].predict(io.BytesIO(data))
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(status_code=400, detail="invalid image file") from e
