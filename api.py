from fastapi import FastAPI
from pydantic import BaseModel
import cv2
import numpy as np
import tensorflow as tf
from fastapi import UploadFile
from fastapi import File
from fastapi import Form 
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
from fastapi.middleware.cors import CORSMiddleware

import matplotlib.pyplot as plt

from router import generate_answer


app = FastAPI(
    title="AI Lung Disease RAG API",
    description="Medical RAG API using LangChain + Groq + FAISS",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "best_cnn_model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

le = joblib.load(LABEL_ENCODER_PATH)
class_names = le.classes_

print("Classes:" , le.classes_)

IMG_SIZE = 224

## Clahe image processing

def apply_clahe(img):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l,a,b))

    enhanced = cv2.cvtColor(

        enhanced,

        cv2.COLOR_LAB2BGR
    )

    return enhanced

def preprocess_image(image_bytes):

    # BYTES → NUMPY
    nparr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    # READ IMAGE
    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise ValueError("Invalid image")

    # RESIZE
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    # CLAHE
    img = apply_clahe(img)

    # FLOAT
    img = img.astype(np.float32)

    # EFFICIENTNET PREPROCESS
    img = preprocess_input(img)

    # ADD BATCH DIMENSION
    img = np.expand_dims(img, axis=0)

    return img

@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        age: float = Form(...),
        gender: str = Form(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)
        # age = age / 100.0
        # gender = gender.lower()

        # if gender in ["male", "m"]:

        #     gender_value = 1

        # elif gender in ["female", "f"]:

        #     gender_value = 0

        # else:

        #     gender_value = 0.5

        # meta = np.array([

        #     [age, gender_value]

        # ], dtype=np.float32)

        pred = model.predict(
            img,
            verbose=0
        )[0]

        top3_idx = np.argsort(pred)[::-1][:3]

        results = []

        for idx in top3_idx:

            results.append({

                "disease": str(class_names[idx]),

                "confidence": round(
                    float(pred[idx]) * 100,
                    2
                )
            })

            return {
            "success": True,
            "top_predictions": results
        }
    except Exception as e:
        return {

            "success": False,
            "error": str(e)
        }



    

# Initialize FastAPI



# Request Body
class QueryRequest(BaseModel):
    question: str


# Home Route
@app.get("/")
def home():
 
    return {
        "message": "AI Lung Disease RAG API is running successfully"
    }


# RAG Route
@app.post("/ask")
def ask_question(request: QueryRequest):

    user_question = request.question

    answer = generate_answer(user_question)

    return {
        "question": user_question,
        "answer": answer
    }

