import io
import pickle
from contextlib import asynccontextmanager
from http import HTTPStatus

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model_session, input_names, output_names, idx2labels
    # Load onnx model
    provider_list = ["CUDAExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
    model_session = rt.InferenceSession("models/cnn_model.onnx", providers=provider_list)
    input_names = [i.name for i in model_session.get_inputs()]
    output_names = [i.name for i in model_session.get_outputs()]

    idx2labels = np.load(f"data/processed/cards-dataset/label_converter.npy", allow_pickle=True).item()

    # run application
    yield

    # Clean up
    del model_session
    del input_names
    del output_names
    del idx2labels


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


def predict_image(img) -> list[str]:
    """Predict image class (or classes) given image and return the result."""
    batch = {input_names[0]: img}
    output = model_session.run(output_names, batch)[0]

    # get probabilities
    e_x = np.exp(output - np.max(output))
    predicted_p = (e_x.T / e_x.sum(axis=1)).max(axis=0)
    predicted_idx = np.argmax(output, axis=1)

    labels = []
    for label_idx in predicted_idx:
        labels.append(idx2labels[label_idx])
    return predicted_p, labels  # output.softmax(dim=-1)


# FastAPI endpoint for image classification
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        byte_img = await file.read()
        img = Image.open(io.BytesIO(byte_img)).resize((224, 224))
        img = ((img - np.mean(img)) / np.std(img)).astype(np.float32)
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        img = img.transpose(0, 3, 1, 2)  # > batch,3,244,244
        probabilities, prediction = predict_image(img)
        return {"filename": file.filename, "prediction": prediction, "probabilities": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500) from e


# from enum import Enum
# import re
# import json
# class ItemEnum(Enum):
#     alexnet = "alexnet"
#     resnet = "resnet"
#     lenet = "lenet"

# app = FastAPI()

# @app.get("/")
# def root():
#     """ Health check."""
#     response = {
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#     }
#     return response

# # @app.get("/items/{item_id}")
# # def read_item(item_id: int):
# #     return {"item_id": item_id}

# # @app.get("/restric_items/{item_id}")
# # def read_item(item_id: ItemEnum):
# #     return {"item_id": item_id}

# @app.get("/query_items")
# def read_item(item_id: int):
#     return {"item_id": item_id}

# database = {'username': [ ], 'password': [ ]}

# @app.post("/login/")
# def login(username: str, password: str):
#     username_db = database['username']
#     password_db = database['password']
#     if username not in username_db and password not in password_db:
#         with open('database.csv', "a") as file:
#             file.write(f"{username}, {password} \n")
#         username_db.append(username)
#         password_db.append(password)
#     return "login saved"

# @app.get("/text_model/")
# def contains_email(data: str):
#     regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#     response = {
#         "input": data,
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#         "regex": regex,
#         "is_email": re.fullmatch(regex,data) is not None
#     }
#     return response

# from fastapi import UploadFile, File
# from typing import Optional
