import io
from contextlib import asynccontextmanager

# from http import HTTPStatus
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import Counter, Histogram, Summary, make_asgi_app

# Error metrics
error_counter = Counter("prediction_error", "Number of prediction errors")
request_counter = Counter("prediction_requests", "Number of prediction requests")
request_size = Histogram("predict_request_size", "Predict request size by shape")
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
review_summary = Summary("review_length_summary", "Review length summary")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model_session, input_names, output_names, idx2labels
    # Load onnx model
    provider_list = ["CUDAExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"]
    model_session = rt.InferenceSession("resnet18_model.onnx", providers=provider_list)
    input_names = [i.name for i in model_session.get_inputs()]
    output_names = [i.name for i in model_session.get_outputs()]

    idx2labels = np.load("label_converter.npy", allow_pickle=True).item()

    # run application
    yield

    # Clean up
    del model_session
    del input_names
    del output_names
    del idx2labels


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


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
    predicted_p = e_x.T / e_x.sum(axis=1)
    predicted_i = np.argsort(predicted_p, axis=0)[::-1][:3]
    predicted_c = predicted_i[0]

    prediction = []
    prob = {}
    for i in range(len(predicted_c)):
        prediction.append(idx2labels[predicted_c[i]])
        [print(j) for j in predicted_i[:, i]]
        prob[i] = {idx2labels[int(j)]: round(predicted_p[j, i].item() * 100, 4) for j in predicted_i[:, i]}

    return prob, prediction  # output.softmax(dim=-1)


# FastAPI endpoint for image classification
@app.post("/classify/")
async def classify_image(img_files: list[UploadFile] = list[File(...)]):
    """Classify image endpoint."""
    request_counter.inc()
    try:
        img_arr = []
        name_arr = []
        for file in img_files:
            byte_img = await file.read()
            img = Image.open(io.BytesIO(byte_img))
            request_size.observe(img.size)
            img = img.resize((224, 224))
            img = ((img - np.mean(img)) / np.std(img)).astype(np.float32)
            img_arr.append(img)
            name_arr.append(file.filename)
        img_arr = np.asarray(img_arr)
        img_arr = img_arr.transpose(0, 3, 1, 2)  # > batch,3,244,244
        probabilities, prediction = predict_image(img_arr)
        return {"filename": name_arr, "prediction": prediction, "probabilities": probabilities}
    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
