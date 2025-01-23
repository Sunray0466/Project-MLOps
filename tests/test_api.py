import io
import json
import subprocess

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from project_mlops.api import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from the backend!"}


def test_classify_image_invalid_file():
    # Send invalid data as a file
    response = client.post(
        "/classify/",
        files={"file": ("test_invalid.txt", b"Not an image file", "text/plain")},
    )
    assert response.status_code == 500


def test_classify_image():
    curl_command = [
        "curl",
        "-X",
        "POST",
        "https://backend-474989323251.europe-west1.run.app/classify/",
        "-H",
        "accept: application/json",
        "-H",
        "Content-Type: multipart/form-data",
        "-F",
        "file=@/Users/veroonika/VisualStudioCodeProjects/dtu/Project-MLOps/tests/images/kingofhearts.jpg;type=image/jpeg",
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0

    response_json = json.loads(result.stdout)

    # Assert probabilities are <= 1
    for prob in response_json["probabilities"]:
        assert prob <= 1.0, f"Probability {prob} exceeds 1.0"
        assert prob >= 0.0, f"Probability {prob} is negative"

    # Additional assertions on the response
    assert response_json["filename"] == "kingofhearts.jpg"
    assert response_json["prediction"] == ["king of hearts"]
    assert len(response_json["probabilities"]) > 0
