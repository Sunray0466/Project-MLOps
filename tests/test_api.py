import io
import json
import os
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
    project_dir = os.getcwd()
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
        f"img_files=@{project_dir}/tests/images/kingofhearts.jpg;type=image/jpeg",
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0

    response_json = json.loads(result.stdout)

    # Assert probabilities are <= 1
    for prob_key, prob_value in response_json["probabilities"]["0"].items():
        assert prob_value <= 100.0, f"Probability {prob_value} exceeds 1.0"
        assert prob_value >= 0.0, f"Probability {prob_value} is negative"

    # Additional assertions on the response
    assert response_json["filename"] == ["kingofhearts.jpg"]
    assert response_json["prediction"] == ["king of hearts"]
    assert len(response_json["probabilities"]) > 0
