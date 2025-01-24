import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2

st.markdown(
    """
<style>
    .stAppHeader {
        display: none;
    }
    .stMainBlockContainer {
        height: 300vh;
        max-width: 60rem;
        background: linear-gradient(90deg,#050,#0a0 20% 80%,#050);
        padding: 6rem 6rem 10rem;
    }
    .st-key-upload_button {
        text-align: right;
    }
    .stImage {
        padding: 20px;
        border: solid 5px white;
        border-radius: 20px;
        aspect-ratio: 4 / 5;
        align-content: center;
        &>div>div {
            color: white;
            font-size: 20px;
            font-weight: bold;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/mlops-448220/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri
    name = os.environ.get("backend", None)
    return name


def classify_image(image_files, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    response = requests.post(
        predict_url, files=[("img_files", file) for file in image_files] + [("is_byte", True)], timeout=10
    )
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)
    st.title("Card classifier")

    uploaded_file = st.file_uploader("Upload card(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    ucl, ucr = st.columns([0.85, 0.15])
    pressed = ucr.button("Upload", key="upload_button")

    if pressed and len(uploaded_file) != 0:
        bimgs = [file.read() for file in uploaded_file]
        result = classify_image(bimgs, backend=backend)
        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            for (
                img,
                prob,
            ) in probabilities.items():
                lc, rc = st.columns([0.35, 0.65], gap="small")
                img = int(img)
                lc.image(bimgs[img], caption="Card " + str(img + 1))
                rc.subheader(
                    f"Prediction: {prediction[img]}",
                )

                # make a nice bar chart
                data = prob.items()
                df = pd.DataFrame(data)
                df.rename({0: "Card", 1: "Probabilities"}, axis=1, inplace=True)
                rc.altair_chart(
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x="Probabilities",
                        y=alt.X("Card", sort=None),
                    )
                    .configure(background="#0005")
                )
        else:
            ucl.text("Failed to get prediction!")
    elif pressed:
        ucl.text("No images uploaded!")


if __name__ == "__main__":
    main()
