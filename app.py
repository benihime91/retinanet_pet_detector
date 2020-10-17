import argparse
import warnings
from typing import *

import numpy as np
import streamlit as st
from PIL import Image

from references.display_preds import Visualizer
from references.utils import get_model, get_preds, label_dict, load_yaml_config

warnings.filterwarnings("ignore")

# do computation on CPU
DEVICE = "cpu"
# Instantiate the visualizer
viz = Visualizer(label_dict)

def load_model(args: argparse.Namespace):
    " loads in the pre-trained RetinaNet Model "
    model = get_model(args)
    return model


# Loads in the user Input Image
def load_image() -> np.array:
    "Loads in an image using the streamlit API"
    _prompt_ = "Choose a png or jpg image"
    uploaded_image = st.file_uploader(_prompt_, type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.markdown("## Uploaded Image")
        # Make sure image is RGB
        image = Image.open(uploaded_image).convert("RGB")
        # Convert PIL image to array
        image = np.array(image)
        # Display the uploaded Image
        st.image(image, use_column_width=True)
        return image


# Draw prediciton on the given image
def draw_preds_on_image(uploaded_image, boxes, labels, scores):
    "draws predicitons on the Given Image"
    image = viz.draw_bboxes(
        uploaded_image, boxes, labels, scores, save=False, show=False, return_fig=True
    )
    return image


def _init_app() -> None:
    """
    Fn that Initializes the App.
    Prints some Fancy Stuff to StreamLit page, Display demo Image and train data stats
    """
    # set title
    st.title("Detecting Pet Faces 👁 🐶 🐱",)
    st.markdown(
        "This application detects the faces of some common Pet Breeds using a **RetinaNet**."
    )
    st.write("## How does it work?")
    st.write(
        "Upload an image of a pet (cat or dog) and the app will draw the dounding box where it detects the objects:"
    )
    # Display demo Image
    st.image(
        Image.open("images/res_3.png"), caption="Example", use_column_width=True,
    )
    st.write("## Upload your own image")
    st.write(
        "**Note:** The model has been trained on pets breeds given in the `The Oxford-IIIT Pet Dataset`"
        " and therefore will only with those kind of images."
    )
    st.markdown("**To be more precise the model has been trained on these breeds:**")
    # Show Train data Statistics
    st.image(Image.open("images/breed_count.jpg"), use_column_width=True,)


def main() -> None:
    # Start the app
    _init_app()

    # Load in the Image
    image = load_image()
    st.sidebar.markdown("# Set Detection Parameters")
    # select model architecture
    st.sidebar.markdown("**Select Model Architeture:**")
    model_arch = st.sidebar.selectbox(
        " Currently supports RetinaNet model with resnet50 & resnet34 backbone :",
        ("resnet50", "resnet34"),
    )
    ps = {"resnet50": "config/resnet50.yaml", "resnet34": "config/resnet34.yaml"}
    _path = ps[model_arch]

    # Set Score threshold
    st.sidebar.markdown("**Score threshold to filter detections**")
    score_threshold = st.sidebar.slider(
        label="Detections with score < score_threshold are discarded",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
    )

    # Set IOU Threshold
    st.sidebar.markdown("**iou threshold for detection bounding boxes**")
    nms_thres = st.sidebar.slider(label="IOU threshold", min_value=0.1, max_value=1.0, value=0.4,)

    # Set maximum detections
    st.sidebar.markdown("**Max number of bounding boxes to be detected in the given Image**" )
    md = st.sidebar.slider(label="max detections", min_value=1, max_value=500, value=100,)

    # general usage Info
    st.sidebar.header("General Usage Info:")
    st.sidebar.subheader("**Many bounding boxes generated ?**")
    st.sidebar.markdown(f"* **try increasing the `score threshold`**\n * **reducing the `IOU threshold`**")
    st.sidebar.subheader("**Bounding boxes not being detected ?**")
    st.sidebar.markdown(
        f"* **try reducing the `score threshold`**\n * **increasing the `IOU threshold`** \n * **select a different `model architecture` from above given list**"
    )

    # If image is Upladed make predictions when button is clicked
    if image is not None:
        st.info("You can modify the default detection parameters using the left side-bar !")
        if st.button("Generate predictions"):
            # Load the model and set model parameters
            _prompt_ = "Loading model ... It might take some time to download the model if using for the 1st time.."
            conf_dict = load_yaml_config(_path)
            conf_dict["score_thres"] = score_threshold
            conf_dict["nms_thres"] = nms_thres
            conf_dict["max_detections"] = md

            args = argparse.Namespace(**conf_dict)

            with st.spinner(_prompt_):
                # Load in the model
                model = get_model(args=args)

            with st.spinner("Generating results ... "):
                # Get instance predictions for the uploaded Image
                bb, lb, sc = get_preds(model, image)

            # Print Results
            st.markdown("**Results:**")
            st.markdown(
                f"* **Number of bounding boxes detected in Image** : {len(bb)}\n"
                f"* **Breeds detected in Image** : {[label_dict[idx] for idx in lb]}"
            )
            with st.spinner("Writing results on the Image:"):
                # Draw on predictions to image
                res = draw_preds_on_image(image, bb, lb, sc)
                st.write(res)


if __name__ == "__main__":
    # run the app
    main()
