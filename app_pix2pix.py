
from pathlib import Path
import re

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from flask import (
    Flask, abort, flash, redirect, render_template,
    request, send_from_directory, url_for
)
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 256
BASE_FILTERS = 16

CHECKPOINT_DIR = ROOT / "checkpoints"
UPLOAD_DIR = ROOT / "upload"
GENERATED_DIR = ROOT / "generated"
TEMPLATE_DIR = ROOT / "templates"

for folder in (UPLOAD_DIR, GENERATED_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
app.secret_key = "local-sketch-app-change-this-key"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def build_generator(base=BASE_FILTERS):
    inputs = tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = inputs
    skips = []

    for i, multiplier in enumerate([1, 2, 4, 8, 8, 8, 8]):
        x = tf.keras.layers.Conv2D(
            base * multiplier, 4, strides=2,
            padding="same", use_bias=(i == 0)
        )(x)

        if i != 0:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

        if i < 6:
            skips.append(x)

    for skip, multiplier, dropout in zip(
        reversed(skips),
        [8, 8, 8, 4, 2, 1],
        [0.5, 0.5, 0, 0, 0, 0],
    ):
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(
            base * multiplier, 4, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        if dropout:
            x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(
        3, 4, padding="same", activation="tanh"
    )(x)

    return tf.keras.Model(inputs, outputs, name="generator")


def find_latest_weights():
    files = list(
        CHECKPOINT_DIR.glob("generator_epoch_*.weights.h5")
    )

    if not files:
        raise FileNotFoundError(
            f"No generator checkpoints found in {CHECKPOINT_DIR}"
        )

    def epoch_number(path):
        match = re.search(
            r"generator_epoch_(\d+)\.weights\.h5$", path.name
        )
        return int(match.group(1)) if match else -1

    return max(files, key=epoch_number)


def load_model():
    weights = find_latest_weights()
    model = build_generator()
    model.load_weights(weights)
    print(f"Loaded generator checkpoint: {weights.name}")
    return model


def preprocess_sketch(path):
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32)

    return array / 127.5 - 1.0


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/models/<model_name>", methods=["GET", "POST"])
def models(model_name):
    if model_name != "sketch2faces":
        abort(404)

    if request.method == "GET":
        return render_template(
            "generator_page.html",
            model_name="Sketch to Faces",
            images=False,
            generated_image=None,
            original_image=None,
        )

    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        flash("Please choose a face sketch.")
        return redirect(request.url)

    extension = Path(uploaded.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        flash("Please upload a PNG, JPG, JPEG, WEBP or BMP image.")
        return redirect(request.url)

    safe_name = secure_filename(uploaded.filename) or "sketch.png"
    original_path = UPLOAD_DIR / safe_name
    output_path = GENERATED_DIR / "sample.png"

    try:
        with Image.open(uploaded.stream) as image:
            image.convert("RGB").save(original_path, format="PNG")

        input_array = preprocess_sketch(original_path)[None, ...]
        prediction = generator(
            input_array, training=False
        ).numpy()[0]

        prediction = np.clip(
            (prediction + 1.0) * 127.5, 0, 255
        ).astype(np.uint8)

        Image.fromarray(prediction, mode="RGB").save(output_path)

    except Exception as exc:
        app.logger.exception("Image generation failed")
        flash(f"Could not generate image: {exc}")
        return redirect(request.url)

    return render_template(
        "generator_page.html",
        model_name="Sketch to Faces",
        images=True,
        generated_image=url_for("generated", filename="sample.png"),
        original_image=url_for("uploaded", filename=safe_name),
    )


@app.route("/generated/<path:filename>")
def generated(filename):
    return send_from_directory(GENERATED_DIR, filename)


@app.route("/upload/<path:filename>")
def uploaded(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    generator = load_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
