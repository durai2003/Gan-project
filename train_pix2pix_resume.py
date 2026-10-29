
import argparse
import random
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMAGE_SIZE = 256
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_pairs(root, split):
    photo_dir = root / split / "photos"
    sketch_dir = root / split / "sketches"

    if not photo_dir.is_dir() or not sketch_dir.is_dir():
        raise FileNotFoundError(
            f"Missing photo or sketch folder for {split}: {root / split}"
        )

    photos = {
        p.name: p for p in photo_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }
    sketches = {
        p.name: p for p in sketch_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }

    names = sorted(set(photos) & set(sketches))
    print(f"{split}: {len(names)} matched pairs")

    if not names:
        raise ValueError(f"No matched photo/sketch filenames in {split}")

    return [(sketches[n], photos[n]) for n in names]


def read_image(path, size=IMAGE_SIZE):
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (size, size),
            method=Image.Resampling.BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32)

    return array / 127.5 - 1.0


def load_batch(pairs, indices, augment=True):
    sketches, photos = [], []

    for index in indices:
        sketch_path, photo_path = pairs[index]

        sketch = read_image(sketch_path)
        photo = read_image(photo_path)

        # Apply the same flip to both members of the pair.
        if augment and random.random() < 0.5:
            sketch = np.fliplr(sketch).copy()
            photo = np.fliplr(photo).copy()

        sketches.append(sketch)
        photos.append(photo)

    return np.stack(sketches), np.stack(photos)


def build_generator(base=16):
    inputs = tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = inputs
    skips = []

    # Encoder
    for i, multiplier in enumerate([1, 2, 4, 8, 8, 8, 8]):
        filters = base * multiplier
        x = tf.keras.layers.Conv2D(
            filters, 4, strides=2, padding="same",
            use_bias=(i == 0)
        )(x)

        if i != 0:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

        if i < 6:
            skips.append(x)

    # Decoder
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


def build_discriminator(base=16):
    sketch_input = tf.keras.layers.Input(
        (IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    photo_input = tf.keras.layers.Input(
        (IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    x = tf.keras.layers.Concatenate()(
        [sketch_input, photo_input]
    )

    for i, multiplier in enumerate([1, 2, 4, 8]):
        x = tf.keras.layers.Conv2D(
            base * multiplier,
            4,
            strides=2,
            padding="same",
            use_bias=(i == 0),
        )(x)

        if i != 0:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    outputs = tf.keras.layers.Conv2D(
        1, 4, padding="same"
    )(x)

    return tf.keras.Model(
        [sketch_input, photo_input], outputs,
        name="discriminator"
    )


def save_examples(generator, pairs, output_path, count=3):
    canvas = Image.new(
        "RGB", (IMAGE_SIZE * 3, IMAGE_SIZE * min(count, len(pairs)))
    )

    for row, (sketch_path, photo_path) in enumerate(pairs[:count]):
        sketch = read_image(sketch_path)
        photo = read_image(photo_path)

        generated = generator(
            np.expand_dims(sketch, axis=0), training=False
        ).numpy()[0]

        def to_pil(array):
            array = np.clip((array + 1) * 127.5, 0, 255)
            return Image.fromarray(array.astype(np.uint8))

        y = row * IMAGE_SIZE
        canvas.paste(to_pil(sketch), (0, y))
        canvas.paste(to_pil(photo), (IMAGE_SIZE, y))
        canvas.paste(to_pil(generated), (IMAGE_SIZE * 2, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print("Saved comparison:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="data/paired_dataset"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=5)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Load the latest saved generator/discriminator weights and continue with additional epochs.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.steps_per_epoch < 0:
        parser.error("steps-per-epoch cannot be negative")

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    root = Path(args.dataset)
    train_pairs = get_pairs(root, "train")
    val_pairs = get_pairs(root, "validation")

    print("TensorFlow:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())
    print("Building GAN...")

    generator = build_generator(args.base_filters)
    discriminator = build_discriminator(args.base_filters)

    checkpoint_dir = Path("checkpoints")
    output_dir = Path("training_outputs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume:
        gen_files = {
            int(m.group(1)): p
            for p in checkpoint_dir.glob("generator_epoch_*.weights.h5")
            if (m := re.search(r"generator_epoch_(\d+)\.weights\.h5$", p.name))
        }
        disc_files = {
            int(m.group(1)): p
            for p in checkpoint_dir.glob("discriminator_epoch_*.weights.h5")
            if (m := re.search(r"discriminator_epoch_(\d+)\.weights\.h5$", p.name))
        }
        common_epochs = sorted(set(gen_files) & set(disc_files))
        if not common_epochs:
            raise SystemExit(
                f"--resume was requested but no matching generator/discriminator weights were found in {checkpoint_dir.resolve()}"
            )
        last_epoch = common_epochs[-1]
        generator.load_weights(gen_files[last_epoch])
        discriminator.load_weights(disc_files[last_epoch])
        start_epoch = last_epoch + 1
        print(f"Resuming model weights from epoch {last_epoch}.")
        print("Note: optimizer momentum/state is restarted; only model weights are resumed.")

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def train_step(sketches, real_photos):
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake_photos = generator(sketches, training=True)

            real_logits = discriminator(
                [sketches, real_photos], training=True
            )
            fake_logits = discriminator(
                [sketches, fake_photos], training=True
            )

            gan_loss = bce(tf.ones_like(fake_logits), fake_logits)
            l1_loss = tf.reduce_mean(
                tf.abs(real_photos - fake_photos)
            )
            g_loss = gan_loss + 100.0 * l1_loss

            d_real = bce(tf.ones_like(real_logits), real_logits)
            d_fake = bce(tf.zeros_like(fake_logits), fake_logits)
            d_loss = (d_real + d_fake) / 2.0

        g_grads = gt.gradient(
            g_loss, generator.trainable_variables
        )
        d_grads = dt.gradient(
            d_loss, discriminator.trainable_variables
        )

        gen_optimizer.apply_gradients(
            zip(g_grads, generator.trainable_variables)
        )
        disc_optimizer.apply_gradients(
            zip(d_grads, discriminator.trainable_variables)
        )

        return g_loss, gan_loss, l1_loss, d_loss

    for epoch in range(start_epoch, start_epoch + args.epochs):
        order = list(range(len(train_pairs)))
        random.shuffle(order)

        full_steps = (
            len(order) + args.batch_size - 1
        ) // args.batch_size

        steps = (
            full_steps if args.steps_per_epoch == 0
            else min(args.steps_per_epoch, full_steps)
        )

        print(f"\nEpoch {epoch}/{args.epochs}; steps={steps}")

        for step in range(steps):
            indices = order[
                step * args.batch_size:
                (step + 1) * args.batch_size
            ]
            if not indices:
                break

            sketches, photos = load_batch(
                train_pairs, indices, augment=True
            )

            losses = train_step(
                tf.convert_to_tensor(sketches),
                tf.convert_to_tensor(photos),
            )

            values = [float(loss.numpy()) for loss in losses]
            print(
                f"Step {step + 1}/{steps}: "
                f"G={values[0]:.4f}, "
                f"GAN={values[1]:.4f}, "
                f"L1={values[2]:.4f}, "
                f"D={values[3]:.4f}"
            )

        generator.save_weights(
            checkpoint_dir
            / f"generator_epoch_{epoch:03d}.weights.h5"
        )
        discriminator.save_weights(
            checkpoint_dir
            / f"discriminator_epoch_{epoch:03d}.weights.h5"
        )

        save_examples(
            generator,
            val_pairs,
            output_dir / f"epoch_{epoch:03d}_examples.jpg",
        )

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
