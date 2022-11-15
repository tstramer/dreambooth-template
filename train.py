from argparse import ArgumentParser
from pathlib import Path
import shutil
from typing import List
import io
import json
import mimetypes
import os
import tempfile
import time
import urllib.request
import zipfile


import replicate

MODEL_NAME = "replicate/dreambooth"
MODEL_VERSION = "053b31c5f2a2648a37485ca0f0d543aa22c082cf4fdfceee689d7bfcce2b4013"


def train(class_prompt, instance_prompt, training_data, optional_train_args):
    print(f"Training on Replicate using model: {MODEL_NAME}@{MODEL_VERSION}")
    print(f"https://replicate.com/{MODEL_NAME}/versions/{MODEL_VERSION}")
    model = replicate.models.get(MODEL_NAME)
    version = model.versions.get(MODEL_VERSION)

    # TODO generate class images using regular stable diffusion, in parallel

    prediction = replicate.predictions.create(
        version=version,
        input={
            "instance_prompt": instance_prompt,
            "class_prompt": class_prompt,
            "instance_data": open(training_data, "rb"),
            **(optional_train_args or {}),
        },
    )

    # TODO print details about the prediction -- ID, a curl command to cancel it, tail the logs
    print(
        f"Training run starting. ID: {prediction.id}\n"
        "  To cancel the run, use the following command:\n\n"
        "  curl -s -X POST \\\n"
        '       -H "Authorization: Token $REPLICATE_API_TOKEN" \\\n'
        f"       https://api.replicate.com/v1/predictions/{prediction.id}/cancel"
    )

    while prediction.status == "starting":
        time.sleep(1)
        prediction.reload()

    print("Training started")

    logs = ""
    while prediction.status == "processing":
        time.sleep(5)
        prediction.reload()
        # print any new logs
        if prediction.logs:
            new_logs = prediction.logs[len(logs) :]
            if new_logs:
                print(new_logs)
                logs = prediction.logs

    if prediction.status != "succeeded":
        print(
            f"Training failed: id={prediction.id}, status={prediction.status}, error={prediction.error}"
        )
        exit(1)

    weights_url = prediction.output
    print(f"Done training. Weights saved at: {weights_url}")

    # download and unzip trained weights
    # need to set user-agent header to avoid 403
    req = urllib.request.Request(weights_url, headers={"User-Agent": "python"})
    with urllib.request.urlopen(req) as response, open("weights.zip", "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    with zipfile.ZipFile("weights.zip", "r") as zip_ref:
        zip_ref.extractall("weights")

    # TODO: download generated weights


def _is_image(file: Path) -> bool:
    file_type, _ = mimetypes.guess_type(file)
    return file_type and file_type.startswith("image/")


def _zip_images(images: List[Path]) -> str:
    print(f"Bundling {len(images)} training images into training data ZIP")

    z = os.path.join(tempfile.gettempdir(), "data.zip")
    with zipfile.ZipFile(z, "w") as zip:
        for file in images:
            # write the file into the archive under the path data/
            # TODO: put them at the root of the archive so it's easier to make by hand
            zip.write(file, os.path.join("data", os.path.basename(file)))

    return z


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--class-prompt",
        "-c",
        required=True,
        help=(
            "The prompt to specify images in the same class as provided instance images. "
            "For example, you might use 'a photo of a dog'."
        ),
    )
    parser.add_argument(
        "--instance-prompt",
        "-p",
        required=True,
        help=(
            "The prompt with an identifier for the subject in your training data. "
            "For example, you might use 'a photo of sks dog' as your instance prompt. "
            "Then you could generate outputs with prompts like 'a photo of sks dog wearing a tuxedo'."
        ),
    )

    # multiple ways to provide training data
    parser.add_argument(
        "--images", "-i", nargs="*", help="A list of images to use to train your model."
    )
    parser.add_argument(
        "--instance-data",
        "-d",
        help="A ZIP file of images to use to train your model. The images should be in the root.",
    )

    # optional training args
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=500,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--prior-loss-weight",
        type=float,
        default=1.0,
        help="Weight of prior preservation loss.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=1.0,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    args = parser.parse_args()

    print(repr(args))

    if args.instance_data and args.images:
        print("Only one of instance-data or image should be provided")
        parser.print_help()
        exit(1)
    elif args.instance_data:
        training_data = args.instance_data
    elif args.images:
        training_data = _zip_images(args.images)
    else:
        print("No training data provided, looking for images in data/")
        images = [os.path.join("data", f) for f in os.listdir("data") if _is_image(f)]
        print(repr(images))
        if len(images) == 0:
            print("No training data provided via arguments, and none found in data/")
            exit(1)
        training_data = _zip_images(images)

    optional_train_args = {
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "max_train_steps": args.max_train_steps,
        "num_train_epochs": args.num_train_epochs,
        "prior_loss_weight": args.prior_loss_weight,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    optional_train_args = {
        k: v for k, v in optional_train_args.items() if v is not None
    }

    train(
        class_prompt=args.class_prompt,
        instance_prompt=args.instance_prompt,
        training_data=training_data,
        optional_train_args=optional_train_args,
    )
