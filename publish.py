import os
from argparse import ArgumentParser
from pathlib import Path
from subprocess import CalledProcessError, check_output
from typing import Tuple
import re
import uuid

import replicate

# This is the location train.py puts the weights it downloads
DEFAULT_WEIGHTS_PATH = "weights/data/500"


MODEL_VERSION_LOG_PATTERN = re.compile("^.*?@sha256:(?P<model_version>.*)$")


def publish(model_name: str, weights: Path) -> None:
    print(f"Publishing dreambooth model to https://replicate.com/{model_name}")

    api_token = os.getenv("REPLICATE_API_TOKEN")
    if api_token is None:
        print(
            "Failed to publish model:\n"
            "  Replicate API token not set. Grab your token from\n"
            "  https://replicate.com/account and set it as an environment variable:\n\n"
            "    export REPLICATE_API_TOKEN=<your-token>"
        )
        exit(1)

    # if not _cog_installed() and not _install_cog():
    #     print("Failed to publish model:\n" "  Cog is not installed.")
    #     exit(1)

    # TODO use this to check if the user is in the beta
    if not _cog_login(api_token):
        print("Failed to publish model:\n" "  Could not log in to Replicate with Cog.")
        exit(1)

    if not _model_exists(model_name):
        print(
            "Failed to publish model:\n"
            f"  Model `{model_name}` doesn't exist on Replicate. Visit\n"
            "  https://replicate.com/create to create it. (You'll need beta access\n"
            "  to do this. If you don't have access, email us at team@replicate.com)"
        )
        exit(1)

    # TODO move the weights into the right place

    tag_name = str(uuid.uuid4())

    if not _build_model(model_name, tag_name):
        exit(1)

    if not _push_model(model_name, tag_name):
        exit(1)

    model_version = _get_model_version(model_name, tag_name)
    if model_version is None:
        exit(1)
    print(f"Model version: {model_version}")

    if not _remove_local_image(model_name, tag_name):
        exit(1)

    exit(0)


# TODO: this doesn't work on 4.14.294-220.533.amzn2.aarch64
def _cog_installed() -> bool:
    _, success = _run_command(["command", "-v", "cog"])
    return success


def _cog_login(api_token: str) -> None:
    _, success = _run_command(
        ["cog", "login", "--token-stdin"], input=api_token.encode()
    )
    return success


def _install_cog() -> None:
    # TODO offer to install cog
    return False


def _model_exists(model_name: str) -> bool:
    try:
        model = replicate.models.get(model_name)
        # this will 404 if the model doesn't exist
        versions = model.versions.list()
        return True
    except Exception:
        return False


def _build_model(model_name, tag_name) -> bool:
    _, success = _run_command(["cog", "build", "-t", f"r8.im/{model_name}:{tag_name}"])
    return success


def _push_model(model_name, tag_name) -> bool:
    _, success = _run_command(["docker", "push", f"r8.im/{model_name}:{tag_name}"])
    return success


def _remove_local_image(model_name, tag_name) -> bool:
    _, success = _run_command(
        ["docker", "image", "rm", f"r8.im/{model_name}:{tag_name}"]
    )
    return success


def _get_model_version(model_name, tag_name) -> str:
    command = f"docker inspect --format='{{{{index .RepoDigests 0}}}}' r8.im/{model_name}:{tag_name}"
    output, success = _run_command(command, shell=True)
    if not success:
        return None

    output = output.decode("utf8").strip()
    match = MODEL_VERSION_LOG_PATTERN.match(output)
    if match is None:
        return None
    return match.group("model_version")


def _run_command(*args, **kwargs) -> Tuple[any, bool]:
    try:
        output = check_output(*args, **kwargs)
        return output, True
    except CalledProcessError:
        return None, False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--weights", "-w")
    args = parser.parse_args()

    if args.weights:
        weights = Path(args.weights)
    else:
        weights = Path(DEFAULT_WEIGHTS_PATH)

    publish(model_name=args.model_name, weights=weights)
