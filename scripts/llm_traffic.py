import logging
import json
import argparse

import numpy as np
import requests
import pandas as pd
from litgpt import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse the arguments for the script.

    Returns:
        The arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description="Generate fake traffic data using a simulated LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-1_5",
        help="The model to use.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/penguins.csv",
        help="The path to the data file.",
    )
    parser.add_argument(
        "--target-uri",
        type=str,
        default="http://127.0.0.1:8080/invocations",
        help="The target URI for the model.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=150,
        help="The number of samples to generate.",
    )
    return parser.parse_args()


def get_prompt(data: pd.DataFrame) -> str:
    """Get the prompt for the model.

    Args:
        data: A pandas DataFrame with the data to use for the prompt.

    Returns:
        The prompt for the model.
    """
    return f"""
    You are given a json file several entries. Your task is to generate another json
    file with several fake entries based on the schema of the first file. For the
    fields that have string values, use the values from the original file to generate
    fake values. For the fields that have numeric values, generate fake values that are
    within the range of the original values. Output only the json file, no other text.
    Do not include any other text in your response.

    The original file is:
    {data.to_json(orient="records")}
    """


def get_mock_response(prompt: str, n_samples: int = 10) -> list[dict]:
    """Get a mock response for the model.

    Args:
        prompt: The prompt for the model. It will only be used to seed the random number
            generator.
        n_samples: The number of samples to generate.

    Returns:
        A list of dictionaries with the mock response.
    """
    rng = np.random.default_rng(abs(hash(prompt)))
    return [
        {
            "species": rng.choice(["Adelie", "Chinstrap", "Gentoo"]),
            "island": rng.choice(["Dream", "Biscoe", "Torgersen"]),
            "culmen_length_mm": 52.3 + rng.standard_normal() * 1.0,
            "culmen_depth_mm": 19.3 + rng.standard_normal() * 1.0,
            "flipper_length_mm": 194.0 + rng.standard_normal() * 10.0,
            "body_mass_g": 3650.0 + rng.standard_normal() * 100.0,
            "sex": rng.choice(["MALE", "FEMALE"]),
        }
        for _ in range(n_samples)
    ]


def get_payload(text: list[dict]) -> dict:
    """Get the payload for the model.

    Args:
        text: A list of dictionaries with the text to use for the payload.

    Returns:
        A dictionary with the payload.
    """
    payload = {}
    payload["inputs"] = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()} for row in text
    ]
    payload["params"] = {"data_capture": True}
    return payload


def post_request(payload: dict, target_uri: str) -> dict:
    """Post a request to the model.

    Args:
        payload: A dictionary with the payload.

    Returns:
        A dictionary with the response.
    """
    predictions = requests.post(
        url=target_uri,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=5,
    )
    return predictions.json()


def main():
    logger.info("Parsing arguments...")
    args = parse_arguments()
    logger.info("Arguments parsed.")

    logger.info("Loading data...")
    data = pd.read_csv(args.data_path)
    logger.info("Data loaded.")

    logger.info("Loading model...")
    model = LLM.load(args.model)  # noqa: F841
    logger.info("Model loaded.")

    logger.info("Generatin prompt...")
    # The model output is not reliable so we use a mock response.
    prompt = get_prompt(data.sample(30, random_state=42))
    logger.info("Prompt generated.")
    # text = model.generate(prompt)
    logger.info("Generating fake traffic data...")
    text = get_mock_response(prompt, n_samples=args.n_samples)
    logger.info("Fake traffic data generated.")

    logger.info("Sending fake traffic data to the model...")
    payload = get_payload(text)
    response = post_request(payload, args.target_uri)
    logger.info(
        "Fake traffic data sent to the model and successfully registered "
        f"{len(response['predictions'])} predictions.",
    )


if __name__ == "__main__":
    main()
