import logging
import os
import requests
import tempfile

import pandas as pd


logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/penguins.csv")
NROWS = 10

URL = "http://0.0.0.0:8080/invocations"
DATA_CAPTURE = False  # TODO: Specify params with CSV file input


def main():
    logging.basicConfig(level=logging.INFO)
    data = pd.read_csv(DATA_PATH, nrows=NROWS)
    data.drop(columns=["species"], inplace=True)
    logger.info("Data loaded with %d rows.", len(data))

    with tempfile.NamedTemporaryFile(delete=True, suffix=".csv") as tmp_file:
        data.to_csv(tmp_file.name, index=False)
        logger.info(f"Data saved to {tmp_file.name}")

        response = requests.post(
            URL,
            data=tmp_file,
            headers={"Content-Type": "text/csv"},
        )
        logger.info(
            "Sent request for %d rows with data capture=%s, got response: %s",
            NROWS,
            DATA_CAPTURE,
            response.json(),
        )


if __name__ == "__main__":
    main()
