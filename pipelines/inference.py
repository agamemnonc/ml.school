import json
import logging
import logging.config
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModelContext


class Model(mlflow.pyfunc.PythonModel):
    """A custom model that can be used to make predictions.

    This model implements an inference pipeline with three phases: preprocessing,
    prediction, and postprocessing. The model will optionally store the input requests
    and predictions in a SQLite database.

    The [Custom MLflow Models with mlflow.pyfunc](https://mlflow.org/blog/custom-pyfunc)
    blog post is a great reference to understand how to use custom Python models in
    MLflow.
    """

    def __init__(
        self,
        *,
        data_collection_uri: str | None = "penguins.db",
        data_collection_output_file: str | None = None,
        data_capture: bool = False,
    ) -> None:
        """Initialize the model.

        By default, the model will not collect the input requests and predictions. This
        behavior can be overwritten on individual requests.

        This constructor expects the connection URI to the storage medium where the data
        will be collected. By default, the data will be stored in a SQLite database
        named "penguins" and located in the root directory from where the model runs.
        Alternatively, you can specify the output file for the collected data by using
        the `data_collection_output_file` parameter. The file will be written in JSON
        lines format.

        You can override the location by using the `DATA_COLLECTION_URI` and
        `DATA_COLLECTION_OUTPUT_FILE` environment variables.
        """
        self.data_capture = data_capture
        self.data_collection_uri = data_collection_uri
        self.data_collection_output_file = data_collection_output_file

    def _configure_data_collection(self) -> None:
        """Configure data collection behavior.

        Supports one of the two options, not both:
        - Storing the data in a SQLite database.
        - Writing the data in a JSON lines file.

        If both are specified, an exception will be raised.
        """
        if self.data_collection_uri and self.data_collection_output_file:
            raise ValueError(
                "Both data collection URI and output file are specified. Please specify "
                "only one of them."
            )

        if self.data_collection_uri is not None:
            logging.info("Data collection URI: %s", self.data_collection_uri)
        elif self.data_collection_output_file is not None:
            logging.info(
                "Data collection output file: %s", self.data_collection_output_file
            )

    def load_context(self, context: PythonModelContext) -> None:
        """Load the transformers and the Keras model specified as artifacts.

        This function is called only once as soon as the model is constructed.
        """
        # By default, we want to use the JAX backend for Keras. You can use a different
        # backend by setting the `KERAS_BACKEND` environment variable.
        if not os.getenv("KERAS_BACKEND"):
            os.environ["KERAS_BACKEND"] = "jax"

        import keras

        self._configure_logging()
        logging.info("Loading model context...")

        # If the DATA_COLLECTION_URI environment variable is set, we should use it
        # to specify the database filename. Otherwise, we'll use the default filename
        # specified when the model was instantiated.
        self.data_collection_uri = os.environ.get(
            "DATA_COLLECTION_URI",
            self.data_collection_uri,
        )
        self.data_collection_output_file = os.environ.get(
            "DATA_COLLECTION_OUTPUT_FILE",
            self.data_collection_output_file,
        )
        self._configure_data_collection()

        logging.info("Keras backend: %s", os.environ.get("KERAS_BACKEND"))
        logging.info("Data collection URI: %s", self.data_collection_uri)
        logging.info(
            "Data collection output file: %s", self.data_collection_output_file
        )

        # First, we need to load the transformation pipelines from the artifacts. These
        # will help us transform the input data and the output predictions. Notice that
        # these transformation pipelines are the ones we fitted during the training
        # phase.
        self.features_transformer = joblib.load(
            context.artifacts["features_transformer"],
        )
        self.target_transformer = joblib.load(context.artifacts["target_transformer"])

        # Then, we can load the Keras model we trained.
        self.model = keras.saving.load_model(context.artifacts["model"])

        logging.info("Model is ready to receive requests")

    def predict(
        self,
        context: PythonModelContext,  # noqa: ARG002
        model_input: pd.DataFrame | dict | list,
        params: dict[str, Any] | None = None,
    ) -> list:
        """Handle the request received from the client.

        This method is responsible for processing the input data received from the
        client, making a prediction using the model, and returning a readable response
        to the client.

        The caller can specify whether we should capture the input request and
        prediction by using the `data_capture` parameter when making a request.
        """
        if isinstance(model_input, list | dict):
            model_input = pd.DataFrame(model_input)

        logging.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )

        model_output = []

        transformed_payload = self.process_input(model_input)
        if transformed_payload is not None:
            logging.info("Making a prediction using the transformed payload...")
            predictions = self.model.predict(transformed_payload, verbose=0)

            model_output = self.process_output(predictions)

        # If the caller specified the `data_capture` parameter when making the
        # request, we should use it to determine whether we should capture the
        # input request and prediction.
        if (
            params
            and params.get("data_capture", False) is True
            or not params
            and self.data_capture
        ):
            self.capture(model_input, model_output)

        logging.info("Returning prediction to the client")
        logging.debug("%s", model_output)

        return model_output

    def process_input(self, payload: pd.DataFrame) -> pd.DataFrame:
        """Process the input data received from the client.

        This method is responsible for transforming the input data received from the
        client into a format that can be used by the model.
        """
        logging.info("Transforming payload...")

        # We need to transform the payload using the transformer. This can raise an
        # exception if the payload is not valid, in which case we should return None
        # to indicate that the prediction should not be made.
        try:
            result = self.features_transformer.transform(payload)
        except Exception:
            logging.exception(
                "There was an error processing the payload. Trying to process each row "
                "individually..."
            )
            out = []
            # Use iterrows() to get both index and row as Series
            for _, row in payload.iterrows():
                try:
                    out.append(self.features_transformer.transform(pd.DataFrame([row])))
                except Exception:
                    logging.exception(
                        "There was an error processing a row from the payload."
                    )

            result = np.array(out) if out else np.array([])

        return result

    def process_output(self, output: np.ndarray) -> list:
        """Process the prediction received from the model.

        This method is responsible for transforming the prediction received from the
        model into a readable format that will be returned to the client.
        """
        logging.info("Processing prediction received from the model...")

        result = []
        if output is not None:
            prediction = np.argmax(output, axis=1)
            confidence = np.max(output, axis=1)

            # Let's transform the prediction index back to the
            # original species. We can use the target transformer
            # to access the list of classes.
            classes = self.target_transformer.named_transformers_[
                "species"
            ].categories_[0]
            prediction = np.vectorize(lambda x: classes[x])(prediction)

            # We can now return the prediction and the confidence from the model.
            # Notice that we need to unwrap the numpy values so we can serialize the
            # output as JSON.
            result = [
                {"prediction": p.item(), "confidence": c.item()}
                for p, c in zip(prediction, confidence, strict=True)
            ]

        return result

    def capture(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save the input request and output prediction to the database.

        This method will save the input request and output prediction to a SQLite
        database. If the database doesn't exist, this function will create it.
        """
        logging.info("Storing input payload and predictions in the database...")

        # Let's create a copy from the model input so we can modify the
        # DataFrame before storing it in the database.
        data = model_input.copy()

        # We need to add the current time, the prediction and confidence columns
        # to the DataFrame to store everything together.
        data["date"] = datetime.now(timezone.utc)

        # Let's initialize the prediction and confidence columns with None. We
        # 'll overwrite them later if the model output is not empty.
        data["prediction"] = None
        data["confidence"] = None

        # Let's also add a column to store the ground truth. This column can be
        # used by the labeling team to provide the actual species for the data.
        data["ground_truth"] = None

        # If the model output is not empty, we should update the prediction and
        # confidence columns with the corresponding values.
        if model_output is not None and len(model_output) > 0:
            data["prediction"] = [item["prediction"] for item in model_output]
            data["confidence"] = [item["confidence"] for item in model_output]

        # Let's automatically generate a unique identified for each row in the
        # DataFrame. This will be helpful later when labeling the data.
        data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]

        if self.data_collection_uri is not None:
            connection = None
            try:
                connection = sqlite3.connect(self.data_collection_uri)

                # Save the data to the database.
                data.to_sql("data", connection, if_exists="append", index=False)

            except sqlite3.Error:
                logging.exception(
                    "There was an error saving the input request and output prediction "
                    "in the database.",
                )
            finally:
                if connection:
                    connection.close()

        elif self.data_collection_output_file is not None:
            with open(self.data_collection_output_file, "a") as f:
                # Convert DataFrame to records (list of dicts) and write each row
                for _, row in data.iterrows():
                    # Convert datetime to ISO format string for JSON serialization
                    row_dict = row.to_dict()
                    row_dict["date"] = row_dict["date"].isoformat()
                    json.dump(row_dict, f)
                    f.write("\n")

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys
        from pathlib import Path

        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )
