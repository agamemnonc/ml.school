import logging
import logging.config
import os
import sys
import time
from io import StringIO
from pathlib import Path
from glob import glob
from typing import Any, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from keras.models import Model as KerasModel
from metaflow import S3, Parameter, current
import keras

PYTHON = "3.12"

PACKAGES = {
    "scikit-learn": "1.5.2",
    "pandas": "2.2.3",
    "numpy": "2.1.1",
    "keras": "3.6.0",
    "jax[cpu]": "0.4.35",
    "boto3": "1.35.32",
    "packaging": "24.1",
    "mlflow": "2.17.1",
    "setuptools": "75.1.0",
    "requests": "2.32.3",
    "evidently": "0.4.33",
    "azure-ai-ml": "1.19.0",
    "azureml-mlflow": "1.57.0.post1",
    "python-dotenv": "1.0.1",
    "psutil": "6.1.0",
}

TRAINING_EPOCHS = 50
TRAINING_BATCH_SIZE = 32

DEBUG_TRAINING_EPOCHS = 2
DEBUG_N_SPLITS = 2
DEBUG_DATA_FRAC = 0.1


class FlowMixin:
    """Base class used to share code across multiple pipelines."""

    data_path = Parameter(
        "data-path",
        help=(
            "Directory containing CSV files to process. All CSV files will be "
            "concatenated into a single dataset."
        ),
        default="data",
    )

    def load_dataset(self):
        """Load and prepare the dataset.

        When running in production mode, this function reads every CSV file available in
        the supplied S3 location and concatenates them into a single dataframe. When
        running in development mode, this function reads the dataset from the supplied
        string parameter.
        """
        import numpy as np

        if current.is_production:
            data_path = os.environ.get("DATA_PATH", self.data_path)

            with S3(s3root=data_path) as s3:
                files = s3.get_all()
                logging.info("Found %d file(s) in remote location", len(files))
                raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
        else:
            # When running in development mode, the raw data is passed as a string,
            # so we can convert it to a DataFrame.
            csv_files = glob(os.path.join(self.data_path, "*.csv"))
            logging.info("Found %d CSV file(s) in local directory", len(csv_files))
            raw_data = [pd.read_csv(f) for f in csv_files]

        data = pd.concat(raw_data, ignore_index=True)

        # Replace extraneous values in the sex column with NaN. We can handle missing
        # values later in the pipeline.
        data["sex"] = data["sex"].replace(".", np.nan)

        # We want to shuffle the dataset. For reproducibility, we can fix the seed value
        # when running in development mode. When running in production mode, we can use
        # the current time as the seed to ensure a different shuffle each time the
        # pipeline is executed.
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(data))

        return data


def packages(*names: str):
    """Return a dictionary of the specified packages and their version.

    This function is useful to set up the different pipelines while keeping the
    package versions consistent and centralized in a single location.
    """
    return {name: PACKAGES[name] for name in names if name in PACKAGES}


def configure_logging():
    """Configure logging handlers and return a logger instance."""
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), [0])],
    )


def build_features_transformer():
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        # We can use the `handle_unknown="ignore"` parameter to ignore unseen categories
        # during inference. When encoding an unknown category, the transformer will
        # return an all-zero vector.
        OneHotEncoder(handle_unknown="ignore"),
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                # We'll apply the numeric transformer to all columns that are not
                # categorical (object).
                make_column_selector(dtype_exclude="object"),
            ),
            (
                "categorical",
                categorical_transformer,
                # We want to make sure we ignore the target column which is also a
                # categorical column. To accomplish this, we can specify the column
                # names we only want to encode.
                ["island", "sex"],
            ),
        ],
    )


def build_model(input_shape, learning_rate=0.01):
    """Build and compile the neural network to predict the species of a penguin."""
    from keras import Input, layers, models, optimizers

    model = models.Sequential(
        [
            Input(shape=(input_shape,)),
            layers.Dense(10, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ],
    )

    model.compile(
        optimizer=optimizers.SGD(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
    )

    return model

@keras.saving.register_keras_serializable(package="MyModels")
class KerasEnsemble(KerasModel):
    def __init__(self, models: list[KerasModel], **kwargs) -> None:
        super().__init__(**kwargs)
        self.models = models
    
    def call(self, inputs, training=None):
        predictions = [model(inputs, training=training) for model in self.models]
        return np.mean(predictions, axis=0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "models": [keras.saving.serialize_keras_object(model) for model in self.models]
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        models_config = config.pop("models")
        models = [keras.saving.deserialize_keras_object(c) for c in models_config]
        return cls(models=models, **config)

def build_ensemble_model(models: list[KerasModel]):
    """Build an ensemble model from a list of models."""
    return KerasEnsemble(models=models)