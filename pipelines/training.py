import logging
import os
from pathlib import Path

import pandas as pd

from common import (
    PYTHON,
    TRAINING_BATCH_SIZE,
    TRAINING_EPOCHS,
    TRAINING_EXPERIMENT_ID,
    DEBUG_N_SPLITS,
    DEBUG_TRAINING_EPOCHS,
    DEBUG_DATA_FRAC,
    DEBUG_EXPERIMENT_ID,
    FlowMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
    configure_logging,
    packages,
    build_ensemble_model,
)
from inference import Model
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    environment,
    project,
    pypi_base,
    resources,
    step,
)
from metaflow.cards import Image

configure_logging()


@project(name="penguins")
@pypi_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "jax[cpu]",
        "boto3",
        "packaging",
        "mlflow",
        "setuptools",
        "python-dotenv",
        "psutil",
    ),
)
class Training(FlowSpec, FlowMixin):
    """Training pipeline.

    This pipeline trains, evaluates, and registers a model to predict the species of
    penguins.
    """

    accuracy_threshold = Parameter(
        "accuracy-threshold",
        help=(
            "Minimum accuracy threshold required to register the model at the end of "
            "the pipeline. The model will not be registered if its accuracy is below "
            "this threshold."
        ),
        default=0.7,
    )

    n_splits = Parameter(
        "n-splits",
        help="Number of splits to use for the cross-validation process.",
        default=5,
    )

    debugging_mode = Parameter(
        "debug",
        help="Run the flow in debug mode.",
        default=False,
        is_flag=True,
    )

    register_only_if_better = Parameter(
        "register-only-if-better",
        help=(
            "Only register the model if its accuracy is better than the best model "
            "registered so far."
        ),
        default=False,
        is_flag=True,
    )

    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": os.getenv(
                "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
                "true",
            ),
        },
    )
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        if self.debugging_mode and current.is_production:
            raise ValueError(
                "Debug mode cannot be used with production mode (--production)"
            )

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true").lower() == "true":
            logging.info("Enabling system metrics logging")
            mlflow.enable_system_metrics_logging()

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        self.data = self.load_dataset()

        if self.debugging_mode:
            self.data = self.data.sample(frac=DEBUG_DATA_FRAC)

        try:
            # Let's start a new MLFlow run to track everything that happens during the
            # execution of this flow. We want to set the name of the MLFlow
            # experiment to the Metaflow run identifier so we can easily
            # recognize which experiment corresponds with each run.
            experiment_id = (
                DEBUG_EXPERIMENT_ID if self.debugging_mode else TRAINING_EXPERIMENT_ID
            )
            run = mlflow.start_run(experiment_id=1, run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        # This is the configuration we'll use to train the model. We want to set it up
        # at this point so we can reuse it later throughout the flow.
        self.training_parameters = {
            "epochs": DEBUG_TRAINING_EPOCHS if self.debugging_mode else TRAINING_EPOCHS,
            "batch_size": TRAINING_BATCH_SIZE,
        }

        # Now that everything is set up, we want to run a cross-validation process
        # to evaluate the model and train a final model on the entire dataset. Since
        # these two steps are independent, we can run them in parallel.
        self.next(self.cross_validation, self.transform)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        from sklearn.model_selection import KFold
        from sklearn.model_selection import train_test_split

        if self.n_splits > 1:
            logging.info("Using %d-fold cross-validation", self.n_splits)
            # We are going to use a 5-fold cross-validation process to evaluate the
            # model, so let's set it up. We'll shuffle the data before splitting it into
            # batches.
            kfold = KFold(
                n_splits=DEBUG_N_SPLITS if self.debugging_mode else self.n_splits,
                shuffle=True,
            )

            # We can now generate the indices to split the dataset into training and
            # test sets. This will return a tuple with the fold number and the training
            # and test indices for each of 5 folds.
            self.folds = list(enumerate(kfold.split(self.data)))
        else:
            logging.info("Using 20%% test size with stratification")
            # If we are not using cross-validation, we'll split the data into training
            # and test sets using a 20% test size and stratify the target column.
            train_idx, test_idx = train_test_split(
                range(len(self.data)), test_size=0.2, stratify=self.data.species
            )
            self.folds = [(0, (train_idx, test_idx))]

        # We want to transform the data and train a model using each fold, so we'll use
        # `foreach` to run every cross-validation iteration in parallel. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.transform_fold, foreach="folds")

    @step
    def transform_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """
        # Let's start by unpacking the indices representing the training and test data
        # for the current fold. We computed these values in the previous step and passed
        # them as the input to this step.
        self.fold, (self.train_indices, self.test_indices) = self.input

        logging.info("Transforming fold %d...", self.fold)

        # We need to turn the target column into a shape that the Scikit-Learn
        # pipeline understands.
        species = self.data.species.to_numpy().reshape(-1, 1)

        # We can now build the SciKit-Learn pipeline to process the target column,
        # fit it to the training data and transform both the training and test data.
        target_transformer = build_target_transformer()
        self.y_train = target_transformer.fit_transform(
            species[self.train_indices],
        )
        self.y_test = target_transformer.transform(
            species[self.test_indices],
        )

        # Finally, let's build the SciKit-Learn pipeline to process the feature columns,
        # fit it to the training data and transform both the training and test data.
        features_transformer = build_features_transformer()
        self.x_train = features_transformer.fit_transform(
            self.data.iloc[self.train_indices],
        )
        self.x_test = features_transformer.transform(
            self.data.iloc[self.test_indices],
        )

        # Store transformers as attributes
        self.features_transformer = features_transformer
        self.target_transformer = target_transformer

        # After processing the data and storing it as artifacts in the flow, we want
        # to train a model.
        self.next(self.train_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @resources(memory=4096)
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        logging.info("Training fold %d...", self.fold)

        # Let's track the training process under the same experiment we started at the
        # beginning of the flow. Since we are running cross-validation, we can create
        # a nested run for each fold to keep track of each separate model individually.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):
            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model from this fold.
            self.mlflow_fold_run_id = run.info.run_id

            # Let's configure the autologging for the training process. Since we are
            # training the model corresponding to one of the folds, we won't log the
            # model itself.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the training data. Notice how we are
            # using the training data we processed and stored as artifacts in the
            # `transform` step.
            self.model = build_model(self.x_train.shape[1])
            self.model.fit(
                self.x_train,
                self.y_train,
                verbose=0,
                **self.training_parameters,
            )

        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data for this fold.
        """
        import mlflow
        import numpy as np
        from sklearn import metrics

        logging.info("Evaluating fold %d...", self.fold)

        # Also store ground truth values and predictions for later analysis
        self.y_true = self.y_test
        self.y_proba = self.model.predict(self.x_test)
        self.y_pred = np.argmax(self.y_proba, axis=1)

        # Let's evaluate the model using the test data we processed and stored as
        # artifacts during the `transform` step.
        self.loss = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=2,
        )

        self.accuracy = metrics.accuracy_score(self.y_true, self.y_pred)
        self.recall = metrics.recall_score(self.y_true, self.y_pred, average="macro")
        self.precision = metrics.precision_score(
            self.y_true, self.y_pred, average="macro"
        )
        self.average_precision = metrics.average_precision_score(
            self.y_true, self.y_proba, average="macro"
        )
        self.auc = metrics.roc_auc_score(
            self.y_true, self.y_proba, average="macro", multi_class="ovr"
        )

        logging.info(
            "Fold %d - loss: %f - accuracy: %f - recall: %f - precision: %f - "
            "average_precision: %f - auc: %f",
            self.fold,
            self.loss,
            self.accuracy,
            self.recall,
            self.precision,
            self.average_precision,
            self.auc,
        )

        # Let's log everything under the same nested run we created when training the
        # current fold's model.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics(
                {
                    "test_loss": self.loss,
                    "test_accuracy": self.accuracy,
                },
            )

        # When we finish evaluating every fold in the cross-validation process, we want
        # to evaluate the overall performance of the model by averaging the scores from
        # each fold.
        self.next(self.evaluate_model)

    @card
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process.

        This function averages the score computed for each individual model to
        determine the final model performance.
        """
        import mlflow
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        # We need access to the `mlflow_run_id` and `mlflow_tracking_uri` artifacts
        # that we set at the start of the flow, but since we are in a join step, we
        # need to merge the artifacts from the incoming branches to make them
        # available.
        self.merge_artifacts(inputs, include=["mlflow_run_id", "mlflow_tracking_uri"])

        # Let's calculate the mean and standard deviation of the accuracy and loss from
        # all the cross-validation folds. Notice how we are accumulating these values
        # using the `inputs` parameter provided by Metaflow.
        metrics = [
            [i.loss, i.accuracy, i.recall, i.precision, i.average_precision, i.auc]
            for i in inputs
        ]

        (
            self.loss,
            self.accuracy,
            self.recall,
            self.precision,
            self.average_precision,
            self.auc,
        ) = np.mean(metrics, axis=0)
        (
            self.loss_std,
            self.accuracy_std,
            self.recall_std,
            self.precision_std,
            self.average_precision_std,
            self.auc_std,
        ) = np.std(metrics, axis=0)

        y_true_all = np.concatenate([i.y_true for i in inputs]).astype(int)
        y_pred_all = np.concatenate([i.y_pred for i in inputs])

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_all, ax=ax)
        ax.set_title("Confusion Matrix")

        current.card.append(Image.from_matplotlib(fig))
        plt.close(fig)

        logging.info("Loss: %f ±%f", self.loss, self.loss_std)
        logging.info("Accuracy: %f ±%f", self.accuracy, self.accuracy_std)
        logging.info("Recall: %f ±%f", self.recall, self.recall_std)
        logging.info("Precision: %f ±%f", self.precision, self.precision_std)
        logging.info(
            "Average Precision: %f ±%f",
            self.average_precision,
            self.average_precision_std,
        )
        logging.info("AUC: %f ±%f", self.auc, self.auc_std)
        # Let's log the model metrics on the parent run.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "cross_validation_accuracy": self.accuracy,
                    "cross_validation_accuracy_std": self.accuracy_std,
                    "cross_validation_loss": self.loss,
                    "cross_validation_loss_std": self.loss_std,
                    "cross_validation_recall": self.recall,
                    "cross_validation_recall_std": self.recall_std,
                    "cross_validation_precision": self.precision,
                    "cross_validation_precision_std": self.precision_std,
                    "cross_validation_average_precision": self.average_precision,
                    "cross_validation_average_precision_std": self.average_precision_std,
                    "cross_validation_auc": self.auc,
                    "cross_validation_auc_std": self.auc_std,
                },
            )

        # Store transformers from first fold for inference
        self.features_transformer = inputs[0].features_transformer
        self.target_transformer = inputs[0].target_transformer

        # Create ensemble from all fold models
        fold_models = [fold.model for fold in inputs]
        self.ensemble_model = build_ensemble_model(fold_models)

        self.next(self.register_model)

    @card
    @step
    def transform(self):
        """Apply the transformation pipeline to the entire dataset.

        This function transforms the columns of the entire dataset because we'll
        use all of the data to train the final model.

        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference.
        """
        # Let's build the SciKit-Learn pipeline to process the target column and use it
        # to transform the data.
        self.target_transformer = build_target_transformer()
        self.y = self.target_transformer.fit_transform(
            self.data.species.to_numpy().reshape(-1, 1),
        )

        # Let's build the SciKit-Learn pipeline to process the feature columns and use
        # it to transform the training.
        self.features_transformer = build_features_transformer()
        self.x = self.features_transformer.fit_transform(self.data)

        # Now that we have transformed the data, we can train the final model.
        self.next(self.train_model)

    @card
    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @resources(memory=4096)
    @step
    def train_model(self):
        """Train the model that will be deployed to production.

        This function will train the model using the entire dataset.
        """
        import mlflow

        # Let's log the training process under the experiment we started at the
        # beginning of the flow.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Let's disable the automatic logging of models during training so we
            # can log the model manually during the registration step.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the entire dataset.
            self.model = build_model(self.x.shape[1])
            self.model.fit(
                self.x,
                self.y,
                verbose=2,
                **self.training_parameters,
            )

            # Let's log the training parameters we used to train the model.
            mlflow.log_params(self.training_parameters)

        # After we finish training the model, we want to register it.
        self.next(self.register_model)

    @environment(
        vars={
            "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "jax"),
        },
    )
    @step
    def register_model(self, inputs):
        """Register the model in the Model Registry.

        This function will prepare and register the final model in the Model Registry.
        This will be the model that we trained using the entire dataset.

        We'll only register the model if its accuracy is above a predefined threshold.
        """
        import tempfile
        import mlflow

        # Since this is a join step, we need to merge the artifacts from the incoming
        # branches to make them available here.
        # Explicitly set transformers (using the ones from full training)
        # needs to happen before merge_artifacts
        cv_input = [i for i in inputs if hasattr(i, "ensemble_model")][0]
        full_input = [i for i in inputs if hasattr(i, "model")][0]

        self.features_transformer = full_input.features_transformer
        self.target_transformer = full_input.target_transformer

        self.ensemble_model = cv_input.ensemble_model
        self.model = full_input.model
        self.accuracy = cv_input.accuracy  # Use CV accuracy for threshold check

        self.merge_artifacts(inputs)

        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register where we'll register the final version of
        # the model.
        better_than_best = (
            self._check_run_better_than_best(accuracy=self.accuracy)
            if self.register_only_if_better
            else True
        )

        # We only want to register the model if its accuracy is above the threshold
        # specified by the `accuracy_threshold` parameter and if accuracy is better
        # than the best model registered so far.
        if self.accuracy >= self.accuracy_threshold and better_than_best:
            logging.info("Registering model...")

            # We'll register the model under the experiment we started at the beginning
            # of the flow. We also need to create a temporary directory to store the
            # model artifacts.
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                # Register ensemble model
                mlflow.pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    registered_model_name="penguins_ensemble",
                    artifact_path="model",
                    code_paths=[
                        (Path(__file__).parent / "inference.py").as_posix(),
                        (Path(__file__).parent / "common.py").as_posix(),
                    ],
                    artifacts=self._get_model_artifacts(directory, is_ensemble=True),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    # Our model expects a Python dictionary, so we want to save the
                    # input example directly as it is by setting`example_no_conversion`
                    # to `True`.
                    example_no_conversion=True,
                )
            # Register single model
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                mlflow.pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    registered_model_name="penguins_single",
                    artifact_path="single_model",
                    code_paths=[
                        (Path(__file__).parent / "inference.py").as_posix(),
                        (Path(__file__).parent / "common.py").as_posix(),
                    ],
                    artifacts=self._get_model_artifacts(directory, is_ensemble=False),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    example_no_conversion=True,
                )
        else:
            if not self.accuracy >= self.accuracy_threshold:
                logging.info(
                    "The accuracy of the model (%.2f) is lower than the accuracy "
                    "threshold (%.2f). Skipping model registration.",
                    self.accuracy,
                    self.accuracy_threshold,
                )
            if not better_than_best:
                logging.info(
                    "The current model has lower accuracy than the best model "
                    "registered so far. Skipping model registration."
                )

        # Let's now move to the final step of the pipeline.
        self.next(self.end)

    def _check_run_better_than_best(self, accuracy: float) -> bool:
        """Check if the current run is better than the best registered run.

        Args:
            accuracy: The accuracy of the current run.

        Returns:
            True if the current run is better than the last run, False otherwise.
        """
        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        runs = self._get_previous_runs(mlflow, self.mlflow_run_id)

        if runs.empty:
            return True  # First run should always register

        register = accuracy > runs["metrics.cross_validation_accuracy"].max()
        if not register:
            logging.info(
                f"The accuracy of the model ({accuracy:.2f}) is lower than the accuracy "
                f"of the best model registered so far ({runs['metrics.cross_validation_accuracy'].max():.2f}). "
                "Skipping model registration.",
            )
        return register

    @staticmethod
    def _get_previous_runs(mlflow_client, current_run_id: str) -> pd.DataFrame:
        """Get previous non-nested runs, excluding the current run and its children.

        Args:
            mlflow_client: MLflow client instance
            current_run_id: ID of the current run to exclude

        Returns:
            DataFrame containing filtered runs
        """
        from mlflow.entities import ViewType

        # Get all runs first
        all_runs = mlflow_client.search_runs(
            experiment_ids=[
                mlflow_client.get_experiment_by_name("penguins").experiment_id
            ],
            filter_string=(
                "status = 'FINISHED' "
                f"AND run_id != '{current_run_id}'"  # Exclude current main run
            ),
            order_by=["start_time DESC"],
            run_view_type=ViewType.ACTIVE_ONLY,
        )

        # Filter out nested runs and runs where current run is the parent
        return all_runs[
            (
                ~all_runs["tags.mlflow.runName"].str.contains(
                    "cross-validation", na=False
                )
            )
            & (
                ~all_runs["tags.mlflow.parentRunId"]
                .fillna("")
                .str.contains(current_run_id)
            )
        ]

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str, is_ensemble: bool):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        model_path = (Path(directory) / "model.joblib").as_posix()
        if is_ensemble:
            joblib.dump(self.ensemble_model, model_path)
        else:
            joblib.dump(self.model, model_path)

        # We also want to save the Scikit-Learn transformers so we can package them
        # with the model and use them during inference.
        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_signature(self):
        """Return the model's signature.

        The signature defines the expected format for model inputs and outputs. This
        definition serves as a uniform interface for appropriate and accurate use of
        a model.
        """
        from mlflow.models import infer_signature

        return infer_signature(
            model_input={
                "island": "Biscoe",
                "culmen_length_mm": 48.6,
                "culmen_depth_mm": 16.0,
                "flipper_length_mm": 230.0,
                "body_mass_g": 5800.0,
                "sex": "MALE",
            },
            model_output={"prediction": "Adelie", "confidence": 0.90},
            params={"data_capture": False},
        )

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}"
            for package, version in packages(
                "scikit-learn",
                "pandas",
                "numpy",
                "keras",
                "jax[cpu]",
            ).items()
        ]


if __name__ == "__main__":
    Training()
