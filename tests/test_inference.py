import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock
import json

import numpy as np
import pandas as pd
import pytest

from pipelines.inference import Model


@pytest.fixture
def mock_keras_model(monkeypatch):
    """Return a mock Keras model."""
    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr("keras.saving.load_model", lambda _: mock_model)

    return mock_model


@pytest.fixture
def mock_transformers(monkeypatch):
    """Return mock transformer instances."""
    mock_features_transformer = Mock()
    mock_features_transformer.transform = Mock()

    mock_species_transformer = Mock()
    mock_species_transformer.categories_ = [["Adelie", "Chinstrap", "Gentoo"]]

    mock_target_transformer = Mock()
    mock_target_transformer.named_transformers_ = {"species": mock_species_transformer}

    def mock_load(path):
        return (
            mock_features_transformer
            if path == "features_transformer"
            else mock_target_transformer
        )

    monkeypatch.setattr("joblib.load", mock_load)
    return mock_features_transformer, mock_target_transformer


@pytest.fixture
def model(mock_keras_model, mock_transformers):
    """Return a model instance."""
    directory = tempfile.mkdtemp()
    data_collection_uri = Path(directory) / "database.db"

    model = Model(data_collection_uri=data_collection_uri, data_capture=False)

    mock_context = Mock()
    mock_context.artifacts = {
        "model": "model",
        "features_transformer": "features_transformer",
        "target_transformer": "target_transformer",
    }

    model.load_context(mock_context)

    assert model.model == mock_keras_model
    assert model.features_transformer == mock_transformers[0]
    assert model.target_transformer == mock_transformers[1]

    return model


def fetch_data(model):
    connection = sqlite3.connect(model.data_collection_uri)
    cursor = connection.cursor()
    cursor.execute("SELECT island, prediction, confidence FROM data;")
    data = cursor.fetchone()
    connection.close()
    return data


def test_process_input(model):
    model.features_transformer.transform = Mock(
        return_value=np.array([[0.1, 0.2]]),
    )
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    # Ensure the transform method is called with the input data.
    model.features_transformer.transform.assert_called_with(input_data)
    assert model.features_transformer.transform.call_count == 1

    # The function should return the transformed data.
    assert np.array_equal(result, np.array([[0.1, 0.2]]))


def test_process_input_return_empty_array_on_exception(model):
    model.features_transformer.transform = Mock(side_effect=Exception("Invalid input"))
    input_data = pd.DataFrame({"island": ["Torgersen"]})
    result = model.process_input(input_data)

    # The transform method should be called twice: once for the entire DataFrame and
    # once for each row individually (fallback mechanism).
    assert model.features_transformer.transform.call_count == 2

    # Check that it was called with both the DataFrame and a single-row DataFrame
    call_args_list = model.features_transformer.transform.call_args_list
    assert isinstance(
        call_args_list[0][0][0], pd.DataFrame
    )  # First call with DataFrame
    assert isinstance(
        call_args_list[1][0][0], pd.DataFrame
    )  # Second call with single-row DataFrame

    # Since there was an exception, the function should return an empty array.
    assert result.size == 0


def test_process_output(model):
    output = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    result = model.process_output(output)

    assert result == [
        {"prediction": "Adelie", "confidence": 0.6},
        {"prediction": "Chinstrap", "confidence": 0.7},
    ]


def test_process_output_return_empty_list_on_none(model):
    assert model.process_output(None) == []


def test_predict_return_empty_list_on_invalid_input(model, monkeypatch):
    mock_process_input = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict_return_empty_list_on_invalid_prediction(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    model.model.predict = Mock(return_value=None)
    monkeypatch.setattr(model, "process_input", mock_process_input)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)
    assert result == []


def test_predict(model, monkeypatch):
    mock_process_input = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    mock_process_output = Mock(
        return_value=[{"prediction": "Adelie", "confidence": 0.6}],
    )
    model.model.predict = Mock(return_value=np.array([[0.6, 0.3, 0.1]]))
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)

    input_data = [{"island": "Torgersen", "culmen_length_mm": 39.1}]
    result = model.predict(context=None, model_input=input_data)

    assert result == [{"prediction": "Adelie", "confidence": 0.6}]
    mock_process_input.assert_called_once()
    mock_process_output.assert_called_once()
    model.model.predict.assert_called_once()


@pytest.mark.parametrize(
    ("default_data_capture", "request_data_capture", "database_exists"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_data_capture(
    model,
    default_data_capture,
    request_data_capture,
    database_exists,
):
    model.data_capture = default_data_capture
    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": request_data_capture},
    )

    assert Path(model.data_collection_uri).exists() == database_exists


def test_capture_stores_data_in_database(model):
    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": True},
    )

    data = fetch_data(model)
    assert data == ("Torgersen", "Adelie", 0.6)


def test_capture_on_invalid_output(model, monkeypatch):
    mock_process_output = Mock(return_value=None)
    monkeypatch.setattr(model, "process_output", mock_process_output)

    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": True},
    )

    data = fetch_data(model)

    # The prediction and confidence columns should be None because the output
    # from the model was empty
    assert data == ("Torgersen", None, None)


def test_init_with_invalid_data_collection_config():
    """Test that initializing with both URI and output file raises an error."""
    with pytest.raises(ValueError) as exc_info:
        Model(
            data_collection_uri="test.db",
            data_collection_output_file="output.jsonl",
            data_capture=True,
        )
    assert (
        "both data collection uri and output file are specified. please specify "
        "only one of them." in str(exc_info.value).lower()
    )


def test_capture_to_jsonl_file(model, tmp_path):
    """Test data capture to JSONL file."""
    output_file = tmp_path / "output.jsonl"
    model.data_collection_uri = None
    model.data_collection_output_file = str(output_file)

    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": True},
    )

    assert output_file.exists()
    with open(output_file) as f:
        data = json.loads(f.readline())
        assert data["island"] == "Torgersen"
        assert data["prediction"] == "Adelie"
        assert data["confidence"] == 0.6
        assert "date" in data
        assert "uuid" in data


@pytest.mark.parametrize(
    ("env_uri", "env_file", "expected_uri", "expected_file"),
    [
        ("env.db", None, "env.db", None),
        (None, "env.jsonl", None, "env.jsonl"),
        (None, None, "penguins.db", None),
    ],
)
def test_environment_variables_override(
    mock_keras_model,
    mock_transformers,
    monkeypatch,
    env_uri,
    env_file,
    expected_uri,
    expected_file,
):
    """Test environment variables override model configuration."""
    if env_uri:
        monkeypatch.setenv("DATA_COLLECTION_URI", env_uri)
    if env_file:
        monkeypatch.setenv("DATA_COLLECTION_OUTPUT_FILE", env_file)

    model = Model(
        data_collection_uri=expected_uri, data_collection_output_file=expected_file
    )
    mock_context = Mock()
    mock_context.artifacts = {
        "model": "model",
        "features_transformer": "features_transformer",
        "target_transformer": "target_transformer",
    }

    model.load_context(mock_context)

    assert model.data_collection_uri == expected_uri
    assert model.data_collection_output_file == expected_file


def test_capture_handles_database_error(model, monkeypatch):
    """Test that database errors are handled gracefully during capture."""

    def mock_connect(*args, **kwargs):
        raise sqlite3.Error("Database error")

    monkeypatch.setattr("sqlite3.connect", mock_connect)

    # Should not raise an exception
    model.predict(
        context=None,
        model_input=[{"island": "Torgersen"}],
        params={"data_capture": True},
    )


def test_predict_with_dict_input(model):
    """Test that predict handles dictionary input correctly."""
    input_data = {"island": "Torgersen", "culmen_length_mm": 39.1}
    result = model.predict(context=None, model_input=input_data)
    assert len(result) == 1
    assert result[0]["prediction"] == "Adelie"


@pytest.mark.parametrize(
    ("input_data", "expected_len"),
    [
        (pd.DataFrame(), 0),
        (pd.DataFrame([{"island": "Invalid"}]), 1),
        (pd.DataFrame([{"island": "A"}, {"island": "B"}]), 2),
    ],
)
def test_process_input_with_different_sizes(model, input_data, expected_len):
    """Test process_input handles different input sizes correctly."""
    model.features_transformer.transform = Mock(
        return_value=np.zeros((expected_len, 2))
    )
    result = model.process_input(input_data)
    assert len(result) == expected_len
