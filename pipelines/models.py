import keras
import numpy as np
from keras.models import Model as KerasModel


@keras.saving.register_keras_serializable(package="MyModels")
class KerasEnsemble(KerasModel):
    def __init__(self, models: list["KerasModel"], **kwargs) -> None:
        super().__init__(**kwargs)
        self.models = models

    def call(self, inputs, training=None):
        predictions = [model(inputs, training=training) for model in self.models]
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "models": [
                    keras.saving.serialize_keras_object(model) for model in self.models
                ]
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        models_config = config.pop("models")
        models = [keras.saving.deserialize_keras_object(c) for c in models_config]
        return cls(models=models, **config)


def build_ensemble_model(models: list[KerasModel]) -> KerasEnsemble:
    """Build an ensemble model from a list of models."""
    return KerasEnsemble(models=models)
