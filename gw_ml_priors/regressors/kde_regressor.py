from abc import abstractmethod
from typing import Dict, List, Optional

import cloudpickle
import numpy as np
import pandas as pd
from scipy import stats

from ..logger import logger
from .regressor import Regressor

RANDOM_STATE = 42


class KDERegressor(Regressor):
    def __init__(
        self,
        input_parameters: List[str],
        output_parameters: List[str],
        outdir: str,
        model_hyper_param: Optional[Dict] = {},
    ):
        super().__init__(input_parameters, output_parameters, outdir)
        if len(input_parameters) > 2 or len(output_parameters) > 1:
            raise ValueError(
                "The KDE Regressor can only take up to 2  inputs and provide 1 output"
            )
        self.model = None

    def train(self, data: pd.DataFrame) -> None:
        super().train(data)
        train, test, train_labels, test_labels = self.train_test_split(data)
        self.model = stats.gaussian_kde(train.values.T)
        logger.info("Training complete")
        self.test(test, test_labels)

    def test(self, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        predicted_labels = self.model(data.values.T)
        errors = abs(predicted_labels - labels.values)
        model_testing_data_mae = round(np.mean(errors), 2)
        logger.info(f"MODEL TESTING: " f"Mean Abosulte Error={model_testing_data_mae}")

    def save(self) -> None:
        if self.model is None:
            raise ValueError("Empty model, not saving")
        with open(self.savepath, "wb") as f:
            cloudpickle.dump(self.model, f)

    def load(self):
        with open(self.savepath, "rb") as f:
            self.model = cloudpickle.load(f)

    def visualise(self):
        raise NotImplementedError()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.model(data.values.T)

    @property
    def n_trees(self) -> int:
        return np.nan
