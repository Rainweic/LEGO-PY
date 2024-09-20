from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append("..")

from dags.pipeline import Pipeline
from dags.stage import BaseStage


class DataIngestor(BaseStage):

    @staticmethod
    def download_data():
        data = load_wine()
        features = data['data']
        targets = data['target']
        return features, targets

    def forward(self, *args, **kwargs):
        features, targets = self.download_data()
        return features, targets


class DataPreprocessor(BaseStage):
    @staticmethod
    def normalize(features):
        return MinMaxScaler().fit_transform(features)

    @staticmethod
    def split(features, targets):
        return train_test_split(features, targets, test_size=0.2)

    def forward(self, features, targets, *args, **kwargs):

        xtr, xte, ytr, yte = self.split(features, targets)

        xtr = self.normalize(xtr)
        xte = self.normalize(xte)

        data = {
            'xtr': xtr, 'xte': xte,
            'ytr': ytr, 'yte': yte
        }
        return data


class ModelTrainer(BaseStage):
    def __init__(self, *args, **kwargs):
        super(ModelTrainer, self).__init__(*args, **kwargs)

        self.model = None

    def train_model(self, xtr, ytr):
        self.model = RandomForestClassifier().fit(xtr, ytr)

    def test_model(self, xte, yte):
        acc = self.model.score(xte, yte)
        return acc

    def forward(self, preprocessed_data, *args, **kwargs):
        
        xtr = preprocessed_data['xtr']
        xte = preprocessed_data['xte']
        ytr = preprocessed_data['ytr']
        yte = preprocessed_data['yte']

        self.train_model(xtr, ytr)

        acc = self.test_model(xte, yte)

        print('Accuracy:', acc)


def build_pipeline():

    data_ingestor = DataIngestor().set_default_outputs(2)

    data_preprocessor = DataPreprocessor() \
        .after(data_ingestor) \
        .set_inputs(data_ingestor.output_data_names) \
        .set_default_outputs(1)

    model_trainer = ModelTrainer() \
        .after(data_preprocessor) \
        .set_inputs(data_preprocessor.output_data_names)

    pipeline = Pipeline()
    pipeline.add_stages([data_ingestor, data_preprocessor, model_trainer])

    return pipeline


p = build_pipeline()

p.start()