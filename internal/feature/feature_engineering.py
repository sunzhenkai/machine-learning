import pandas as pd
from sklearn.preprocessing import LabelEncoder
from enum import Enum


class FeatureType(Enum):
    UNKNOWN = 0,
    ID = 1,
    LABEL = 2,
    VALUE_DISCRETE = 100,
    VALUE_CONTINUOUS = 10,
    IDENTITY_ORDERED = 200,
    IDENTITY_UNORDERED = 201


class Feature:
    def __init__(self, name: str, tp: FeatureType):
        self.name = name
        self.type = tp

    def __str__(self):
        return f"Feature{{name={self.name}, type={self.type}}}"


class FeatureSet:
    def __init__(self, features_=None):
        if features_ is None:
            features_ = []
        self.features = features_

    def __str__(self):
        return f"FeatureSet{{features=[{', '.join(str(i) for i in self.features)}]}}"

    def feature_names(self):
        return [i.name for i in self.features]

    def feature_size(self):
        return len(self.features)

    def check(self) -> str:
        # duplicated features check
        _s = set()
        _duplicated = []
        for _f in self.features:
            if _f.name not in _s:
                _s.add(_f.name)
            else:
                _duplicated.append(_f)

        _msg = ''
        if _duplicated:
            _msg += f"duplicated features: {str(FeatureSet(_duplicated))}"

        if not _msg:
            _msg = 'OK'
        return _msg

    def get_features_by_types(self, *args):
        return FeatureSet(list(filter(lambda x: x.type in args, self.features)))

    def get_features_without_types(self, *args):
        return FeatureSet(list(filter(lambda x: x.type not in args, self.features)))


class DataManager:
    def __init__(self, feature_set: FeatureSet, train_df: pd.DataFrame, feature_encoder=None, *args, **kwargs):
        self.feature_set: FeatureSet = feature_set
        self.origin_train_x: pd.DataFrame = train_df
        if feature_encoder is None:
            self.feature_encoder = LabelEncoder()

        self.label_feature_set = self.feature_set.get_features_by_types(FeatureType.LABEL)
        self.input_feature_set = self.feature_set.get_features_without_types(FeatureType.LABEL, FeatureType.ID)

        self.train_x: pd.DataFrame = self.origin_train_x.loc[:, self.input_feature_set.feature_names()]
        self.train_y: pd.DataFrame = self.origin_train_x[self.label_feature_set.features[0].name].to_frame()
        self.train_x_encoded = pd.DataFrame()
        self._encode(self.train_x_encoded, self.train_x)

        self.shape = self.train_x.shape[1:]
        print('Shape:', self.shape, ', Rows:', self.train_x.shape[0])

    def _encode(self, to_df: pd.DataFrame, from_df: pd.DataFrame):
        for c in from_df.columns:
            to_df[c] = self.feature_encoder.fit_transform(from_df[c])

    def format_input(self, df: pd.DataFrame):
        result = pd.DataFrame()
        selected = df.loc[:, self.input_feature_set.feature_names()]
        self._encode(result, selected)
        return result
