from sklearn.model_selection import train_test_split
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


def convert_to_str_df(df):
    for c in df.columns:
        df[c] = df[c].astype(str)
    label_encoder = LabelEncoder()
    for c in df.columns:
        df[c] = label_encoder.fit_transform(df[c])


class DataManager:
    def __init__(self, feature_set_: FeatureSet, train_df_, test_df_, test_size=0.3, radom_state=10):
        self.feature_set = feature_set_
        self.origin_train_x = train_df_
        self.origin_test_df = test_df_

        self.label_feature_set = self.feature_set.get_features_by_types(FeatureType.LABEL)
        self.input_feature_set = self.feature_set.get_features_without_types(FeatureType.LABEL, FeatureType.ID)

        self.train_x = train_df_.loc[:, self.input_feature_set.feature_names()]
        self.train_y = train_df_[self.label_feature_set.features[0].name].to_frame()
        self.test_x = test_df_.loc[:, self.input_feature_set.feature_names()]

        convert_to_str_df(self.train_x)
        convert_to_str_df(self.test_x)

        #self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(features_df_, label_df_,
        #                                                                        test_size=test_size,
        #                                                                        random_state=radom_state)
