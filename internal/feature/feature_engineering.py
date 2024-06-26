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


def convert_to_str_df(df):
    for c in df.columns:
        df[c] = df[c].astype(str)
    label_encoder = LabelEncoder()
    for c in df.columns:
        df[c] = label_encoder.fit_transform(df[c])


class DataManager:
    def __init__(self, feature_set_: FeatureSet, train_df_, test_df_, test_size=0.3, radom_state=10):
        self.feature_set = feature_set_
        label_ = feature_set_.get_features_by_types(FeatureType.LABEL).features[0]
        # print(label_, train_df_.columns, train_df_.columns.difference([label_.name]))
        features_df_ = train_df_.loc[:, train_df_.columns.difference([label_.name])]
        convert_to_str_df(features_df_)
        label_df_ = train_df_[label_.name]

        self.origin_train_x = train_df_
        self.origin_test_df = test_df_
        self.train_x = features_df_
        self.train_y = label_df_.to_frame()
        # print(features_df_.dtypes)
        #self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(features_df_, label_df_,
        #                                                                        test_size=test_size,
        #                                                                        random_state=radom_state)
