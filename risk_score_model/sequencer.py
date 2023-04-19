# Python imports
from itertools import chain

# Third-party imports
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import RobustScaler

# Package imports


def build_padded_sequences(data,
                           maxlen=100,
                           include_dates=False,
                           training=False,
                           padding_how='post'):

    patient_sequences = list(data['concept_id'].values)

    patient_sequences = keras.preprocessing.sequence.pad_sequences(
        patient_sequences, value=0, maxlen=maxlen, padding=padding_how)

    if not include_dates:
        x = patient_sequences
    else:
        concept_dates = keras.preprocessing.sequence.pad_sequences(
            data['concept_date_int'],
            value=0,
            maxlen=maxlen,
            padding=padding_how)

        if training:
            dates_for_scaler = np.array(
                list(chain(*data['concept_date_int'].tolist()))).reshape(
                    -1, 1)
            robust_scaler = RobustScaler().fit(dates_for_scaler)

        x = [patient_sequences, concept_dates]

    if include_dates and training:
        return x, robust_scaler
    else:
        return x


def normalize_dates(df, grouped_cols, date_col):
    grouped = df.groupby(grouped_cols)[date_col]
    grouped = grouped.apply(
        lambda x: (x.max() - x).astype('timedelta64[D]').astype(int) + 1)
    df[f'{date_col}_int'] = grouped


def condense_sequences(data, labels=None):
    data = data.sort_values(by=['person_id', 'drug_era_id', 'concept_date'],
                            ascending=[True, True, True])
    normalize_dates(data, ['person_id', 'drug_era_id'], 'concept_date')

    data_grouped1 = data.groupby(['person_id', 'drug_era_id'
                                  ])['concept_id'].apply(list).to_frame()
    data_grouped2 = data.groupby(
        ['person_id',
         'drug_era_id'])['concept_date_int'].apply(list).to_frame()

    if labels is not None:
        data_grouped = data_grouped1.merge(labels,
                                           on=['person_id', 'drug_era_id'])
    else:
        data_grouped = data_grouped1

    data_grouped = data_grouped.merge(data_grouped2,
                                      on=['person_id', 'drug_era_id'])

    return data_grouped
