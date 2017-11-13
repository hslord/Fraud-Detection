import pandas as pd
from numpy import nan_to_num
pd.options.display.max_columns = 100


class CompletePreprocessor():
    def __init__(self):
        self.keys = None

    def fit_transform(self, df):
        df = preprocessing(df)
        all_labels = df.columns.tolist()
        y_label = 'acct_type'
        X_labels = all_labels
        X_labels.remove(y_label)

        y = df[y_label].astype(float).values
        X = df[X_labels].astype(float).values
        self.keys = X_labels
        return nan_to_num(X), y

    def transform(self, df):
        X = df[self.keys].astype(float).values
        return nan_to_num(X)


def preprocessing(df):

    # Drop columns from feature matrix based on preliminary EDA
    df.drop(['approx_payout_date', 'country', 'channels', 'currency', 'description', 'email_domain', 'event_created',
             'event_end', 'event_published', 'event_start', 'name', 'name_length', 'object_id', 'org_name',
             'payee_name', 'sale_duration2', 'user_type', 'venue_address', 'venue_country', 'venue_latitude',
             'venue_longitude', 'venue_state', 'previous_payouts', 'user_created'], axis=1, inplace=True)

    # New binary column for fraud or not fraud
    df['acct_type'] = df['acct_type'].apply(lambda row: 1 if (
        (row == 'fraudster') or (row == 'fraudster_event')) else 0)

    # Change listed column to binary values for yes or no
    df['listed'] = df['listed'].apply(lambda row: 1 if (row == 'y') else 0)

    # Change org_desc column to binary values for not blank or blank
    df['org_desc'] = df['org_desc'].apply(lambda row: 0 if (row == '') else 1)

    # Change payout_type column to binary values for not blank or blank
    df['payout_type'] = df['payout_type'].apply(
        lambda row: 0 if (row == '') else 1)

    # Change delivery method column to binary values for 0.0 or not 0.0
    df['delivery_method'] = df['delivery_method'].apply(
        lambda row: 0 if (row == 0.0) else 1)

    # Change venue name column to binary values for not blank or blank
    df['venue_name'] = df['venue_name'].apply(
        lambda row: 0 if (row == "") else 1)

    # Binarize to has/doesn't have Facebook (0 means no Facebook)
    df['org_facebook'].fillna(0).apply(
        lambda row: 0 if (row == 0.0) else 1)

    # Binarize to has/doesn't have Twitter (0 means no Twitter)
    df['org_twitter'].fillna(0).apply(
        lambda row: 0 if (row == 0.0) else 1)

    # sum all tickets sold across different ticket types for each event
    qty_sold = []
    for row in df['ticket_types']:
        sold = 0
        for dicts in row:
            for key, val in dicts.iteritems():
                if key == 'quantity_sold':
                    sold += val
        qty_sold.append(sold)

    df['tix_sold'] = qty_sold
    df.drop('ticket_types', axis=1, inplace=True)

    return df


if __name__ == '__main__':

    # df = pd.read_json(u'/Users/brendanwong/galvanize/team-case-study-repos/dsi-fraud-detection-case-study/example.copy.json')
    df = pd.read_json(
        u'/Users/hslord/galvanize/dsi-fraud-detection-case-study/data/data.json')

    df = preprocessing(df)

    df.head()
