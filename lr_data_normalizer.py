import pandas as pd

def normalize_data(train_file, test_file, output_train_file, output_test_file):
    train_df = pd.read_csv(train_file + '.csv')
    test_df = pd.read_csv(test_file + '.csv')
    feat_col = train_df.col.difference(['target_dcp'])
    # Only train
    mean = train_df[feat_col].mean()
    std = train_df[feat_col].std()

    train_df[feat_col] = (train_df[feat_col] - mean) / std
    test_df[feat_col] = (test_df[feat_col] - mean) / std
    train_df.to_csv(output_train_file + '.csv', index=False)
    test_df.to_csv(output_test_file + '.csv', index=False)

normalize_data('train_12_hour_lr_data', 'test_12_hour_lr_data', 'train_12_hour_lr_normalized', 'test_12_hour_lr_normalized')
normalize_data('train_24_hour_lr_data', 'test_24_hour_lr_data', 'train_24_hour_lr_normalized', 'test_24_hour_lr_normalized')
normalize_data('train_48_hour_lr_data', 'test_48_hour_lr_data', 'train_48_hour_lr_normalized', 'test_48_hour_lr_normalized')
normalize_data('train_6_hour_lr_data', 'test_6_hour_lr_data','train_6_hour_lr_normalized', 'test_6_hour_lr_normalized')
normalize_data('train_1_hour_lr_data', 'test_1_hour_lr_data','train_1_hour_lr_normalized', 'test_1_hour_lr_normalized')


