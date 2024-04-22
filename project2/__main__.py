
from test_case import train_transformer, train_lstm
from load_data import load_all_data
if __name__ == "__main__":
    test_ds, val_ds, train_ds = load_all_data()
    #train_transformer(test_ds, val_ds, train_ds, embed_dim=64, num_heads=2, ff_dim=32, path='result', seed=3)
    train_lstm(test_ds, val_ds, train_ds, n_lstm=4, embed_dim=32, recurrent_dropout=0.2, path='result', seed=3)
