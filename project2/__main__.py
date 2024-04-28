
from test_case import train_transformer, train_lstm
from load_data import load_all_data
if __name__ == "__main__":
    test_ds, val_ds, train_ds = load_all_data()
    embed_dims = [16, 64, 256]
    n_lstm = [16, 64, 128]
    num_heads = [2, 16, 32]
    ff_dims = [32, 128, 256]
    seeds = [3, 5, 7, 12, 22]
    dropouts = [0, 0.1, 0.4]
    # for seed in seeds:
    #     for dim in embed_dims:
    #         path = 'transformer-dim-' + str(dim) 
    #         print(path) 
    #         train_transformer(test_ds, val_ds, train_ds, embed_dim=dim, num_heads=16, ff_dim=128, path=path, seed=seed)
    #     for n_head in num_heads:
    #         path = 'transformer-heads-' + str(dim)
    #         print(path)
    #         train_transformer(test_ds, val_ds, train_ds, embed_dim=64,
    #                           num_heads=n_head, ff_dim=128, path=path, seed=seed)
    #     for ff in ff_dims:
    #         path = 'transformer-ff-' + str(dim)
    #         print(path)
    #         train_transformer(test_ds, val_ds, train_ds, embed_dim=64,
    #                           num_heads=16, ff_dim=ff, path=path, seed=seed)
    for seed in seeds:
        for n in n_lstm:
            path = 'lstm-n-' + str(n)
            print(path)
            train_lstm(test_ds, val_ds, train_ds, n_lstm=n, embed_dim=256, recurrent_dropout=0.2, path=path, seed=seed)
        for dim in embed_dims:
            path = 'lstm-embed-dim-' + str(dim)
            print(path)
            train_lstm(test_ds, val_ds, train_ds, n_lstm=128, embed_dim=dim,
                       recurrent_dropout=0.2, path=path, seed=seed)
        for dr in dropouts:
            path = 'lstm-dropout-' + str(dr)
            print(path)
            train_lstm(test_ds, val_ds, train_ds, n_lstm=128, embed_dim=256,
                       recurrent_dropout=dr, path=path, seed=seed)
