
from test_case import train_transformer, train_lstm
from load_data import load_all_data
if __name__ == "__main__":
    # if doing binary 
    test_ds, val_ds, train_ds = load_all_data(binary=True)
    embed_dims = [16, 64, 256]
    n_lstm = [16, 64, 128]
    num_heads = [16, 64, 128]
    ff_dims = [32, 128, 256]
    seeds = [5, 7]
    dropouts = [0, 0.1, 0.4]
    # for seed in seeds:
        # for dim in embed_dims:
        #     path = 'transformer-dim-' + str(dim) 
        #     print(path) 
        #     train_transformer(test_ds, val_ds, train_ds, embed_dim=dim, num_heads=128, ff_dim=128, path=path, seed=seed)
        # for n_head in num_heads:
        #     path = 'transformer-heads-' + str(n_head)
        #     print(path)
        #     train_transformer(test_ds, val_ds, train_ds, embed_dim=64,
        #                       num_heads=n_head, ff_dim=128, path=path, seed=seed)
        # for ff in ff_dims:
        #     path = 'transformer-ff-' + str(ff)
        #     print(path)
        #     train_transformer(test_ds, val_ds, train_ds, embed_dim=64,
        #                       num_heads=128, ff_dim=ff, path=path, seed=seed)
    for seed in seeds:
        path = 'bin-lstm'
        train_lstm(test_ds, val_ds, train_ds, n_lstm=128, embed_dim=64, recurrent_dropout=0.2, path=path, seed=seed, binary = True)
        # for n in n_lstm:
        #     path = 'bin-lstm-n-' + str(n)
        #     print(path)
        #     train_lstm(test_ds, val_ds, train_ds, n_lstm=n, embed_dim=256, recurrent_dropout=0.2, path=path, seed=seed, binary = True)
        # for dim in embed_dims:
        #     path = 'bin-lstm-embed-dim-' + str(dim)
        #     print(path)
        #     train_lstm(test_ds, val_ds, train_ds, n_lstm=128, embed_dim=dim,
        #                recurrent_dropout=0.2, path=path, seed=seed, binary=True)
        # for dr in dropouts:
        #     path = 'bin-lstm-dropout-' + str(dr)
        #     print(path)
        #     train_lstm(test_ds, val_ds, train_ds, n_lstm=128, embed_dim=256,
        #                recurrent_dropout=dr, path=path, seed=seed, binary=True)
