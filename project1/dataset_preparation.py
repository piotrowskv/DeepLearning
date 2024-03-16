from tensorflow.keras import utils

DATA_DIR = 'data'


def get_train_data(batch_size):
    print("Getting train data...")
    train_ds = utils.image_dataset_from_directory(
        DATA_DIR + '/train',
        image_size=(32, 32),
        label_mode='categorical',
        batch_size=batch_size)

    return train_ds


def get_validation_data():
    print("Getting validation data...")
    val_ds = utils.image_dataset_from_directory(
        DATA_DIR + '/valid',
        label_mode='categorical',
        image_size=(32, 32))
    return val_ds


def get_test_data():
    print("Getting test data...")
    val_ds = utils.image_dataset_from_directory(
        DATA_DIR + '/test',
        label_mode='categorical',
        image_size=(32, 32))
    return val_ds
