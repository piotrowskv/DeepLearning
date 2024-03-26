from tensorflow.keras import utils

DATA_DIR = 'data'


def get_train_data(batch_size, categorical=True):
    print("Getting train data...")
    if categorical:
        train_ds = utils.image_dataset_from_directory(
            DATA_DIR + '/train',
            label_mode='categorical',
            image_size=(32, 32))
    else:
        train_ds = utils.image_dataset_from_directory(
            DATA_DIR + '/train',
            image_size=(32, 32))
    return train_ds



def get_augmented_train_data(batch_size):
    print("Getting train data...")
    train_ds = utils.image_dataset_from_directory(
        DATA_DIR + '/train-augmented',
        image_size=(32, 32),
        label_mode='categorical')

    return train_ds


def get_validation_data(categorical=True):
    print("Getting validation data...")
    if categorical: 
        val_ds = utils.image_dataset_from_directory(
            DATA_DIR + '/valid',
            label_mode='categorical',
            image_size=(32, 32))
    else:
        val_ds = utils.image_dataset_from_directory(
            DATA_DIR + '/valid',
            image_size=(32, 32))
    return val_ds


def get_test_data(categorical=True):
    print("Getting test data...")
    if categorical: 
        val_ds = utils.image_dataset_from_directory(
            DATA_DIR + '/test',
            label_mode='categorical',
            shuffle=False,
            image_size=(32, 32))
    else:
        val_ds = utils.image_dataset_from_directory(
            DATA_DIR + '/test',
            shuffle=False,
            image_size=(32, 32))
    return val_ds

def get_train_data_single_class(class_name):
    print("Getting train data for " + class_name + "...")
    train_ds = utils.image_dataset_from_directory(
        DATA_DIR + '/train/' + class_name,
        image_size=(32, 32),
        label_mode='categorical')

    return train_ds