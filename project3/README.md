## Project of bedroom generation using diffusion model

### Installation:
Download and unpack data from [https://www.kaggle.com/datasets/jhoward/lsun_bedroom](https://www.kaggle.com/datasets/jhoward/lsun_bedroom). The data must be readable by tensorflow, it is recommended to unload it as:
```
all_images/
    train/
        bed/
            image1.png
            image2.png
            ...
```
You can do so by running function provided in `load_data.py` file.

Next, start the tests by running `__main__.py`. You can change the parameters as you wish.