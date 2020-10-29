# SMETER ML

This is a machine learning framework for creating models to predict the HTC of a domestic building using smart meter data + data from internal and external temperature sensors.

It is assumed that the training and testing data has already been collected or generated. So long as the data follows the expected schema, it should be possible to use the framework to experiment with different models and train ensembles of them. The exact data schema is detailed in the section [Train/test data](#train-test-data). See also the sample data in `data/sample/`.

The general interface is through `make` and one or more model configuration files which are stored in `model-config/`. Once a configuration file has been created for a model, you can train it on training data, test it on test data and perform crossvalidation on it. For more details, see the [Usage](#usage) section.

## Environment

You will need a suitable environment with the (mostly Python) dependencies installed. The only non-Python dependency is GNU Make, which is used to run the various tasks.

If you are using the [nix package manager](https://nixos.org) then simply run `nix-shell` in the project root. The necessary packages, which are specified in `default.nix` will be installed and a shell started in which the packages are available.

Otherwise, ensure you have Python>=3.6 and GNU Make installed on your system and use `venv` + the `requirements.txt` file.
```
python 3 -m venv smeter-ml-env
source smeter-ml-venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train/test data

No training data is supplied in this repository. It is assumed that it has been generated/collected separately. Training and testing data should be saved in `data/train/` and `data/test/`. In each case it is expected that the following files exist:

**`indoor_temp.npy`** -- (Required). Contains numpy array of internal temperature time series. Array has shape (\<number of houses\>, \<number of timesteps\>).

**`outdoor_temp.npy`** -- (Required). Contains numpy array of external temperature time series. Array has same shape as indoor_temp.npy.

**`elec_kwh.npy`** -- (Required). Contains numpy array of electricity consumption time series. Array has same shape as indoor_temp.npy.

**`gas_kwh.npy`** -- (Required). Contains numpy array of gas consumption time series. Array has same shape as indoor_temp.npy.

**`htc.csv`** -- (Required). It is expected that this contains the columns `mean_htc, htc_0, htc_1, htc_2, ...`, where `htc_n` is the HTC computed using data starting at the nth day. E.g. if the data collected runs from October 1st to March 31st, and HTCs are computed using 4 weeks of data, then `htc_10` will be the HTC calculated using data from Oct 11-Nov 7.

**`building_type.csv`** -- (Optional). Has one column, `building_type`, whose values are one of: SEMI, MID_TERRACE, END_TERRACE, DETACHED, BUNGALOW.

**`sqm.csv`** -- (Optional). Sq. m values for each of the houses.

**Note:** Each of the data files must contain their data for the houses in the same order.

### Model configs

Configuration files for constructing models are saved in `model-config/`. They are simple key-value text files with a number of options, such as ensemble size, number of epochs, static features and other hyperparameters.

See the README in `model-config/` for details on the available options.

### make commands

The primary interface is through `make`. There are a number of tasks which can be run to perform tasks such as training and crossvalidating models.

#### `make ensemble`
Create an ensemble of CNNs -- train the models and save their weights in `models/ensemble/<model name>`.
The config for the ensemble is read from a text file in `model-config`. For more info about what the config can contain, see the README in `model-config/`.
Example usage:
```
make ensemble config=model0.txt
```

#### `make crossvalidate`

Perform 5-fold crossvalidation on a model.
The config for the model is read from a text file in `model-config/`. For more info about what the
config can contain, see the README in `model-config/`.

Note: crossvalidation is only performed on single models, not ensembles, so the value of "ensemble_size"
in the model config will be ignored.

Example usage:
```
make crossvalidate config=model0.txt
```

#### `make sample-loss-function-params`

Try out different parameters for the loss function.

There are two different parameters, _s_ and _lambda_. Supply 0, 1 or both of them. The default for both _s_ and _lambda_ values to try is [0.1, 1, 10, 100]. Note values for _s_ and _lambda_ should be supplied as comma separated lists with *no spaces*.

Example usage:
```
make sample-loss-function-params
make sample-loss-function-params s=1,2,4,8
make sample-loss-function-params lambda=2,4,6,8,10
make sample-loss-function-params s=1,2,4,8 lambda=2,4,6,8,10
```

#### `make clean-models`
Delete all the saved model files in `models/`. Use with caution.


## Developing

While a number of configurable options are exposed through the model config files, you may wish to experiment with other variations which it is not possible to set in this way, for example the structure of the CNNs themselves. Something like this will require a change to the source code, which is contained within a small number of modules in `src/ml/`.

You are encouraged to look through the source code and amend it as necessary to suit your needs. It should be fairly self-explanatory which module is responsible for which job and if you need an additional hint it is worth looking at what gets called by the different targets in the `makefile`.

## TODO

- [ ] The program is only set up to accomodate a single static feature in a model. Ideally it should be possible to train a model using multiple static features. These could be read from `data/(train|test)/static_features.csv`, which can have arbitrary static features as its columns.
- [ ] Make neural network structure configurable through config file. Would this be too complicated?
- [ ] Installing deps via requirements.txt fails.
