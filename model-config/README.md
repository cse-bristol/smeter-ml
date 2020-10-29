# model-config

Configuration files for ensemble models should be placed in this directory.

Ensembles consist of n models of the same general structure, trained with different initial weights and, if specified, trained on different subsets of the overall training data set. The basic model is a convolutional neural network, whose structure can be seen in `src/ml/model/cnn.py`.

Ensembles can be configured in a number of different ways, the options for which are detailed below. See also the example configuration file, `example-config.txt`.

The basic workflow is to create a model config file, say `my_model.txt`, and save it in this directory. You can then use one of the `make` commands, such as `make ensemble config=my_model`, to (in this case) train an ensemble as per the config options and save the individual model files in `models/ensemble/my_model/`. The model files consist of a network structure file, `model_n.json`, and a network weights file, `model_<n>.h5`, for each of the n members of the ensemble.

The available configuration options are:

| Option | Type | Description | Example |
| ------ | ---- | ----------- | ------- |
| ensemble_size             | int    | Number of ensemble members                   | 10    |
| epochs                    | int    | Number of training epochs                    | 20    |
| input_days                | int    | Number of days to run model on               | 21    |
| training_repeats          | int    | Number of training samples to make per house | 4     |
| loss_fn_lambda            | float  | lambda param for loss function               | 100   |
| loss_fn_s                 | float  | s param for loss function                    | 0.104 |
| training_htc_lower_bound  | float  | Only train on houses with htc >= value       | 200   |
| training_htc_upper_bound  | float  | Only train on houses with htc <= value       | 400   |
| channels                  | list*  | Which channels the model will use            | mean_temp,outdoor_temp |
| validation_split          | float  | validation_split param in keras              | None  |
| early_stopping_variable   | str    | For keras EarlyStopping, e.g. val_loss       | None  |
| early_stopping_patience   | int    | For keras EarlyStopping, how many iterations to wait before stopping early | None |
| ensemble_type             | str    | Either regular or bagged                     | None  |
| static_features**         | list   | Either floor_area or proxy (proxy is `sum(gas) / mean(in_temp - out_temp)` over whole data series) | floor_area |
| scale_inputs              | bool   | Whether or not to scale inputs into [0, 1]   | False |
| scale_targets             | bool   | Whether or not to scale targets into [0, 1]  | False |
| single_output_only        | bool   | Whether or not to only output HTC, rather than upper and lower bounds | False |
| building_types            | list   | Subset of SEMI, MID_TERRACE, END_TERRACE, DETACHED, BUNGALOW. Only these building types will be used in training data. | SEMI,END_TERRACE |

\* All options of type list should be comma-separated lists with no spaces.

\*\* At the moment it is only possible to use a single static feature (either floor_area or proxy).
