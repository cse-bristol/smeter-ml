# Create an ensemble of CNNs -- train the models and save their weights in models/ensemble/<model name>.
# The config for the ensemble is read from a text file in model-config/. For more info about what the
# config can contain, see the README in model-config.
#
# Example usage:
# ```
# make ensemble config=model0.txt
# ```
.PHONY: ensemble
ensemble: data/train/outdoor_temp.npy data/train/indoor_temp.npy data/train/elec_kwh.npy data/train/gas_kwh.npy data/train/building_type.csv data/train/sqm.csv data/train/htc.csv data/train/scalefactors.json
	PYTHONPATH="src" python3 -m src.ml.model.ensemble $(config)

# Perform 5-fold crossvalidation on a model.
# The config for the model is read from a text file in model-config/. For more info about what the
# config can contain, see the README in model-config.
#
# Note: crossvalidation is only performed on single models, not ensembles, so the value of "ensemble_size"
# in the model config will be ignored.
#
# Example usage:
# ```
# make crossvalidate config=model0.txt
# ```
.PHONY: crossvalidate
crossvalidate: data/train/outdoor_temp.npy data/train/indoor_temp.npy data/train/elec_kwh.npy data/train/gas_kwh.npy data/train/building_type.csv data/train/sqm.csv data/train/htc.csv data/train/scalefactors.json
	PYTHONPATH="src" python3 -m src.ml.model.crossvalidate $(config)

# Try out different parameters for the loss function
# There are two different parameters, s and lambda.
# Supply 0, 1 or both of them. The default for both s and lambda values to try is [0.1, 1, 10, 100]
# Note values for s and lam should be supplied as comma separated lists with *no spaces*.
#
# Example usage:
# ```
# make sample-loss-function-params
# make sample-loss-function-params s=1,2,4,8
# make sample-loss-function-params lambda=2,4,6,8,10
# make sample-loss-function-params s=1,2,4,8 lambda=2,4,6,8,10
# ```
.PHONY: sample-loss-function-params
sample-loss-function-params:
	PYTHONPATH="src" python3 -m src.ml.model.hyperparam_selection s=$(s) lam=$(lambda)

# Delete all the saved model files
.PHONY: clean-models
clean-models:
	rm -r models/ensemble
	mkdir -p models/ensemble

# Compute normalisation factors for each of the input features + target feature and save them in a file.
data/train/scalefactors.json: data/train/outdoor_temp.npy data/train/indoor_temp.npy data/train/elec_kwh.npy data/train/gas_kwh.npy data/train/building_type.csv data/train/sqm.csv data/train/htc.csv
	PYTHONPATH="src" python3 -m src.ml.features "MAKE_FEATURE_SCALE_FACTORS"


# Lint python files using flake8
# args:
#   dir - optional, directory or file to scan, defaults to 'src'
.PHONY: lint
lint:
	flake8 --max-line-length=120 $(if $(dir),$(dir),src)

# Attempt automatic formatting and fix of linting issues using black
# args:
#   dir - optional, directory or file to scan, defaults to 'src'
.PHONY: format
format:
	black -l 120 $(if $(dir),$(dir),src)
