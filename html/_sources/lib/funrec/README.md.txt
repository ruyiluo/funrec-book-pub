# FunRec: Unified Recommendation Training Pipeline

FunRec provides a simplified, unified interface for training and evaluating recommendation models. The entire pipeline is condensed into 5 simple steps with a clean, modular architecture.

## Usage

```python
import funrec

# Load complete configuration once
config = funrec.load_config("funrec/config/config_dssm_ml1m.yaml")

# Load raw data using data section
train_data, test_data = funrec.load_data(config.data)

# Prepare features using feature section
feature_columns, processed_data = funrec.prepare_features(config.features, train_data, test_data)

# Train model using training section
model = funrec.train_model(config.training, feature_columns, processed_data)

# Evaluate using evaluation section
metrics = funrec.evaluate_model(model, processed_data, config.evaluation)
```

## Package Structure

```
funrec/
├── __init__.py                 # Main package exports
├── README.md                   # This documentation
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── config.py              # Configuration loading
│   └── config_dssm_ml1m.yaml  # Sample configuration
├── data/                       # Data loading and access
│   ├── __init__.py
│   ├── loaders.py             # Data loading functions
│   ├── data_config.py         # Dataset configurations
│   └── data_utils.py          # Data utilities
├── features/                   # Feature engineering
│   ├── __init__.py
│   ├── feature_column.py      # FeatureColumn class definition
│   └── processors.py          # Feature preparation logic
├── models/                     # Model definitions and utilities
│   ├── __init__.py
│   ├── dssm.py               # DSSM model implementation
│   ├── layers.py             # Neural network layers
│   └── utils.py              # Model building utilities
├── training/                   # Training functionality
│   ├── __init__.py
│   └── trainer.py            # Model training logic
└── evaluation/                 # Model evaluation
    ├── __init__.py
    ├── evaluator.py          # Evaluation logic
    └── metrics.py            # Evaluation metrics
```

## Testing

Run the test script to verify the pipeline:

```bash
python test_funrec.py
```

## Configuration

The configuration file should contain four sections:

- `data`: Dataset configuration
- `features`: Feature engineering configuration  
- `training`: Model training configuration with `build_function` path
- `evaluation`: Evaluation metrics configuration

### Example Configuration

```yaml
training:
  build_function: "funrec.models.dssm.build_dssm_model"  # Function path
  model_params:
    dnn_units: [128, 64, 32]
    dropout_rate: 0.2
  optimizer: "adam"
  # ... other training parameters
```

## Adding New Models

1. Create your model in `funrec/models/new_model.py`:
```python
def build_new_model(feature_columns, model_config):
    # Extract parameters from config
    param1 = model_config.get('param1', default_value)
    # Build and return (main_model, user_model, item_model)
    return main_model, user_model, item_model
```

2. Reference it in your config:
```yaml
training:
  build_function: "funrec.models.new_model.build_new_model"
  model_params:
    param1: value1
```

## Supported Models

Currently supports:
- DSSM (Deep Structured Semantic Model) for retrieval/recall tasks

## Design Principles

- **Configuration-Driven**: Everything specified in YAML config
- **Modular Architecture**: Clear separation of concerns
- **No Over-Engineering**: Simple, direct function calls
- **Easy Extension**: Add new models with minimal code