# Configuration Structure

This directory contains all configuration files for the Nowcasting system. The configuration follows a **unified structure** where experiment configs contain all necessary settings including preprocessing and series configurations.

## Directory Structure

```
config/
├── experiment/     # Experiment configurations (unified structure)
├── model/         # Model-specific hyperparameters
├── preprocess/    # Preprocessing templates (for reference)
└── series/        # Series configurations (legacy, now embedded in experiments)
```

## Unified Configuration Structure

### Experiment Configs (`config/experiment/`)

Experiment configs now contain **all** configuration in a single file:

```yaml
# Experiment Metadata
name: experiment_name
description: "Experiment description"
model_type: dfm  # dfm, ddfm, arma, tft, xgboost

# Preprocess Configuration
preprocess:
  metadata:
    data_path: data/sample_data.csv
    target_column: KOGDP...D
    exclude_columns: [date]
    sample_start: null
    sample_end: null
  
  data_quality:
    remove_sparse_rows: true
    sparse_threshold: 0.8
    clip_outliers: true
    outlier_std: 3.0
  
  feature_engineering:
    window_summarizer:
      enabled: true
      lags: [1, 2, 3, 5, 10]
      windows: [5, 10, 20, 40]
      functions: [mean, std, min, max, median, skew, kurt]
  
  global_preprocessing:
    imputation:
      method: ffill_bfill  # For non-DFM/DDFM models
    scaling:
      enabled: true
      type: robust
    transformations:
      enabled: []

# Series Configuration
series:
  - series_id: KO3YEARC
    frequency: m
    transformation: chg
    blocks: [Block_Global, Block_Investment]
    release: 1
    scaler: null  # null = use global scaler
    impute: null  # null = use global or Kalman filter (DFM/DDFM)
```

### Key Features

1. **Unified Structure**: All experiment settings in one file
2. **Conditional Imputation**: 
   - DFM/DDFM: Uses Kalman filter (built-in, no explicit imputation needed)
   - Other models: Uses explicit imputation (ffill_bfill, mean, median, forecaster)
3. **Per-Series Overrides**: Each series can override global scaler/impute settings
4. **Window Summarizer**: Automatic feature engineering with lags and rolling windows

## Model Configs (`config/model/`)

Model-specific hyperparameters:
- `dfm.yaml` - DFM model parameters
- `ddfm.yaml` - DDFM model parameters
- `arma.yaml` - ARMA model parameters
- `tft.yaml` - Temporal Fusion Transformer parameters
- `xgboost.yaml` - XGBoost parameters

## Preprocess Configs (`config/preprocess/`)

**Note**: These are now templates/reference files. In the new structure, preprocessing is embedded in experiment configs. These files are kept for:
- Reference and documentation
- Backward compatibility
- Template generation

## Series Configs (`config/series/`)

**Note**: These are legacy files. In the new structure, series configurations are embedded in experiment configs. These files are kept for:
- Reference
- Migration purposes
- Template generation

## Configuration Hierarchy

1. **Experiment Config** (primary) - Contains all settings
2. **Model Config** (referenced) - Model-specific hyperparameters
3. **Preprocess Config** (legacy) - Reference templates only
4. **Series Config** (legacy) - Reference templates only

## Migration Guide

### Old Structure
```yaml
# experiment/{id}.yaml
data:
  path: data/sample_data.csv
preprocessing:
  config_path: config/preprocess/{id}.yaml
series: [KO3YEARC, ...]

# preprocess/{id}.yaml (separate file)
imputation:
  method: ffill_bfill
scaling:
  type: robust

# series/{series_id}.yaml (separate files)
frequency: m
transformation: chg
blocks: [Block_Global]
```

### New Structure
```yaml
# experiment/{id}.yaml (unified)
preprocess:
  global_preprocessing:
    imputation:
      method: ffill_bfill
    scaling:
      type: robust
series:
  - series_id: KO3YEARC
    frequency: m
    transformation: chg
    blocks: [Block_Global]
```

## Best Practices

1. **Use Experiment Configs**: All new experiments should use the unified structure
2. **Model Type Awareness**: 
   - DFM/DDFM: Imputation handled by Kalman filter
   - Other models: Specify explicit imputation method
3. **Per-Series Overrides**: Use sparingly, prefer global settings
4. **Window Summarizer**: Enable for feature-rich models (XGBoost, TFT)
5. **Scaling**: Use RobustScaler for economic data with outliers

## UI Integration

The UI left sidebar displays the config tree:
```
Experiment
├── Name, Model Type, Description
├── Preprocess
│   ├── Metadata (data path, target, exclude columns)
│   ├── Data Quality
│   ├── Feature Engineering (Window Summarizer)
│   └── Global Preprocessing (imputation, scaling)
└── Series
    ├── KO3YEARC (frequency, transformation, blocks, release, scaler, impute)
    └── [Other series...]
```

Right side: Training progress and monitoring.

