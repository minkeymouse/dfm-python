# sktime (Polars 지원) Transformations 통합 계획

## 목표

sktime의 `ColumnTransformer` (Polars 지원)를 활용하여 dfm-python의 transformation과 standardization을 통합하고, polars DataFrame을 직접 사용할 수 있도록 구조 개선.

## 현재 구조 분석

### 현재 전처리 위치
1. **`dataloader/loader.py`**: `transform_data()` - transformation만 적용 (config 기반)
2. **`core/estimation.py`**: `standardize_data()` - standardization만 적용
3. **`models/dfm.py`**: `DFMLinear.fit()` - standardization 적용
4. **`models/ddfm.py`**: `DDFM.fit()` - standardization 적용

### 문제점
- Transformation과 standardization이 분리되어 있음
- `DFMLinear.fit()`에서는 transformation이 적용되지 않음 (config의 transformation 무시)
- Polars DataFrame → numpy array 변환 필요
- 중복 코드 (여러 곳에서 standardization)

## 제안 구조

### 1. 새로운 모듈 구조

```
dfm-python/src/dfm_python/
├── transformations/
│   ├── __init__.py          # Public API export
│   ├── scaler.py            # DFMScaler 클래스 (핵심)
│   ├── transformers.py       # Custom transformation 함수들
│   └── sktime_wrappers.py   # sktime optional dependency 처리
```

### 2. DFMScaler 클래스 설계

```python
# transformations/scaler.py

class DFMScaler:
    """DFM Scaler using sktime transformers with Polars support.
    
    This class handles:
    1. Series-specific transformations (lin, log, chg, ch1, pch, pc1, pca, cch, cca)
    2. Frequency-aware transformations (based on series frequency)
    3. Global standardization (mean=0, std=1 for entire dataset)
    
    Uses sktime's ColumnTransformer with Polars support for efficient processing.
    """
    
    def __init__(self, config: DFMConfig):
        """Initialize scaler from DFMConfig.
        
        Parameters
        ----------
        config : DFMConfig
            DFM configuration containing series transformations and frequencies
        """
        pass
    
    def fit(self, X: pl.DataFrame, y=None) -> 'DFMScaler':
        """Fit the scaler to data.
        
        Parameters
        ----------
        X : pl.DataFrame
            Input data (T x N), where T is time periods and N is number of series.
            Column names should match series_ids from config.
        y : optional
            Not used, present for sklearn compatibility
        
        Returns
        -------
        self : DFMScaler
            Fitted scaler instance
        """
        pass
    
    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Transform data using fitted scaler.
        
        Parameters
        ----------
        X : pl.DataFrame
            Input data (T x N)
        
        Returns
        -------
        X_transformed : pl.DataFrame
            Transformed and standardized data
        """
        pass
    
    def fit_transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Fit and transform in one step."""
        pass
    
    def inverse_transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Inverse transform (for forecasting/backtesting).
        
        Note: Some transformations (pch, pc1, etc.) are not fully invertible
        due to division by zero handling. Returns best-effort reconstruction.
        """
        pass
    
    @property
    def Mx(self) -> np.ndarray:
        """Mean values used for standardization (N,)"""
        pass
    
    @property
    def Wx(self) -> np.ndarray:
        """Standard deviation values used for standardization (N,)"""
        pass
```

### 3. Optional Dependency 처리

```python
# transformations/sktime_wrappers.py

try:
    from sktime.transformations.compose import (
        ColumnTransformer,
        TransformerPipeline
    )
    from sktime.transformations.series.log import LogTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.transformations.series.func_transform import FunctionTransformer
    from sklearn.preprocessing import StandardScaler
    HAS_SKTIME = True
except ImportError:
    HAS_SKTIME = False
    ColumnTransformer = None
    TransformerPipeline = None
    LogTransformer = None
    Differencer = None
    FunctionTransformer = None
    StandardScaler = None

def check_sktime_available():
    """Check if sktime is available."""
    if not HAS_SKTIME:
        raise ImportError(
            "sktime is required for DFMScaler. "
            "Install it with: pip install dfm-python[transform]"
        )
```

### 4. 통합 전략

#### 전략 A: DFMScaler를 필수로 사용 (권장)

**장점**:
- 일관된 transformation/standardization 처리
- Polars 직접 지원
- 코드 중복 제거

**구현**:
1. `DFMLinear.fit()`에서 `DFMScaler` 사용
2. `core/estimation.py`의 `standardize_data()` 제거 (또는 deprecated)
3. `dataloader/loader.py`의 `transform_data()` 제거 (또는 deprecated)

**단점**:
- sktime이 optional dependency이므로, 없으면 fallback 필요

#### 전략 B: DFMScaler를 선택적으로 사용

**장점**:
- Backward compatibility 유지
- sktime 없이도 동작 가능

**구현**:
1. `DFMLinear.fit()`에서 `use_sktime` 플래그로 선택
2. sktime 없으면 기존 방식 사용

**단점**:
- 두 가지 코드 경로 유지 필요
- 복잡도 증가

## 권장 구현: 전략 A (DFMScaler 필수)

### 단계별 구현

#### Step 1: transformations 모듈 생성

```python
# transformations/__init__.py
from .scaler import DFMScaler
from .transformers import (
    make_pch_transformer,
    make_pc1_transformer,
    make_pca_transformer,
    make_cch_transformer,
    make_cca_transformer,
)

__all__ = [
    'DFMScaler',
    'make_pch_transformer',
    'make_pc1_transformer',
    'make_pca_transformer',
    'make_cch_transformer',
    'make_cca_transformer',
]
```

#### Step 2: DFMScaler 구현

```python
# transformations/scaler.py

import numpy as np
import polars as pl
from typing import Optional
from ..config import DFMConfig
from .sktime_wrappers import (
    check_sktime_available,
    ColumnTransformer,
    TransformerPipeline,
    LogTransformer,
    Differencer,
    FunctionTransformer,
    StandardScaler,
)
from .transformers import (
    make_pch_transformer,
    make_pc1_transformer,
    make_pca_transformer,
    make_cch_transformer,
    make_cca_transformer,
    log_transform,
)

# Frequency → lag 매핑
FREQ_TO_LAG_YOY = {'m': 12, 'q': 4, 'sa': 2, 'a': 1}
FREQ_TO_LAG_STEP = {'m': 1, 'q': 3, 'sa': 6, 'a': 12}

def get_periods_per_year(freq: str) -> int:
    """Get number of periods per year for a frequency"""
    return FREQ_TO_LAG_YOY.get(freq, 12)

def get_annual_factor(freq: str, step: int) -> float:
    """Get annualization factor for a frequency and step"""
    periods_per_year = get_periods_per_year(freq)
    return periods_per_year / step

class DFMScaler:
    """DFM Scaler using sktime transformers with Polars support."""
    
    def __init__(self, config: DFMConfig):
        check_sktime_available()
        
        self.config = config
        self.transformations = [s.transformation for s in config.series]
        self.frequencies = [s.frequency for s in config.series]
        self.series_ids = config.get_series_ids()
        
        # 각 series별로 TransformerPipeline 구성
        transformers = []
        for i, (trans, freq) in enumerate(zip(self.transformations, self.frequencies)):
            trafo_pipeline = self._create_transformation_pipeline(trans, freq)
            transformers.append((f'series_{i}', trafo_pipeline, i))
        
        # ColumnTransformer 구성 (Polars 지원!)
        self.column_transformer = ColumnTransformer(transformers)
        
        # 전체 pipeline: ColumnTransformer → StandardScaler
        self.pipeline = TransformerPipeline([
            ("transform", self.column_transformer),
            ("scaler", StandardScaler())
        ])
        
        # Polars 출력 설정
        self.pipeline.set_output(transform="polars")
        
        # Standardization 파라미터 (fit 후 설정됨)
        self._Mx: Optional[np.ndarray] = None
        self._Wx: Optional[np.ndarray] = None
    
    def _create_transformation_pipeline(self, trans: str, freq: str) -> TransformerPipeline:
        """Create transformation pipeline for a single series."""
        if trans == 'lin':
            return TransformerPipeline([
                FunctionTransformer(func=lambda x: x)
            ])
        elif trans == 'log':
            return TransformerPipeline([
                FunctionTransformer(func=log_transform)
            ])
        elif trans == 'chg':
            lag = FREQ_TO_LAG_STEP.get(freq, 1)
            return TransformerPipeline([
                Differencer(lags=[lag])
            ])
        elif trans == 'ch1':
            lag = FREQ_TO_LAG_YOY.get(freq, 1)
            return TransformerPipeline([
                Differencer(lags=[lag])
            ])
        elif trans == 'pch':
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            return TransformerPipeline([
                make_pch_transformer(step)
            ])
        elif trans == 'pc1':
            year_step = FREQ_TO_LAG_YOY.get(freq, 12)
            return TransformerPipeline([
                make_pc1_transformer(year_step)
            ])
        elif trans == 'pca':
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = get_annual_factor(freq, step)
            return TransformerPipeline([
                make_pca_transformer(step, annual_factor)
            ])
        elif trans == 'cch':
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            return TransformerPipeline([
                make_cch_transformer(step)
            ])
        elif trans == 'cca':
            step = FREQ_TO_LAG_STEP.get(freq, 1)
            annual_factor = get_annual_factor(freq, step)
            return TransformerPipeline([
                make_cca_transformer(step, annual_factor)
            ])
        else:
            # Default: Identity
            return TransformerPipeline([
                FunctionTransformer(func=lambda x: x)
            ])
    
    def fit(self, X: pl.DataFrame, y=None) -> 'DFMScaler':
        """Fit the scaler to data."""
        # Ensure column order matches series_ids
        if list(X.columns) != self.series_ids:
            X = X.select(self.series_ids)
        
        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X, y)
        
        # Extract StandardScaler parameters
        self.scaler = self.pipeline.steps[1][1]  # StandardScaler
        self._Mx = self.scaler.mean_
        self._Wx = self.scaler.scale_
        
        return self
    
    def transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Transform data using fitted scaler."""
        # Ensure column order matches series_ids
        if list(X.columns) != self.series_ids:
            X = X.select(self.series_ids)
        
        return self.pipeline.transform(X, y)
    
    def fit_transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        """Inverse transform (for forecasting/backtesting)."""
        # Note: Some transformations are not fully invertible
        # This is a best-effort implementation
        return self.pipeline.inverse_transform(X, y)
    
    @property
    def Mx(self) -> np.ndarray:
        """Mean values used for standardization (N,)"""
        if self._Mx is None:
            raise ValueError("Scaler has not been fitted yet.")
        return self._Mx
    
    @property
    def Wx(self) -> np.ndarray:
        """Standard deviation values used for standardization (N,)"""
        if self._Wx is None:
            raise ValueError("Scaler has not been fitted yet.")
        return self._Wx
```

#### Step 3: DFMLinear.fit() 수정

```python
# models/dfm.py

from ..transformations import DFMScaler

class DFMLinear(BaseFactorModel):
    def fit(self, X: Union[np.ndarray, pl.DataFrame], config: Optional[DFMConfig] = None, ...):
        # ... (기존 초기화 코드) ...
        
        # Convert to polars DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pl.DataFrame(X, schema=series_ids)
        elif not isinstance(X, pl.DataFrame):
            raise TypeError(f"X must be np.ndarray or pl.DataFrame, got {type(X)}")
        
        # Create and fit scaler
        scaler = DFMScaler(config)
        X_transformed = scaler.fit_transform(X)
        
        # Convert to numpy for estimation (필요한 경우)
        X_numpy = X_transformed.to_numpy()
        
        # Extract standardization parameters
        Mx = scaler.Mx
        Wx = scaler.Wx
        
        # ... (기존 estimation 코드) ...
        
        # 결과에 scaler 저장 (inverse_transform용)
        result.scaler = scaler
        result.Mx = Mx
        result.Wx = Wx
        
        return result
```

#### Step 4: core/estimation.py 수정

```python
# core/estimation.py

def _dfm_core(X: np.ndarray, config: DFMConfig, ...):
    """DFM estimation core function.
    
    Note: This function now expects pre-transformed and standardized data.
    Use DFMScaler before calling this function.
    """
    # ... (기존 코드, standardization 제거) ...
    
    # X는 이미 transformed + standardized 되어 있다고 가정
    # standardization 코드 제거
```

### 5. Backward Compatibility

#### Option 1: Deprecation Warnings

```python
# dataloader/loader.py

def transform_data(...):
    """Transform data using config transformations.
    
    .. deprecated:: 0.2.0
        Use DFMScaler instead. This function will be removed in 0.3.0.
    """
    warnings.warn(
        "transform_data() is deprecated. Use DFMScaler instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... (기존 코드) ...
```

#### Option 2: Fallback 구현

```python
# transformations/scaler.py

class DFMScaler:
    def __init__(self, config: DFMConfig, use_sktime: bool = True):
        if use_sktime:
            try:
                check_sktime_available()
                self._use_sktime = True
            except ImportError:
                if use_sktime:
                    raise
                self._use_sktime = False
        else:
            self._use_sktime = False
        
        if not self._use_sktime:
            # Fallback to native implementation
            self._init_native(config)
        else:
            self._init_sktime(config)
```

## 마이그레이션 가이드

### 사용자 관점

#### Before (현재)
```python
from dfm_python import DFM
from dfm_python.dataloader import load_data

model = DFM()
X, Time, Z = load_data('data.csv', config)
# load_data 내부에서 transformation 적용
result = model.fit(X, config)
# fit 내부에서 standardization 적용
```

#### After (새로운 구조)
```python
from dfm_python import DFM
from dfm_python.transformations import DFMScaler
import polars as pl

model = DFM()
X = pl.read_csv('data.csv')  # Polars 직접 사용
scaler = DFMScaler(config)
X_transformed = scaler.fit_transform(X)
result = model.fit(X_transformed, config)
# standardization은 이미 scaler에서 적용됨
```

## 테스트 전략

1. **Unit Tests**: DFMScaler의 각 transformation 테스트
2. **Integration Tests**: DFMLinear.fit()과의 통합 테스트
3. **Backward Compatibility Tests**: 기존 코드와의 호환성 테스트
4. **Performance Tests**: Polars vs numpy 변환 성능 비교

## 의존성 관리

### pyproject.toml

```toml
[project.optional-dependencies]
transform = [
    "sktime>=0.27.0",
]
all = [
    "dfm-python[hydra,db,dev,deep,transform]",
]
```

### 설치 방법

```bash
# sktime 포함 설치
pip install dfm-python[transform]

# 또는 전체 설치
pip install dfm-python[all]
```

## 장점 요약

1. **Polars 직접 지원**: pandas 변환 불필요
2. **일관된 처리**: transformation + standardization 통합
3. **코드 중복 제거**: 여러 곳에서 standardization 하지 않음
4. **확장성**: 새로운 transformation 쉽게 추가 가능
5. **표준 인터페이스**: sklearn/sktime 호환
6. **성능**: Polars의 빠른 처리 속도 활용

## 다음 단계

1. ✅ 구조 설계 (현재)
2. ⏳ transformations 모듈 구현
3. ⏳ DFMScaler 구현
4. ⏳ DFMLinear.fit() 통합
5. ⏳ 테스트 작성
6. ⏳ 문서화
7. ⏳ 마이그레이션 가이드 작성

