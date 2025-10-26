# robustloss-lab

Robust classification loss toolkit (CCE/SCCE, Focal, GCE) with simple experiment runners for noisy labels and outliers.  
노이즈/아웃라이어 환경에서 강건한 분류 실험을 위한 손실 함수 라이브러리와 실험 러너를 제공합니다.

---

## ✨ Features
- Losses: CE, GCE, Focal, **CCE**, **SCCE**
- Experiment runners: `run_experiment`, `run_clean_vs_noise`, `run_clean_vs_outlier`
- Noise / Outlier injection utilities and metadata logging
- Minimal, sklearn-like workflow with PyTorch backend

---

## 📦 Install

> **Note**: PyTorch는 CUDA/CPU 환경에 맞춰 별도 설치하세요.  
> 예:  
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

```bash
pip install robustloss-lab
```

---

## 🚀 Quick start

```python
from robustloss import (
    DatasetSchema, TaskType,
    make_loss, run_experiment, plot_history,
    NoiseConfig, OutlierConfig, pct_drop
)

# 1) Define dataset schema
schema = DatasetSchema(
    name="uci_wine",
    target_name="class",
    task_type=TaskType.MULTICLASS
)

# 2) Choose a loss (e.g., SCCE)
loss_fn = make_loss("scce", eps=1e-3)

# 3) Run a quick experiment
model, hist, report = run_experiment(
    df, schema,
    loss_fn=loss_fn, loss_name="SCCE",
    epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4,
    optimizer_name="adam", seed=42
)

print(report)  # dict(test_acc=..., test_f1=..., noise_meta=..., outlier_meta=...)
plot_history([hist], ["SCCE"])
```

---

## 📂 Modules

**Core (required submodules)**  
- `schemas.py` — 데이터 스키마 정의  
- `preprocess.py`, `datamod.py` — 전처리 / split  
- `loss_functions.py` — CE, GCE, Focal, CCE, SCCE  
- `models.py` — Logistic/Softmax linear classifiers  
- `train_many.py` — 학습 루프, early stopping, 시각화  

**Optional submodules**  
- `noise_types.py`, `apply_noise.py` — 라벨/피처 노이즈  
- `outliers.py`, `apply_outliers.py` — 아웃라이어 생성/주입  

> 선택 모듈은 환경에 따라 미포함일 수 있으며, 패키지 import 시 `None`으로 바인딩될 수 있습니다.

---


## 🧩 API Sketch (요약)

### Experiment Runners
```python
model, hist, report = run_experiment(df, schema, loss_fn, ...)

[h_c, h_n], labels, df_res = run_clean_vs_noise(df, schema, loss_fn=loss_fn, ...)

[h_c, h_o], labels, df_res = run_clean_vs_outlier(df, schema, loss_fn=loss_fn, ...)
```

### Metadata
- `noise_meta`: 라벨/피처 노이즈 적용 정보 (전이행렬, 인덱스, 시드 등)  
- `outlier_meta`: 아웃라이어 주입 요약 (비율, |z|-통계, m범위/시드 등)

---

## 🧪 Important Prototypes

### **Setting**

#### **DatasetSchema**
```python
@dataclass(frozen=True, slots=True)
class DatasetSchema(
    name: str
    target_name: str
    task_type: Optional[TaskType] = None       # Task_Type.BINARY / Task_Type.MULTICLASS  None일 시 전처리 시 자동감지 
    numeric_features: Optional[Sequence[str]] = None
    categorical_features: Optional[Sequence[str]] = None
    drop_features: Sequence[str] = field(default_factory=tuple)
)
```

##### **예시**
```python
schema = DatasetSchema(
    name="uci_wine",
    target_name="class",
    task_type=TaskType.MULTICLASS
)
```

#### **NoiseConfig**
```python
@dataclass(slots=True)
class NoiseConfig:
    kind: Literal["none", "label", "feature", "both"] = "none"

    # --- Label Noise ---
    label_mode: Optional[LabelMode] = None     # 노이즈 종류 ["symmetric", "pairflip", "classdep", "instancedep"]
    label_rate: float = 0.0                    # 노이즈율 η
    seed_label: Optional[int] = None           # 라벨 노이즈 랜덤 시드
    pairflip_pairs: Optional[Dict[int, int]] = None  # pairflip용 클래스 쌍
    classdep_etas: Optional[np.ndarray] = None       # class-dependent 노이즈율 벡터
    instancedep_tau: float = 1.0                     # instance-dependent scaling factor

    # --- Feature Noise ---
    feature_mode: Optional[FeatureMode] = None # 노이즈 종류 ["gaussian", "spike"]
    seed_feature: Optional[int] = None         # 피처 노이즈 랜덤 시드
    feature_frac: float = 0.0                  # 전체 샘플 중 노이즈 적용 비율
    feature_scale: float = 0.0                 # Gaussian scale (std 비율)
    spike_frac: float = 0.0                    # Spike 적용 비율
    spike_value: float = 10.0                  # Spike 값 (outlier 크기)
```

#### **OutlierConfig**
```python
@dataclass(frozen=True, slots=True)
class OutlierConfig:
    spike_value: float = 10.0                  #
    rate: float = 0.1                          # outlier 비율 (0.1=10%)
    zmin: float = 3.0                          # z-score 하한 (3σ)
    zmax: float = 5.0                          # z-score 상한 (5σ)
    mmin: int = 1                              # 한 행에서 변조할 feature 최소 개수
    mmax: Optional[int] = None                 # 한 행에서 변조할 feature 최대 개수 (None=전체)
    two_side: bool = True                      # True: ±, False: +만
    seed_outlier: Optional[int] = 42           # Outlier 시드
    target: Iterable[str] = ("train",)         # 주입할 split ("train","val","test")
```

#### **run_experiment**
```python
run_experiment(
    df,                                        # Dataframe
    schema_or_name: Union[str, DatasetSchema], # DatasetSchema 객체 또는 registry.py 의 str
    loss_fn,                       # 사용할 손실 함수 (예: CE, GCE, CCE 등)

    # -------------------------
    # 학습 하이퍼파라미터 (기본 프리셋)
    # -------------------------
    epochs: int = 50,              # 최대 학습 epoch 수
    batch_size: int = 64,          # 미니배치 크기
    lr: float = 1e-3,              # 학습률 (learning rate)
    weight_decay: float = 1e-4,    # L2 정규화 강도 (weight decay)

    optimizer_name: str = "adam",  # 옵티마이저 종류 ("adam" | "sgd" | "sgd_momentum")
    loss_name: str = "loss",       # 손실 함수 이름 (로그 출력/플롯 라벨링용)
    patience: int = 10,            # Early Stopping patience (val_loss 개선 없을 시 중단)

    # -------------------------
    # 실행 환경
    # -------------------------
    seed: int = 42,                # 랜덤 시드 (재현성 보장)
    device: str | None = None,     # 연산 장치 지정 ("cuda", "cpu", None이면 자동)

    # -------------------------
    # noise setting
    # -------------------------
    noise: Optional[NoiseConfig] = None,        # 노이즈 구성 객체 (label/feature 종류, 비율, 시드 등)
    noise_targets: Iterable[str] = ("train",),  # 노이즈 적용 대상 split ("train","val","test" 중 선택)

    # -------------------------
    # outlier setting
    # -------------------------
    outliers: Optional[OutlierConfig] = None,   # outlier 구성 객체
):

return model, hist, dict(test_acc=test_acc, test_f1=test_f1, noise_meta=noise_meta, outlier_meta=outlier_meta)

```

#### **noise_meta**
```python
noise_meta: Dict[str, Any] = {
    "train": {   # train split에 노이즈 적용 시
        "kind": "label" | "feature" | "both",   # 적용된 노이즈 종류
        # --- Label Noise ---
        "label_mode": "symmetric" | "pairflip" | "classdep" | "instancedep",
        "label_rate": float,        # 노이즈율 η
        "seed_label": int,          # 라벨 노이즈 시드
        "label_idx": np.ndarray,    # 변경된 샘플 인덱스
        "label_orig": np.ndarray,   # 변경 전 라벨
        "transition": np.ndarray,   # 전이행렬 (pairflip/classdep 모드에서 사용)

        # --- Feature Noise ---
        "feature_mode": "gaussian" | "spike",
        "seed_feature": int,        # 피처 노이즈 시드
        "feature_idx": np.ndarray,  # 노이즈가 적용된 샘플 인덱스
    },
    "val": None | {...},   # val split에도 적용된 경우
    "test": None | {...},  # test split에도 적용된 경우
}
```

#### **outlier_meta**
```python
outlier_meta: Dict[str, Any] = {
    "train": {   # train split에 outlier 적용 시
        "n_added": int,         # 추가된 outlier 행 개수
        "rate": float,          # 전체 샘플 대비 outlier 비율
        "m_avg": float,         # 한 행에서 변조된 feature 수 평균
        "m_min": int,           # 변조된 feature 수 최소
        "m_max": int,           # 변조된 feature 수 최대

        # --- Z-score 요약 (절대값 기준) ---
        "z_avg": float,         # 모든 outlier feature |z| 평균
        "z_std": float,         # 모든 outlier feature |z| 표준편차
        "z_min": float,         # 관측된 |z| 최소값
        "z_max": float,         # 관측된 |z| 최대값

        # --- Config 기록 ---
        "zmin": float,          # 설정된 z-score 하한
        "zmax": float,          # 설정된 z-score 상한
        "two_side": bool,       # ± 방향 허용 여부
        "seed_outlier": int,    # outlier 생성 시드
    },
    "val": None | {...},   # val split에도 적용된 경우
    "test": None | {...},  # test split에도 적용된 경우
}
```

#### **run_clean_vs_noise**
```python
run_clean_vs_noise(
    df,                            # Dataframe
    schema_or_name,                # DatasetSchema 객체 또는 registry.py 의 str
    *,
    loss_fn,                       # 사용할 손실 함수 (예: CE, GCE, CCE 등)
    loss_name: str = "loss",       # 손실 함수 이름 (로그 출력/플롯 라벨링용)
    seed: int = 42,                # 랜덤 시드 (재현성 보장)

    # -------------------------
    # 학습 하이퍼파라미터 (기본 프리셋)
    # -------------------------
    epochs: int = 50,              # 최대 학습 epoch 수
    batch_size: int = 64,          # 미니배치 크기
    lr: float = 1e-3,              # 학습률 (learning rate)
    weight_decay: float = 1e-4,    # L2 정규화 강도 (weight decay)

    optimizer_name: str = "adam",  # 옵티마이저 종류 ("adam" | "sgd" | "sgd_momentum")
    patience: int = 10,            # Early Stopping patience (val_loss 개선 없을 시 중단)
    device: str | None = None,

    # -------------------------
    # noise setting
    # -------------------------
    noise_cfg: Optional["NoiseConfig"] = None,
    noise_targets: Iterable[str] = ("train",),
):

return ( [hist_c, hist_n], ["CLEAN", "NOISE"], df_results )

```

#### **run_clean_vs_outlier**
```python
run_clean_vs_outlier(
    df,                               # 전체 데이터셋 (pandas DataFrame)
    schema_or_name,                   # DatasetSchema 객체 또는 str (등록된 스키마 이름)

    *,
    loss_fn,                          # 사용할 손실 함수 (callable, 예: ce_loss, gce_loss 등)
    loss_name: str = "loss",          # 손실 함수 이름 (로그/라벨링용 표시)

    seed: int = 42,                   # 랜덤 시드 (데이터 분할/학습 재현성 보장)

    # -------------------------
    # 학습 하이퍼파라미터
    # -------------------------
    epochs: int = 50,                 # 최대 학습 epoch 수
    batch_size: int = 64,             # 미니배치 크기
    lr: float = 1e-3,                 # 학습률 (learning rate)
    weight_decay: float = 1e-4,       # L2 정규화 강도 (weight decay)

    optimizer_name: str = "adam",     # 옵티마이저 종류 ("adam" | "sgd" | "sgd_momentum")
    patience: int = 10,               # Early Stopping patience (val_loss 개선 없을 시 중단)

    # -------------------------
    # 실행 환경
    # -------------------------
    device: str | None = None,        # 연산 장치 ("cuda", "cpu", None → 자동 감지)

    # -------------------------
    # 아웃라이어 설정
    # -------------------------
    outlier_cfg: Optional[OutlierConfig] = None,  # OutlierConfig 객체 (rate, z 범위, m 범위 등 설정)
):

return ([hist_c, hist_o], ["CLEAN", "OUTLIER"], df_results)

```

---

## 📝 Patch Notes
- 1.0.0: First release  
- 1.0.1: 로그 수정  
- 2.0.0: 라이브러리 모듈화, 노이즈 추가 ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.0.0))  
- 2.0.1: 버그 수정  
- 2.0.2: 버그 수정  
- 2.0.3: 버그 수정  
- 2.0.4: 버그 수정 ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.0.4))  
- 2.1.0: Outliers 모듈 추가 ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.0))  
- 2.1.1: 얕은 복사로 인한 경고 해결 ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.1))  
- 2.1.2: Outlier 메타 추가 ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.2))  
- 2.1.3: init 수정 ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.3))  
- 2.1.4: SCCE 함수 clamp 1로 재정의, 구분을 위해 q_t로 변경 ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.4))  
- 2.1.5: PyPI 패키징 정리, README 개선, 배포명 `robustloss-lab`, import `robustloss` ([패치내역](https://github.com/RosePasta22/ML-DL-Seminar/releases/tag/v2.1.5))
- 2.1.6: Move package into new repo; update pyproject ([패치내역](https://github.com/RosePasta22/Robustloss-Lab/releases/tag/v2.1.6))
- **2.1.7: Update README** ([패치내역](https://github.com/RosePasta22/Robustloss-Lab/releases/tag/v2.1.7))

---

## 📎 Links
- [Latest Release](https://github.com/RosePasta22/Robustloss-Lab/releases/tag/v2.1.7)  
- [Homepage / Source](https://github.com/RosePasta22/Robustloss-Lab)
