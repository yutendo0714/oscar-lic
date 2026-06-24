# Implementation Interfaces

## Core dataclasses

```python
@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    sha256: str
    width: int
    height: int
    dataset_id: str
    split: str

@dataclass
class StreamSection:
    section_type: int
    payload: bytes
    codec: int = 0
    flags: int = 0
    dependency_mask: int = 0

@dataclass
class CodecResult:
    reconstruction: Tensor
    bitstream_path: str
    total_bytes: int
    section_bytes: dict[str, int]
    encode_seconds: float
    decode_seconds: float
    metadata: dict

@dataclass
class OCRResult:
    raw_text: str
    normalized_text: str
    confidence: float | None
    boxes: list
    model_id: str
    model_revision: str
    normalization_profile: str
```

## Base codec protocol

```python
class Codec(Protocol):
    def compress(self, image: Tensor, output_path: Path) -> CodecResult: ...
    def decompress(self, bitstream_path: Path) -> CodecResult: ...
    def estimate_rate(self, image: Tensor) -> dict[str, float]: ...
```

## OCR protocol

```python
class OCRSystem(Protocol):
    def recognize_crops(self, crops: Tensor) -> list[OCRResult]: ...
    def detect_and_recognize(self, image: Tensor) -> OCRResult: ...
    def differentiable_outputs(self, image: Tensor) -> dict[str, Tensor]: ...
```

Held-out systems need not implement differentiable outputs.

## Utility protocol

```python
class CandidateLayout(Protocol):
    def enumerate(self, latent_shape) -> list[Candidate]: ...
    def gather(self, latent, candidate_ids): ...
    def scatter(self, values, candidate_ids, output_shape): ...

class UtilityPredictor(nn.Module):
    def forward(self, base_features, candidates) -> UtilityDistribution: ...

class Allocator(Protocol):
    def select(self, utilities, predicted_bits, budget_bits) -> list[int]: ...
```

## Evaluation output

One JSON per image per operating point. Aggregation never reads console text. Schema is in `evaluation/REPORTING_SCHEMA.json`.

## Error behavior

- missing checkpoint: fail fast
- unknown bitstream version: fail fast
- OCR timeout: mark metric missing with reason, do not treat as empty string
- detector finds no text: distinguish correct no-text image from false negative using annotations
- invalid UTF-8 metadata: reject
