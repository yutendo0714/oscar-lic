# Threat Model for OCR-Aware Learned Compression

## Assets

- correct decoded visual content
- exact textual semantics
- bounded file size and decode cost
- model/checkpoint integrity
- privacy of text content
- bitstream interoperability

## Adversaries and failures

### T1: Benign image corruption

Noise, blur, resizing, low light, color shifts and camera/ISP artifacts before encoding.

### T2: Rate-inflation input attack

Small input perturbation causes excessive latent entropy/file size while preserving visible input similarity.

### T3: Semantic distortion attack

Perturbation causes decoded text/OCR to change under bounded input norm or perceptual difference.

### T4: Malicious bitstream

Corrupt lengths, oversized dimensions, crafted entropy payload, packet loss, random/burst flips.

### T5: Supply-chain attack

Backdoored checkpoint, dependency, external repository or pretrained OCR model.

### T6: Model drift

Decoder or OCR service is updated after encoding; latent/task interface changes.

### T7: Privacy attacker

Attacker observes bitstream or selected gate and infers sensitive text, even without authorized full decoding.

### T8: Semantic side-channel exposure

Track B plaintext strings or coordinates reveal data more directly than a normal image codec.

## Assumptions

- Cryptographic confidentiality is outside the codec unless an encrypted transport/profile is explicitly added.
- CRC is not authentication.
- The attacker may know architecture and weights.
- White-box and transfer black-box attacks are both evaluated.

## Metrics

- bitrate amplification factor
- PSNR/LPIPS and CER/WER change
- attack success at norm/perceptual constraint
- crash/OOM/time amplification
- text extraction accuracy from bitstream features
- clean RD degradation from defense

## Defenses to evaluate

- input adversarial training for rate and OCR loss
- bounded entropy parameterization and rate cap
- section length validation
- CRC and packetized resynchronization
- unequal FEC for high-utility packets
- signed model manifests and safetensors
- privacy adversary/gradient reversal for utility latent
- decoder/OCR version conformance suite

## Security reporting rule

A defense is not successful if it merely shifts rate inflation to uncounted metadata, destroys clean small-text performance, or assumes the attacked OCR is the same as the training teacher.
