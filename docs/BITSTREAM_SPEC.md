# OSCAR Experimental Bitstream Specification

**Status:** non-normative research format, version 0.1  
**Byte order:** little-endian  
**Primary extension:** `.oscr`

## 1. Goals

- unambiguous total-rate accounting
- independent base decoding
- optional enhancement and error protection
- stream corruption detection
- no plaintext text in Track A
- forward-compatible skipping of unknown optional sections

## 2. File layout

```text
FixedHeader
SectionTable[N]
SectionPayloads
FileCRC32
```

### 2.1 FixedHeader

| Field | Type | Bytes | Meaning |
|---|---:|---:|---|
| magic | char[4] | 4 | ASCII `OSCR` |
| major | uint8 | 1 | incompatible format version |
| minor | uint8 | 1 | backward-compatible version |
| flags | uint16 | 2 | feature flags |
| width | uint32 | 4 | original width |
| height | uint32 | 4 | original height |
| channels | uint8 | 1 | normally 3 |
| bit_depth | uint8 | 1 | normally 8 |
| color_space | uint8 | 1 | enum |
| reserved0 | uint8 | 1 | zero |
| model_id | uint32 | 4 | registry identifier, not a URL |
| model_version | uint32 | 4 | decoder model version |
| section_count | uint16 | 2 | N |
| header_bytes | uint16 | 2 | fixed + table bytes |
| total_bytes | uint64 | 8 | complete file length |
| header_crc32 | uint32 | 4 | CRC through section table with this field zeroed |

Fixed header total: 44 bytes.

### 2.2 Section table entry

| Field | Type | Bytes |
|---|---:|---:|
| section_type | uint16 | 2 |
| codec | uint16 | 2 |
| flags | uint32 | 4 |
| offset | uint64 | 8 |
| length | uint64 | 8 |
| unprotected_length | uint64 | 8 |
| payload_crc32 | uint32 | 4 |
| dependency_mask | uint32 | 4 |

Entry total: 40 bytes.

## 3. Section types

| ID | Name | Required | Description |
|---:|---|---|---|
| 1 | BASE_HYPER | conditional | opaque upstream hyperstream |
| 2 | BASE_MAIN | yes | base latent stream |
| 3 | BASE_AUX | no | upstream auxiliary metadata |
| 10 | TEXT_GATE | no | entropy-coded candidate selection/index |
| 11 | TEXT_HYPER | no | enhancement hyperprior |
| 12 | TEXT_MAIN | no | enhancement payload |
| 13 | TEXT_PROTECTION | no | FEC/parity metadata |
| 20 | MODEL_METADATA | no | nonsemantic decoder parameters allowed by protocol |
| 30 | EXPERIMENT_METADATA | no in publication files | minimal reproducibility identifiers; not used by decoder |

Track A validation rejects sections containing OCR text strings, UTF-8 transcriptions or word coordinates not derivable from decoded base features.

## 4. Flags

- bit 0: text enhancement present
- bit 1: payload FEC present
- bit 2: deterministic refiner seed present
- bit 3: tiled coding
- bit 4: progressive truncation permitted
- bit 5: semantic side-channel — must be 0 for Track A

## 5. Base independence

A decoder that supports the declared base model must reconstruct `x_base` using only base sections and fixed header. Missing or corrupt text sections produce a base-only reconstruction and an error status, not silent partial text output.

## 6. Text gate encoding

The gate section begins with:

```text
candidate_layout_version: uint16
num_candidates: uint32
num_selected: uint32
probability_model_id: uint32
encoded_gate_bytes: remaining payload
```

Candidate geometry is determined by model version; per-image box coordinates are not sent unless encoded as part of the counted gate syntax.

## 7. Progressive truncation

A valid truncation point is a complete section or independently checksummed packet. Byte truncation inside an arithmetic-coded packet is invalid. For progressive experiments, split text payload into ordered packets and include each as a section or subpacket table.

## 8. Integrity and FEC

- CRC detects corruption; it does not correct it.
- FEC overhead is physically included in the section length and total bytes.
- Critical text packets may use stronger protection based on decoder-known ordering, not plaintext identity.
- File CRC32 covers all preceding bytes.

## 9. Rate computation

\[
bpp=\frac{8\times total\_bytes}{width\times height}.
\]

Per-section breakdown uses table lengths. Files with inconsistent `total_bytes`, offsets, overlap or CRC are invalid.

## 10. Model compatibility

`model_id/model_version` maps to a local registry containing decoder weights hash and architecture. A decoder must refuse unknown incompatible major versions rather than guessing.

## 11. Security limits

Before allocation:

- validate dimensions and maximum pixels
- validate section count and lengths against file size
- reject overlapping/out-of-range sections
- cap decompressed symbol counts
- do not deserialize arbitrary Python objects from metadata

## 12. Planned tests

- round-trip header pack/unpack
- section length mismatch rejection
- CRC corruption detection
- base-only decode after removing all text sections
- semantic-string scanner for Track A metadata
- exact total-byte accounting
