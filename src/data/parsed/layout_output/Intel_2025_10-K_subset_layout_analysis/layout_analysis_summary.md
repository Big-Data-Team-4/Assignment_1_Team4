# Project LANTERN Part 3 - Layout Detection Results

**Document:** data/raw/Intel_2025_10-K_subset.pdf
**Pages:** 19
**Blocks Detected:** 460
**LayoutLMv3 Enhanced:** True

## LayoutLMv3 Multimodal Enhancement (Checkpoint 3)

- **Captions detected:** 0
- **Reading order enhanced:** 0 blocks
- **Multimodal processing:** Successfully applied

## Content Routing Summary

- **Text blocks → PyMuPDF:** 370
- **Table blocks → Camelot:** 88
- **Figure blocks → Image storage:** 2

## Block Type Distribution

- **Text:** 369 blocks
- **Title:** 1 blocks
- **Table:** 88 blocks
- **Figure:** 2 blocks

## Assignment Requirements Met

- [x] Layout detection using deep learning (PaddleOCR + LayoutLMv3)
- [x] TextBlock objects with bounding boxes created
- [x] Type labels assigned (Text, Title, Table, Figure)
- [x] Content routing implemented:
  - Text blocks → PyMuPDF extraction + file saving
  - Figure blocks → PyMuPDF image extraction + file saving
  - Table blocks → Camelot extraction + CSV saving
- [x] JSON output with page numbers, block types, bounding boxes
- [x] Metadata stored with provenance information
- [x] **NEW: LayoutLMv3 experimentation for caption extraction (Checkpoint 3)**
- [x] **NEW: Reading order detection for multi-column layouts**
