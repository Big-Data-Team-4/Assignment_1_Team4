# Project LANTERN Part 3 - Layout Detection Results

**Document:** data\raw\Intel_2024_10-K.pdf
**Pages:** 20
**Blocks Detected:** 374

## Content Routing Summary

- **Text blocks → PyMuPDF:** 307
- **Table blocks → Camelot:** 60
- **Figure blocks → Image storage:** 7

## Block Type Distribution

- **Text:** 306 blocks
- **Title:** 1 blocks
- **Table:** 60 blocks
- **Figure:** 7 blocks

## Assignment Requirements Met

- [x] Layout detection using deep learning (PaddleOCR)
- [x] TextBlock objects with bounding boxes created
- [x] Type labels assigned (Text, Title, Table, Figure)
- [x] Content routing implemented:
  - Text blocks → PyMuPDF extraction + file saving
  - Figure blocks → PyMuPDF image extraction + file saving
  - Table blocks → Camelot extraction + CSV saving
- [x] JSON output with page numbers, block types, bounding boxes
- [x] Metadata stored with provenance information
