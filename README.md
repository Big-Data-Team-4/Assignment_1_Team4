# DAMG7245_Team5_Final-Project
 
## Project Overview
 
Project Lantern is a modular, end-to-end pipeline that converts long PDFs (e.g., 10-K filings) into structured data. It performs OCR, layout detection, table extraction, and document cleanup, then writes clean Markdown/JSON plus CSV tables—all with optional concurrency and cost benchmarking.
Key features
Full parsing stack: OCR → layout → tables → doc post-processing (Docling).
Modular adapters: swap implementations via scripts.adapters_team4 without changing the runner.
Batch + parallel runs: process hundreds of pages with 1–N workers.
Robust outputs: per-page JSON, Markdown, and table CSVs under src/data/parsed/.
Benchmarking: benchmarks.py measures runtime, memory, failures, and estimates cloud cost (Azure Document Intelligence / OCR).
Reproducible: single command to re-run on any PDF in src/data/raw/.
 
---
 
## Links 
Codelabs : https://codelabs-preview.appspot.com/?file_id=1Tb0qTeCHsBzMm7QdRG1cyb6hgisS-eIZw4ilpAAza5o#0

Video Link :
https://northeastern-my.sharepoint.com/personal/pekamwar_s_northeastern_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fpekamwar%5Fs%5Fnortheastern%5Fedu%2FDocuments%2FRecordings%2FCall%20with%20Big%20Data%20%2D%20Fall%202025%2D20250926%5F152637%2DMeeting%20Recording%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2Ecaa51e71%2D6c65%2D4d31%2D9b38%2D65985329b75d
---
 
## 🛠️ Technologies Used

[![Azure AI Document Intelligence](https://img.shields.io/badge/Azure_AI-0078D4?style=for-the-badge&logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)
[![Docling](https://img.shields.io/badge/Docling-FF6B6B?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/DS4SD/docling)
[![Camelot](https://img.shields.io/badge/Camelot-2E8B57?style=for-the-badge&logo=python&logoColor=white)](https://camelot-py.readthedocs.io/)
[![LayoutParser](https://img.shields.io/badge/LayoutParser-4B8BBE?style=for-the-badge&logo=python&logoColor=white)](https://layout-parser.readthedocs.io/)
[![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)
 
---
 
## Architecture Diagram
 <img width="2064" height="1294" alt="image" src="https://github.com/user-attachments/assets/2aa130b7-109a-4116-9b98-555caf91a9a0" />
 
---
 
## How to Run Application 
 Running dvc pipeline
 dvc init
 dvc repro

 

## Team Information
| Name            | Student ID    | Contribution |
|----------------|--------------|--------------|
| **Anusha Prakash** | 002306070  | 33.33% |
| **Komal Khairnar**  | 002472617  | 33.33% |
| **Shriya Pekamwar**  | 002059178  | 33.33% |
 
