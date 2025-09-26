import os
import json
from pathlib import Path
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import csv

# Use relative paths for DVC integration
CURRENT_DIR = Path.cwd()  # This will be src/ when DVC runs the script

# Load .env file from current directory or parent
env_path = CURRENT_DIR / '.env'
if not env_path.exists():
    env_path = CURRENT_DIR.parent / '.env'  # Try parent directory
load_dotenv(env_path)

# Azure Document Intelligence configuration
AZURE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

# Initialize client only if credentials are available
client = None
if AZURE_ENDPOINT and AZURE_KEY:
    try:
        client = DocumentAnalysisClient(
            endpoint=AZURE_ENDPOINT,
            credential=AzureKeyCredential(AZURE_KEY)
        )
        print("Azure Document Intelligence client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Azure client: {e}")
        client = None

def extract_document_content(pdf_path, output_folder):
    """Extract text, tables, and layout from PDF using Azure Document Intelligence"""
    
    if not client:
        print("Azure Document Intelligence client not available. Skipping processing.")
        return
    
    # Create output folder structure
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    text_folder = output_path / "Markdown"
    tables_folder = output_path / "tables"
    images_folder = output_path / "images"
    json_folder = output_path / "json"
    
    for folder in [text_folder, tables_folder, images_folder, json_folder]:
        folder.mkdir(exist_ok=True)
    
    base_name = Path(pdf_path).stem
    
    print(f"Processing {pdf_path}...")
    
    try:
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        
        file_size_mb = len(pdf_content) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Process with Azure Document Intelligence
        print("Analyzing document with Azure Document Intelligence...")
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            pdf_content
        )
        result = poller.result()
        
        print(f"Analysis complete! Found {len(result.pages)} pages")
        
        # Extract and save text
        extract_text(result, base_name, text_folder)
        
        # Extract and save tables
        extract_tables(result, base_name, tables_folder)
        
        # Extract figures/images metadata (actual image extraction requires PyMuPDF)
        extract_figures_metadata(result, base_name, json_folder)
        
        # Save complete analysis result
        save_complete_analysis(result, base_name, json_folder)
        
        # Extract actual images if PyMuPDF is installed
        try:
            import fitz
            extract_images_with_pymupdf(pdf_path, base_name, images_folder)
        except ImportError:
            print("Install PyMuPDF to extract actual images: pip install pymupdf")
        
        print(f"Successfully processed {pdf_path}")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def extract_text(result, base_name, output_folder):
    """Extract and save text content in Markdown format"""
    # Save full document text as Markdown
    full_markdown_file = output_folder / f"{base_name}_full_text.md"
    with open(full_markdown_file, 'w', encoding='utf-8') as f:
        f.write(f"# {base_name} Document Text\n\n")
        f.write(result.content.replace('\n', '  \n'))  # Convert newlines to Markdown line breaks
    
    print(f"Extracted full text: {len(result.content)} characters as Markdown")

def extract_tables(result, base_name, output_folder):
    """Extract and save tables, filtering out false positives"""
    if not result.tables:
        print("No tables found in document")
        return
    
    print(f"Found {len(result.tables)} potential tables")
    
    real_tables = []
    false_positives = []
    
    for idx, table in enumerate(result.tables):
        # Get the page number for this table
        page_num = 1
        if table.bounding_regions:
            page_num = table.bounding_regions[0].page_number
        
        # Check if this is likely a real table
        is_real_table = True
        
        # Create table data structure first to analyze content
        table_data = []
        for row_idx in range(table.row_count):
            table_data.append([''] * table.column_count)
        
        # Fill in cell values
        for cell in table.cells:
            table_data[cell.row_index][cell.column_index] = cell.content
        
        # Filtering logic based on common false positives in financial documents:
        
        # 1. Skip tables with very few cells
        if len(table.cells) <= 2:
            is_real_table = False
        
        # 2. Skip tables where any cell contains a very long prose text (>200 chars)
        # Real data tables typically have shorter cell content
        for cell in table.cells:
            if len(cell.content) > 200:
                is_real_table = False
                break
        
        # 3. Skip 1x2 or 2x1 tables (often pull quotes or image/caption pairs)
        if (table.row_count == 1 and table.column_count == 2) or \
           (table.row_count == 2 and table.column_count == 1):
            is_real_table = False
        
        # 4. Check for quote-like content (contains words like "CEO", ends with attribution)
        all_content = ' '.join(cell.content for cell in table.cells)
        quote_indicators = ['CEO', 'President', 'Director', 'Officer', '—', '–']
        if any(indicator in all_content for indicator in quote_indicators) and len(table.cells) <= 4:
            is_real_table = False
        
        # 5. Real financial tables usually have numeric data
        # Check if at least 20% of non-empty cells contain numbers
        if len(table.cells) >= 4:
            numeric_cells = 0
            non_empty_cells = 0
            for cell in table.cells:
                if cell.content.strip():
                    non_empty_cells += 1
                    if any(char.isdigit() for char in cell.content):
                        numeric_cells += 1
            
            if non_empty_cells > 0 and (numeric_cells / non_empty_cells) < 0.2:
                # Less than 20% of cells have numbers - probably not a data table
                if table.row_count < 3 or table.column_count < 3:
                    is_real_table = False
        
        if not is_real_table:
            false_positives.append({
                'page': page_num,
                'rows': table.row_count,
                'cols': table.column_count,
                'preview': all_content[:100] + '...' if len(all_content) > 100 else all_content
            })
            continue
        
        real_tables.append(table)
        
        # Save as CSV
        table_num = len(real_tables)  # Use real table count for numbering
        csv_file = output_folder / f"{base_name}_page{page_num}_table{table_num}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table_data)
        
        # Also save as JSON with metadata
        json_file = output_folder / f"{base_name}_page{page_num}_table{table_num}.json"
        table_json = {
            "table_index": table_num,
            "page_number": page_num,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "cells": table_data
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(table_json, f, indent=2, ensure_ascii=False)
    
    print(f"Extracted {len(real_tables)} real tables")
    print(f"Filtered out {len(false_positives)} false positives")
    
    # Save false positives report for review
    if false_positives:
        report_file = output_folder / f"{base_name}_filtered_tables.json"
        with open(report_file, 'w') as f:
            json.dump({
                "description": "Tables filtered out as likely false positives",
                "filtered_count": len(false_positives),
                "filtered_tables": false_positives
            }, f, indent=2)

def extract_figures_metadata(result, base_name, output_folder):
    """Extract figure/image metadata from layout analysis"""
    figures_data = []
    
    # Check if the result has figures attribute
    if hasattr(result, 'figures') and result.figures:
        print(f"Found {len(result.figures)} figures")
        
        for idx, figure in enumerate(result.figures):
            figure_info = {
                "figure_index": idx + 1,
                "page_number": figure.bounding_regions[0].page_number if figure.bounding_regions else None,
                "caption": figure.caption.content if hasattr(figure, 'caption') and figure.caption else None
            }
            figures_data.append(figure_info)
    
    # Save figures metadata
    if figures_data:
        figures_file = output_folder / f"{base_name}_figures_metadata.json"
        with open(figures_file, 'w', encoding='utf-8') as f:
            json.dump(figures_data, f, indent=2)

def extract_images_with_pymupdf(pdf_path, base_name, output_folder):
    """Extract actual images using PyMuPDF"""
    import fitz
    
    pdf_document = fitz.open(pdf_path)
    image_count = 0
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_file = output_folder / f"{base_name}_page{page_num + 1}_img{img_idx + 1}.{image_ext}"
                with open(image_file, "wb") as f:
                    f.write(image_bytes)
                
                image_count += 1
            except Exception as e:
                print(f"Error extracting image on page {page_num + 1}: {e}")
    
    pdf_document.close()
    print(f"Extracted {image_count} images")

def save_complete_analysis(result, base_name, output_folder):
    """Save the complete analysis as structured JSON"""
    analysis_data = {
        "document_info": {
            "total_pages": len(result.pages),
            "total_tables": len(result.tables) if result.tables else 0,
            "total_characters": len(result.content)
        },
        "pages": [],
        "tables_summary": []
    }
    
    # Page information
    for page in result.pages:
        page_info = {
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "angle": page.angle,
            "lines_count": len(page.lines),
            "words_count": len(page.words) if hasattr(page, 'words') else 0
        }
        analysis_data["pages"].append(page_info)
    
    # Table summaries
    if result.tables:
        for idx, table in enumerate(result.tables):
            table_info = {
                "table_index": idx + 1,
                "page_number": table.bounding_regions[0].page_number if table.bounding_regions else None,
                "rows": table.row_count,
                "columns": table.column_count,
                "total_cells": len(table.cells)
            }
            analysis_data["tables_summary"].append(table_info)
    
    # Save analysis
    analysis_file = output_folder / f"{base_name}_document_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"Saved complete document analysis")

# Main execution for DVC integration
if __name__ == "__main__":
    # Define paths relative to src/ directory (DVC working directory)
    raw_folder = Path("data/raw")
    output_folder = Path("data/parsed/AzureDoc_output")
    
    # Make sure credentials are set
    if not AZURE_ENDPOINT or not AZURE_KEY:
        print(f"Azure Document Intelligence credentials not configured")
        print(f"Set AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY")
        print(f"Looking for .env at: {env_path}")
        print(f"Skipping Azure extraction - creating placeholder output")
        
        # Create empty output directory structure so DVC doesn't fail
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / "Markdown").mkdir(exist_ok=True)
        (output_folder / "tables").mkdir(exist_ok=True)
        (output_folder / "images").mkdir(exist_ok=True)
        (output_folder / "json").mkdir(exist_ok=True)
        
        # Create a placeholder file to indicate Azure was skipped
        placeholder_file = output_folder / "azure_credentials_missing.txt"
        with open(placeholder_file, 'w') as f:
            f.write("Azure Document Intelligence credentials not configured.\n")
            f.write("Set AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY to enable.\n")
            f.write(f"Current working directory: {Path.cwd()}\n")
            f.write(f"Looking for .env at: {env_path}\n")
        
        print(f"Created placeholder output at: {output_folder}")
        
    else:
        # Check if raw folder exists
        if not raw_folder.exists():
            print(f"Directory not found at: {raw_folder}")
            print(f"Current working directory: {Path.cwd()}")
            print(f"Please check the directory path")
            
            # Create placeholder anyway so DVC doesn't fail
            output_folder.mkdir(parents=True, exist_ok=True)
            error_file = output_folder / "raw_directory_missing.txt"
            with open(error_file, 'w') as f:
                f.write(f"Raw directory not found at: {raw_folder}\n")
                f.write(f"Current working directory: {Path.cwd()}\n")
        else:
            # Process all PDF files in the raw folder
            pdf_files = list(raw_folder.glob("*.pdf"))
            if not pdf_files:
                print(f"No PDF files found in {raw_folder}")
                
                # Create placeholder output
                output_folder.mkdir(parents=True, exist_ok=True)
                no_pdfs_file = output_folder / "no_pdfs_found.txt"
                with open(no_pdfs_file, 'w') as f:
                    f.write(f"No PDF files found in {raw_folder}\n")
            else:
                print(f"Found {len(pdf_files)} PDF files to process")
                for pdf_file in pdf_files:
                    print(f"Processing PDF: {pdf_file}")
                    print(f"Output will be saved to: {output_folder}")
                    extract_document_content(str(pdf_file), str(output_folder))