import pdfplumber
import json
import os
import logging
from pathlib import Path
from datetime import datetime
import traceback
from PIL import Image  # For image handling
import fitz  # PyMuPDF
from io import BytesIO

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFToJSONParser:
    def __init__(self, pdf_dir="data/raw", output_dir="data/parsed/pdfparser_output"):
        """
        Initialize PDF to JSON Parser
        
        Args:
            pdf_dir (str): Directory containing PDF files
            output_dir (str): Directory to save output JSON files
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.image_dir = self.output_dir / "images"
        self.markdown_dir = self.output_dir / "markdown"
        self.jsonl_dir = self.output_dir / "jsonl"
        
        # Create directories if they don't exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structure
        self.extracted_data = {
            "source_file": "",
            "metadata": {},
            "pages": [],
            "total_pages": 0,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_tables": 0,
            "total_images": 0,
            "total_text_length": 0
        }
    
    def find_pdf_files(self):
        """
        Find all PDF files in the pdf_dir
        
        Returns:
            list: List of PDF file paths
        """
        try:
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files in {self.pdf_dir}")
            for pdf_file in pdf_files:
                logger.info(f"  - {pdf_file.name}")
            return pdf_files
        except Exception as e:
            logger.error(f"Error finding PDF files: {str(e)}")
            return []
    
    def extract_metadata(self, pdf):
        """Extract PDF metadata"""
        try:
            metadata = pdf.metadata or {}
            self.extracted_data["metadata"] = {
                "title": str(metadata.get("Title", "")),
                "author": str(metadata.get("Author", "")),
                "subject": str(metadata.get("Subject", "")),
                "creator": str(metadata.get("Creator", "")),
                "producer": str(metadata.get("Producer", "")),
                "creation_date": str(metadata.get("CreationDate", "")),
                "modification_date": str(metadata.get("ModDate", ""))
            }
            logger.info("Metadata extracted successfully")
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            self.extracted_data["metadata"] = {}
    
    def extract_text_from_page(self, page):
        """
        Extract text from a single page
        
        Args:
            page: pdfplumber page object
            
        Returns:
            str: Extracted text
        """
        try:
            text = page.extract_text()
            return text if text else ""
        except Exception as e:
            logger.error(f"Error extracting text from page {page.page_number}: {str(e)}")
            return ""
    
    def extract_tables_from_page(self, page):
        """
        Extract tables from a single page
        
        Args:
            page: pdfplumber page object
            
        Returns:
            list: List of tables (each table is a list of rows)
        """
        tables = []
        try:
            # Find tables on the page
            page_tables = page.find_tables()
            
            for i, table in enumerate(page_tables):
                try:
                    # Extract table data
                    table_data = table.extract()
                    
                    # Clean and process table data
                    cleaned_table = []
                    for row in table_data:
                        if row:  # Skip empty rows
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            cleaned_table.append(cleaned_row)
                    
                    if cleaned_table:  # Only add non-empty tables
                        table_info = {
                            "table_id": f"page_{page.page_number}_table_{i+1}",
                            "bbox": list(table.bbox) if table.bbox else [],
                            "data": cleaned_table,
                            "rows": len(cleaned_table),
                            "columns": len(cleaned_table[0]) if cleaned_table else 0
                        }
                        tables.append(table_info)
                        logger.debug(f"Extracted table {i+1} from page {page.page_number}: {len(cleaned_table)} rows")
                
                except Exception as e:
                    logger.error(f"Error processing table {i+1} on page {page.page_number}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error finding tables on page {page.page_number}: {str(e)}")
        
        return tables
    
    def extract_images_from_page(self, page):
        """
        Extract and save images from a single page using PyMuPDF, return JSON-safe metadata
        
        Args:
            page: pdfplumber page object (used for page number and compatibility)
            
        Returns:
            list: List of image metadata (JSON serializable)
        """
        images = []
        try:
            # Open the PDF with PyMuPDF
            pdf_doc = fitz.open(self.extracted_data["source_file"])
            pdf_page = pdf_doc[page.page_number - 1]  # PyMuPDF is 0-indexed
            image_list = pdf_page.get_images(full=True)
            
            for i, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image = Image.open(BytesIO(image_bytes))
                    
                    img_filename = f"page_{page.page_number}_image_{i+1}.{image_ext}"
                    img_path = self.current_image_dir / img_filename
                    image.save(img_path)
                    
                    # Get bounding box (approximate, as PyMuPDF doesn't provide exact bbox)
                    img_info = pdf_page.get_image_info()[i]
                    bbox = img_info.get("bbox", [0, 0, base_image["width"], base_image["height"]])
                    
                    image_info = {
                        "image_id": f"page_{page.page_number}_image_{i+1}",
                        "saved_path": str(img_path.relative_to(self.output_dir)),
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "width": int(base_image["width"]),
                        "height": int(base_image["height"]),
                        "format": image_ext.upper(),
                        "mode": image.mode
                    }
                    images.append(image_info)
                    logger.debug(f"Saved image {i+1} on page {page.page_number} to {img_path}")
                except Exception as e:
                    logger.error(f"Error saving image {i+1} on page {page.page_number}: {str(e)}")
                    continue
            pdf_doc.close()
        except Exception as e:
            logger.error(f"Error extracting images from page {page.page_number}: {str(e)}")
        
        return images
    
    def process_page(self, page):
        """
        Process a single page to extract all content
        
        Args:
            page: pdfplumber page object
            
        Returns:
            dict: Page data
        """
        page_num = page.page_number
        logger.info(f"Processing page {page_num}")
        
        # Extract text
        text = self.extract_text_from_page(page)
        
        # Extract tables
        tables = self.extract_tables_from_page(page)
        
        # Extract images
        images = self.extract_images_from_page(page)
        
        # Get page dimensions
        page_data = {
            "page_number": page_num,
            "width": float(page.width) if page.width else 0,
            "height": float(page.height) if page.height else 0,
            "text": text,
            "text_length": len(text),
            "tables": tables,
            "table_count": len(tables),
            "images": images,
            "image_count": len(images)
        }
        
        return page_data
    
    def extract_pdf_content(self, pdf_path):
        """
        Extract content from PDF file
        
        Args:
            pdf_path (Path): Path to PDF file
        """
        try:
            logger.info(f"Starting PDF extraction from: {pdf_path}")
            
            pdf_name = pdf_path.stem
            self.current_image_dir = self.image_dir / pdf_name
            self.current_image_dir.mkdir(parents=True, exist_ok=True)
            
            # Reset data structure
            self.extracted_data = {
                "source_file": str(pdf_path),
                "metadata": {},
                "pages": [],
                "total_pages": 0,
                "extraction_timestamp": datetime.now().isoformat(),
                "total_tables": 0,
                "total_images": 0,
                "total_text_length": 0
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                self.extract_metadata(pdf)
                
                # Set total pages
                self.extracted_data["total_pages"] = len(pdf.pages)
                logger.info(f"Total pages to process: {len(pdf.pages)}")
                
                # Process each page
                for page in pdf.pages:
                    try:
                        page_data = self.process_page(page)
                        self.extracted_data["pages"].append(page_data)
                        
                        # Update counters
                        self.extracted_data["total_tables"] += page_data["table_count"]
                        self.extracted_data["total_images"] += page_data["image_count"]
                        self.extracted_data["total_text_length"] += page_data["text_length"]
                        
                        # Log progress every 10 pages
                        if page.page_number % 10 == 0:
                            logger.info(f"Progress: {page.page_number}/{len(pdf.pages)} pages")
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page.page_number}: {str(e)}")
                        # Add error page data
                        error_page = {
                            "page_number": page.page_number,
                            "error": str(e),
                            "text": "",
                            "text_length": 0,
                            "tables": [],
                            "table_count": 0,
                            "images": [],
                            "image_count": 0
                        }
                        self.extracted_data["pages"].append(error_page)
                        continue
                
                logger.info("PDF extraction completed successfully")
                
        except Exception as e:
            logger.error(f"Critical error during PDF extraction: {str(e)}")
            raise
    
    def table_to_markdown(self, table_data):
        """
        Convert table data to markdown format
        
        Args:
            table_data (list): List of rows (each row is a list of cells)
            
        Returns:
            str: Markdown formatted table
        """
        if not table_data:
            return ""
        
        markdown_table = []
        
        # Add header row
        if table_data:
            header = "| " + " | ".join(str(cell) for cell in table_data[0]) + " |"
            markdown_table.append(header)
            
            # Add separator row
            separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
            markdown_table.append(separator)
            
            # Add data rows
            for row in table_data[1:]:
                row_md = "| " + " | ".join(str(cell) for cell in row) + " |"
                markdown_table.append(row_md)
        
        return "\n".join(markdown_table)
    
    def generate_markdown_content(self):
        """
        Generate markdown content from extracted data
        
        Returns:
            str: Markdown formatted content
        """
        markdown_content = []
        
        # Document header
        title = self.extracted_data["metadata"].get("title", "PDF Document")
        if title:
            markdown_content.append(f"# {title}")
        else:
            pdf_name = Path(self.extracted_data["source_file"]).stem
            markdown_content.append(f"# {pdf_name}")
        
        markdown_content.append("")
        
        # Metadata section
        metadata = self.extracted_data["metadata"]
        if any(metadata.values()):
            markdown_content.append("## Document Information")
            markdown_content.append("")
            
            if metadata.get("author"):
                markdown_content.append(f"**Author:** {metadata['author']}")
            if metadata.get("subject"):
                markdown_content.append(f"**Subject:** {metadata['subject']}")
            if metadata.get("creator"):
                markdown_content.append(f"**Creator:** {metadata['creator']}")
            if metadata.get("creation_date"):
                markdown_content.append(f"**Created:** {metadata['creation_date']}")
            
            markdown_content.append("")
        
        # Summary section
        markdown_content.append("## Document Summary")
        markdown_content.append("")
        markdown_content.append(f"- **Total Pages:** {self.extracted_data['total_pages']}")
        markdown_content.append(f"- **Total Tables:** {self.extracted_data['total_tables']}")
        markdown_content.append(f"- **Total Images:** {self.extracted_data['total_images']}")
        markdown_content.append(f"- **Total Text Length:** {self.extracted_data['total_text_length']:,} characters")
        markdown_content.append(f"- **Extraction Date:** {self.extracted_data['extraction_timestamp']}")
        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")
        
        # Page content
        for page_data in self.extracted_data["pages"]:
            page_num = page_data["page_number"]
            markdown_content.append(f"## Page {page_num}")
            markdown_content.append("")
            
            # Page text
            if page_data.get("text"):
                markdown_content.append(page_data["text"])
                markdown_content.append("")
            
            # Page tables
            if page_data.get("tables"):
                for i, table in enumerate(page_data["tables"], 1):
                    markdown_content.append(f"### Table {i} (Page {page_num})")
                    markdown_content.append("")
                    
                    table_md = self.table_to_markdown(table["data"])
                    if table_md:
                        markdown_content.append(table_md)
                    else:
                        markdown_content.append("*Table data could not be formatted*")
                    
                    markdown_content.append("")
            
            # Page images
            if page_data.get("images"):
                for i, image in enumerate(page_data["images"], 1):
                    markdown_content.append(f"### Image {i} (Page {page_num})")
                    markdown_content.append("")
                    markdown_content.append(f"![{image['image_id']}]({image['saved_path']})")
                    markdown_content.append("")
                    markdown_content.append(f"- **Dimensions:** {image['width']} x {image['height']}")
                    markdown_content.append(f"- **Format:** {image['format']}")
                    markdown_content.append(f"- **File:** `{image['saved_path']}`")
                    markdown_content.append("")
            
            # Add page separator
            if page_num < self.extracted_data['total_pages']:
                markdown_content.append("---")
                markdown_content.append("")
        
        return "\n".join(markdown_content)
    
    def save_jsonl(self, pdf_path):
        """
        Save extracted data as JSONL file (one JSON object per line)
        
        Args:
            pdf_path (Path): Original PDF file path
            
        Returns:
            Path: Output JSONL file path
        """
        try:
            pdf_name = pdf_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pdf_name}_extracted_{timestamp}.jsonl"
            
            output_path = self.jsonl_dir / filename
            
            logger.info(f"Saving JSONL to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write document metadata as first line
                doc_info = {
                    "record_type": "document_metadata",
                    "source_file": self.extracted_data["source_file"],
                    "metadata": self.extracted_data["metadata"],
                    "total_pages": self.extracted_data["total_pages"],
                    "extraction_timestamp": self.extracted_data["extraction_timestamp"],
                    "total_tables": self.extracted_data["total_tables"],
                    "total_images": self.extracted_data["total_images"],
                    "total_text_length": self.extracted_data["total_text_length"]
                }
                json_line = json.dumps(doc_info, ensure_ascii=False, separators=(',', ':'))
                f.write(json_line + '\n')
                
                # Write each page as separate JSON line
                for page_data in self.extracted_data["pages"]:
                    page_record = {
                        "record_type": "page_data",
                        "source_file": self.extracted_data["source_file"],
                        "document_metadata": self.extracted_data["metadata"],
                        "page_number": page_data["page_number"],
                        "page_width": page_data["width"],
                        "page_height": page_data["height"],
                        "text": page_data["text"],
                        "text_length": page_data["text_length"],
                        "tables": page_data["tables"],
                        "table_count": page_data["table_count"],
                        "images": page_data["images"],
                        "image_count": page_data["image_count"],
                        "extraction_timestamp": self.extracted_data["extraction_timestamp"]
                    }
                    json_line = json.dumps(page_record, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
            
            logger.info(f"JSONL saved successfully to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving JSONL: {str(e)}")
            raise
    
    def save_markdown(self, pdf_path):
        """
        Save extracted data as markdown file
        
        Args:
            pdf_path (Path): Original PDF file path
            
        Returns:
            Path: Output markdown file path
        """
        try:
            pdf_name = pdf_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pdf_name}_extracted_{timestamp}.md"
            
            output_path = self.markdown_dir / filename
            
            logger.info(f"Generating markdown content...")
            markdown_content = self.generate_markdown_content()
            
            logger.info(f"Saving markdown to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown saved successfully to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving markdown: {str(e)}")
            raise
    
    def save_json(self, pdf_path):
        """
        Save extracted data to JSON file
        
        Args:
            pdf_path (Path): Original PDF file path
            
        Returns:
            Path: Output JSON file path
        """
        try:
            pdf_name = pdf_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pdf_name}_extracted_{timestamp}.json"
            
            output_path = self.output_dir / filename
            
            logger.info(f"Saving JSON to: {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON saved successfully to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")
            raise

def main():
    """Main function to run the PDF to JSON parser"""
    
    # Configuration - Updated paths for DVC integration
    PDF_DIR = "data/raw"        # Directory containing PDF files (relative to src/)
    OUTPUT_DIR = "data/parsed/pdfparser_output"   # Directory for JSON and image output (relative to src/)
    
    try:
        # Initialize parser
        parser = PDFToJSONParser(PDF_DIR, OUTPUT_DIR)
        
        # Find PDF files
        pdf_files = parser.find_pdf_files()
        
        if not pdf_files:
            print(f"No PDF files found in {PDF_DIR}")
            print(f"Please place your PDF files in the '{PDF_DIR}' directory")
            return
        
        # Process each PDF file
        processed_files = []
        markdown_files = []
        jsonl_files = []
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                print(f"\nProcessing PDF {i}/{len(pdf_files)}: {pdf_file.name}")
                print("-" * 50)
                
                # Extract PDF content
                parser.extract_pdf_content(pdf_file)
                
                # Save JSON
                json_output = parser.save_json(pdf_file)
                processed_files.append(json_output)
                
                # Save Markdown
                markdown_output = parser.save_markdown(pdf_file)
                markdown_files.append(markdown_output)
                
                # Save JSONL
                jsonl_output = parser.save_jsonl(pdf_file)
                jsonl_files.append(jsonl_output)
                
                # Print summary
                print("\n" + "="*60)
                print("PDF TO JSON & MARKDOWN & JSONL EXTRACTION SUMMARY")
                print("="*60)
                print(f"Source PDF: {pdf_file}")
                print(f"Total pages processed: {parser.extracted_data['total_pages']}")
                print(f"Total tables extracted: {parser.extracted_data['total_tables']}")
                print(f"Total images found and saved: {parser.extracted_data['total_images']}")
                print(f"Total text length: {parser.extracted_data['total_text_length']:,} characters")
                print(f"JSON output: {json_output}")
                print(f"Markdown output: {markdown_output}")
                print(f"JSONL output: {jsonl_output}")
                print(f"JSON file size: {json_output.stat().st_size / 1024 / 1024:.2f} MB")
                print(f"Markdown file size: {markdown_output.stat().st_size / 1024:.2f} KB")
                print(f"JSONL file size: {jsonl_output.stat().st_size / 1024:.2f} KB")
                print(f"Images saved to: {parser.image_dir / pdf_file.stem}")
                print("="*60)
                
                print(f"Successfully processed: {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                print(f"Failed to process {pdf_file.name}: {str(e)}")
                continue
        
        print(f"\nPDF to JSON & Markdown & JSONL processing completed!")
        print(f"Processed {len(processed_files)} PDF files successfully")
        print(f"Files saved to: {OUTPUT_DIR}")
        
        # List output files
        print("\nGenerated files:")
        print("JSON files:")
        for json_file in processed_files:
            print(f"  - {json_file.name}")
        
        print("Markdown files:")
        for md_file in markdown_files:
            print(f"  - {md_file.name}")
        
        print("JSONL files:")
        for jsonl_file in jsonl_files:
            print(f"  - {jsonl_file.name}")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    main()