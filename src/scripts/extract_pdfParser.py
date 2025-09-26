# Portable PDF Parser - works on any machine with just Python packages!
import pdfplumber
import json
import os
import logging
from pathlib import Path
from datetime import datetime
import traceback
from PIL import Image
import fitz  # PyMuPDF
from io import BytesIO

# Pure Python OCR - no system dependencies!
try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_TYPE = "EasyOCR"
except ImportError:
    OCR_AVAILABLE = False
    OCR_TYPE = "None"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PortablePDFParser:
    def __init__(self, pdf_dir="data/raw", output_dir="data/parsed/pdfParser_output"):
        """Portable PDF parser that works anywhere with just pip install"""
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        
        # Create subdirectories
        self.pdfparser_output_dir = self.output_dir / "metadata"
        self.page_texts_dir = self.output_dir / "page_texts"
        self.image_dir = self.output_dir / "images"
        self.logs_dir = self.output_dir / "logs"
        
        # Create directories
        for directory in [self.pdf_dir, self.output_dir, self.pdfparser_output_dir, 
                         self.page_texts_dir, self.image_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR
        self.ocr_reader = None
        self.ocr_available = self.initialize_ocr()
        
        # Settings
        self.ocr_threshold = 30
        self.ocr_log = []
        self.x_density = 150
        self.y_density = 150
        
        # Data structure
        self.extracted_data = {
            "source_file": "",
            "metadata": {},
            "pages": [],
            "total_pages": 0,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_tables": 0,
            "total_images": 0,
            "total_text_length": 0,
            "ocr_pages": [],
            "total_words": 0,
            "ocr_engine": OCR_TYPE
        }
    
    def initialize_ocr(self):
        """Initialize pure Python OCR (no system dependencies)"""
        if not OCR_AVAILABLE:
            logger.warning("EasyOCR not installed - OCR will be disabled")
            logger.info(" To enable OCR: pip install easyocr")
            return False
        
        try:
            # Initialize EasyOCR reader (downloads models automatically first time)
            logger.info("Initializing EasyOCR (first run downloads models)...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU only for compatibility
            logger.info("EasyOCR initialized successfully")
            return True
        except Exception as e:
            logger.error(f"EasyOCR initialization failed: {e}")
            return False
    
    def find_pdf_files(self):
        """Find all PDF files"""
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
    
    def page_to_image(self, pdf_path, page_number):
        """Convert PDF page to PIL Image"""
        try:
            pdf_doc = fitz.open(pdf_path)
            page = pdf_doc[page_number - 1]
            
            # High DPI for better OCR
            zoom = 300 / 72.0  # 300 DPI
            mat = fitz.Matrix(zoom, zoom)
            
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pdf_doc.close()
            
            image = Image.open(BytesIO(img_data))
            logger.debug(f"Page {page_number} converted to {image.size} image")
            return image
            
        except Exception as e:
            logger.error(f"Error converting page {page_number} to image: {str(e)}")
            return None
    
    def apply_easyocr_to_page(self, pdf_path, page_number):
        """Apply EasyOCR (pure Python, no system dependencies)"""
        if not self.ocr_available:
            logger.info(f"Skipping OCR for page {page_number} - EasyOCR not available")
            return ""
        
        try:
            logger.info(f"Applying EasyOCR to page {page_number}")
            
            # Convert page to image
            image = self.page_to_image(pdf_path, page_number)
            if image is None:
                return ""
            
            # Convert PIL to numpy array for EasyOCR
            import numpy as np
            image_array = np.array(image)
            
            # Apply OCR
            results = self.ocr_reader.readtext(image_array)
            
            # Extract text from results
            ocr_text_parts = []
            for (bbox, text, confidence) in results:
                # Only include text with reasonable confidence
                if confidence > 0.3:  # 30% minimum confidence
                    ocr_text_parts.append(text.strip())
            
            ocr_text = '\n'.join(ocr_text_parts)
            
            logger.info(f"EasyOCR extracted {len(ocr_text)} characters from page {page_number}")
            return ocr_text
            
        except Exception as e:
            logger.error(f"Error applying EasyOCR to page {page_number}: {str(e)}")
            return ""
    
    def extract_text_from_page(self, page, pdf_path):
        """Enhanced text extraction with EasyOCR fallback"""
        try:
            page_num = page.page_number
            all_extracted_text = []
            used_ocr = False
            
            # Method 1: Standard pdfplumber
            try:
                text_standard = page.extract_text(x_density=self.x_density, y_density=self.y_density)
                if text_standard and text_standard.strip():
                    all_extracted_text.append(("standard", text_standard))
            except Exception as e:
                logger.warning(f"Standard extraction failed: {e}")
            
            # Method 2: Advanced layout
            try:
                text_layout = page.extract_text(x_density=200, y_density=200, layout=True)
                if text_layout and text_layout.strip() and text_layout != text_standard:
                    all_extracted_text.append(("layout", text_layout))
            except Exception as e:
                logger.warning(f"Layout extraction failed: {e}")
            
            # Method 3: Character-level
            try:
                chars = page.chars
                if chars:
                    lines = {}
                    for char in chars:
                        y = round(char['y0'], 1)
                        if y not in lines:
                            lines[y] = []
                        lines[y].append(char)
                    
                    sorted_lines = sorted(lines.keys(), reverse=True)
                    char_text_lines = []
                    for y in sorted_lines:
                        line_chars = sorted(lines[y], key=lambda c: c['x0'])
                        line_text = ''.join(char['text'] for char in line_chars)
                        if line_text.strip():
                            char_text_lines.append(line_text.strip())
                    
                    char_text = '\n'.join(char_text_lines)
                    if char_text and char_text.strip():
                        all_extracted_text.append(("chars", char_text))
            except Exception as e:
                logger.warning(f"Character extraction failed: {e}")
            
            # Method 4: PyMuPDF
            try:
                pdf_doc = fitz.open(pdf_path)
                fitz_page = pdf_doc[page_num - 1]
                fitz_text = fitz_page.get_text()
                pdf_doc.close()
                
                if fitz_text and fitz_text.strip():
                    all_extracted_text.append(("pymupdf", fitz_text))
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
            
            # Choose best extraction
            best_text = ""
            best_method = "none"
            
            if all_extracted_text:
                best_extraction = max(all_extracted_text, key=lambda x: len(x[1].strip()))
                best_text = best_extraction[1]
                best_method = best_extraction[0]
                
                logger.info(f"Page {page_num}: Best method '{best_method}' got {len(best_text.strip())} chars")
            
            # Check if OCR needed
            if len(best_text.strip()) < self.ocr_threshold:
                logger.info(f"Page {page_num} needs OCR (only {len(best_text.strip())} chars extracted)")
                
                # Apply EasyOCR
                ocr_text = self.apply_easyocr_to_page(pdf_path, page_num)
                
                if len(ocr_text.strip()) > len(best_text.strip()):
                    original_length = len(best_text.strip())
                    best_text = ocr_text
                    used_ocr = True
                    
                    # Log OCR usage
                    ocr_info = {
                        "page_number": page_num,
                        "original_text_length": original_length,
                        "ocr_text_length": len(ocr_text.strip()),
                        "timestamp": datetime.now().isoformat()
                    }
                    self.ocr_log.append(ocr_info)
                    self.extracted_data["ocr_pages"].append(page_num)
                    
                    logger.info(f"Used EasyOCR: {original_length} -> {len(ocr_text.strip())} chars")
            
            return best_text if best_text else "", used_ocr
            
        except Exception as e:
            logger.error(f"Error extracting text from page {page.page_number}: {str(e)}")
            return "", False
    
    def extract_words_with_bbox(self, page):
        """Extract words with bounding boxes"""
        try:
            words = page.extract_words()
            processed_words = []
            for word in words:
                word_data = {
                    "text": str(word.get("text", "")),
                    "x0": float(word.get("x0", 0)),
                    "y0": float(word.get("y0", 0)),
                    "x1": float(word.get("x1", 0)),
                    "y1": float(word.get("y1", 0)),
                    "width": float(word.get("width", 0)),
                    "height": float(word.get("height", 0)),
                    "size": float(word.get("size", 0)),
                    "fontname": str(word.get("fontname", "")),
                }
                processed_words.append(word_data)
            return processed_words
        except Exception as e:
            logger.error(f"Error extracting words: {str(e)}")
            return []
    
    def save_page_text(self, page_text, page_number, pdf_name, used_ocr=False):
        """Save individual page text"""
        try:
            pdf_text_dir = self.page_texts_dir / pdf_name
            pdf_text_dir.mkdir(parents=True, exist_ok=True)
            
            ocr_suffix = "_OCR" if used_ocr else ""
            page_file = pdf_text_dir / f"page_{page_number:03d}{ocr_suffix}.txt"
            
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(page_text)
            
        except Exception as e:
            logger.error(f"Error saving page text: {str(e)}")
    
    def extract_tables_from_page(self, page):
        """Extract tables"""
        tables = []
        try:
            page_tables = page.find_tables()
            for i, table in enumerate(page_tables):
                try:
                    table_data = table.extract()
                    cleaned_table = []
                    for row in table_data:
                        if row:
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            cleaned_table.append(cleaned_row)
                    
                    if cleaned_table:
                        table_info = {
                            "table_id": f"page_{page.page_number}_table_{i+1}",
                            "data": cleaned_table,
                            "rows": len(cleaned_table),
                            "columns": len(cleaned_table[0]) if cleaned_table else 0
                        }
                        tables.append(table_info)
                except Exception:
                    continue
        except Exception:
            pass
        return tables
    
    def extract_images_from_page(self, page):
        """Extract and save images from a single page"""
        images = []
        try:
            pdf_doc = fitz.open(self.extracted_data["source_file"])
            pdf_page = pdf_doc[page.page_number - 1]
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
                    
                    # Get bounding box info
                    try:
                        img_info = pdf_page.get_image_info()[i]
                        bbox = img_info.get("bbox", [0, 0, base_image["width"], base_image["height"]])
                    except (IndexError, KeyError):
                        bbox = [0, 0, base_image["width"], base_image["height"]]
                    
                    image_info = {
                        "image_id": f"page_{page.page_number}_image_{i+1}",
                        "saved_path": str(img_path.relative_to(self.pdfparser_output_dir)),
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
    
    def process_page(self, page, pdf_path, pdf_name):
        """Process a single page"""
        page_num = page.page_number
        logger.info(f"Processing page {page_num}")
        
        # Extract text with OCR fallback
        text, used_ocr = self.extract_text_from_page(page, pdf_path)
        
        # Save page text
        self.save_page_text(text, page_num, pdf_name, used_ocr)
        
        # Extract other content
        words = self.extract_words_with_bbox(page)
        tables = self.extract_tables_from_page(page)
        images = self.extract_images_from_page(page)
        
        self.extracted_data["total_words"] += len(words)
        
        page_data = {
            "page_number": page_num,
            "text": text,
            "text_length": len(text),
            "used_ocr": used_ocr,
            "words": words,
            "word_count": len(words),
            "tables": tables,
            "table_count": len(tables),
            "images": images,
            "image_count": len(images)
        }
        
        return page_data
    
    def save_ocr_log(self, pdf_name):
        """Save OCR log"""
        try:
            if not self.ocr_log:
                return
            
            log_file = self.logs_dir / f"{pdf_name}_ocr_log.json"
            log_data = {
                "pdf_file": pdf_name,
                "ocr_engine": OCR_TYPE,
                "ocr_pages": self.ocr_log,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error saving OCR log: {str(e)}")
    
    def extract_pdf_content(self, pdf_path):
        """Extract content from PDF"""
        try:
            logger.info(f"Starting portable PDF extraction: {pdf_path}")
            
            pdf_name = pdf_path.stem
            
            # Create image directory for this PDF
            self.current_image_dir = self.image_dir / pdf_name
            self.current_image_dir.mkdir(parents=True, exist_ok=True)
            
            # Reset data
            self.extracted_data["source_file"] = str(pdf_path)
            self.extracted_data["pages"] = []
            self.extracted_data["extraction_timestamp"] = datetime.now().isoformat()
            self.ocr_log = []
            
            with pdfplumber.open(pdf_path) as pdf:
                self.extract_metadata(pdf)
                self.extracted_data["total_pages"] = len(pdf.pages)
                logger.info(f"Processing {len(pdf.pages)} pages")
                
                for page in pdf.pages:
                    try:
                        page_data = self.process_page(page, pdf_path, pdf_name)
                        self.extracted_data["pages"].append(page_data)
                        
                        # Update counters
                        self.extracted_data["total_tables"] += page_data["table_count"]
                        self.extracted_data["total_images"] += page_data["image_count"]
                        self.extracted_data["total_text_length"] += page_data["text_length"]
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page.page_number}: {str(e)}")
                        continue
                
                self.save_ocr_log(pdf_name)
                logger.info("PDF extraction completed")
                
        except Exception as e:
            logger.error(f"Critical error: {str(e)}")
            raise
    
    def save_json(self, pdf_path):
        """Save results to JSON"""
        try:
            pdf_name = pdf_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pdf_name}_portable_{timestamp}.json"
            output_path = self.pdfparser_output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")
            raise
    
    def generate_markdown_content(self):
        """Generate markdown summary of PDF content"""
        try:
            pdf_name = Path(self.extracted_data["source_file"]).stem
            
            md_lines = [
                f"# PDF Analysis Report: {pdf_name}",
                "",
                f"**Extraction Date:** {self.extracted_data['extraction_timestamp']}",
                f"**Source File:** {self.extracted_data['source_file']}",
                f"**OCR Engine:** {self.extracted_data['ocr_engine']}",
                "",
                "## Document Overview",
                "",
                f"- **Total Pages:** {self.extracted_data['total_pages']}",
                f"- **Total Words:** {self.extracted_data['total_words']:,}",
                f"- **Total Characters:** {self.extracted_data['total_text_length']:,}",
                f"- **Total Tables:** {self.extracted_data['total_tables']}",
                f"- **Total Images:** {self.extracted_data['total_images']}",
                f"- **Pages Using OCR:** {len(self.extracted_data['ocr_pages'])}",
                "",
            ]
            
            # Add metadata section
            if self.extracted_data["metadata"]:
                md_lines.extend([
                    "## Document Metadata",
                    "",
                ])
                
                for key, value in self.extracted_data["metadata"].items():
                    if value and value.strip():
                        md_lines.append(f"- **{key.title()}:** {value}")
                md_lines.append("")
            
            # Add OCR information if used
            if self.extracted_data["ocr_pages"]:
                md_lines.extend([
                    "## OCR Information",
                    "",
                    f"OCR was applied to {len(self.extracted_data['ocr_pages'])} pages:",
                    f"Pages: {', '.join(map(str, self.extracted_data['ocr_pages']))}",
                    "",
                ])
            
            # Add page summaries
            md_lines.extend([
                "## Page Summaries",
                "",
            ])
            
            for page in self.extracted_data["pages"]:
                page_num = page["page_number"]
                text_preview = page["text"][:200] + "..." if len(page["text"]) > 200 else page["text"]
                
                md_lines.extend([
                    f"### Page {page_num}",
                    "",
                    f"- **Characters:** {page['text_length']:,}",
                    f"- **Words:** {page['word_count']}",
                    f"- **Tables:** {page['table_count']}",
                    f"- **Images:** {page['image_count']}",
                    f"- **OCR Used:** {'Yes' if page['used_ocr'] else 'No'}",
                    "",
                ])
                
                if text_preview.strip():
                    md_lines.extend([
                        "**Text Preview:**",
                        "```",
                        text_preview,
                        "```",
                        "",
                    ])
            
            # Add table information if any
            tables_found = [page for page in self.extracted_data["pages"] if page["table_count"] > 0]
            if tables_found:
                md_lines.extend([
                    "## Tables Found",
                    "",
                ])
                
                for page in tables_found:
                    if page["tables"]:
                        for table in page["tables"]:
                            md_lines.extend([
                                f"### {table['table_id'].replace('_', ' ').title()}",
                                f"- **Rows:** {table['rows']}",
                                f"- **Columns:** {table['columns']}",
                                "",
                            ])
            
            # Add image information if any
            images_found = [page for page in self.extracted_data["pages"] if page["image_count"] > 0]
            if images_found:
                md_lines.extend([
                    "## Images Found",
                    "",
                ])
                
                for page in images_found:
                    if page["images"]:
                        for img in page["images"]:
                            md_lines.extend([
                                f"### {img['image_id'].replace('_', ' ').title()}",
                                f"- **Dimensions:** {img['width']}x{img['height']}",
                                f"- **Format:** {img['format']}",
                                f"- **Saved to:** {img['saved_path']}",
                                "",
                            ])
            
            return "\n".join(md_lines)
        
        except Exception as e:
            logger.error(f"Error generating markdown content: {str(e)}")
            return f"# Error generating markdown\n\n{str(e)}"
    
    def save_markdown(self, pdf_path):
        """Save results to Markdown format"""
        try:
            pdf_name = pdf_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pdf_name}_portable_{timestamp}.md"
            output_path = self.pdfparser_output_dir / filename
            
            # Generate markdown content
            md_content = self.generate_markdown_content()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"Markdown saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving Markdown: {str(e)}")
            raise

def main():
    """Main function - works on any machine!"""
    
    print("PORTABLE PDF PARSER")
    print("=" * 50)
    print(f" OCR Engine: {OCR_TYPE}")
    if not OCR_AVAILABLE:
        print(" To enable OCR: pip install easyocr")
    print("=" * 50)
    
    PDF_DIR = "data/raw"
    OUTPUT_DIR = "data/parsed/pdfParser_output"
    
    try:
        parser = PortablePDFParser(PDF_DIR, OUTPUT_DIR)
        pdf_files = parser.find_pdf_files()
        
        if not pdf_files:
            print(f"No PDF files found in {PDF_DIR}")
            return
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                print(f"\n Processing {i}/{len(pdf_files)}: {pdf_file.name}")
                
                parser.extract_pdf_content(pdf_file)
                json_output = parser.save_json(pdf_file)
                md_output = parser.save_markdown(pdf_file)
                
                print(f"\nRESULTS:")
                print(f" Pages: {parser.extracted_data['total_pages']}")
                print(f" OCR pages: {len(parser.extracted_data['ocr_pages'])}")
                print(f" Words: {parser.extracted_data['total_words']:,}")
                print(f" Text chars: {parser.extracted_data['total_text_length']:,}")
                print(f" Per-page files: {parser.page_texts_dir / pdf_file.stem}")
                print(f" JSON: {json_output}")
                print(f" Markdown: {md_output}")
                print(f" Success!")
                
            except Exception as e:
                print(f" Failed: {pdf_file.name}: {e}")
                continue
        
        print(f"\n PORTABLE PROCESSING COMPLETE!")
        print("\n WORKS ON ANY MACHINE WITH JUST:")
        print("   pip install pdfplumber PyMuPDF Pillow easyocr")
        
    except Exception as e:
        print(f" Fatal error: {e}")

if __name__ == "__main__":
    main()