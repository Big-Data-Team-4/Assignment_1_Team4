#!/usr/bin/env python3
"""
Docling Image Detective - Improved Version
Goal: Extract images and generate clean outputs with proper formatting
Updated for DVC integration with relative paths

Project Structure:
src/
├── scripts/
│   └── extract_docling.py  (this file)
└── data/
    ├── raw/                (input PDFs)
    └── parsed/
        └── docling_output/ (generated outputs)
"""

import os
import json
import jsonlines
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    from PIL import Image
    import fitz  # PyMuPDF for comparison
    print("All packages loaded")
except ImportError as e:
    print(f"Installing: {e}")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "docling", "Pillow", "PyMuPDF", "jsonlines"])
    from docling.document_converter import DocumentConverter
    from PIL import Image
    import fitz


class DoclingProcessor:
    """Clean Docling processor with proper formatting and metadata collection."""
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/parsed/docling_output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.setup_directories()
        self.converter = DocumentConverter()
        
        # For metadata collection
        self.metadata_entries = []
    
    def setup_directories(self):
        """Setup clean directory structure."""
        directories = [
            self.output_dir / "markdown", 
            self.output_dir / "json",
            self.images_dir,
            self.metadata_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def fix_markdown_syntax_highlighting(self, text: str) -> str:
        """
        Fix markdown syntax highlighting issues that cause green text.
        Focus on specific patterns that trigger unwanted highlighting.
        """
        lines = text.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix common patterns that cause green highlighting
            
            # 1. Numbers followed by % (like "14%" -> "14\%")
            line = re.sub(r'(\d+)%', r'\1\\%', line)
            
            # 2. Dollar signs with numbers (like "$54.2" -> "\\$54.2")
            line = re.sub(r'\$(\d)', r'\\$\1', line)
            
            # 3. Years that get highlighted (like "2022" in certain contexts)
            line = re.sub(r'\b(20\d{2})\b', r'\1', line)  # Keep years but break syntax
            
            # 4. Fix standalone backticks that aren't part of code blocks
            if not line.strip().startswith('```'):
                line = line.replace('`', '\\`')
            
            # 5. Fix hash symbols that aren't headers
            if not line.strip().startswith('#'):
                line = line.replace('#', '\\#')
            
            # 6. Fix asterisks that aren't intentional bold/italic
            # Only escape if they're not part of intentional markdown formatting
            line = re.sub(r'(?<!\*)\*(?!\*)', r'\\*', line)
            
            # 7. Fix underscores that trigger unwanted formatting
            line = re.sub(r'(?<!_)_(?!_)', r'\\_', line)
            
            # 8. Fix parentheses with numbers that get highlighted
            line = re.sub(r'\((\$?\d+\.?\d*)\)', r'\\(\1\\)', line)
            
            # 9. Fix ampersands
            line = line.replace('&', '\\&')
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def detect_docling_images(self, doc) -> Dict[str, Any]:
        """Detect images in Docling document."""
        
        image_info = {
            'total_images_detected': 0,
            'images_extracted': 0,
            'extraction_details': [],
            'detection_success': False
        }
        
        if not hasattr(doc, 'pictures') or not doc.pictures:
            print("No images detected by Docling")
            return image_info
        
        pictures = doc.pictures
        image_info['total_images_detected'] = len(pictures)
        image_info['detection_success'] = True
        
        print(f"Docling detected {len(pictures)} images")
        
        # Try to extract each image
        for i, picture in enumerate(pictures):
            extraction_result = self.extract_single_image(picture, f"docling_image_{i+1}")
            image_info['extraction_details'].append(extraction_result)
            
            if extraction_result['extracted']:
                image_info['images_extracted'] += 1
        
        return image_info
    
    def extract_single_image(self, picture, base_name: str) -> Dict[str, Any]:
        """Extract a single image from Docling picture object."""
        
        extraction_result = {
            'image_name': base_name,
            'extracted': False,
            'output_file': None,
            'method_used': None,
            'error': None
        }
        
        # Try different extraction methods
        extraction_methods = [
            ('data', lambda p: getattr(p, 'data', None)),
            ('image', lambda p: getattr(p, 'image', None)),
            ('content', lambda p: getattr(p, 'content', None))
        ]
        
        for method_name, method_func in extraction_methods:
            try:
                data = method_func(picture)
                if isinstance(data, bytes) and len(data) > 100:
                    # Try to save as image
                    try:
                        import io
                        img = Image.open(io.BytesIO(data))
                        output_path = self.images_dir / f"{base_name}.png"
                        img.save(output_path)
                        
                        extraction_result['extracted'] = True
                        extraction_result['output_file'] = str(output_path)
                        extraction_result['method_used'] = method_name
                        
                        print(f"Extracted: {output_path.name}")
                        return extraction_result
                    
                    except Exception as e:
                        extraction_result['error'] = f"Image save failed: {e}"
            
            except Exception as e:
                continue
        
        extraction_result['error'] = "No extractable image data found"
        return extraction_result
    
    def extract_actual_images_from_pdf(self, pdf_path: Path, filename: str) -> Dict[str, Any]:
        """Extract actual images directly from PDF using PyMuPDF."""
        
        actual_images_info = {
            'total_actual_images': 0,
            'images_extracted': 0,
            'extraction_details': []
        }
        
        try:
            pdf_doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_doc.extract_image(xref)
                        
                        image_detail = {
                            'image_name': f"actual_page{page_num+1}_img{img_index+1}",
                            'page': page_num + 1,
                            'width': base_image['width'],
                            'height': base_image['height'],
                            'format': base_image['ext'],
                            'size_bytes': len(base_image['image']),
                            'extracted': False,
                            'output_file': None
                        }
                        
                        # Save the actual image
                        image_bytes = base_image["image"]
                        actual_image_path = self.images_dir / f"actual_{filename}_page{page_num+1}_img{img_index+1}.{base_image['ext']}"
                        
                        with open(actual_image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        image_detail['extracted'] = True
                        image_detail['output_file'] = str(actual_image_path)
                        actual_images_info['images_extracted'] += 1
                        
                        actual_images_info['extraction_details'].append(image_detail)
                        actual_images_info['total_actual_images'] += 1
                        
                        print(f"Saved actual image: {actual_image_path.name}")
                        
                    except Exception as e:
                        print(f"Failed to extract actual image: {e}")
            
            pdf_doc.close()
            
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        return actual_images_info
    
    def count_markdown_image_placeholders(self, markdown_content: str) -> int:
        """Count image placeholders in markdown."""
        return markdown_content.count('<!-- image -->')
    
    def create_metadata_entry(self, pdf_path: Path, docling_images: Dict, actual_images: Dict, 
                            markdown_placeholders: int, processing_time: float) -> Dict[str, Any]:
        """Create consistent metadata entry for JSONL output."""
        
        filename = pdf_path.stem
        file_stats = pdf_path.stat()
        
        metadata = {
            # File information
            'filename': pdf_path.name,
            'file_path': str(pdf_path),
            'file_size_bytes': file_stats.st_size,
            'file_stem': filename,
            
            # Processing information
            'processing_timestamp': datetime.now().isoformat(),
            'processing_time_seconds': round(processing_time, 2),
            'processing_status': 'success',
            
            # Image detection results
            'docling_images_detected': docling_images['total_images_detected'],
            'docling_images_extracted': docling_images['images_extracted'],
            'actual_images_in_pdf': actual_images['total_actual_images'],
            'actual_images_extracted': actual_images['images_extracted'],
            'markdown_image_placeholders': markdown_placeholders,
            
            # Detection accuracy
            'detection_accuracy': docling_images['total_images_detected'] == actual_images['total_actual_images'],
            'extraction_success_rate': (docling_images['images_extracted'] / max(1, docling_images['total_images_detected'])),
            
            # Output files
            'output_files': {
                'markdown': f"markdown/{filename}.md",
                'json': f"json/{filename}.json",
                'metadata': f"metadata/{filename}_metadata.json"
            },
            
            # Detailed results
            'docling_extraction_details': docling_images['extraction_details'],
            'actual_extraction_details': actual_images['extraction_details']
        }
        
        return metadata
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file with clean output."""
        
        filename = pdf_path.stem
        start_time = datetime.now()
        
        print(f"\nProcessing: {pdf_path.name}")
        print("-" * 50)
        
        try:
            # 1. Convert with Docling
            print("1. Converting with Docling...")
            result = self.converter.convert(str(pdf_path))
            doc = result.document
            
            # 2. Get and clean markdown content
            print("2. Generating markdown...")
            raw_markdown = doc.export_to_markdown()
            # Fix markdown syntax highlighting to prevent green text issues
            clean_markdown = self.fix_markdown_syntax_highlighting(raw_markdown)
            
            # Save markdown
            markdown_file = self.output_dir / "markdown" / f"{filename}.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(clean_markdown)
            print(f"Saved: {markdown_file}")
            
            # 3. Save formatted JSON
            print("3. Generating JSON...")
            json_data = doc.model_dump()  # Get as dict first
            json_file = self.output_dir / "json" / f"{filename}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)  # Proper formatting with indent
            print(f"Saved: {json_file}")
            
            # 4. Detect and extract Docling images
            print("4. Processing Docling images...")
            docling_images = self.detect_docling_images(doc)
            
            # 5. Extract actual images from PDF
            print("5. Extracting actual PDF images...")
            actual_images = self.extract_actual_images_from_pdf(pdf_path, filename)
            
            # 6. Count markdown placeholders
            markdown_placeholders = self.count_markdown_image_placeholders(clean_markdown)
            
            # 7. Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 8. Create metadata entry
            metadata = self.create_metadata_entry(
                pdf_path, docling_images, actual_images, 
                markdown_placeholders, processing_time
            )
            
            # Save individual metadata file
            metadata_file = self.metadata_dir / f"{filename}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.metadata_entries.append(metadata)
            
            return {
                'status': 'success',
                'filename': pdf_path.name,
                'processing_time': processing_time,
                'metadata': metadata
            }
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"Error processing {pdf_path.name}: {e}")
            
            # Create error metadata
            error_metadata = {
                'filename': pdf_path.name,
                'file_path': str(pdf_path),
                'file_stem': pdf_path.stem,
                'processing_timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'processing_status': 'failed',
                'error_message': str(e),
                'docling_images_detected': 0,
                'docling_images_extracted': 0,
                'actual_images_in_pdf': 0,
                'actual_images_extracted': 0,
                'markdown_image_placeholders': 0,
                'detection_accuracy': False,
                'extraction_success_rate': 0.0
            }
            
            self.metadata_entries.append(error_metadata)
            
            return {
                'status': 'failed',
                'filename': pdf_path.name,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def save_jsonl_metadata(self):
        """Save all metadata entries to JSONL file with proper structure."""
        jsonl_file = self.metadata_dir / "processing_metadata.jsonl"
        
        # Ensure proper JSONL format: each line is a separate JSON object
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for entry in self.metadata_entries:
                # Write each entry as a single line JSON object (no indentation for JSONL)
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"\nSaved JSONL metadata: {jsonl_file}")
        print(f"Total entries: {len(self.metadata_entries)}")
        
        # Also create a summary JSON for easier reading
        summary_file = self.metadata_dir / "processing_summary.json"
        summary = {
            'total_files_processed': len(self.metadata_entries),
            'successful_processing': len([e for e in self.metadata_entries if e.get('processing_status') == 'success']),
            'failed_processing': len([e for e in self.metadata_entries if e.get('processing_status') == 'failed']),
            'processing_timestamp': datetime.now().isoformat(),
            'entries': self.metadata_entries
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return jsonl_file
    
    def process_all_pdfs(self):
        """Process all PDFs in the input directory."""
        
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.input_dir}")
            print(f"Place your PDF files in the '{self.input_dir}' directory")
            return
        
        print("DOCLING PROCESSOR")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Files to process: {len(pdf_files)}")
        print("=" * 60)
        
        results = []
        successful = 0
        failed = 0
        
        for pdf_file in pdf_files:
            result = self.process_pdf(pdf_file)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
        
        # Save JSONL metadata
        jsonl_file = self.save_jsonl_metadata()
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total files: {len(pdf_files)}")
        
        # Show key statistics
        if successful > 0:
            total_docling_detected = sum(entry.get('docling_images_detected', 0) for entry in self.metadata_entries if entry.get('processing_status') == 'success')
            total_docling_extracted = sum(entry.get('docling_images_extracted', 0) for entry in self.metadata_entries if entry.get('processing_status') == 'success')
            total_actual_images = sum(entry.get('actual_images_in_pdf', 0) for entry in self.metadata_entries if entry.get('processing_status') == 'success')
            
            print(f"\nSUMMARY STATISTICS:")
            print(f"Docling detected: {total_docling_detected} images")
            print(f"Docling extracted: {total_docling_extracted} images")
            print(f"Actual in PDFs: {total_actual_images} images")


def main():
    """Run the processor with configurable paths."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Docling PDF Processor')
    parser.add_argument('--input', '-i', default='data/raw', 
                       help='Input directory containing PDF files (default: data/raw)')
    parser.add_argument('--output', '-o', default='data/parsed/docling_output', 
                       help='Output directory for results (default: data/parsed/docling_output)')
    
    args = parser.parse_args()
    
    processor = DoclingProcessor(input_dir=args.input, output_dir=args.output)
    processor.process_all_pdfs()


if __name__ == "__main__":
    main()