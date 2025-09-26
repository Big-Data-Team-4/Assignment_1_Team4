#!/usr/bin/env python3
"""
Enhanced Camelot Table Extraction for Multiple SEC Filing PDFs
Updated: Batch processing for all PDFs in a directory with enhanced smart filtering
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
import traceback
import re
import numpy as np
import glob

# Required imports with error handling
try:
    import camelot
    # Test if read_pdf method exists
    if not hasattr(camelot, 'read_pdf'):
        raise ImportError("camelot.read_pdf method not found")
except ImportError as e:
    print(f"Camelot installation issue: {e}")
    print("Please run the following commands:")
    print("pip uninstall camelot-py")
    print("pip install 'camelot-py[cv]'")
    print("pip install opencv-python ghostscript")
    print("\nFor macOS, you may also need:")
    print("brew install ghostscript")
    exit(1)

class CamelotSmartExtractor:
    """
    Enhanced Camelot table extractor with selective bullet filtering.
    """
    
    def __init__(self, pdf_path, output_dir="camelot_enhanced_output", log_level=logging.INFO):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.setup_logging(log_level)
        self.setup_output_directory()
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Initialized Enhanced Camelot extractor for: {self.pdf_path.name}")
        
        self.extracted_data = {
            "metadata": {
                "filename": self.pdf_path.name,
                "extraction_date": datetime.now().isoformat(),
                "file_size_mb": round(self.pdf_path.stat().st_size / (1024*1024), 2),
                "extractor": "enhanced camelot with selective bullet removal"
            },
            "tables": {
                "all_detected": [],
                "valid_tables": [],
                "filtered_out": []
            },
            "extraction_stats": {}
        }
    
    def setup_logging(self, level):
        # Create a unique log file for this PDF
        log_filename = f'camelot_extraction_{self.pdf_path.stem}.log'
        log_path = self.output_dir / log_filename
        
        # Create logger specific to this instance
        logger_name = f"{__name__}_{self.pdf_path.stem}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_output_directory(self):
        try:
            # Create subdirectory for this specific PDF
            pdf_output_dir = self.output_dir / self.pdf_path.stem
            pdf_output_dir.mkdir(parents=True, exist_ok=True)
            (pdf_output_dir / "tables_csv").mkdir(exist_ok=True)
            (pdf_output_dir / "table_plots").mkdir(exist_ok=True)
            
            # Update output_dir to the PDF-specific directory
            self.output_dir = pdf_output_dir
            
            self.logger.info(f"Output directory created: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise
    
    def extract_tables_lattice(self):
        self.logger.info("Starting enhanced lattice table extraction...")
        lattice_tables = []
        
        # Strategy 1: Basic lattice
        try:
            self.logger.debug("Trying basic lattice method...")
            tables = camelot.read_pdf(
                str(self.pdf_path), 
                pages='all', 
                flavor='lattice'
            )
            
            self.logger.info(f"Basic lattice method found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                table_data = self._process_table(table, i+1, "lattice_basic")
                if table_data:
                    lattice_tables.append(table_data)
                
        except Exception as e:
            self.logger.error(f"Basic lattice method failed: {e}")
        
        # Strategy 2: Advanced lattice
        try:
            self.logger.debug("Trying advanced lattice method...")
            tables = camelot.read_pdf(
                str(self.pdf_path), 
                pages='all', 
                flavor='lattice',
                split_text=True,
                flag_size=True,
                strip_text='\n',
                line_scale=40
            )
            
            self.logger.info(f"Advanced lattice method found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                table_data = self._process_table(table, i+1, "lattice_advanced")
                if table_data:
                    lattice_tables.append(table_data)
                
        except Exception as e:
            self.logger.error(f"Advanced lattice method failed: {e}")
        
        return lattice_tables
    
    def extract_tables_stream(self):
        self.logger.info("Starting enhanced stream table extraction...")
        stream_tables = []
        
        # Strategy 1: Basic stream
        try:
            self.logger.debug("Trying basic stream method...")
            tables = camelot.read_pdf(
                str(self.pdf_path), 
                pages='all', 
                flavor='stream',
                row_tol=10
            )
            
            self.logger.info(f"Basic stream method found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                table_data = self._process_table(table, i+1, "stream_basic")
                if table_data:
                    stream_tables.append(table_data)
                
        except Exception as e:
            self.logger.error(f"Basic stream method failed: {e}")
        
        # Strategy 2: Stream with edge tolerance
        try:
            self.logger.debug("Trying stream with edge tolerance...")
            tables = camelot.read_pdf(
                str(self.pdf_path), 
                pages='all', 
                flavor='stream',
                edge_tol=50,
                row_tol=5,
                column_tol=5
            )
            
            self.logger.info(f"Stream with edge tolerance found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                table_data = self._process_table(table, i+1, "stream_edge_tol")
                if table_data:
                    stream_tables.append(table_data)
                
        except Exception as e:
            self.logger.error(f"Stream with edge tolerance failed: {e}")
        
        # Strategy 3: Custom stream
        try:
            self.logger.debug("Trying custom stream settings...")
            tables = camelot.read_pdf(
                str(self.pdf_path), 
                pages='all', 
                flavor='stream',
                edge_tol=100,
                row_tol=8,
                column_tol=10,
                strip_text='\n'
            )
            
            self.logger.info(f"Custom stream found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                table_data = self._process_table(table, i+1, "stream_custom")
                if table_data:
                    stream_tables.append(table_data)
                
        except Exception as e:
            self.logger.error(f"Custom stream failed: {e}")
        
        # New Strategy 4: Stream tight for aligned tables
        try:
            self.logger.debug("Trying stream tight settings...")
            tables = camelot.read_pdf(
                str(self.pdf_path), 
                pages='all', 
                flavor='stream',
                row_tol=2,
                column_tol=0,
                strip_text='\n'
            )
            
            self.logger.info(f"Stream tight found {len(tables)} tables")
            
            for i, table in enumerate(tables):
                table_data = self._process_table(table, i+1, "stream_tight")
                if table_data:
                    stream_tables.append(table_data)
                
        except Exception as e:
            self.logger.error(f"Stream tight failed: {e}")
        
        return stream_tables
    
    def _clean_dataframe(self, df):
        """
        Post-process DataFrame: strip whitespace, selectively remove bullets from cells, remove apparent headers/footers.
        New: Only strip prefixes if not financial data.
        """
        if df.empty:
            return df
        
        # Use map for compatibility
        df = df.map(lambda x: str(x).strip() if pd.notna(x) else '')
        
        # New: Selective prefix stripping - skip if original or temp is financial
        bullet_prefixes = r'^[Ã¢â‚¬Â¢Ã¢â€"Ã¢â€"Â¦Ã¢â€"ÂªÃ¢â€"Â«Ã¢â€" Ã¢â€"Â¡\-*+Ã¢â‚¬Â¢]\s*|\d+\.\s*|\([a-zA-Z]+\)\s*'  # Excluded numeric paren
        def selective_strip(cell):
            if not cell or self._is_financial_number(cell):
                return cell  # Preserve if original is financial
            temp_stripped = re.sub(bullet_prefixes, '', cell)
            if self._is_financial_number(temp_stripped):
                return cell  # Preserve original if temp is financial
            return temp_stripped  # Strip if not financial
        
        df = df.map(selective_strip)
        
        # Drop columns that are entirely empty or whitespace
        df = df.loc[:, ~(df == '').all(axis=0)]
        
        # New: Drop columns that are mostly bullet markers or empty (>80%)
        bullet_col_pattern = r'^[Ã¢â‚¬Â¢Ã¢â€"Ã¢â€"Â¦Ã¢â€"ÂªÃ¢â€"Â«Ã¢â€" Ã¢â€"Â¡\-*+Ã¢â‚¬Â¢a-zA-Z\.\(\)]+$'  # Excluded digits
        for col in df.columns:
            col_ratio = df[col].apply(lambda x: bool(re.match(bullet_col_pattern, str(x))) if str(x) else True).mean()
            if col_ratio > 0.8:
                df.drop(col, axis=1, inplace=True)
                self.logger.debug(f"Dropped bullet-heavy column: {col}")
        
        # Drop rows that are entirely empty or whitespace
        df = df.loc[~(df == '').all(axis=1)]
        
        if df.empty:
            return df
        
        # Remove rows that look like page headers/footers
        header_patterns = [
            r'^(UNITED STATES|SECURITIES AND EXCHANGE|WASHINGTON|D\.C\.)',
            r'^FORM 10-K$',
            r'^\(Mark One\)$',
            r'^Page \d+$',
            r'^CONFIDENTIAL$'
        ]
        indices_to_drop = []
        for idx, row in df.iterrows():
            row_text = ' '.join(row.astype(str)).upper()
            if any(re.search(pattern, row_text) for pattern in header_patterns):
                indices_to_drop.append(idx)
        if indices_to_drop:
            df.drop(indices_to_drop, inplace=True)
        
        # Drop rows with only bullets or list markers (now after stripping)
        bullet_rows = 0
        indices_to_drop = []
        bullet_pattern = r'^(?:\s*|\d+\.?\s*|\([a-zA-Z]+\)\s*)$'  # Updated to ignore numeric parens
        for idx, row in df.iterrows():
            if all(re.match(bullet_pattern, str(cell)) for cell in row if str(cell)):
                indices_to_drop.append(idx)
                bullet_rows += 1
        if indices_to_drop:
            df.drop(indices_to_drop, inplace=True)
        
        self.logger.debug(f"Cleaned {bullet_rows} bullet rows and empty elements from DF")
        
        return df
    
    def _process_table(self, table, table_num, method):
        try:
            # Clean the DataFrame first
            if hasattr(table, 'df'):
                original_shape = table.df.shape
                table.df = self._clean_dataframe(table.df)
                cleaned_shape = table.df.shape
                if cleaned_shape != original_shape:
                    self.logger.debug(f"Cleaned table {table_num}: {original_shape} -> {cleaned_shape}")
                if table.df.empty:
                    self.logger.warning(f"Table {table_num} became empty after cleaning")
                    return None
            
            # Basic table information
            table_data = {
                "table_id": f"{method}_{table_num}",
                "page": getattr(table, 'page', 'unknown'),
                "method": method,
                "accuracy": getattr(table, 'accuracy', 0.0),
                "shape": table.df.shape if hasattr(table, 'df') else (0, 0),
                "whitespace": getattr(table, 'whitespace', 0.0),
                "order": getattr(table, 'order', 0),
                "data": table.df.to_dict('records') if hasattr(table, 'df') else [],
                "raw_data": table.df.values.tolist() if hasattr(table, 'df') else [],
                "columns": table.df.columns.tolist() if hasattr(table, 'df') else []
            }
            
            # Apply enhanced validation
            if hasattr(table, 'df') and not table.df.empty:
                validation_result = self._validate_table_structure(table.df, table_data)
                table_data.update(validation_result)
                
                # Save only valid tables to CSV
                if validation_result["is_valid_table"]:
                    csv_filename = f"{method}_table_{table_num}_page_{table.page}.csv"
                    csv_path = self.output_dir / "tables_csv" / csv_filename
                    
                    # FIX: Clean up column names and remove numeric headers
                    df_to_save = table.df.copy()
                    
                    # Check if columns are numeric (0, 1, 2, etc.)
                    if all(str(col).isdigit() for col in df_to_save.columns):
                        # Option 1: Try to use first row as headers if it looks like headers
                        if (not df_to_save.empty and 
                            not all(str(df_to_save.iloc[0, j]).replace('.', '').replace(',', '').replace('$', '').replace('(', '').replace(')', '').isdigit() 
                                   for j in range(min(3, len(df_to_save.columns))))):  # Check first 3 columns
                            
                            # Use first row as column names
                            new_columns = []
                            for j in range(len(df_to_save.columns)):
                                header_val = str(df_to_save.iloc[0, j]).strip()
                                if header_val and header_val not in ['nan', 'NaN', '']:
                                    new_columns.append(header_val)
                                else:
                                    new_columns.append(f"Column_{j+1}")
                            
                            df_to_save.columns = new_columns
                            df_to_save = df_to_save.iloc[1:].reset_index(drop=True)  # Remove first row, reset index
                            self.logger.debug(f"Used first row as headers for table {table_num}")
                        
                        else:
                            # Option 2: Replace with generic column names
                            df_to_save.columns = [f"Column_{i+1}" for i in range(len(df_to_save.columns))]
                            self.logger.debug(f"Replaced numeric columns with generic names for table {table_num}")
                    
                    # Ensure no empty column names
                    clean_columns = []
                    for i, col in enumerate(df_to_save.columns):
                        clean_col = str(col).strip()
                        if not clean_col or clean_col in ['nan', 'NaN', '']:
                            clean_columns.append(f"Column_{i+1}")
                        else:
                            clean_columns.append(clean_col)
                    df_to_save.columns = clean_columns
                    
                    # Save CSV without index (prevents 0,1,2... row numbers)
                    df_to_save.to_csv(csv_path, index=False)
                    table_data["csv_file"] = str(csv_path)
                    
                    # Update the stored data with cleaned version
                    table_data["data"] = df_to_save.to_dict('records')
                    table_data["raw_data"] = df_to_save.values.tolist()
                    table_data["columns"] = df_to_save.columns.tolist()
                    
                    self.logger.info(f"Valid table saved: {csv_path}")
                else:
                    table_data["csv_file"] = None
                    reasons = ", ".join(validation_result["filter_reasons"])
                    self.logger.info(f"Filtered table: {table_data['table_id']} - Reasons: {reasons}")
                
                # Generate table plot (only for valid)
                if validation_result["is_valid_table"]:
                    try:
                        plot_filename = f"{method}_table_{table_num}_page_{table.page}.png"
                        plot_path = self.output_dir / "table_plots" / plot_filename
                        table.plot(str(plot_path))
                        table_data["plot_saved"] = str(plot_path)
                    except Exception as plot_error:
                        self.logger.debug(f"Could not save plot for table {table_num}: {plot_error}")
                        table_data["plot_saved"] = None
            else:
                self.logger.warning(f"Empty table found: {method}_table_{table_num}")
                table_data["is_valid_table"] = False
                table_data["filter_reasons"] = ["empty_after_cleaning"]
                table_data["confidence_score"] = 0
                return None
            
            return table_data
            
        except Exception as e:
            self.logger.error(f"Error processing table {table_num}: {e}")
            return {
                "table_id": f"{method}_{table_num}",
                "error": str(e),
                "method": method,
                "is_valid_table": False,
                "filter_reasons": ["processing_error"]
            }
    
    def _validate_table_structure(self, df, table_data):
        try:
            rows, cols = df.shape
            validation_result = {
                "is_valid_table": True,
                "confidence_score": 0,
                "filter_reasons": [],
                "validation_details": {}
            }
            
            # Quick elimination - relaxed for single-row
            if rows < 1 or cols < 2:
                validation_result["is_valid_table"] = False
                validation_result["filter_reasons"].append("insufficient_size")
                return validation_result
            
            # FIRST: Preserve known good SEC tables
            if self._is_known_sec_table(df):
                validation_result["is_valid_table"] = True
                validation_result["confidence_score"] = 10
                validation_result["filter_reasons"] = []
                return validation_result
            
            # THEN: Enhanced list checks
            if self._is_bulleted_list(df):
                validation_result["is_valid_table"] = False
                validation_result["filter_reasons"].append("bulleted_list")
                return validation_result
            
            if self._is_section_header_table(df):
                validation_result["is_valid_table"] = False
                validation_result["filter_reasons"].append("section_header")
                return validation_result
            
            if self._is_extra_text_table(df):
                validation_result["is_valid_table"] = False
                validation_result["filter_reasons"].append("extra_text_above")
                return validation_result
            
            # Positive scoring
            score = 0
            details = {}
            
            # 1. Size
            if rows >= 4 and cols >= 3:
                score += 4
            elif rows >= 3 and cols >= 2:
                score += 2
            details["size_score"] = f"{rows}x{cols}"
            
            # 2. Non-empty columns
            non_empty_cols = self._count_non_empty_columns(df)
            if non_empty_cols >= 3:
                score += 3
            elif non_empty_cols >= 2:
                score += 1
            details["non_empty_columns"] = non_empty_cols
            
            # 3. Numeric columns
            numeric_columns = self._count_numeric_columns(df)
            if numeric_columns >= 2:
                score += 5
            elif numeric_columns >= 1:
                score += 3
            details["numeric_columns"] = numeric_columns
            
            # 4. Financial keywords
            financial_score = self._count_financial_keywords(df)
            score += financial_score * 2
            details["financial_keywords"] = financial_score
            
            # 5. Headers
            if self._has_proper_headers(df):
                score += 2
            details["has_proper_headers"] = self._has_proper_headers(df)
            
            # 6. Row consistency
            consistency_score = self._check_row_consistency(df)
            score += consistency_score
            details["row_consistency"] = consistency_score
            
            # 7. Accuracy bonus
            accuracy = table_data.get("accuracy", 0)
            if accuracy > 0.85:
                score += 3
            elif accuracy > 0.7:
                score += 2
            details["camelot_accuracy"] = accuracy
            
            # 8. Tabular patterns
            if self._has_tabular_content_patterns(df):
                score += 3
            details["tabular_patterns"] = self._has_tabular_content_patterns(df)
            
            validation_result["confidence_score"] = score
            validation_result["validation_details"] = details
            
            # Decision
            if score >= 8:
                validation_result["is_valid_table"] = True
            elif score >= 5 and numeric_columns >= 1:
                validation_result["is_valid_table"] = True
            elif score >= 4 and self._has_proper_headers(df):
                validation_result["is_valid_table"] = True
            else:
                validation_result["is_valid_table"] = False
                validation_result["filter_reasons"].append(f"low_confidence_score_{score}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return {
                "is_valid_table": False,
                "confidence_score": 0,
                "filter_reasons": ["validation_error"],
                "validation_details": {"error": str(e)}
            }
    
    def _is_bulleted_list(self, df):
        """
        Enhanced: Focus on first column for bullet artifacts.
        """
        all_text = ' '.join(df.astype(str).values.flatten())
        list_indicators = [
            r'^(?:[Ã¢â‚¬Â¢Ã¢â€"Ã¢â€"Â¦Ã¢â€"ÂªÃ¢â€"Â«Ã¢â€" Ã¢â€"Â¡\-*+]|\d+\.| [a-zA-Z]\)| \(\w+\) )\s+[A-Z]',
            r'^\s*[a-zA-Z]\.\s+[A-Z]',
            r'^\s*\([ivx]+\)\s+[A-Z]',
            r'^\s*-{2,}\s*$',
        ]
        indicator_matches = sum(1 for pattern in list_indicators if re.search(pattern, all_text, re.MULTILINE))
        
        # Enhanced: Prioritize first column for bullet checks
        first_col_bullets = 0
        bullet_marker = r'^(?:[Ã¢â‚¬Â¢Ã¢â€"Ã¢â€"Â¦Ã¢â€"ÂªÃ¢â€"Â«Ã¢â€" Ã¢â€"Â¡\-*+]|\d+\.|\([a-zA-Z]+\))'  # Ignore numeric parens
        if len(df.columns) > 0:
            first_col = df.iloc[:, 0].dropna().astype(str)
            first_col_bullets = sum(1 for cell in first_col if re.match(bullet_marker, cell.strip()))
        
        first_col_ratio = first_col_bullets / len(df) if len(df) > 0 else 0
        
        # Overall row check
        bullet_row_count = 0
        for _, row in df.iterrows():
            if any(re.match(bullet_marker, str(cell).strip()) for cell in row if str(cell).strip()):
                bullet_row_count += 1
        
        row_ratio = bullet_row_count / len(df) if len(df) > 0 else 0
        
        # Stricter: High first-col ratio OR high overall + indicators
        is_list = (first_col_ratio > 0.7) or (indicator_matches >= 1 and row_ratio > 0.7)
        self.logger.debug(f"Bulleted list check: first_col_ratio={first_col_ratio:.2f}, row_ratio={row_ratio:.2f}, indicators={indicator_matches}, is_list={is_list}")
        return is_list
    
    def _is_section_header_table(self, df):
        if len(df) <= 2:
            header_text = ' '.join(df.astype(str).values.flatten()).upper()
            if re.match(r'^(PART|ITEM|SECTION|SUBPART)\s+\w+', header_text) or len(header_text.split()) < 5:
                return True
        return False
    
    def _is_extra_text_table(self, df):
        first_row = ' '.join(df.iloc[0].astype(str)).lower()
        extra_patterns = [
            r'indicate by check mark',
            r'well-known seasoned issuer',
            r'emerging growth company',
            r'shell company'
        ]
        return any(pattern in first_row for pattern in extra_patterns)
    
    def _is_known_sec_table(self, df):
        all_text = ' '.join(df.astype(str).values.flatten()).lower()
        known_patterns = [
            r'(item|part)\s+\d+\.\s+\w+',
            r'trading symbol[s]?',
            r'common stock.*new york stock exchange',
            r'notes due \d{4}'
        ]
        matches = any(re.search(pattern, all_text) for pattern in known_patterns)
        self.logger.debug(f"Known SEC table check: matches={matches}")
        return matches
    
    def _count_non_empty_columns(self, df):
        non_empty_count = 0
        for col in df.columns:
            col_data = df[col].dropna()
            meaningful_data = [str(val).strip() for val in col_data if str(val).strip()]
            if len(meaningful_data) / len(df) > 0.3:
                non_empty_count += 1
        return non_empty_count
    
    def _count_numeric_columns(self, df):
        numeric_cols = 0
        for col in df.columns:
            col_data = df[col].dropna().astype(str)
            if len(col_data) == 0:
                continue
            numeric_count = 0
            for value in col_data:
                value_clean = value.strip()
                if self._is_financial_number(value_clean):
                    numeric_count += 1
            if len(col_data) > 0 and numeric_count / len(col_data) > 0.5:
                numeric_cols += 1
        return numeric_cols
    
    def _is_financial_number(self, value):
        if not value or len(value) > 50:
            return False
        financial_patterns = [
            r'\$[\d,]+\.?\d*',
            r'[\d,]+\.?\d*%?',
            r'\([\d,]+\.?\d*\)',
            r'[\d,]+[MK]?(?:\s*\(000s\))?',
            r'^\d{4}-\d{2}-\d{2}',
            r'^\d{1,2}/\d{1,2}/\d{4}',
            r'december 31, \d{4}',
        ]
        is_financial = any(re.search(pattern, value) for pattern in financial_patterns)
        is_list_marker = re.match(r'^[\d+\.\(\)\w\s\-*Ã¢â‚¬Â¢]{0,5}$', value.strip()) and len(value.strip()) < 3
        return is_financial and not is_list_marker
    
    def _count_financial_keywords(self, df):
        if df.empty:
            return 0
        header_text = ' '.join(str(x) for x in list(df.columns) + list(df.iloc[0])).lower()
        financial_terms = [
            'revenue', 'net income', 'cash flow', 'assets', 'liabilities', 'equity',
            'dividend', 'shares', 'eps', 'operating', 'consolidated', 'balance sheet',
            'income statement', 'market risk', 'cybersecurity', 'executive', 'compensation',
            'fiscal year', 'december 31', 'million', 'billion', 'thousands', 'trading symbol', 'stock exchange', 'tax', 'rate'
        ]
        keyword_count = sum(1 for term in financial_terms if term in header_text)
        return min(keyword_count, 5)
    
    def _has_proper_headers(self, df):
        if df.empty:
            return False
        headers = list(df.columns) + list(df.iloc[0].astype(str))
        descriptive_headers = sum(1 for h in headers 
                                 if isinstance(h, str) and len(h.strip()) > 3 
                                 and not re.match(r'^\d+$', h.strip()))
        return descriptive_headers >= 2
    
    def _check_row_consistency(self, df):
        if len(df) < 2:
            return 0
        column_patterns = []
        for col in df.columns:
            col_data = df[col].dropna().astype(str)
            if len(col_data) == 0:
                continue
            numeric_count = sum(1 for val in col_data if self._is_financial_number(val))
            numeric_ratio = numeric_count / len(col_data)
            if numeric_ratio > 0.7:
                column_patterns.append("numeric")
            elif numeric_ratio > 0.3:
                column_patterns.append("mixed")
            else:
                column_patterns.append("text")
        if len(set(column_patterns)) <= 2 and len(column_patterns) >= 2:
            return 2
        elif len(column_patterns) >= 2:
            return 1
        else:
            return 0
    
    def _has_tabular_content_patterns(self, df):
        if df.empty:
            return False
        all_text = ' '.join(df.astype(str).values.flatten()).lower()
        table_indicators = [
            'total', 'subtotal', 'average', 'as of', 'ended', 'consolidated',
            'shares outstanding', 'market value', 'trading symbol'
        ]
        indicator_matches = sum(1 for indicator in table_indicators if indicator in all_text)
        return indicator_matches >= 2
    
    def _apply_smart_filtering(self, all_tables):
        valid_tables = []
        filtered_out = []
        for table in all_tables:
            if table.get("is_valid_table", True):
                valid_tables.append(table)
            else:
                filtered_out.append(table)
        self.logger.info(f"Enhanced filtering: {len(valid_tables)} valid, {len(filtered_out)} filtered out")
        return {
            "valid_tables": valid_tables,
            "filtered_out": filtered_out
        }
    
    def generate_extraction_stats(self):
        all_detected = self.extracted_data["tables"]["all_detected"]
        valid_tables = self.extracted_data["tables"]["valid_tables"]
        filtered_out = self.extracted_data["tables"]["filtered_out"]
        
        method_counts = {}
        accuracy_scores = []
        confidence_scores = []
        page_distribution = {}
        
        for table in valid_tables:
            method = table.get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1
            
            if "accuracy" in table and table["accuracy"] > 0:
                accuracy_scores.append(table["accuracy"])
            
            if "confidence_score" in table:
                confidence_scores.append(table["confidence_score"])
            
            page = table.get("page", "unknown")
            page_distribution[str(page)] = page_distribution.get(str(page), 0) + 1
        
        filter_reason_counts = {}
        for table in filtered_out:
            for reason in table.get("filter_reasons", []):
                filter_reason_counts[reason] = filter_reason_counts.get(reason, 0) + 1
        
        stats = {
            "total_detected": len(all_detected),
            "valid_tables": len(valid_tables),
            "filtered_out": len(filtered_out),
            "filter_rate": len(filtered_out) / len(all_detected) if all_detected else 0,
            "methods_used": list(method_counts.keys()),
            "valid_tables_by_method": method_counts,
            "average_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "accuracy_range": {
                "min": min(accuracy_scores) if accuracy_scores else 0,
                "max": max(accuracy_scores) if accuracy_scores else 0
            },
            "confidence_range": {
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0
            },
            "pages_with_valid_tables": len(page_distribution),
            "page_distribution": page_distribution,
            "high_confidence_tables": len([t for t in valid_tables if t.get("confidence_score", 0) >= 8]),
            "filter_reasons_breakdown": filter_reason_counts
        }
        
        self.extracted_data["extraction_stats"] = stats
        return stats
    
    def save_results(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camelot_enhanced_extraction_{self.pdf_path.stem}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        try:
            # Save JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"JSON results saved to: {output_path}")
            
            # Save Markdown
            markdown_path = self.save_as_markdown(filename)
            self.logger.info(f"Markdown results saved to: {markdown_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def save_as_markdown(self, json_filename):
        """Convert extraction results to Markdown format and save."""
        markdown_filename = json_filename.replace('.json', '.md')
        markdown_path = self.output_dir / markdown_filename
        
        try:
            md_content = self._generate_markdown_content()
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            return markdown_path
            
        except Exception as e:
            self.logger.error(f"Failed to save markdown: {e}")
            raise
    
    def _generate_markdown_content(self):
        """Generate markdown content from extraction data."""
        data = self.extracted_data
        metadata = data.get("metadata", {})
        stats = data.get("extraction_stats", {})
        valid_tables = data.get("tables", {}).get("valid_tables", [])
        filtered_tables = data.get("tables", {}).get("filtered_out", [])
        
        # Start building markdown
        md_content = f"# Camelot Table Extraction Report\n\n"
        md_content += f"**Document**: {metadata.get('filename', 'Unknown')}\n"
        md_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md_content += f"**Project**: LANTERN - FinTrust Analytics\n\n"
        md_content += "---\n\n"
        
        # Metadata section
        md_content += self._format_metadata_section(metadata)
        
        # Statistics section
        md_content += self._format_stats_section(stats)
        
        # Valid tables summary
        md_content += self._format_table_summary(valid_tables, "Valid Tables")
        
        # Filtered tables summary
        if filtered_tables:
            md_content += self._format_table_summary(filtered_tables, "Filtered Tables", show_reasons=True)
        
        # Detailed table information
        if valid_tables:
            md_content += self._format_table_details(valid_tables, "Valid Table Details")
        
        return md_content
    
    def _format_metadata_section(self, metadata):
        """Format the metadata section for markdown."""
        md_content = "## Extraction Metadata\n\n"
        
        md_content += f"- **Filename**: `{metadata.get('filename', 'Unknown')}`\n"
        md_content += f"- **Extraction Date**: {metadata.get('extraction_date', 'Unknown')}\n"
        md_content += f"- **File Size**: {metadata.get('file_size_mb', 'Unknown')} MB\n"
        md_content += f"- **Extractor**: {metadata.get('extractor', 'Unknown')}\n\n"
        
        return md_content
    
    def _format_stats_section(self, stats):
        """Format the extraction statistics section for markdown."""
        md_content = "## Extraction Statistics\n\n"
        
        # Overview stats
        md_content += "### Overview\n"
        md_content += f"- **Total Tables Detected**: {stats.get('total_detected', 0)}\n"
        md_content += f"- **Valid Tables**: {stats.get('valid_tables', 0)}\n"
        md_content += f"- **Filtered Out**: {stats.get('filtered_out', 0)}\n"
        md_content += f"- **Filter Rate**: {stats.get('filter_rate', 0):.1%}\n"
        md_content += f"- **High Confidence Tables**: {stats.get('high_confidence_tables', 0)}\n\n"
        
        # Method breakdown
        if stats.get('valid_tables_by_method'):
            md_content += "### Valid Tables by Method\n"
            for method, count in stats.get('valid_tables_by_method', {}).items():
                md_content += f"- **{method}**: {count} tables\n"
            md_content += "\n"
        
        # Quality metrics
        md_content += "### Quality Metrics\n"
        md_content += f"- **Average Accuracy**: {stats.get('average_accuracy', 0):.3f}\n"
        md_content += f"- **Average Confidence**: {stats.get('average_confidence', 0):.1f}\n"
        
        acc_range = stats.get('accuracy_range', {})
        conf_range = stats.get('confidence_range', {})
        md_content += f"- **Accuracy Range**: {acc_range.get('min', 0):.3f} - {acc_range.get('max', 0):.3f}\n"
        md_content += f"- **Confidence Range**: {conf_range.get('min', 0)} - {conf_range.get('max', 0)}\n\n"
        
        # Page distribution
        page_dist = stats.get('page_distribution', {})
        if page_dist:
            md_content += "### Page Distribution\n"
            md_content += f"- **Pages with Valid Tables**: {stats.get('pages_with_valid_tables', 0)}\n"
            
            # Sort pages numerically
            sorted_pages = sorted(page_dist.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
            page_list = [f"Page {page} ({count} tables)" for page, count in sorted_pages[:10]]  # Limit to first 10
            if len(sorted_pages) > 10:
                page_list.append(f"... and {len(sorted_pages) - 10} more pages")
            
            md_content += f"- **Distribution**: {', '.join(page_list)}\n\n"
        
        # Filter reasons
        filter_reasons = stats.get('filter_reasons_breakdown', {})
        if filter_reasons:
            md_content += "### Filter Reasons Breakdown\n"
            for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
                md_content += f"- **{reason.replace('_', ' ').title()}**: {count} tables\n"
            md_content += "\n"
        
        return md_content
    
    def _format_table_summary(self, tables, title="Tables", show_reasons=False):
        """Format a summary of tables for markdown."""
        if not tables:
            return f"## {title}\n\nNo tables found.\n\n"
        
        md_content = f"## {title} ({len(tables)} tables)\n\n"
        
        # Create summary table
        if show_reasons:
            md_content += "| Table ID | Page | Method | Shape | Confidence | Accuracy | Filter Reasons |\n"
            md_content += "|----------|------|--------|-------|------------|----------|----------------|\n"
        else:
            md_content += "| Table ID | Page | Method | Shape | Confidence | Accuracy | CSV File |\n"
            md_content += "|----------|------|--------|-------|------------|----------|----------|\n"
        
        for table in tables:
            table_id = table.get('table_id', 'Unknown')
            page = table.get('page', 'N/A')
            method = table.get('method', 'Unknown')
            shape = table.get('shape', [0, 0])
            confidence = table.get('confidence_score', 0)
            accuracy = table.get('accuracy', 0)
            
            # Format shape
            shape_str = f"{shape[0]}×{shape[1]}" if isinstance(shape, list) and len(shape) >= 2 else str(shape)
            
            if show_reasons:
                reasons = table.get('filter_reasons', [])
                reasons_str = ', '.join(reasons) if reasons else 'N/A'
                md_content += f"| {table_id} | {page} | {method} | {shape_str} | {confidence} | {accuracy:.3f} | {reasons_str} |\n"
            else:
                csv_file = table.get('csv_file', '')
                csv_name = Path(csv_file).name if csv_file else 'N/A'
                md_content += f"| {table_id} | {page} | {method} | {shape_str} | {confidence} | {accuracy:.3f} | `{csv_name}` |\n"
        
        md_content += "\n"
        
        return md_content
    
    def _format_table_details(self, tables, title="Table Details"):
        """Format detailed information for each table in markdown."""
        if not tables:
            return ""
        
        md_content = f"## {title}\n\n"
        
        for i, table in enumerate(tables, 1):
            table_id = table.get('table_id', f'Table {i}')
            md_content += f"### {table_id}\n\n"
            
            # Basic info
            md_content += f"- **Page**: {table.get('page', 'Unknown')}\n"
            md_content += f"- **Method**: {table.get('method', 'Unknown')}\n"
            md_content += f"- **Shape**: {table.get('shape', 'Unknown')}\n"
            md_content += f"- **Confidence Score**: {table.get('confidence_score', 0)}\n"
            md_content += f"- **Camelot Accuracy**: {table.get('accuracy', 0):.3f}\n"
            
            # Files
            if table.get('csv_file'):
                csv_path = Path(table['csv_file']).name
                md_content += f"- **CSV File**: `{csv_path}`\n"
            
            if table.get('plot_saved'):
                plot_path = Path(table['plot_saved']).name
                md_content += f"- **Plot File**: `{plot_path}`\n"
            
            # Validation details
            val_details = table.get('validation_details', {})
            if val_details:
                md_content += f"- **Non-empty Columns**: {val_details.get('non_empty_columns', 'N/A')}\n"
                md_content += f"- **Numeric Columns**: {val_details.get('numeric_columns', 'N/A')}\n"
                md_content += f"- **Financial Keywords**: {val_details.get('financial_keywords', 'N/A')}\n"
                md_content += f"- **Has Proper Headers**: {val_details.get('has_proper_headers', 'N/A')}\n"
            
            # Data preview
            data = table.get('data', [])
            if data and len(data) > 0:
                md_content += "\n**Data Preview:**\n\n"
                
                # Convert to DataFrame for better formatting
                try:
                    df = pd.DataFrame(data)
                    
                    # Limit preview size
                    preview_df = df.head(3)  # First 3 rows
                    if len(df.columns) > 5:  # Limit columns too
                        preview_df = preview_df.iloc[:, :5]
                        preview_df['...'] = '...'
                    
                    # Convert to markdown table
                    md_table = preview_df.to_markdown(index=False)
                    md_content += md_table + "\n\n"
                    
                    if len(data) > 3:
                        md_content += f"*Showing first 3 of {len(data)} rows*\n\n"
                    
                except Exception as e:
                    md_content += f"*Error formatting data preview: {e}*\n\n"
            
            md_content += "---\n\n"
        
        return md_content
    
    def run_extraction(self):
        self.logger.info("=== Starting Enhanced Camelot Extraction ===")
        
        try:
            lattice_tables = self.extract_tables_lattice()
            stream_tables = self.extract_tables_stream()
            all_tables = [t for t in lattice_tables + stream_tables if t]
            filtered_results = self._apply_smart_filtering(all_tables)
            
            self.extracted_data["tables"]["all_detected"] = all_tables
            self.extracted_data["tables"]["valid_tables"] = filtered_results["valid_tables"]
            self.extracted_data["tables"]["filtered_out"] = filtered_results["filtered_out"]
            
            stats = self.generate_extraction_stats()
            output_file = self.save_results()
            
            self.logger.info("=== Enhanced Extraction Completed ===")
            self.logger.info(f"Summary: {stats['valid_tables']} valid tables saved to CSV")
            
            return self.extracted_data
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            self.logger.error(traceback.format_exc())
            raise


class BatchCamelotProcessor:
    """
    Batch processor for multiple PDF files.
    """
    
    def __init__(self, input_dir, output_dir, log_level=logging.INFO):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.log_level = log_level
        
        # Setup main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup batch logging
        self.setup_batch_logging()
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    def setup_batch_logging(self):
        batch_log_path = self.output_dir / "batch_processing.log"
        
        # Create batch logger
        self.batch_logger = logging.getLogger("batch_processor")
        self.batch_logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in self.batch_logger.handlers[:]:
            self.batch_logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(batch_log_path)
        file_handler.setLevel(self.log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.batch_logger.addHandler(file_handler)
        self.batch_logger.addHandler(console_handler)
    
    def find_pdf_files(self):
        """Find all PDF files in the input directory."""
        pdf_files = []
        
        # Search for PDFs in the input directory
        pdf_patterns = ['*.pdf', '*.PDF']
        for pattern in pdf_patterns:
            pdf_files.extend(glob.glob(str(self.input_dir / pattern)))
        
        # Convert to Path objects and sort
        pdf_files = [Path(f) for f in pdf_files]
        pdf_files.sort()
        
        self.batch_logger.info(f"Found {len(pdf_files)} PDF files in {self.input_dir}")
        for pdf_file in pdf_files:
            self.batch_logger.info(f"  - {pdf_file.name}")
        
        return pdf_files
    
    def process_all_pdfs(self):
        """Process all PDF files in the input directory."""
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            self.batch_logger.warning("No PDF files found to process!")
            return {}
        
        batch_results = {
            "batch_metadata": {
                "start_time": datetime.now().isoformat(),
                "input_directory": str(self.input_dir),
                "output_directory": str(self.output_dir),
                "total_pdfs": len(pdf_files)
            },
            "processing_results": {},
            "batch_summary": {}
        }
        
        successful_extractions = 0
        failed_extractions = 0
        total_valid_tables = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            self.batch_logger.info(f"\n{'='*60}")
            self.batch_logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_file.name}")
            self.batch_logger.info(f"{'='*60}")
            
            try:
                # Create extractor for this PDF
                extractor = CamelotSmartExtractor(
                    pdf_path=pdf_file,
                    output_dir=self.output_dir,
                    log_level=self.log_level
                )
                
                # Run extraction
                extraction_results = extractor.run_extraction()
                
                # Store results
                batch_results["processing_results"][pdf_file.name] = {
                    "status": "success",
                    "extraction_data": extraction_results,
                    "output_directory": str(extractor.output_dir)
                }
                
                successful_extractions += 1
                valid_tables = extraction_results.get("extraction_stats", {}).get("valid_tables", 0)
                total_valid_tables += valid_tables
                
                self.batch_logger.info(f"Successfully processed {pdf_file.name}: {valid_tables} valid tables")
                
            except Exception as e:
                self.batch_logger.error(f"Failed to process {pdf_file.name}: {e}")
                self.batch_logger.error(traceback.format_exc())
                
                batch_results["processing_results"][pdf_file.name] = {
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                
                failed_extractions += 1
        
        # Generate batch summary
        batch_results["batch_metadata"]["end_time"] = datetime.now().isoformat()
        batch_results["batch_summary"] = {
            "total_pdfs_processed": len(pdf_files),
            "successful_extractions": successful_extractions,
            "failed_extractions": failed_extractions,
            "success_rate": successful_extractions / len(pdf_files) if pdf_files else 0,
            "total_valid_tables_extracted": total_valid_tables,
            "average_tables_per_pdf": total_valid_tables / successful_extractions if successful_extractions else 0
        }
        
        # Save batch results
        self.save_batch_results(batch_results)
        
        return batch_results
    
    def save_batch_results(self, batch_results):
        """Save the batch processing results in both JSON and Markdown formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        batch_results_file = self.output_dir / f"batch_extraction_results_{timestamp}.json"
        try:
            with open(batch_results_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
            self.batch_logger.info(f"Batch JSON results saved to: {batch_results_file}")
        except Exception as e:
            self.batch_logger.error(f"Failed to save batch JSON results: {e}")
        
        # Save Markdown
        markdown_file = self.output_dir / f"batch_extraction_results_{timestamp}.md"
        try:
            md_content = self._generate_batch_markdown(batch_results)
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(md_content)
            self.batch_logger.info(f"Batch Markdown results saved to: {markdown_file}")
        except Exception as e:
            self.batch_logger.error(f"Failed to save batch Markdown results: {e}")
    
    def _generate_batch_markdown(self, batch_results):
        """Generate markdown content for batch processing results."""
        metadata = batch_results.get("batch_metadata", {})
        summary = batch_results.get("batch_summary", {})
        processing_results = batch_results.get("processing_results", {})
        
        # Start building markdown
        md_content = f"# Batch Camelot Extraction Report\n\n"
        md_content += f"**Project**: LANTERN - FinTrust Analytics\n"
        md_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        md_content += f"**Batch Processing**: {metadata.get('total_pdfs', 0)} PDF files\n\n"
        md_content += "---\n\n"
        
        # Batch metadata
        md_content += "## Batch Metadata\n\n"
        md_content += f"- **Input Directory**: `{metadata.get('input_directory', 'Unknown')}`\n"
        md_content += f"- **Output Directory**: `{metadata.get('output_directory', 'Unknown')}`\n"
        md_content += f"- **Start Time**: {metadata.get('start_time', 'Unknown')}\n"
        md_content += f"- **End Time**: {metadata.get('end_time', 'Unknown')}\n"
        md_content += f"- **Total PDFs**: {metadata.get('total_pdfs', 0)}\n\n"
        
        # Batch summary
        md_content += "## Batch Summary\n\n"
        md_content += f"- **Total PDFs Processed**: {summary.get('total_pdfs_processed', 0)}\n"
        md_content += f"- **Successful Extractions**: {summary.get('successful_extractions', 0)}\n"
        md_content += f"- **Failed Extractions**: {summary.get('failed_extractions', 0)}\n"
        md_content += f"- **Success Rate**: {summary.get('success_rate', 0):.1%}\n"
        md_content += f"- **Total Valid Tables**: {summary.get('total_valid_tables_extracted', 0)}\n"
        md_content += f"- **Average Tables per PDF**: {summary.get('average_tables_per_pdf', 0):.1f}\n\n"
        
        # Processing results table
        md_content += "## Individual PDF Results\n\n"
        md_content += "| PDF File | Status | Valid Tables | Output Directory |\n"
        md_content += "|----------|--------|--------------|------------------|\n"
        
        for pdf_name, result in processing_results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                extraction_data = result.get('extraction_data', {})
                stats = extraction_data.get('extraction_stats', {})
                valid_tables = stats.get('valid_tables', 0)
                status_icon = "Success"
            else:
                valid_tables = "N/A"
                status_icon = "Failed"
            
            output_dir = result.get('output_directory', 'N/A')
            output_name = Path(output_dir).name if output_dir != 'N/A' else 'N/A'
            
            md_content += f"| `{pdf_name}` | {status_icon} | {valid_tables} | `{output_name}` |\n"
        
        md_content += "\n"
        
        # Failed extractions details
        failed_results = {k: v for k, v in processing_results.items() if v.get('status') == 'failed'}
        if failed_results:
            md_content += "## Failed Extractions\n\n"
            for pdf_name, result in failed_results.items():
                md_content += f"### {pdf_name}\n\n"
                md_content += f"**Error**: {result.get('error', 'Unknown error')}\n\n"
                md_content += "---\n\n"
        
        return md_content
    
    def print_batch_summary(self, batch_results):
        """Print a summary of the batch processing results."""
        summary = batch_results.get("batch_summary", {})
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETED!")
        print(f"{'='*70}")
        print(f"Total PDFs processed: {summary.get('total_pdfs_processed', 0)}")
        print(f"Successful extractions: {summary.get('successful_extractions', 0)}")
        print(f"Failed extractions: {summary.get('failed_extractions', 0)}")
        print(f"Success rate: {summary.get('success_rate', 0):.1%}")
        print(f"Total valid tables extracted: {summary.get('total_valid_tables_extracted', 0)}")
        print(f"Average tables per PDF: {summary.get('average_tables_per_pdf', 0):.1f}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Check individual PDF folders for:")
        print(f"  - JSON results (camelot_enhanced_extraction_*.json)")
        print(f"  - Markdown reports (camelot_enhanced_extraction_*.md)")
        print(f"  - CSV tables (tables_csv/ folder)")
        print(f"  - Table plots (table_plots/ folder)")
        print(f"Also check batch summary files (JSON + Markdown) in main output directory.")


def main():
    # Configuration - Updated paths
    INPUT_DIR = "/Users/anushaprakash/Desktop/Bigdata/src/data/raw"
    OUTPUT_DIR = "/Users/anushaprakash/Desktop/Bigdata/src/data/parsed/camelot_output"
    
    try:
        # Create batch processor
        processor = BatchCamelotProcessor(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            log_level=logging.INFO
        )
        
        # Process all PDFs
        batch_results = processor.process_all_pdfs()
        
        # Print summary
        processor.print_batch_summary(batch_results)
        
        return batch_results
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Batch processing failed: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()