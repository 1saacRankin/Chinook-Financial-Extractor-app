import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import re
import io
import base64
from datetime import datetime
from pdf2image import convert_from_bytes
import pytesseract
from collections import defaultdict


# Fill missing values in table -- silence warning
pd.set_option('future.no_silent_downcasting', True)

# ====================================
# Streamlit Configuration
# ====================================
st.set_page_config(page_title="Chinook PDF Financial Data Extractor", layout="wide")
st.title("Chinook PDF Financial Data Extractor")
st.text("Upload PDFs for monthly or yearly financial documents." + 
        "\nIf uploading scanned documents, accuracy is not guaranteed.")

# ====================================
# Shared Helper Functions
# ====================================
amount_re = re.compile(r'\(?-?\$?[\d,]+(?:\.\d+)?\)?|-')

def clean_label(label: str) -> str:
    label = re.sub(r"\$", "", label)
    label = re.sub(r"-?[0-9,]*\.?[0-9]+|\([0-9,]*\.?[0-9]+\)", "", label)
    label = label.replace("/", " ")
    label = " ".join(label.split())
    return label.capitalize()

def clean_label_for_match(line: str) -> str:
    line = re.sub(r"\$", "", line)
    line = re.sub(r"-?[0-9,]*\.?[0-9]+|\([0-9,]*\.?[0-9]+\)", "", line)
    line = line.replace("/", " ")
    line = " ".join(line.split())
    return line.strip().lower()

def parse_amount(token: str):
    if token is None:
        return np.nan
    t = token.strip()
    if t == "-" or t == "":
        return np.nan
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    t = t.replace("$", "").replace(",", "").replace(" ", "")
    if t == "" or t == "-":
        return np.nan
    try:
        val = float(t)
    except Exception:
        return np.nan
    return -val if neg else val

# Find the month and year from file names
def extract_month_year(filename: str):
    patterns = [
        r"(\b\w{3,9}\b)[\s_-]?(\d{2,4})",  # Apr 22 or April 2022
        r"(\d{4})[\s_-]?(\d{2})",          # 2022-04 or 202204
    ]
    for pat in patterns:
        match = re.search(pat, filename, re.IGNORECASE)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 2:
                    if groups[0].isalpha():
                        year = groups[1] if len(groups[1]) == 4 else "20" + groups[1]
                        return datetime.strptime(f"{groups[0][:3]} {year}", "%b %Y")
                    else:
                        return datetime.strptime(f"{groups[0]}-{groups[1]}", "%Y-%m")
            except:
                continue
    return None

# Find year from file name
def extract_year(filename: str):
    fn_year_match = re.search(r"(20\d{2})", filename)
    return int(fn_year_match.group(1)) if fn_year_match else None

# Display documents in viewer
def display_pdf(pdf_bytes_io):
    base64_pdf = base64.b64encode(pdf_bytes_io.read()).decode('utf-8')
    pdf_display = f'''
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="600"
            type="application/pdf"
        ></iframe>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

# If large document, display page by page
def display_pdf_preview_all_pages(pdf_bytes_io, dpi=50, max_pages=None):
    try:
        pages = convert_from_bytes(pdf_bytes_io.read(), dpi=dpi)
        if max_pages:
            pages = pages[:max_pages]

        with st.expander("Show document preview", expanded=True):
            for i, page in enumerate(pages, start=1):
                st.image(page, caption=f"Page {i}", use_container_width=True)

    except Exception as e:
        st.error(f"Could not render PDF preview: {e}")

# Extract text if possible, otherwise use OCR
def extract_text_with_fallback(file_bytes, selected_pages=None):
    import warnings
    warnings.filterwarnings('ignore')
    
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_to_extract = selected_pages if selected_pages else range(len(pdf.pages))
            for page_num in pages_to_extract:
                if page_num < len(pdf.pages):
                    t = pdf.pages[page_num].extract_text(layout=False)
                    if t:
                        text += t + "\n"
    except Exception:
        pass
    
    if text.strip():
        return text
    
    try:
        images = convert_from_bytes(file_bytes)
        pages_to_extract = selected_pages if selected_pages else range(len(images))
        for page_num in pages_to_extract:
            if page_num < len(images):
                text += pytesseract.image_to_string(images[page_num]) + "\n"
    except Exception:
        pass
    
    return text

def extract_text_by_page(file_bytes):
    """Extract text from each page separately and return dict of page_num -> text"""
    import warnings
    warnings.filterwarnings('ignore')
    
    page_texts = {}
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                t = page.extract_text(layout=False)
                if t:
                    page_texts[page_num] = t
    except Exception:
        pass
    
    # If no text found, try OCR
    if not page_texts:
        try:
            images = convert_from_bytes(file_bytes)
            for page_num, img in enumerate(images):
                text = pytesseract.image_to_string(img)
                if text:
                    page_texts[page_num] = text
        except Exception:
            pass
    
    return page_texts

def find_duplicate_fields_in_document(file_bytes, file_name):
    """Find fields that appear on multiple pages within a document"""
    page_texts = extract_text_by_page(file_bytes)
    
    # Track which pages each field appears on
    field_pages = defaultdict(list)
    
    for page_num, text in page_texts.items():
        for line in text.splitlines():
            if re.search(r"[0-9]", line):
                field = clean_label(line)
                if field and len(field) >= 3:
                    field_pages[field].append(page_num + 1)  # Convert to 1-indexed
    
    # Find duplicates
    duplicates = {field: pages for field, pages in field_pages.items() if len(pages) > 1}
    
    return duplicates

def get_pdf_page_count(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return len(pdf.pages)

# Improved frequency detection
def detect_frequency(filenames):
    monthly_count = sum(1 for f in filenames if extract_month_year(f) is not None)
    yearly_count = sum(1 for f in filenames if extract_year(f) is not None and extract_month_year(f) is None)
    
    if monthly_count > yearly_count:
        return "monthly"
    elif yearly_count > 0:
        return "yearly"
    else:
        if filenames and extract_month_year(filenames[0]):
            return "monthly"
        return "yearly"

def get_field_order_from_latest_document(pdfs_with_info, freq):
    """
    Get fields ordered by appearance in the most recent document.
    For fields not in latest doc, append alphabetically at the end.
    ONLY returns fields from the selected pages of the latest document.
    """
    # Find the most recent document
    if freq == "monthly":
        dated_pdfs = [(extract_month_year(name), pdf_bytes, pages, name) 
                     for name, pdf_bytes, pages in pdfs_with_info if extract_month_year(name)]
        if dated_pdfs:
            dated_pdfs.sort(key=lambda x: x[0], reverse=True)
            latest_pdf = dated_pdfs[0]
            latest_bytes, latest_pages = latest_pdf[1], latest_pdf[2]
        else:
            latest_bytes, latest_pages = pdfs_with_info[0][1], pdfs_with_info[0][2]
    else:  # yearly
        dated_pdfs = [(extract_year(name), pdf_bytes, pages, name) 
                     for name, pdf_bytes, pages in pdfs_with_info if extract_year(name)]
        if dated_pdfs:
            dated_pdfs.sort(key=lambda x: x[0], reverse=True)
            latest_pdf = dated_pdfs[0]
            latest_bytes, latest_pages = latest_pdf[1], latest_pdf[2]
        else:
            latest_bytes, latest_pages = pdfs_with_info[0][1], pdfs_with_info[0][2]
    
    # Extract fields ONLY from latest document's selected pages
    text = extract_text_with_fallback(latest_bytes, latest_pages)
    latest_fields = []
    seen = set()
    
    for line in text.splitlines():
        if re.search(r"[0-9]", line):
            field = clean_label(line)
            if field and field not in seen and len(field) >= 3:
                latest_fields.append(field)
                seen.add(field)
    
    # Return ONLY fields from latest document (no remaining fields from other docs)
    return latest_fields


# ====================================
# Yearly Extraction
# ====================================
def extract_rows_from_text(text, file_year, file_prev_year, file_name):
    rows = []
    for line in text.splitlines():
        s = line.rstrip()
        if not s.strip():
            continue
        
        # Extract all amounts from the line
        matches = list(amount_re.finditer(s))
        if len(matches) < 2:
            continue
        
        # Get description (everything before first amount)
        desc = s[:matches[0].start()].strip()
        if not desc or len(desc) < 3:  # Filter out very short descriptions
            continue
        
        # Parse all amounts
        amounts = [parse_amount(m.group(0)) for m in matches]
        
        # Skip if all amounts are NaN
        if all(pd.isna(amt) for amt in amounts):
            continue
        
        # Build row with primary years and any extra columns
        row = {
            "Description": clean_label(desc),
            f"{file_year} ({file_name})": amounts[-2] if len(amounts) >= 2 else np.nan,
            f"{file_prev_year} ({file_name})": amounts[-1] if len(amounts) >= 1 else np.nan,
        }
        
        # Add extra columns if more than 2 amounts found
        for i in range(len(amounts) - 2):
            row[f"Column_{i+1} ({file_name})"] = amounts[i]
        
        rows.append(row)
    return rows

def extract_pdf_data_yearly(file_bytes, file_name, rows_of_interest=None, selected_pages=None):
    file_year = extract_year(file_name)
    if not file_year:
        return pd.DataFrame()
    file_prev_year = file_year - 1
    text = extract_text_with_fallback(file_bytes, selected_pages)
    rows = extract_rows_from_text(text, file_year, file_prev_year, file_name)
    df = pd.DataFrame(rows)
    
    if df.empty:
        return df
    
    # Remove duplicate descriptions (keep first occurrence)
    df = df.drop_duplicates(subset=['Description'], keep='first')
    
    if rows_of_interest:
        interest_clean = [clean_label_for_match(x) for x in rows_of_interest]
        df = df[df["Description"].apply(lambda d: any(term in clean_label_for_match(d) for term in interest_clean))]
    
    return df

def values_match(val1, val2, tolerance=0.01):
    """Check if two values match within a tolerance, handling NaN"""
    if pd.isna(val1) and pd.isna(val2):
        return True
    if pd.isna(val1) or pd.isna(val2):
        return False
    try:
        return abs(float(val1) - float(val2)) <= tolerance
    except:
        return val1 == val2

def consolidate_yearly_data(dfs_with_sources):
    """
    Consolidate yearly data, handling duplicate years intelligently.
    Returns dataframe and conflict information for highlighting.
    """
    if not dfs_with_sources:
        return pd.DataFrame(), {}
    
    # Merge all dataframes on Description
    result_df = dfs_with_sources[0].copy()
    for df in dfs_with_sources[1:]:
        result_df = pd.merge(result_df, df, on="Description", how="outer", suffixes=('', '_dup'))
    
    # Merge duplicate rows immediately after combining dataframes
    result_df = merge_duplicate_rows(result_df)
    
    # Group columns by year (including source info)
    year_pattern = re.compile(r'(\d{4})')
    year_columns = {}  # year -> [(col_name, values)]
    
    for col in result_df.columns:
        if col == 'Description':
            continue
        match = year_pattern.search(col)
        if match:
            year = match.group(1)
            if year not in year_columns:
                year_columns[year] = []
            year_columns[year].append(col)
    
    # Track conflicts: {(row_desc, year): [col1, col2, ...]}
    conflicts = {}
    
    # Process each year
    final_columns = ['Description']
    for year in sorted(year_columns.keys()):
        cols = year_columns[year]
        
        if len(cols) == 1:
            # Single column - keep as is with source info
            final_columns.append(cols[0])
        else:
            # Multiple columns for same year - check if identical
            all_identical = True
            
            for idx, row in result_df.iterrows():
                desc = row['Description']
                values = [row[col] for col in cols]
                
                first_val = values[0]
                for val in values[1:]:
                    if not values_match(first_val, val):
                        all_identical = False
                        # Track conflict for highlighting
                        conflicts[(desc, year)] = cols
                        break
            
            if all_identical:
                # Rename first column to just year, drop duplicates
                result_df[year] = result_df[cols[0]]
                result_df.drop(columns=cols, inplace=True)
                final_columns.append(year)
            else:
                # Keep all columns with their source names
                for col in cols:
                    final_columns.append(col)
    
    # Reorder columns
    final_columns = [c for c in final_columns if c in result_df.columns]
    result_df = result_df[final_columns]
    
    # Sort by year
    def extract_year_for_sort(col):
        if col == 'Description':
            return 0
        match = re.search(r'(\d{4})', col)
        return int(match.group(1)) if match else 9999
    
    cols_sorted = sorted(result_df.columns, key=extract_year_for_sort)
    result_df = result_df[cols_sorted]
    
    return result_df, conflicts

def merge_duplicate_rows(df):
    """Merge rows with the same Description, combining their data intelligently"""
    if df.empty or 'Description' not in df.columns:
        return df
    
    # For each unique description, combine all duplicate rows
    merged_rows = []
    
    for desc in df['Description'].unique():
        mask = df['Description'] == desc
        rows = df[mask]
        
        if len(rows) == 1:
            # No duplicates, keep as is
            merged_rows.append(rows.iloc[0])
        else:
            # Multiple rows - merge them
            merged_row = {'Description': desc}
            
            # For each column, combine non-NaN values
            for col in df.columns:
                if col != 'Description':
                    # Get all non-NaN values for this column across duplicate rows
                    values = rows[col].dropna()
                    
                    if len(values) == 0:
                        merged_row[col] = np.nan
                    elif len(values) == 1:
                        merged_row[col] = values.iloc[0]
                    else:
                        # Multiple non-NaN values - check if they're all the same
                        if values.nunique() == 1:
                            merged_row[col] = values.iloc[0]
                        else:
                            # Different values - take the first non-NaN
                            merged_row[col] = values.iloc[0]
            
            merged_rows.append(merged_row)
    
    result_df = pd.DataFrame(merged_rows)
    return result_df

def remove_sparse_columns(df, min_coverage=0.10):
    """Remove columns with less than min_coverage (default 10%) non-NaN values compared to the fullest column"""
    if df.empty or 'Description' not in df.columns:
        return df, []
    
    # Find the column with the most data (excluding Description)
    data_cols = [col for col in df.columns if col != 'Description']
    if not data_cols:
        return df, []
    
    max_count = max(df[col].notna().sum() for col in data_cols)
    threshold = max_count * min_coverage
    
    # Keep Description and columns that meet the threshold
    cols_to_keep = ['Description']
    removed_cols = []
    
    for col in data_cols:
        count = df[col].notna().sum()
        if count >= threshold:
            cols_to_keep.append(col)
        else:
            removed_cols.append(f"{col} ({count}/{max_count} entries)")
    
    result_df = df[cols_to_keep]
    
    return result_df, removed_cols


# ====================================
# Monthly Extraction
# ====================================
def extract_pdf_data_monthly(file_bytes, file_name, selected_fields, selected_pages=None):
    text = extract_text_with_fallback(file_bytes, selected_pages)
    date_obj = extract_month_year(file_name)
    col_name = date_obj.strftime("%B %Y") if date_obj else file_name
    data = {}
    
    for field in selected_fields:
        found = False
        for line in text.splitlines():
            line_clean = clean_label_for_match(line)
            field_clean = clean_label_for_match(field)
            
            if field_clean in line_clean or line_clean in field_clean:
                value_matches = re.findall(r"-?[0-9,]*\.?[0-9]+|\([0-9,]*\.?[0-9]+\)", line)
                if value_matches:
                    # Extract all values found in the line
                    values = []
                    for raw_val in value_matches:
                        raw_val = raw_val.replace(",", "")
                        num = -float(raw_val.strip("()")) if raw_val.startswith("(") and raw_val.endswith(")") else float(raw_val)
                        if line.strip().startswith("-") and num > 0:
                            num = -num
                        values.append(num)
                    
                    # Store all values if multiple, otherwise just the single value
                    if len(values) == 1:
                        data[field] = values[0]
                    else:
                        # Primary value is the last one
                        data[field] = values[-1]
                        # Store additional columns
                        for i, val in enumerate(values[:-1]):
                            data[f"{field} (Col {i+1})"] = val
                    
                    found = True
                    break
        
        if not found:
            data[field] = np.nan
    
    return col_name, data

# ====================================
# Session State
# ====================================
if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = []
if "extracted_df" not in st.session_state:
    st.session_state["extracted_df"] = None
if "conflicts" not in st.session_state:
    st.session_state["conflicts"] = {}
if "page_selections" not in st.session_state:
    st.session_state["page_selections"] = {}
if "all_fields" not in st.session_state:
    st.session_state["all_fields"] = []
if "selected_fields" not in st.session_state:
    st.session_state["selected_fields"] = []

# ====================================
# Layout
# ====================================
left_col, right_col = st.columns([2, 1])

with left_col:
    
    # Upload all PDFs first
    st.subheader("Step 1: Upload All PDFs")
    multiple_pdfs = st.file_uploader("Upload all PDF files", type=["pdf"], accept_multiple_files=True, key="batch_upload")
    
    if multiple_pdfs:
        # Store PDFs efficiently - only read once
        if "pdf_data" not in st.session_state or len(st.session_state.get("pdf_data", [])) != len(multiple_pdfs):
            st.session_state["pdf_data"] = [{"name": f.name, "bytes": f.read()} for f in multiple_pdfs]
        
        st.session_state["uploaded_pdfs"] = st.session_state["pdf_data"]
        
        # Improved frequency detection
        all_names = [p["name"] for p in st.session_state["uploaded_pdfs"]]
        detected_freq = detect_frequency(all_names)
        
        # Allow manual override
        freq = st.radio("Document frequency:", ["monthly", "yearly"], 
                       index=0 if detected_freq == "monthly" else 1,
                       horizontal=True)
        
        # Duplicate field detection
        st.subheader("Step 2: Duplicate Field Detection")
        with st.expander("View fields appearing on multiple pages", expanded=False):
            st.info("These fields appear on multiple pages within the same document. Consider selecting specific pages to avoid duplicates.")
            
            for pdf_file in st.session_state["uploaded_pdfs"]:
                duplicates = find_duplicate_fields_in_document(pdf_file["bytes"], pdf_file["name"])
                if duplicates:
                    st.write(f"**{pdf_file['name']}:** ({len(duplicates)} duplicates found)")
                    for field, pages in duplicates.items():
                        st.write(f"  • {field}: Pages {', '.join(map(str, pages))}")
                else:
                    st.write(f"**{pdf_file['name']}:** No duplicates detected")
        
        # Page selection for each document
        st.subheader("Step 3: Select Pages (Optional)")
        with st.expander("Configure page extraction per document", expanded=False):
            st.info("By default, all pages are extracted. Select specific pages to reduce false positives.")
            for pdf_file in st.session_state["uploaded_pdfs"]:
                page_count = get_pdf_page_count(pdf_file["bytes"])
                selected = st.multiselect(
                    f"{pdf_file['name']} (Total: {page_count} pages)",
                    options=list(range(1, page_count + 1)),
                    default=None,
                    key=f"pages_{pdf_file['name']}"
                )
                if selected:
                    st.session_state["page_selections"][pdf_file["name"]] = [p - 1 for p in selected]
                else:
                    st.session_state["page_selections"][pdf_file["name"]] = None
        
        # Extract preview data from all PDFs
        st.subheader("Step 4: Select Fields to Extract")
        
        # Cache field extraction - use selected pages for each document
        if "all_fields" not in st.session_state or not st.session_state["all_fields"]:
            with st.spinner("Loading fields from selected pages..."):
                if freq == "monthly":
                    all_fields = set()
                    for pdf_file in st.session_state["uploaded_pdfs"]:
                        # Get selected pages for this document, or None (all pages) if not specified
                        selected_pages = st.session_state["page_selections"].get(pdf_file["name"])
                        text = extract_text_with_fallback(pdf_file["bytes"], selected_pages)
                        candidate_lines = [clean_label(line) for line in text.splitlines() if re.search(r"[0-9]", line)]
                        all_fields.update(candidate_lines)
                    st.session_state["all_fields"] = sorted(all_fields)
                else:
                    all_descriptions = set()
                    for pdf_file in st.session_state["uploaded_pdfs"]:
                        # Get selected pages for this document, or None (all pages) if not specified
                        selected_pages = st.session_state["page_selections"].get(pdf_file["name"])
                        df_preview = extract_pdf_data_yearly(pdf_file["bytes"], pdf_file["name"], selected_pages=selected_pages)
                        if not df_preview.empty:
                            all_descriptions.update(df_preview["Description"].unique())
                    st.session_state["all_fields"] = sorted(all_descriptions)
        
        all_fields = st.session_state["all_fields"]
        

        # Field ordering option
        sort_order = st.radio(
            "Order fields by:", 
            ["Alphabetical", "Appearance in latest document"], 
            horizontal=True,
            help="'Latest document' orders by most recent file, with remaining fields alphabetically at the end"
        )
        
        if sort_order == "Appearance in latest document":
            pdfs_with_info = [(p["name"], p["bytes"], st.session_state["page_selections"].get(p["name"])) 
                              for p in st.session_state["uploaded_pdfs"]]
            all_fields = get_field_order_from_latest_document(pdfs_with_info, freq)
        
        # Select All button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✓ Select All", help="Select all fields"):
                st.session_state["selected_fields"] = all_fields.copy()
        with col2:
            if st.button("✗ Clear All", help="Clear all selections"):
                st.session_state["selected_fields"] = []
        
        # Multiselect with session state
        selected_fields = st.multiselect(
            f"Select fields to extract ({len(all_fields)} available):",
            options=all_fields,
            default=st.session_state["selected_fields"],
            key="field_selector"
        )
        
        # Update session state
        st.session_state["selected_fields"] = selected_fields


        # Extract data from all PDFs
        if st.button("Extract Data", type="primary"):
            with st.spinner("Extracting data..."):
                
                # When data is monthly
                if freq == "monthly":
                    months_found = []
                    data_dict = {}
                    for pdf_file in st.session_state["uploaded_pdfs"]:
                        selected_pages = st.session_state["page_selections"].get(pdf_file["name"])
                        col_name, data = extract_pdf_data_monthly(pdf_file["bytes"], pdf_file["name"], selected_fields, selected_pages)
                        data_dict[col_name] = data
                        date_obj = extract_month_year(pdf_file["name"])
                        if date_obj:
                            months_found.append(date_obj)

                    # Check duplicates
                    duplicates = sorted(set([m for m in months_found if months_found.count(m) > 1]))
                    if duplicates:
                        st.warning(f"⚠️ Duplicate months detected: {', '.join(d.strftime('%B %Y') for d in duplicates)}")

                    # Check missing
                    if months_found:
                        all_months = pd.date_range(start=min(months_found), end=max(months_found), freq="MS")
                        missing = [m for m in all_months if m not in months_found]
                        if missing:
                            st.error(f"❌ Missing months: {', '.join(m.strftime('%B %Y') for m in missing)}")
                        for miss in missing:
                            data_dict[miss.strftime("%B %Y")] = {field: np.nan for field in selected_fields}

                    df = pd.DataFrame(data_dict)
                    df = df[sorted(df.columns, key=lambda c: extract_month_year(c) or datetime.max)]
                    
                    # Merge duplicate rows
                    df = merge_duplicate_rows(df)
                    
                    # Remove sparse columns
                    df, removed = remove_sparse_columns(df)
                    
                    st.session_state["extracted_df"] = df
                    st.session_state["conflicts"] = {}

                # When data is yearly
                else:  
                    years_found = []
                    dfs_with_sources = []
                    
                    for pdf_file in st.session_state["uploaded_pdfs"]:
                        selected_pages = st.session_state["page_selections"].get(pdf_file["name"])
                        df = extract_pdf_data_yearly(pdf_file["bytes"], pdf_file["name"], selected_fields, selected_pages)
                        if not df.empty:
                            year_main = extract_year(pdf_file["name"])
                            if year_main is not None:
                                years_found.extend([year_main, year_main - 1])
                            dfs_with_sources.append(df)

                    # Check for missing years
                    missing = []
                    if years_found:
                        all_years_range = range(min(years_found), max(years_found) + 1)
                        missing = [y for y in all_years_range if y not in years_found]
                        if missing:
                            st.error(f"❌ Missing years: {', '.join(str(y) for y in missing)}")

                    # Use robust consolidation
                    final_df, conflicts = consolidate_yearly_data(dfs_with_sources)
                    
                    # Add missing years as blank columns
                    for y in missing:
                        col_name = str(y)
                        if col_name not in final_df.columns:
                            final_df[col_name] = np.nan

                    # Sort columns by year
                    def year_key(col):
                        if col == 'Description':
                            return 0
                        try:
                            return int(re.search(r"\d{4}", col).group())
                        except:
                            return 9999

                    cols_sorted = sorted(final_df.columns, key=year_key)
                    final_df = final_df[cols_sorted]
                    
                    # Merge duplicate rows
                    final_df = merge_duplicate_rows(final_df)
                    
                    # Remove sparse columns
                    final_df, removed = remove_sparse_columns(final_df)

                    st.session_state["extracted_df"] = final_df
                    st.session_state["conflicts"] = conflicts
                
                # Show extraction summary
                rows, cols = st.session_state["extracted_df"].shape
                st.success(f"✅ Data extraction complete! {rows} rows × {cols} columns")
                
                if rows * cols > 100000:
                    st.info(f"ℹ️ Large dataset detected ({rows * cols:,} cells). Cell highlighting disabled for performance.")

# ====================================
# Right Column - PDF Viewer
# ====================================

with right_col:
    st.subheader("Document Viewer")
    
    if st.session_state.get("uploaded_pdfs"):
        def sort_key(doc):
            date_obj = extract_month_year(doc.get("name", ""))
            return date_obj or datetime.max

        sorted_docs = sorted(st.session_state["uploaded_pdfs"], key=sort_key)
        doc_names = [p.get("name", "Unnamed Document") for p in sorted_docs]

        doc_to_view = st.selectbox(
            "Select a document to view",
            doc_names,
            key="viewer_select",
        )

        selected_doc = next(p for p in sorted_docs if p.get("name") == doc_to_view)
        pdf_bytes = selected_doc.get("bytes")

        if pdf_bytes:
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            if pdf_size_mb > 0.50:
                st.info(f"Scanned document detected: showing image preview.")
                display_pdf_preview_all_pages(io.BytesIO(pdf_bytes), dpi=75)
            else:
                display_pdf(io.BytesIO(pdf_bytes))
    else:
        st.info("Upload documents to view them here.")


# ====================================
# Full Width - Table editor
# ====================================
if st.session_state["extracted_df"] is not None:
    st.subheader("Step 5: Edit & Download Data")
    st.info("The table below is fully editable. You can modify the extracted values directly.")

    fill_value = st.number_input("Fill all missing values with:", value=0)
    if st.button("Fill Missing Values"):
        st.session_state["extracted_df"] = st.session_state["extracted_df"].fillna(fill_value)

    df = st.session_state["extracted_df"]
    
    # Simple data editor
    st.info(f"To delete rows, select rows in the left most column and then click the trash can in the top right corner.")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        #use_container_width=True,
        key="table_editor",
        hide_index=True
    )
    
    st.session_state["extracted_df"] = edited_df

    file_name = st.text_input("Output file name (press Enter to update):", value="financial_data.xlsx")

    output = io.BytesIO()
    edited_df.to_excel(output, index=False)
    output.seek(0)
    st.download_button(
        label="📥 Download Excel",
        data=output,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )


# Run: streamlit run Extractor.py
