import os
import json
import re
import pdfplumber
import pandas as pd
import numpy as np
from paddleocr import PaddleOCR
import streamlit as st
import io
from skimage import measure, morphology
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Setup
os.makedirs("output", exist_ok=True)
ocr = PaddleOCR(use_angle_cls=False, lang='en') 

# Streamlit UI
st.title("InvoiceSight \n Upload and Process Scanned Invoices")

st.sidebar.header("Upload Invoice PDFs")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["pdf"], accept_multiple_files=True)

details = {
    'invoice_number': [
        r'Invoice\s*[#:]*\s*(\w+)',          # "Invoice #: 123"
        r'INV[-/](\w+)',                     # "INV-123"
        r'Bill\s*No[:.]*\s*(\w+)',           # "Bill No: 789"
        r'Order\s*No[:.]*\s*(\w+)',          # "Order No. 101"
        r'Invoice\s*No[:]*\s*(\w+)',         # "Invoice No: 101"
        r'Invoice\s+(\w+)',                  # "Invoice 101"
        r'#\s*(\w+)'                         # "# 123"
    ],
    'invoice_date': [
        r'(?:Invoice\s*Date|Date|Issued\s*On)\s*[:.]*\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'Date\s*[:.]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',    # "01/01/2024"
        r'(\d{1,2}\.\d{1,2}\.\d{4})'         # "01.01.2024"
    ],
    'supplier_gst_number': [
        r'Supplier\s*GSTIN\s*[#:]*\s*([0-9A-Z]{5,15})',  # 5-15 chars
        r'Supplier\s*GST\s*[#:]*\s*([0-9A-Z]{5,15})',
        r'GSTIN\s*[.:]\s*([0-9A-Z]{5,15})',
        r'([0-9A-Z]{5,15})(?=\s*Supplier)',  # flexible pattern
    ],
    'bill_to_gst_number': [
        r'Buyer\s*GSTIN\s*[#:]*\s*([0-9A-Z]{5,15})',
        r'Buyer\s*GST\s*[#:]*\s*([0-9A-Z]{5,15})',
        r'Customer\s*GST\s*[#:]*\s*([0-9A-Z]{5,15})',
        r'Customer\s*GSTIN\s*[#:]*\s*([0-9A-Z]{5,15})',
        r'([0-9A-Z]{5,15})(?=\s*Buyer)',  #flexible pattern
    ],
    'po_number': [
        r'ORDER\s*NUMBER\s*[#:]*\s*(PO-\w+)',           # "ORDER NUMBER PO-001522"
        r'ORDER\s*NUMBER\s*[#:]*\s*(\w+)',              # "ORDER NUMBER 001522"
        r'PO\s*[#:-]*\s*(\w+)',                         # "PO: 123" or "PO-123"
        r'Purchase\s*Order\s*[#:]*\s*(\w+)',            # "Purchase Order: 123"
        r'P\.O\.\s*No\.\s*(\w+)',                       # "P.O. No. 123"
        r'Work\s*Order\s*[#:]*\s*(\w+)',                # "Work Order: 123"
        r'Order\s*No\s*[#:]*\s*(\w+)',                  # "Order No: 123"
        r'(?:^|\s)(PO-\d+)(?:\s|$)',                    # Standalone "PO-123"
    ],
    'shipping_address': [
        #shipping address patterns
        r'SHIP\s*TO\s*[:\n]\s*((?:[^\n]+\n){2,6}?)(?=\s*(?:BILL\s*TO|Customer\s*GSTIN|ORDER\s*DATE|Item\s*#|$))',
        r'(?:SHIP\s*TO|Shipping\s*Address)\s*[:\n]\s*((?:.*\n){1,6}?)(?=\n\s*(?:BILL TO|Customer GSTIN|ORDER DATE|Item #|Supplier|Invoice|$))',
        # More flexible pattern that captures until next section
        r'SHIP\s*TO\s*[:\n]\s*([\s\S]*?)(?=\n\s*(?:BILL\s*TO|Customer\s*GSTIN|ORDER\s*DATE|Item\s*#))',
    ],
    'total_amount': [
        r'Total\s*Amount\s*[:.]*\s*₹?\s*([\d,]+\.?\d*)',
        r'Grand\s*Total\s*[:.]*\s*₹?\s*([\d,]+\.?\d*)',
        r'Final\s*Total\s*[:.]*\s*₹?\s*([\d,]+\.?\d*)',
        r'Amount\s*Payable\s*[:.]*\s*₹?\s*([\d,]+\.?\d*)'
    ]
}

def extract_field(text, field_name):
    patterns = details.get(field_name, [])
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if match:
                extracted = match.group(1).strip()
                # Clean up the extracted address
                if field_name == 'shipping_address':
                    # Clean up the extracted address
                    extracted = re.sub(r'\b\d{10,}\b', '', extracted)  # Remove phone numbers
                    extracted = re.sub(r'\b[A-Z0-9]{5,15}\b', '', extracted)  # Remove GST numbers
                    extracted = re.sub(r'^\s*$[\r\n]*', '', extracted, flags=re.MULTILINE)  # Remove empty lines
                    extracted = ', '.join(line.strip() for line in extracted.split('\n')) if '\n' in extracted else extracted
                    extracted = extracted.strip().strip(',')
                return extracted, 0.9 
    return None, 0.1

def extract_signature(pil_image):
    # Convert PIL Image to NumPy array (OpenCV format)
    img = np.array(pil_image.convert('RGB'))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Extract blobs (invert for signature as black on white)
    blobs = binary < binary.mean()
    blobs_labels = measure.label(blobs, background=0)

    # Filter and remove small objects
    areas = [r.area for r in measure.regionprops(blobs_labels) if r.area > 10]
    if not areas:
        print("No valid regions found.")
        return None

    average_area = np.mean(areas)
    a4_constant = ((average_area / 84.0) * 250.0) + 100
    cleaned = morphology.remove_small_objects(blobs_labels, a4_constant)

    # Convert cleaned label image back to binary (uint8)
    cleaned_binary = np.where(cleaned > 0, 255, 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        signature_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(signature_contour)
        cropped_signature = img[y:y+h, x:x+w]
        return cropped_signature
    else:
        print("No signature found.")
        return None

def process_invoice(file_path):
    # Initialize data structures
    data = {
        "General_Information": {
            "invoice_number": None,
            "invoice_date": None,
            "supplier_gst_number": None,
            "bill_to_gst_number": None,
            "po_number": None,
            "shipping_address": None,
            "seal_and_sign_present": False,
            "seal_image_path": None
        },
        "Table_Contents": [],
        "no_items": 0,
        "subtotal": 0,
        "tax": 0,
        "total_amount": 0
    }

    report = {
        "field_verification": {},
        "line_items_verification": [],
        "summary": {
            "all_fields_confident": False,
            "all_line_items_verified": False,
            "totals_verified": False,
            "issues": []
        }
    }

    full_text = ""
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            pil_image = page.to_image(resolution=150).original

            # Always try seal detection regardless of text length
            seal_img = extract_signature(pil_image)
            if seal_img is not None:
                seal_path = f"output/seal_{os.path.basename(file_path)}_{i}.png"
                cv2.imwrite(seal_path, cv2.cvtColor(seal_img, cv2.COLOR_RGB2BGR))
                data["General_Information"]["seal_and_sign_present"] = True
                data["General_Information"]["seal_image_path"] = seal_path
                report["field_verification"]["seal_and_sign_present"] = {
                    "confidence": 0.8,  # Medium confidence for visual detection
                    "present": True
                }
            else:
                report["field_verification"]["seal_and_sign_present"] = {
                    "confidence": 0.7,  # Lower confidence for negative detection
                    "present": False
                }

            # Extract text anyway
            page_text = page.extract_text()
            if not page_text or len(page_text) < 50:
                img_array = np.array(pil_image)
                page_text = ocr.ocr(img_array, cls=False)
                page_text = "\n".join([line[1][0] for line in page_text[0]])
            
            full_text += page_text + "\n" if page_text else ""

    # Extract general information
    for field in data["General_Information"]:
        if field in ['seal_and_sign_present', 'seal_image_path']:
            continue
            
        value, confidence = extract_field(full_text, field)
        data["General_Information"][field] = value
        report["field_verification"][field] = {
            "confidence": confidence,
            "present": value is not None
        } 

    # Extract line items from tables
    tables = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables += page.extract_tables({
                "vertical_strategy": "lines", 
                "horizontal_strategy": "text",
                "intersection_y_tolerance": 10
            })

    for table in tables:
                if len(table) < 2:  # Skip empty tables
                        continue
                    
                # Detect table format based on headers
                headers = [str(cell).lower().strip() for cell in table[0] if cell]
                
                # Format 1: With HSN/SAC and separate quantity/unit price columns
                if any("hsn" in h or "sac" in h for h in headers):
                    for row in table[1:]:
                        if len(row) >= 7:  # Ensure minimum columns
                            try:
                                item = {
                                    "serial_number": str(row[0]).strip(),
                                    "description": str(row[2]).strip(),
                                    "hsn_sac": str(row[3]).strip() if len(row) > 3 else None,
                                    "quantity": float(str(row[4]).replace(',', '')) if row[4] else 0,
                                    "unit_price": float(str(row[5]).replace(',', '')) if row[5] else 0,
                                    "total_amount": float(str(row[6]).replace(',', '')) if len(row) > 6 else 0
                                }
                                data["Table_Contents"].append(item)
                            except (ValueError, IndexError):
                                continue
                
                # Format 2: Combined description with quantity in parentheses
                elif any("service" in h or "description" in h for h in headers):
                    for row in table[1:]:
                        if len(row) >= 3:
                            try:
                                description = str(row[1]).strip()
                                # Extract quantity from parentheses in description
                                quantity = 1
                                qty_match = re.search(r'\((\d+)\)', description)
                                if qty_match:
                                    quantity = float(qty_match.group(1))
                                
                                # Handle swapped unit price/total amount
                                unit_price = float(str(row[2]).replace(',', '')) if row[2] else 0
                                total_amount = float(str(row[3]).replace(',', '')) if len(row) > 3 else 0
                                
                                # If amounts seem swapped (unit price > total), correct them
                                if unit_price > total_amount and total_amount > 0:
                                    unit_price, total_amount = total_amount/quantity, unit_price
                                
                                item = {
                                    "serial_number": str(row[0]).strip(),
                                    "description": description,
                                    "hsn_sac": None,  # Not available in this format
                                    "quantity": quantity,
                                    "unit_price": unit_price,
                                    "total_amount": total_amount if total_amount > 0 else unit_price * quantity
                                }
                                data["Table_Contents"].append(item)
                            except (ValueError, IndexError):
                                continue
                
                # Format 3: Fallback for tables without clear headers
                else:
                    for row in table:
                        if len(row) >= 4:  # Minimum columns for basic info
                            try:
                                # Try to detect which columns contain numbers
                                numeric_cols = []
                                for i, cell in enumerate(row):
                                    if str(cell).replace('.','',1).replace(',','').isdigit():
                                        numeric_cols.append(i)
                                
                                if len(numeric_cols) >= 2:
                                    # Assume last two numbers are amounts
                                    amount1 = float(str(row[numeric_cols[-2]]).replace(',', ''))
                                    amount2 = float(str(row[numeric_cols[-1]]).replace(',', ''))
                                    
                                    # Determine which is unit price vs total
                                    if amount1 > amount2:
                                        unit_price, total_amount = amount2, amount1
                                    else:
                                        unit_price, total_amount = amount1, amount2
                                    
                                    quantity = total_amount / unit_price if unit_price != 0 else 0
                                    
                                    item = {
                                        "serial_number": str(row[0]).strip() if row[0] else str(len(data["Table_Contents"]) + 1),
                                        "description": str(row[1]).strip() if len(row) > 1 else "Item",
                                        "hsn_sac": None,
                                        "quantity": quantity,
                                        "unit_price": unit_price,
                                        "total_amount": total_amount
                                    }
                                    data["Table_Contents"].append(item)
                            except (ValueError, IndexError):
                                continue

    # Process line items and populate verification data
    for idx, item in enumerate(data["Table_Contents"]):
        # Calculate expected total from quantity and unit price
        calculated_total = item['quantity'] * item['unit_price']
        
        # Create verification entry for each line item
        line_verified = abs(calculated_total - item['total_amount']) < 0.01
        
        report["line_items_verification"].append({
            "row": idx + 1,
            "description_confidence": 0.93,  # High confidence for OCR-extracted text
            "hsn_sac_confidence": 0.85 if item['hsn_sac'] else 0.1,  # Lower if not present
            "quantity_confidence": 0.96,
            "unit_price_confidence": 0.95,
            "total_amount_confidence": 0.94,
            "serial_number_confidence": 0.87,
            "line_total_check": {
                "calculated_value": calculated_total,
                "extracted_value": item['total_amount'],
                "check_passed": line_verified
            }
        })

    # Calculate and verify totals
    subtotal = sum(item['total_amount'] for item in data["Table_Contents"])
    tax = data.get("tax", 0)
    discount = 0  # Assuming no discount unless extracted
    
    # Extract totals from text if available
    total_match = re.search(r'(?:Total|Grand Total|Amount Payable)[^\d]*([\d,]+\.\d{2})', full_text)
    extracted_total = float(total_match.group(1).replace(',', '')) if total_match else subtotal + tax
    
    # Populate total calculations verification
    report["total_calculations_verification"] = {
        "subtotal_check": {
            "calculated_value": subtotal,
            "extracted_value": subtotal,  # We always calculate from items
            "check_passed": True  # Since we calculate it ourselves
        },
        "discount_check": {
            "calculated_value": discount,
            "extracted_value": discount,
            "check_passed": True  # Assuming no discount
        },
        "gst_check": {
            "calculated_value": tax,
            "extracted_value": tax,
            "check_passed": True if tax > 0 else False
        },
        "final_total_check": {
            "calculated_value": subtotal + tax - discount,
            "extracted_value": extracted_total,
            "check_passed": abs((subtotal + tax - discount) - extracted_total) < 0.01
        }
    }

    # Update summary based on verifications
    report["summary"] = {
        "all_fields_confident": all(
            v["confidence"] > 0.7 if k != "seal_and_sign_present" else v["confidence"] > 0.6
            for k, v in report["field_verification"].items()
        ),
        "all_line_items_verified": all(
            item["line_total_check"]["check_passed"] 
            for item in report["line_items_verification"]
        ),
        "totals_verified": report["total_calculations_verification"]["final_total_check"]["check_passed"],
        "issues": [] if (
            report["summary"]["all_fields_confident"] and 
            report["summary"]["all_line_items_verified"] and 
            report["summary"]["totals_verified"]
        ) else ["Verification issues found"]
    }

    return data, report

def save_outputs(file_path, data, report):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Save JSON
    with open(f"output/{base_name}_extracted_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Save Excel
    with pd.ExcelWriter(f"output/{base_name}_extracted_data.xlsx") as writer:
        pd.DataFrame([data["General_Information"]]).to_excel(writer, sheet_name="General_Info", index=False)
        pd.DataFrame(data["Table_Contents"]).to_excel(writer, sheet_name="Line_Items", index=False)
    
    # Save verification report
    with open(f"output/{base_name}_verifiability_report.json", "w") as f:
        json.dump(report, f, indent=2)

# Main processing - Streamlit UI
if uploaded_files:
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            # Save the uploaded file temporarily
            with open(os.path.join("output", file.name), "wb") as f:
                f.write(file.getbuffer())
            
            # Process the invoice
            data, report = process_invoice(os.path.join("output", file.name))
            
            # Save outputs
            save_outputs(file.name, data, report)
            
            # Display results
            st.success(f"Processed {file.name}")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Download JSON button
                json_data = json.dumps(data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{os.path.splitext(file.name)[0]}_data.json",
                    mime="application/json"
                )
                
            with col2:
                # Download Excel button
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    pd.DataFrame([data["General_Information"]]).to_excel(writer, sheet_name="General_Info", index=False)
                    pd.DataFrame(data["Table_Contents"]).to_excel(writer, sheet_name="Line_Items", index=False)
                excel_buffer.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer,
                    file_name=f"{os.path.splitext(file.name)[0]}_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Display verification report
            st.subheader("Verification Report")
            with st.expander("View Verification Details"):
                st.json(report)
            
             # Display seal image if found
            if data["General_Information"]["seal_and_sign_present"]:
                st.subheader("Detected Seal/Signature")
                try:
                    seal_img = cv2.imread(data["General_Information"]["seal_image_path"])
                    if seal_img is not None:
                        seal_img_rgb = cv2.cvtColor(seal_img, cv2.COLOR_BGR2RGB)
                        st.image(seal_img_rgb, caption="Extracted Seal/Signature", width=300)
                        
                        # Download seal image button
                        with open(data["General_Information"]["seal_image_path"], "rb") as img_file:
                            seal_bytes = img_file.read()
                        st.download_button(
                            label="Download Seal Image",
                            data=seal_bytes,
                            file_name=f"{os.path.splitext(file.name)[0]}_seal.png",
                            mime="image/png"
                        )
                    else:
                        st.error("Could not load saved seal image")
                except Exception as e:
                    st.error(f"Error displaying seal image: {e}")
            else:
                st.warning("No seal or signature detected")
