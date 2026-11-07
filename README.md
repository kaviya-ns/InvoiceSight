# InvoiceSight

**Invoice data extraction and verification from scanned PDFs.**

InvoiceSight reads image-based (non-searchable) invoice PDFs, extracts structured fields and line items with open-source OCR, detects the seal/signature, and produces a verifiability report with confidence scores so you know what to trust.

![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![PaddleOCR](https://img.shields.io/badge/PaddleOCR-0062B0) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

> 🎥 Demo: [`Demo.mp4`](Demo.mp4)

---

## Features

- **Batch upload** — drop several scanned PDFs into the Streamlit sidebar
- **OCR** — PaddleOCR on each page, with pdfplumber for any embedded text
- **Field extraction** — robust regex patterns for invoice number, invoice date, supplier GSTIN, buyer GSTIN, PO number, shipping address and total amount
- **Line items** — description, HSN/SAC, quantity, unit price, line total
- **Seal & signature detection** — connected-component analysis with scikit-image/OpenCV, cropped and saved as an image
- **Verifiability report** — per-field and per-line-item confidence, line-item totals check and grand-total check
- **Exports** — JSON, Excel and the verifiability report for every invoice

## Sample output

From `invoice-2-1.pdf`:

```json
"field_verification": {
  "seal_and_sign_present": { "confidence": 0.8, "present": true },
  "invoice_number":        { "confidence": 0.9, "present": true },
  "supplier_gst_number":   { "confidence": 0.9, "present": true },
  ...
}
```

See [`invoice-2-1_extracted_data.json`](invoice-2-1_extracted_data.json), [`invoice-2-1_extracted_data.xlsx`](invoice-2-1_extracted_data.xlsx) and [`invoice-2-1_verifiability_report.json`](invoice-2-1_verifiability_report.json).

## Run it

```bash
git clone https://github.com/kaviya-ns/InvoiceSight.git
cd InvoiceSight
pip install -r requirements.txt
streamlit run main.py
```

Upload one or more PDFs from the sidebar. Results are written to `output/`.

---

*Built for the May 2025 hackathon — see `Problem_Statement_1_May_2025_Hackathon.pdf`.*
