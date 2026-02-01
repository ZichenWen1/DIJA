#!/usr/bin/env python3
"""Convert PDF figures to PNG for web display"""
import fitz  # PyMuPDF
from pathlib import Path

def pdf_to_png(pdf_path, output_path, dpi=150):
    """Convert PDF to PNG"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    # Calculate matrix for desired DPI
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    
    pix.save(output_path)
    doc.close()
    print(f"Converted: {pdf_path} -> {output_path}")

# List of figures to convert
figures = [
    ("fig_attack_with_defense.pdf", "teaser.png"),
    ("fig_dllm_vs_arm.pdf", "fig1_comparison.png"),
    ("fig_ours_vs_PAIR.pdf", "fig2_vs_pair.png"),
    ("fig_dllm_vs_arm_asr_e.pdf", "fig3_asr.png"),
    ("fig_mask_token_num_ours.pdf", "fig4_mask_tokens.png"),
]

print("Converting PDF figures to PNG...")
for pdf_name, png_name in figures:
    pdf_path = Path(pdf_name)
    if pdf_path.exists():
        try:
            pdf_to_png(str(pdf_path), png_name, dpi=200)
        except Exception as e:
            print(f"Error converting {pdf_name}: {e}")
    else:
        print(f"Not found: {pdf_name}")

print("\nDone! PNG files created in homepage/")
