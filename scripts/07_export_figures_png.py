"""
07_export_figures_png.py
将 output/figures/ 下的 PDF 图表导出为 PNG，供 README 与 GitHub 页面展示。

说明:
  LaTeX 论文使用 PDF 矢量图，而 GitHub Markdown 无法内联渲染 PDF，
  因此这里用 PyMuPDF 把已有 PDF 栅格化为 200 DPI 的 PNG 副本。
  本脚本不重新计算任何结果，只做格式转换。

输出: output/figures/png/*.png
"""

import os
import fitz  # PyMuPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(PROJECT_ROOT, "output", "figures")
PNG_DIR = os.path.join(PDF_DIR, "png")
os.makedirs(PNG_DIR, exist_ok=True)

DPI = 200
ZOOM = DPI / 72.0  # PDF 用户单位为 1/72 inch

print("=" * 60)
print("07 导出 PNG 图表 (供 README 展示)")
print("=" * 60)

pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
if not pdf_files:
    raise SystemExit(f"未在 {PDF_DIR} 找到 PDF 图表，请先运行 02/05/06 脚本。")

for pdf_name in pdf_files:
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    png_name = os.path.splitext(pdf_name)[0] + ".png"
    png_path = os.path.join(PNG_DIR, png_name)

    doc = fitz.open(pdf_path)
    page = doc[0]  # 图表均为单页
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
    pix.save(png_path)
    doc.close()

    size_kb = os.path.getsize(png_path) / 1024
    print(f"  {pdf_name:38s} -> png/{png_name:38s} "
          f"{pix.width}x{pix.height} ({size_kb:.0f} KB)")

print(f"\n共导出 {len(pdf_files)} 张 PNG 至: {PNG_DIR}")