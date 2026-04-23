from pathlib import Path
from pypdf import PdfWriter

folder = Path(r"C:/")   # change this
output_file = folder / "merged.pdf"

pdf_files = sorted(folder.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in the folder.")

writer = PdfWriter()

for pdf in pdf_files:
    if pdf.name != output_file.name:  # avoid including the output file if rerun
        writer.append(str(pdf))

with open(output_file, "wb") as f:
    writer.write(f)

writer.close()

print(f"Merged {len(pdf_files)} PDFs into: {output_file}")

from PyPDF2 import PdfMerger
import os


#Create an instance of PdfFileMerger() class
merger = PdfMerger()

#Define the path to the folder with the PDF files
path_to_files = r'C:'

folder_name = os.path.split(path_to_files.rstrip('/'))[-1]

#Get the file names in the directory
for file_name in os.listdir(path_to_files):

    if file_name.endswith(".pdf"):
        merger.append(path_to_files + file_name)
        
#Write out the merged PDF file
merger.write(path_to_files + "/" + folder_name + ".pdf")
merger.close()

# from PyPDF2 import PdfMerger
# import os

# #Create an instance of PdfFileMerger() class
# merger = PdfMerger()

# #Define the path to the folder with the PDF files
# path_to_files = r'C/'

# #Get the file names in the directory
# for root, dirs, file_names in os.walk(path_to_files):
#     #Iterate over the list of the file names
#     for file_name in file_names:
#         #Append PDF files
#         merger.append(path_to_files + file_name)

# #Write out the merged PDF file
# merger.write("/merged_all_pages.pdf")
# merger.close()
