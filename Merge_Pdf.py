from PyPDF2 import PdfMerger
import os

#Create an instance of PdfFileMerger() class
merger = PdfMerger()

#Define the path to the folder with the PDF files
path_to_files = r'C:/Users/ChimieENS/Downloads/Documents_RP/'

#Get the file names in the directory
for root, dirs, file_names in os.walk(path_to_files):
    #Iterate over the list of the file names
    for file_name in file_names:
        #Append PDF files
        merger.append(path_to_files + file_name)

#Write out the merged PDF file
merger.write("C:/Users/ChimieENS/Downloads/Documents_RP/merged_all_pages.pdf")
merger.close()