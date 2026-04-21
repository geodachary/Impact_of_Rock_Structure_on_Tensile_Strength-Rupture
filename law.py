# Create RIS file content
ris_content = """TY  - STAT
AU  - Nepal Rastra Bank
PY  - 2016
TI  - Merger and Acquisition Bylaw for Banks and Financial Institutions, 2073
PB  - Nepal Rastra Bank
SP  - 1
EP  - 21
ER  -
"""

file_path = "nepal_rastra_bank_merger_acquisition_2073.ris"
with open(file_path, "w") as f:
    f.write(ris_content)

file_path