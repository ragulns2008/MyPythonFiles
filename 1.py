
import pandas as pd

# -------- Importing Data --------

# Import data from CSV file
csv_data = pd.read_csv(r"D:\FML python files\FMLsample.csv")

print("CSV Data:")
print(csv_data.head())

# Import data from Excel file
excel_data = pd.read_excel(
    r"D:\FML python files\ook.xlsx.xlsx",
    sheet_name="sample"
)

print("\nExcel Data:")
print(excel_data.head())

print("\nData imported successfully!")

# -------- Exporting Data --------

# Export data to CSV file
csv_data.to_csv(
    r"D:\FML python files\exported_data.csv",
    index=False
)

# Export data to Excel file
excel_data.to_excel(
    r"D:\FML python files\exported_data.xlsx",
    index=False
)

print("\nData exported successfully!")
