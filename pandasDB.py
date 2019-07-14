import pandas as pd


def getMultipleExceLSheets(xlsxfile):
    #xlsxfile = full path to excelfile  
    xl = pd.ExcelFile(xlsxfile)
    #print(xl.sheet_names)
    exceldata=xl.parse(xl.sheet_names[0])
    for sheet in xl.sheet_names[1:]:
        exceldata = exceldata.append(xl.parse(sheet, encoding='utf-8'))
    exceldata.reset_index(inplace = True)
    return exceldata
