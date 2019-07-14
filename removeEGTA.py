import pandas as pd

file = '/media/daniel/Seagate Backup Plus Drive2/JData/A/A12_19F01-Gal4_CaMPARI/A12 Data/scoreROIlist.xlsx'

frame = pd.read_excel(file)

frame = frame[frame['Imaging Protocol'].str.contains('ramp')]
frame.to_excel(file)
