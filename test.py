import numpy as np
from openpyxl import Workbook
category=['epidural','healthy','intraparenchymal','intraventricular','subarachnoid','subdural']


f=open('result.txt','r')
lines=[]
k=f.readlines()
k.sort()
y_pred=np.zeros(600)
name=[]
i=0
for line in k:
    s=line.split(' ')
    y_pred[i]=int(s[1].split('\n')[0])
    name.append(s[0].split('.')[0])
    i+=1 



wb = Workbook()
sheet=wb.active
i=0
for row in sheet.iter_rows(min_row=1, min_col=1, max_row=600, max_col=2):
    j=0
    for cell in row:
        if j==0:
            cell.value=name[i]
        else:
            cell.value=category[int(y_pred[i])]
        j=j+1
    i=i+1
    print(i)
wb.save("testing_submission_trail3_CT_G11.xlsx")

