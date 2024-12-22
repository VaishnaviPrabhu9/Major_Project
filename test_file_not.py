import os
#os.path.getsize(fullpathhere) > 0

if os.stat("testnow.csv").st_size == 0:
    print('empty file')
else:
    print('notempty')



