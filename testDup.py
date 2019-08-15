from preprocessing.touchCount import touchCount
from preprocessing.data_utils import get_ETL_data
from getPixel import printPixel
import time
max_records = 320
writers_per_char=160
chars, labs, spts=get_ETL_data('1' ,categories=range(0, max_records), writers_per_char=writers_per_char,database='ETL8B2',get_scripts=True)
item=chars[6213]
print('Function start!')
start=time.time()
print(touchCount(item))
end=time.time()
printPixel(item)
print('Time for touchCount: '+str(end-start))