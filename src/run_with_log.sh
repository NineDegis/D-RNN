#!/bin/bash

DATE=`python -c "from datetime import datetime; dt = datetime.now(); print('{}{}{}{}{}{}{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond))"`
FILE_NAME=log_$DATE.txt

echo The result would be saved in src/logs/$FILE_NAME
python -u $1 > ../logs/$FILE_NAME
echo Finished
