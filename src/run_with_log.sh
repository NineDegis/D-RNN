#!/bin/bash

DATE=`python -c "from datetime import datetime; dt = datetime.now(); print('{}{}{}{}{}{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))"`
FOLDER_NAME=evallog
FILE_NAME=log_$DATE.txt

echo The result would be saved in ./$FOLDER_NAME/$FILE_NAME
python -u $1 > $FOLDER_NAME/$FILE_NAME &
disown
echo Finished
