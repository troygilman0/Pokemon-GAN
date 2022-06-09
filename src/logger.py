from datetime import datetime


def start_log(text='', log=True):
    start_time = datetime.now()
    if log:
        print(f'{start_time} [INFO]\
            {text}')
    return start_time


def end_log(start_time=None, text='', log=True):
    if log:
        end_time = datetime.now()
        time_elapsed = round(datetime.timestamp(end_time) - datetime.timestamp(start_time), 3)
        print(f'{end_time} [INFO]\
            {text}\
            Duration: {time_elapsed:.2f}s')
