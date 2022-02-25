import time
import sys

for i in range(10):
    sys.stdout.write("\r" + str(i))
    sys.stdout.flush()
    time.sleep(0.5)

'''
import time
import sys
import os

load_str = "starting your console application..."
ls_len = len(load_str)

animation = "|/-\\"
anicount = 0

counttime = 0

i = 0

while (counttime != 100):
    time.sleep(0.075)
    load_str_list = list(load_str) 
    x = ord(load_str_list[i])
    y = 0
    if x != 32 and x != 46:             
        if x>90:
            y = x-32
        else:
            y = x + 32
        load_str_list[i]= chr(y)
    res =''             
    for j in range(ls_len):
        res = res + load_str_list[j]
    sys.stdout.write("\r"+res + animation[anicount])
    sys.stdout.flush()
    load_str = res

      
    anicount = (anicount + 1)% 4
    i =(i + 1)% ls_len
    counttime = counttime + 1

os.system("cls") if os.name =="nt" else os.system("clear")

s ="David"
sys.stdout.write("Hello "+str(s)+"\n")
'''