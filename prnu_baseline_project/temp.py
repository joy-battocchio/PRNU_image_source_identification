# import OS module
import os
 
# Get the list of all files and directories
path = "./"
dir_list = os.listdir(path)
for img in dir_list:
    if img.__contains__("D31_I_flat_"):
        x = img.split("_")
        os.rename(img, "Samsung_S4Mini_"+x[3])
    
