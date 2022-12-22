import sys
import os
import numpy as np
from args            import argument_parser

#just a small function to print a .txt report file

def savetxt(t,d):
    with open(d, 'w') as f:
        f.write(t)

def saveMe(args,    #args from the code
         info="", #any extra information we want to save
         dic={"the" : ["end"]}
        ):   #a vector dictionary, you can name the vector withe the key so it gets printed here.
    
    text = ""
    text = text + "########## argument sheet ######################################## \n"
    for arg in vars(args):
        text = text + f"#{arg:>15}  :  {str(getattr(args, arg))} \n"
    text = text + "################################################################## \n"
    
    
    text = text +"\n"+"\n" + info +"\n"+"\n"
    for k in dic.keys():
        text = text + k + "\n"
        for vec in dic[k]:
            text = text + "\t" + str(vec) + "\n"
        text = text + "\n"
    
    savetxt(text,f"{args.save_dir}/{args.name}/report.txt")
    
        
    