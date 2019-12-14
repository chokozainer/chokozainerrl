import pandas as pd
import glob
import os

def table(filestr):
    score_files = glob.glob(filestr)
    score_files.sort(key=os.path.getmtime)
    score_file = score_files[-1]
    df = pd.read_csv(score_file, delimiter='\t' )
    return df

def graph(filestr,strx,stry):
    score_files = glob.glob(filestr)
    score_files.sort(key=os.path.getmtime)
    score_file = score_files[-1]
    df = pd.read_csv(score_file, delimiter='\t' )
    df.plot(x=strx,y=stry)