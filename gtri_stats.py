import sys
import os
from datetime import date
import numpy as np
import pandas as pd
from PIL import Image
import time
from datetime import timedelta
#from multiprocessing import ThreadPool
from multiprocessing import Pool
import random

pd.set_option("display.precision", 1)
pd.set_option('display.max_rows', None)

MASTER_FOLDER = 'C:/Users/KSH06/Desktop/2022-09-07/'
#MASTER_FOLDER = 'X:/'

FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information']

class GTRI_stats:

    def __init__(self,startDate):
        self.startDate = startDate

    def parse_aei(self,aei_path):
        
        aeiFile = open(aei_path, 'r')
        aeiLines = aeiFile.readlines()
        speedArray = []
        for lineStr in aeiLines[1:]:    #Skip first line of file
            C = lineStr.split('*')
            if len(C) == 15:        # Process only well formatted line 
                speedArray.append(int(C[12]))
        return speedArray # Array with Speed Data
    
    def aei_stats(self):

        pd.set_option("display.precision", 1)

        folderArray = []
        speedArray = []

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:
            aeiPath = MASTER_FOLDER + f + '/' + 'aeiData.txt'
            if os.path.exists(aeiPath):
                aeiSpeeds = self.parse_aei(aeiPath)
                speedArray.extend(aeiSpeeds)
                folderArray.extend([f] * len(aeiSpeeds))            
            else:
                print("Missing AEI Data: " + f)

        #Print Stats
        d={
            'carSpeed': speedArray,
            'folder': folderArray
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        print('------------------------ AEI DATA ------------------------')
        print(df.groupby(['folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp]}))       
    
    def get_file_count(self, folder_path):
        return len([entry for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))])
    
    def file_count_stats(self):
        
        print('\n------------------------ VIEW COUNTS ------------------------')

        folderArray = []
        viewArray = []
        countArray = []

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                continue

            viewsInFolder = [name for name in os.listdir(MASTER_FOLDER+f) if os.path.isdir(os.path.join(MASTER_FOLDER+f, name))]
            #print(viewsInFolder)
            for view in viewsInFolder:
                fullPath = MASTER_FOLDER + f + '/' + view

                fileCnt = self.get_file_count(fullPath)

                folderArray.append(f)
                viewArray.append(view)
                countArray.append(fileCnt)
        
        #Print Stats
        d={
            'view': viewArray,
            'folder': folderArray,
            'fileCnt': countArray
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        print(df.reset_index().pivot('folder', 'view', 'fileCnt'))
        #print(df.groupby(['folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp]}))           

    def image_stats(self):
        
        #pd.set_option("display.precision", 2)
        pd.options.display.float_format = '{:.1%}'.format

        print('\n-------------- PORTION OF IMAGES OVEREXPOSED -------------------')

        folderArray = []
        viewArray = []
        clipArray = []

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                continue

            viewsInFolder = [name for name in os.listdir(MASTER_FOLDER+f) if os.path.isdir(os.path.join(MASTER_FOLDER+f, name))]

            print("Processing: " + f, end =" ")
            start = time.time()
            #Make This Parallel
            for view in viewsInFolder:

                fullPath = MASTER_FOLDER + f + '/' + view

                file_names = [fn for fn in os.listdir(fullPath) if fn.endswith('jpeg') ]
                #Get All Image Files in Folder
                for fn in file_names:
                    fullImagePath = fullPath + '/' + fn

                    vals = np.asarray(Image.open(fullImagePath))
                    counts, bins = np.histogram(vals, range(257))
                    counts = counts/np.sum(counts)
                    p255 = counts[-1]

                    folderArray.append(f)
                    viewArray.append(view)
                    clipArray.append(p255)
            elapsed = (time.time() - start)
            print(str(timedelta(seconds=elapsed)))

        
        #Print Stats
        d={
            'view': viewArray,
            'folder': folderArray,
            'pClipped': clipArray
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        df['exp_pass'] = df.apply(lambda row: row.pClipped > 0.05, axis=1)
        df['exp_pass'] = df['exp_pass'].astype(int)

        sumTable = df.pivot_table(index='folder',columns='view',values='exp_pass',aggfunc='sum')
        cntTable = df.pivot_table(index='folder',columns='view',values='exp_pass',aggfunc='count')
        portionTable = sumTable.div(cntTable)

        print(portionTable)


        #return df
        #print(df.reset_index().pivot('folder', 'view', 'fileCnt'))
        #print(df.groupby(['folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp]}))             

    def get_img_stats(self,f):
        folderArray = []
        viewArray = []
        clipArray = []       
 
        viewsInFolder = [name for name in os.listdir(MASTER_FOLDER+f) if os.path.isdir(os.path.join(MASTER_FOLDER+f, name))]

        print("Processing: " + f)

        #Make This Parallel
        for view in viewsInFolder:

            fullPath = MASTER_FOLDER + f + '/' + view
 
            file_names = [fn for fn in os.listdir(fullPath) if fn.endswith('jpeg') ]

            #Shuffle file_names
            random.shuffle(file_names)
            #n_samples = np.min(100,len(file_names))
            n_samples = 200
            cnt = 0
            for fn in file_names:

                fullImagePath = fullPath + '/' + fn

                vals = np.asarray(Image.open(fullImagePath))
                counts, bins = np.histogram(vals, range(257))
                counts = counts/np.sum(counts)
                p255 = counts[-1]

                folderArray.append(f)
                viewArray.append(view)
                clipArray.append(p255)

                cnt = cnt + 1
                if cnt > n_samples:
                    break

        
        return folderArray, viewArray, clipArray

    def image_stats_multi(self):
        
        #pd.set_option("display.precision", 2)
        pd.options.display.float_format = '{:.0%}'.format

        print('\n-------------- PORTION OF IMAGES OVEREXPOSED -------------------')

        folderArray = []
        viewArray = []
        clipArray = []

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                dirsInFolder.remove(f)
                continue

        #Loop Through All folders, if aei file, then add to array
        start = time.time()
        with Pool(processes=6) as pool:
            for result in pool.imap_unordered(self.get_img_stats,dirsInFolder):
                folderArray.extend(result[0])
                viewArray.extend(result[1])
                clipArray.extend(result[2])
        elapsed = (time.time() - start)
        print(str(timedelta(seconds=elapsed)))        
        
        #Print Stats
        d={
            'view': viewArray,
            'folder': folderArray,
            'pClipped': clipArray
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        df['exp_pass'] = df.apply(lambda row: row.pClipped > 0.05, axis=1)
        df['exp_pass'] = df['exp_pass'].astype(int)

        sumTable = df.pivot_table(index='folder',columns='view',values='exp_pass',aggfunc='sum')
        cntTable = df.pivot_table(index='folder',columns='view',values='exp_pass',aggfunc='count')
        portionTable = sumTable.div(cntTable)

        print(portionTable)


    def run(self):
        
        self.aei_stats()
        self.file_count_stats()
        #self.image_stats()
        self.image_stats_multi()



if __name__ == "__main__":
    
    #Check for command line argument specifying date to process
    if len(sys.argv) > 1:
        startDate = sys.argv[-1]
    else:
        startDate = date.today().strftime("%Y-%m-%d")
    
    gtri_stats = GTRI_stats(startDate)
    gtri_stats.run()





