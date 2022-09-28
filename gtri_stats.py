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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


pd.set_option("display.precision", 1)
pd.set_option('display.max_rows', None)

MASTER_FOLDER = 'C:/Users/KSH06/Desktop/2022-09-07/'
#MASTER_FOLDER = 'X:/'
#MASTER_FOLDER = 'X:/Left to Right Trains/'
#MASTER_FOLDER = 'X:/Right to Left Trains/'

N_SAMPLES = 200

#FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information']
FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information','Left to Right Trains','Right to Left Trains']

class GTRI_stats:

    def __init__(self,startDate):
        self.startDate = startDate


    def parse_aei(self,aei_path):
        
        aeiFile = open(aei_path, 'r')
        aeiLines = aeiFile.readlines()
        speedArray = []
        indexArray = []
        idx = 0
        for lineStr in aeiLines[1:]:    #Skip first line of file
            C = lineStr.split('*')
            if len(C) == 15:        # Process only well formatted line 
                speedArray.append(int(C[12]))
                idx = idx + 1
                indexArray.append(idx)
        return speedArray,indexArray # Array with Speed Data

    def get_daypart(self,folder_name):
        
        E = folder_name.split('_')
        time_str = E[1] + '-' + E[2] + '-' + E[3] + ' ' + E[4] +  ':' + E[5]
        a = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
        if (a.hour > 6) and (a.hour < 20):
            dayPart = 'Daytime'
        else:
            dayPart = 'Nighttime'            

        return dayPart # Array with Speed Data    

    def aei_stats(self):

        pd.set_option("display.precision", 1)

        folderArray = []
        speedArray = []
        indexArray = []
        daypartArray = []

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:
            aeiPath = MASTER_FOLDER + f + '/' + 'aeiData.txt'
            if os.path.exists(aeiPath):
                aeiSpeeds,aeiIndex = self.parse_aei(aeiPath)
                speedArray.extend(aeiSpeeds)
                indexArray.extend(aeiIndex)
                folderArray.extend([f] * len(aeiSpeeds))
                daypartArray.extend([self.get_daypart(f)] * len(aeiSpeeds))             
            else:
                print("Missing AEI Data: " + f)

        #Print AEI Stats
        d={
            'carSpeed': speedArray,
            'folder': folderArray,
            'trainIndex':indexArray,
            'daypart': daypartArray
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        print('------------------------ AEI DATA ------------------------')
        #print(df.groupby(['folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp]}))
        print(df.groupby(['daypart','folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp]}))
        #umTable = df.pivot_table(index=['daypart','folder'],columns='view',values='exp_pass',aggfunc='sum')

        return df 
    

    def get_file_count(self, folder_path):
        
        allFiles = [entry for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))]
        allTimes = []

        for f in allFiles:

            D = f.split('_')
            
            splitLength = D[0].split('-')
            
            if len(splitLength) == 4:
                E = D[0].split('-')
                time_str = E[0] + '-' + E[1] + '-' + E[2] + ' ' + E[3][0:2] +  ':' + E[3][2:4] + ':' + E[3][4:6] + '.' + E[3][6:9]
            elif len(splitLength) == 5:
                E = D[0].split('-')
                time_str = E[0] + '-' + E[1] + '-' + E[2] + ' ' + E[3][0:2] +  ':' + E[3][2:4] + ':' + E[3][4:6] + '.' + E[4]
            else:
                E = D[1].split('-')
                time_str = E[0] + '-' + E[1] + '-' + E[2] + ' ' + E[3][0:2] +  ':' + E[3][2:4] + ':' + E[3][4:6] + '.' + E[4]
            
            a = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S.%f')
            allTimes.append(a)
            #print(a)


        return len(allFiles), allTimes
    

    def file_count_stats(self):
        
        pd.set_option("display.precision", 0)
        print('\n------------------------ VIEW COUNTS ------------------------')

        folderArray = []
        viewArray = []
        countArray = []
        daypartArray = []

        folderArrayFPS = []
        viewArrayFPS = []
        timeArrayFPS = []        

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                continue

            viewsInFolder = [name for name in os.listdir(MASTER_FOLDER+f) if os.path.isdir(os.path.join(MASTER_FOLDER+f, name))]
            #print(viewsInFolder)
            for view in viewsInFolder:
                fullPath = MASTER_FOLDER + f + '/' + view

                fileCnt,allTimes = self.get_file_count(fullPath)

                #Folder Count Summary
                folderArray.append(f)
                viewArray.append(view)
                countArray.append(fileCnt)
                daypartArray.append(self.get_daypart(f))

                #FPS Summary Info
                folderArrayFPS.extend([f] * len(allTimes))
                viewArrayFPS.extend([view] * len(allTimes))
                timeArrayFPS.extend(allTimes)  

        #Print Stats
        d={
            'view': viewArray,
            'folder': folderArray,
            'fileCnt': countArray,
            'daypart': daypartArray,
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        #print(df.reset_index().pivot('folder', 'view', 'fileCnt'))
        print(df.pivot_table(columns='view',values='fileCnt',index=['daypart','folder']))

        #FPS Dataframe
        d={
            'view': viewArrayFPS,
            'folder': folderArrayFPS,
            'datetime': timeArrayFPS
        }

        dfFPS = pd.DataFrame.from_dict(d,orient='index').transpose()
        return dfFPS

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
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

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
            n_samples = N_SAMPLES
            cnt = 0
            for fn in file_names:

                fullImagePath = fullPath + '/' + fn
                try:
                    vals = np.asarray(Image.open(fullImagePath))
                except:
                    continue
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

        #daypartArray.append(self.get_daypart(f))

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]
        
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

        df['daypart'] = df.apply(lambda row: self.get_daypart(row.folder), axis=1)

        #sumTable = df.pivot_table(index='folder',columns='view',values='exp_pass',aggfunc='sum')
        #cntTable = df.pivot_table(index='folder',columns='view',values='exp_pass',aggfunc='count')

        sumTable = df.pivot_table(index=['daypart','folder'],columns='view',values='exp_pass',aggfunc='sum')
        cntTable = df.pivot_table(index=['daypart','folder'],columns='view',values='exp_pass',aggfunc='count')
        portionTable = sumTable.div(cntTable)

        print('\n-------------- PORTION OF IMAGES OVEREXPOSED (Sample N = ' + str(N_SAMPLES) + ') -------------------')
        print(portionTable)


    def plot_FPS(self,aeiDf,dfFPS):

        palette = itertools.cycle(sns.color_palette())
        pd.options.mode.chained_assignment = None
        sns.set()
        sns.set(font_scale=0.8)
        plt.rcParams["figure.figsize"] = (15,8)
        fig, axes = plt.subplots(3, 5)

        colorDict = {
        "Isometric": "r",
        "side_bottom": "g"
        }

        maxSpeed = aeiDf['carSpeed'].max()

        #f = 'Train_2022_08_29_22_31'
        axCnt = 0
        for f in aeiDf['folder'].unique()[0:15]:
            axCnt = axCnt + 1
            axLive = plt.subplot(3, 5, axCnt)
            select = aeiDf['folder'] == f
            subAeiDf = aeiDf[select]
            #print(subAeiDf)

            sns.lineplot(data=subAeiDf, x="trainIndex", y="carSpeed", color="b",ax=axLive).set(title=f)
            axLive.set(ylim=(0, maxSpeed))
            #axLive.ylabel("carSpeed",color="b")
            axLive.yaxis.label.set_color('b')

            ax2 = axLive.twinx()
            color=next(palette) #remove blue
            ax2.set(ylim=(0, 25))
            for view in dfFPS['view'].unique():
                #view = 'side_bottom'
                select = (dfFPS['folder'] == f) & (dfFPS['view'] == view)
                subDfFPS = dfFPS[select]
                subDfFPS = subDfFPS.sort_values(by=['datetime'])
                #subDfFPS = subDfFPS.reset_index(drop=True, inplace=True)
                #print(subDfFPS)
                #subDfFPS['delta'] = (subDfFPS['datetime']-subDfFPS['datetime'].shift()).fillna(pd.Timedelta('0 days'))
                subDfFPS['delta'] = subDfFPS['datetime'].diff().dt.microseconds
                subDfFPS =  subDfFPS.assign(FPS = lambda x: (1000000/x['delta']))
                subDfFPS.index = pd.RangeIndex(len(subDfFPS.index))
                #print(subDfFPS)

                #print("Length:",len(subAeiDf))
                x = len(subAeiDf)*subDfFPS.index.to_numpy()/len(subDfFPS.index.to_numpy())

                #ax2 = plt.twinx()
                #ax2.set(ylim=(0, 40))
                sns.lineplot(data=subDfFPS, x=x, y="FPS",color=colorDict[view],ax=ax2,label=view)

                ax2.legend([],[], frameon=False)
                #ax2.grid(None)
                ax2.grid(False)
        
        # ax2.legend()
        plt.tight_layout()
        plt.show()        

        #plt.plot([1, 2, 3, 4])
        #plt.ylabel('some numbers')
        #plt.show()
        #print("TEST")


    def run(self):
        
        aeiDf = self.aei_stats()
        dfFPS = self.file_count_stats()
        self.image_stats_multi()
        self.plot_FPS(aeiDf,dfFPS)




if __name__ == "__main__":
    
    #Check for command line argument specifying date to process
    if len(sys.argv) > 1:
        startDate = sys.argv[-1]
    else:
        startDate = date.today().strftime("%Y-%m-%d")
    
    gtri_stats = GTRI_stats(startDate)
    gtri_stats.run()





