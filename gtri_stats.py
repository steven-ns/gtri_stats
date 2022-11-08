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
import cv2
from scipy.signal import find_peaks
import shutil
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
import dataframe_image as dfi
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from random import sample

pd.set_option("display.precision", 1)
pd.set_option('display.max_rows', None)

MASTER_FOLDER = 'C:/Users/KSH06/Desktop/2022-09-07/'
#MASTER_FOLDER = 'X:/'
#MASTER_FOLDER = 'X:/Left to Right Trains/'
#MASTER_FOLDER = 'X:/Right to Left Trains/'

SAVE_FOLDER = 'C:/Users/KSH06/Desktop/SAVES/'

N_SAMPLES = 200

#FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information']
#FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information','Left to Right Trains','Right to Left Trains']
FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information','2022-09-30-1058_omron','2022-09-30-1058','Left to Right Trains','Right to Left Trains','2022-09-30-0847','2022-09-30-0855','2022-09-30-1025','2022-09-30-1025_omron']

#TODO: Add code for ignore folders that don't start with Train_; or to select only from certain day
FOLDER_IGNORE_LIST = [name for name in os.listdir(MASTER_FOLDER) if name.split('_')[0] != 'Train']

trigger_dict = {'Brake_lower':2,'Brake_upper':2,'Isometric':0,'UC_isometric':1,'crosskey':3,'ls_truck':-1,'side_bottom':-1,'side_top':-1,'truck':4,'under_up':1}

class GTRI_stats:

    def __init__(self,startDate):
        self.startDate = startDate


    def parse_aei(self,aei_path):
        
        aeiFile = open(aei_path, 'r')
        aeiLines = aeiFile.readlines()
        
        speedArray = []
        indexArray = []
        axelArray = []
        EOC_detect = 0
        EOT_detect = 0

        idx = 0
        for lineStr in aeiLines[1:]:    #Skip first line of file
            C = lineStr.split('*')

            if C[0] == 'EOT':
                EOT_detect = 1

            if C[0] == 'EOC':
                EOC_detect = 1

            if len(C) == 15:        # Process only well formatted line 
                speedArray.append(int(C[12]))
                axelArray.append(int(C[13]))
                idx = idx + 1
                indexArray.append(idx)
        return speedArray,indexArray,axelArray,EOT_detect,EOC_detect # Array with Speed Data

    def get_daypart(self,folder_name):
        
        E = folder_name.split('_')
        time_str = E[1] + '-' + E[2] + '-' + E[3] + ' ' + E[4] +  ':' + E[5][0:2]
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
        axelArray = []
        EOC_array = []
        EOT_array = []

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:
            aeiPath = MASTER_FOLDER + f + '/' + 'aeiData.txt'
            if os.path.exists(aeiPath):
                aeiSpeeds,aeiIndex,axelCnt,EOT_detect,EOC_detect = self.parse_aei(aeiPath)
                speedArray.extend(aeiSpeeds)
                indexArray.extend(aeiIndex)
                axelArray.extend(axelCnt)
                folderArray.extend([f] * len(aeiSpeeds))
                daypartArray.extend([self.get_daypart(f)] * len(aeiSpeeds))

                EOC_array.extend([EOC_detect] * len(aeiSpeeds)) 
                EOT_array.extend([EOT_detect] * len(aeiSpeeds)) 


            else:
                print("Missing AEI Data: " + f)

        #Print AEI Stats
        d={
            'carSpeed': speedArray,
            'folder': folderArray,
            'trainIndex':indexArray,
            'daypart': daypartArray,
            'axelCnt':axelArray,
            'EOC':EOC_array,
            'EOT':EOT_array
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        #TODO: Set data types

        print('\n------------------------ AEI DATA ------------------------')
        #print(df.groupby(['folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp]}))
        #print(df.groupby(['daypart','folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp],'axelCnt' : [np.sum],'EOC' : [np.max],'EOT' : [np.max]}))
        pDf = df.groupby(['daypart','folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp],'axelCnt' : [np.sum],'EOC' : [np.max],'EOT' : [np.max]})
        #umTable = df.pivot_table(index=['daypart','folder'],columns='view',values='exp_pass',aggfunc='sum')

        print(pDf)
        #print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

        return df 
    

    def get_file_count(self, folder_path):
        
        allFiles = [entry for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))]
        allFiles = [entry for entry in allFiles if entry != 'Thumbs.db']
        allTimes = []
        
        #print(folder_path)
        
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
            last_time_str = time_str
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
        daypartFPS = []        

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
                daypartFPS.extend([self.get_daypart(f)] * len(allTimes))

        #Print Stats
        d={
            'view': viewArray,
            'folder': folderArray,
            'fileCnt': countArray,
            'daypart': daypartArray,
        }

        #TODO: Set data types

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        #print(df.reset_index().pivot('folder', 'view', 'fileCnt'))
        pd.options.display.float_format = '{:.0f}'.format
        print(df.pivot_table(columns='view',values='fileCnt',index=['daypart','folder']))

        #FPS Dataframe
        d={
            'view': viewArrayFPS,
            'folder': folderArrayFPS,
            'datetime': timeArrayFPS,
            'daypart': daypartFPS
        }

        dfFPS = pd.DataFrame.from_dict(d,orient='index').transpose()
        #TODO: Set data types

        print('\n------------------------ FOLDER DURATIONS (Max-Min Time in Minutes) ------------------------')
        #dfFPS['datetime_min'] = dfFPS['datetime'].dt.total_seconds()

        #print(dfFPS['datetime'].head(10).dt.total_seconds())
        #pd.set_option("display.precision", 3)
        #print(dfFPS.groupby(['daypart','folder','view'])['datetime'].apply(lambda x: (x.max() - x.min()).total_seconds()/60))
        #print(dfFPS.pivot_table(columns='view',values='datetime_min',index=['daypart','folder'],aggfunc={'datetime': np.ptp}))
        #pd.set_option("display.precision", 1)
        pd.options.display.float_format = '{:,.1f}'.format
        print(dfFPS.pivot_table(columns='view',values='datetime',index=['daypart','folder'],aggfunc={lambda x: (x.max() - x.min()).total_seconds()/60}))
        
        #print(dfFPS.head(10))
        #print(dfFPS.pivot_table(columns='view',values='datetime_min',index=['daypart','folder'],aggfunc={'datetime': np.ptp}))
        #print(dfFPS.groupby(['folder','view']).agg({'datetime' : [np.ptp]}))   


        return dfFPS, df

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


    def plot_FPS(self,aeiDf,dfFPS,dfSpeed):

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

        maxSpeed = aeiDf['carSpeed'].max() + 5

        #f = 'Train_2022_08_29_22_31'
        axCnt = 0
        for f in aeiDf['folder'].unique()[-15:]:
            axCnt = axCnt + 1
            axLive = plt.subplot(3, 5, axCnt)
            select = aeiDf['folder'] == f
            subAeiDf = aeiDf[select]
            #print(subAeiDf)

            sns.lineplot(data=subAeiDf, x="trainIndex", y="carSpeed", color="b",ax=axLive,zorder=100).set(title=f)
            axLive.set(ylim=(0, maxSpeed))
            plt.setp(axLive.lines, zorder=100)
            #axLive.ylabel("carSpeed",color="b")
            axLive.yaxis.label.set_color('b')
            axLive.xaxis.label.set_text('Railcar Index')

            #Plot SpeedArray
            select = dfSpeed['folder'] == f
            subDfSpeed = dfSpeed[select]

            x = len(subAeiDf)*np.linspace(0,1,num=len(subDfSpeed))
            sns.lineplot(x=x, y=subDfSpeed['speed'], color="k",ax=axLive,zorder=100,linestyle='--')

            ax2 = axLive.twinx()
            color=next(palette) #remove blue
            ax2.set(ylim=(0, 25))
            for view in dfFPS['view'].unique():
                
                if view in colorDict.keys():
                    pass
                else:
                    continue
            
                #view = 'side_bottom'
                select = (dfFPS['folder'] == f) & (dfFPS['view'] == view)
                subDfFPS = dfFPS[select]
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
                sns.lineplot(data=subDfFPS, x=x, y="FPS",color=colorDict[view],ax=ax2,label=view,zorder=0,alpha  = 0.5)

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
    
    def get_direction(self,folder):

        file_names = [fn for fn in os.listdir(folder) if fn.endswith('jpeg') ]
        print(folder,len(file_names))

        if len(file_names) < 1:
            return 0

        show_video = False
        if len(file_names) > 25:
            start_frame = int(len(file_names)/2)
            end_frame = start_frame + 10
        else:
            return 0

        old_frame = cv2.imread(folder+file_names[start_frame])
        old_frame = cv2.resize(old_frame, (400, 346)) 
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        #First Features
        #sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.04)
        sift = cv2.SIFT_create(contrastThreshold = 0.04)
        keypoints_1, descriptors_1 = sift.detectAndCompute(old_gray,None)

        color = np.random.randint(0, 255, (100, 3))
        avgDist = []

        for i in range(start_frame+1,end_frame):

            frame = cv2.imread(folder+file_names[i])
            frame = cv2.resize(frame, (400, 346)) 
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            keypoints_2, descriptors_2 = sift.detectAndCompute(frame_gray,None)
            
            #feature matching
            try:
                bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                matches = bf.match(descriptors_1,descriptors_2)
                matches = sorted(matches, key = lambda x:x.distance)
            except:
                continue
            
            #Calculate Distance
            x_dst = []
            for match in matches:
                p1 = keypoints_1[match.queryIdx].pt
                p2 = keypoints_2[match.trainIdx].pt
                x_dst.append(p2[0]-p1[0])
            
            if len(x_dst) > 0:
                avgDist.append(np.mean(np.array(x_dst)))

            #Show window
            if show_video:

                img3 = cv2.drawMatches(old_gray, keypoints_1, frame_gray, keypoints_2, matches[:50], frame_gray, flags=2)
                cv2.imshow("frame", img3)
                k = cv2.waitKey(25) & 0xFF
                if k == 27:
                    break

            #Copy to old for next iter
            old_gray = frame_gray.copy()
            keypoints_1 = keypoints_2 
            descriptors_1 = descriptors_2

        #print(avgDist)
        #print("Direction: ",np.mean(np.array(avgDist)))

        return np.mean(np.array(avgDist))

    def get_direction_summary(self):
        
        print('\n----------------- DIRECTION CALCS ---------------------')

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        trainName = []
        trainDirection = []
        daypartArray = []

        for f in dirsInFolder:
            dirResult = 'NaN'
            iso_folder = MASTER_FOLDER + f + '/Isometric/'
            #print(iso_folder)
            dirVal = self.get_direction(iso_folder)
            if dirVal < 0:
                dirResult = 'Left'
            if dirVal > 0:
                dirResult = 'Right'
            
            trainName.append(f)
            trainDirection.append(dirResult)
            daypartArray.append(self.get_daypart(f))
        
        #Print Stats
        d={
            'folder': trainName,
            'direction': trainDirection,
            'daypart':daypartArray
        }

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        #print(df.groupby(['daypart','folder'])['direction'].agg(pd.Series.mode))
        print('\n----------------- DIRECTION OF TRAIN, ISOMETRIC ---------------------')
        print(df.groupby(['daypart','folder']).agg(pd.Series.mode))

        return df
        #print(df)
        #dirTable = df.pivot_table(index=['daypart','folder'],columns='view',values='exp_pass',aggfunc='sum')
        #dirTable = df.pivot_table(index=['daypart','folder'],columns='direction',values='direction')
        #print(dirTable)    

    def get_intensity_ts(self,isLeft,folder):

        file_names = [fn for fn in os.listdir(folder) if fn.endswith('jpeg') ]
        print(folder,len(file_names))

        if len(file_names) < 1:
            return 0

        imgSumArray = []
        locArray = []

        for i in range(0,len(file_names)):
            #print(i)
            frame = cv2.imread(folder+file_names[i])
            imgSumArray.append(np.average(frame, axis=0)[0][0])
            locArray.append(i)
        
        signal_ts = np.array(imgSumArray)
        loc = np.array(locArray)

        peaks, properties = find_peaks(-signal_ts, prominence=10, width=1, rel_height=0.9, distance=15)

        #Plot Signal
        # fig, ax = plt.subplots(figsize = (14,7))
        # plt.plot(loc,signal_ts)

        # for i in range(0,len(peaks)):
        #     plt.text(loc[peaks[i]]-15, signal_ts[peaks[i]]-1, str(i+1), fontsize = 10,color = 'red')
        # plt.show()

        #print(peaks)

        for i in range(0,len(peaks)):
            for k in range(0,3):
                if isLeft:
                    fullImgPath = folder + file_names[peaks[i]-5+k]
                    img = Image.open(fullImgPath)
                    savePath = SAVE_FOLDER + file_names[peaks[i]-5+k]
                    img.save(savePath, "jpeg")
                else:
                    fullImgPath = folder + file_names[peaks[i]+1+k]
                    img = Image.open(fullImgPath)
                    savePath = SAVE_FOLDER + file_names[peaks[i]-5+k]
                    img.save(savePath, "jpeg")                

        #Plot Collage of Images

        # W = 10*400
        # H = 8*346

        # collage = Image.new("RGB", (W,H))
        # cnt = -1
        # maxImgCnt = min(len(peaks),80)-1

        # for i in range(0,W,400):
        #     for j in range(0,H,346):
        #         cnt = cnt + 1

        #         if cnt >= maxImgCnt:
        #             break
                
        #         if isLeft:
        #             fullImgPath = folder + file_names[peaks[cnt]-3]
        #         else:
        #             fullImgPath = folder + file_names[peaks[cnt]+3]
        #         img = Image.open(fullImgPath)
        #         img = img.resize((400,346))

        #         collage.paste(img, (i,j))
        
        # collage.show()

    def process_intensity(self,dfDirection):

        print('\n----------------- INTENSITY PROCESSING ---------------------')

        for i in range(len(dfDirection)):

            f = dfDirection['folder'].iloc[i]
            direction = dfDirection['direction'].iloc[i]

            print(f,direction)

            fullFolderPath = MASTER_FOLDER + dfDirection['folder'].iloc[i] + '/Isometric/'

            if direction == 'Left':                
                self.get_intensity_ts(True,fullFolderPath)
            elif direction == 'Right':
                self.get_intensity_ts(False,fullFolderPath)
            else:
                pass

    def estimate_stretch(self,folder):

        print('\n----------------- STRETCH PROCESSING ---------------------')

        print(folder)

        file_names = [fn for fn in os.listdir(folder) if fn.endswith('jpeg') ]
        print(folder,len(file_names))

        if len(file_names) < 1:
            return 0

        imgSumArray = []
        locArray = np.array([])
        imgProfile = np.array([])

        imgSize = 400

        #for i in range(0,len(file_names)):
        for i in range(303,315):
            #print(i)
            frame = cv2.imread(folder+file_names[i])
            frame = cv2.resize(frame, (imgSize,imgSize), interpolation = cv2.INTER_AREA)
            #print(frame[:,0,0])
            imgProfile = np.concatenate([imgProfile, frame[:,0,0]])
            locArray = np.concatenate([locArray,i+np.linspace(0,1,num = imgSize)])

            fromPath = folder + file_names[i]
            savePath = 'C:/Users/KSH06/Desktop/to_stitch/' + file_names[i]
            shutil.copy(fromPath,savePath)


            #imgSumArray.append(np.average(frame, axis=0)[0][0])
            #locArray.append(i
        #plt.plot(imgProfile)

        #kernel = np.ones(30)
        #imgProfile = np.convolve(imgProfile, kernel)

        #print("Len:",len(locArray))

        #plt.acorr(imgProfile, maxlags = 10000)
        #plt.show()

        peaks, properties = find_peaks(imgProfile, prominence=200, width=5, rel_height=0.9, distance=20)

        #plt.scatter(locArray[peaks][1:],np.diff(peaks))
        # ax = plt.gca()
        # ax.set_ylim(0, 100)

        #plt.plot(np.diff(peaks))

        plt.plot(imgProfile)
        for i in range(0,len(peaks)):
            plt.text(peaks[i],imgProfile[peaks[i]], str(i+1), fontsize = 10, color = 'red')

        plt.show()

        # np.savetxt("C:/Users/KSH06/Desktop/signal.csv", imgProfile, delimiter=",")

    def getSpeedArray(self,f):

        #print('\n----------------- PROCESSING AXEL LOGS ---------------------')

        triggerTimeArray = []
        sensorArray = []

        for i in range(6,8):
            x = np.fromfile(MASTER_FOLDER + f + "/Log"+str(i)+".bin", dtype=np.longlong)
            for val in x:
                triggerTimeArray.append(val)
                sensorArray.append(i)

        d={
            'sensor': sensorArray,
            'triggerTime': triggerTimeArray,
        }

        df=pd.DataFrame.from_dict(d,orient='index').transpose()
        df = df.sort_values(by='triggerTime', ascending=True)
        df = df.reset_index(drop=True)

        speedArrayLR = []
        Left_to_Right = True

        for i in range(1,len(df)):
            
            incBool = df['sensor'].iloc[i] > df['sensor'].iloc[i-1]
            
            if incBool == Left_to_Right:
                
                speed = abs(0.0568182*5.09375/((df['triggerTime'].iloc[i] - df['triggerTime'].iloc[i-1])/1000000000))
                speedArrayLR.append(speed)
        
        speedArrayRL = []
        Left_to_Right = False

        for i in range(1,len(df)):
            
            incBool = df['sensor'].iloc[i] > df['sensor'].iloc[i-1]
            
            if incBool == Left_to_Right:
                
                speed = abs(0.0568182*5.09375/((df['triggerTime'].iloc[i] - df['triggerTime'].iloc[i-1])/1000000000))
                speedArrayRL.append(speed)        
        
        is_Left_to_Right = 0
        if np.mean(np.array(speedArrayRL)) > np.mean(np.array(speedArrayLR)):
            speedArray = speedArrayRL
            is_Left_to_Right = 0
        else:
            speedArray = speedArrayLR
            is_Left_to_Right = 1

        return np.array(speedArray), is_Left_to_Right
        #plt.plot(speedArray)
        #plt.show()

    def axel_trigger_proc(self):
        #daypartArray.append(self.get_daypart(f))

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                dirsInFolder.remove(f)
                continue

        speedArray = []
        folderArray = []
        daypartArray = []
        directionArray = []

        for f in dirsInFolder:
            
            try:
                speedData, is_Left_to_Right = self.getSpeedArray(f)

                folderArray.extend([f] * len(speedData))
                speedArray.extend(speedData)
                daypartArray.extend([self.get_daypart(f)] * len(speedData))
                directionArray.extend([is_Left_to_Right] * len(speedData))

            except:
                print("Missing Speed Data")


        #Print Stats
        d={
            'speed': speedArray,
            'folder': folderArray,
            'daypart': daypartArray,
            'direction': directionArray
        }

        dfSpeed = pd.DataFrame.from_dict(d,orient='index').transpose()

        #pd.set_option("display.precision", 1)
        pd.options.display.float_format = '{:.1f}'.format
        sDf = dfSpeed.groupby(['daypart','folder']).agg({'speed' : [np.size, np.mean, np.max, np.min, np.ptp],'direction' : [np.max]})
        print('\n----------------- AXEL COUNTER DATA ---------------------')
        print(sDf)

        return dfSpeed


    def single_plot(self,aeiDf,dfFPS,dfSpeed,f):

        palette = itertools.cycle(sns.color_palette())
        pd.options.mode.chained_assignment = None
        sns.set()
        sns.set(font_scale=0.8)
        plt.rcParams["figure.figsize"] = (8,5)
        fig, axes = plt.subplots(3, 5)

        colorDict = {
        "Isometric": "r",
        "side_bottom": "g"
        }

        maxSpeed = aeiDf['carSpeed'].max() + 5

        #f = 'Train_2022_08_29_22_31'
        axCnt = 0
        axCnt = axCnt + 1
        axLive = plt.subplot(1, 1, axCnt)
        select = aeiDf['folder'] == f
        subAeiDf = aeiDf[select]
        #print(subAeiDf)

        sns.lineplot(data=subAeiDf, x="trainIndex", y="carSpeed", color="b",ax=axLive,zorder=100).set(title=f)
        axLive.set(ylim=(0, maxSpeed))
        plt.setp(axLive.lines, zorder=100)
        #axLive.ylabel("carSpeed",color="b")
        axLive.yaxis.label.set_color('b')
        axLive.xaxis.label.set_text('Railcar Index')

        #Plot SpeedArray
        select = dfSpeed['folder'] == f
        subDfSpeed = dfSpeed[select]

        x = len(subAeiDf)*np.linspace(0,1,num=len(subDfSpeed))
        sns.lineplot(x=x, y=subDfSpeed['speed'], color="k",ax=axLive,zorder=100,linestyle='--')

        ax2 = axLive.twinx()
        color=next(palette) #remove blue
        ax2.set(ylim=(0, 25))
        for view in dfFPS['view'].unique():
            
            if view in colorDict.keys():
                pass
            else:
                continue
        
            #view = 'side_bottom'
            select = (dfFPS['folder'] == f) & (dfFPS['view'] == view)
            subDfFPS = dfFPS[select]
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
            sns.lineplot(data=subDfFPS, x=x, y="FPS",color=colorDict[view],ax=ax2,label=view,zorder=0,alpha  = 0.5)

            ax2.legend([],[], frameon=False)
            #ax2.grid(None)
            ax2.grid(False)
        
        # ax2.legend()
        plt.tight_layout()
        plt.savefig('./plots/plot1.png', dpi=300)
        #plt.show()



    def buildPDF(self,aeiDf,dfFPS,dfSpeed,dfCounts):
        
        print("Building PDF...")

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                dirsInFolder.remove(f)
                continue

        for f in dirsInFolder[0:2]:

            print(f)

            #Base Image
            base_img = Image.new('RGB', (1920, 1080),(255, 255, 255))

            #Train Info
            train_info = {'Folder': f,'Direction':'Left_to_Right','Cold Start':'Yes','Cold Stop':'No','Amtrack':'TBD','Daypart':'Nighttime'}
            dfInfo = pd.DataFrame(list(train_info.items()),columns = ['Attribute','Value'])
            print (dfInfo)
            dfi.export(dfInfo,'./plots/info_table.png', dpi=300)
            infoTable = Image.open('./plots/info_table.png')
            infoTable = ImageOps.contain(infoTable, (400,400))
            base_img.paste(infoTable, (60, 60))

            #Speed Plot
            self.single_plot(aeiDf,dfFPS,dfSpeed,f)

            plot1 = Image.open('./plots/plot1.png')
            plot1 = ImageOps.contain(plot1, (900,900))
            base_img.paste(plot1, (700, 10))

            #Speed Plot Legend
            legend1 = Image.open('./plots/Legend.jpg')
            legend1 = ImageOps.contain(legend1, (250,250))
            base_img.paste(legend1, (1600, 50))

            #AEI Speed Table
            select = aeiDf['folder'] == f
            subAeiDf = aeiDf[select]

            pDf = subAeiDf.groupby(['folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp],'axelCnt' : [np.sum],'EOC' : [np.max],'EOT' : [np.max]})
            print(pDf)
            dfi.export(pDf,'./plots/table2.png', dpi=300)

            table2 = Image.open('./plots/table2_header.png')
            table2 = ImageOps.contain(table2, (900,900))
            base_img.paste(table2, (20, 550))

            table2 = Image.open('./plots/table2.png')
            table2 = ImageOps.contain(table2, (750,750))
            base_img.paste(table2, (20, 625))
            

            #Axel Speed Table
            select = dfSpeed['folder'] == f
            subdfSpeed = dfSpeed[select]

            pDf = subdfSpeed.groupby(['folder']).agg({'speed' : [np.size, np.mean, np.max, np.min, np.ptp],'direction' : [np.max]})
            print(pDf)
            dfi.export(pDf,'./plots/table3.png', dpi=300)

            table3 = Image.open('./plots/table3_header.png')
            table3 = ImageOps.contain(table3, (900,900))
            base_img.paste(table3, (20, 800))

            table3 = Image.open('./plots/table3.png')
            table3 = ImageOps.contain(table3, (650,650))
            base_img.paste(table3, (20, 850))


            #Image Counts Table
            select = dfCounts['folder'] == f
            subdfCounts = dfCounts[select]

            pd.options.display.float_format = '{:.0f}'.format
            pDf = subdfCounts.groupby(['view']).agg({'fileCnt' : [np.sum]})
            #pDf = subdfCounts.pivot_table(columns='fileCnt',values='fileCnt',index=['view'])
            print(pDf)
            dfi.export(pDf,'./plots/table4.png', dpi=300)

            table4 = Image.open('./plots/table4_header.png')
            table4 = ImageOps.contain(table4, (900,900))
            base_img.paste(table4, (900, 600))

            table4 = Image.open('./plots/table4.png')
            table4 = ImageOps.contain(table4, (200,1000))
            base_img.paste(table4, (1150, 700))

            #Save Image
            base_img.save('./reports/' + f + '.jpg', quality=95)


    def get_mosiacs_stats(self):
        
        pd.set_option("display.precision", 0)
        print('\n------------------------ Mosaics Stats ------------------------')

        folderArray = []
        viewArray = []
        countArray = []
        daypartArray = []
        mTypeArray = []
     

        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                continue

            viewsInFolder = ['side_bottom','side_top']

            for view in viewsInFolder:
                
                mosaicPath = MASTER_FOLDER + f + '/' + view + '/mosaics/'

                #subDirInFolder = [name for name in os.listdir(mosaicPath) if os.path.isdir(os.path.join(mosaicPath, name))]
                subDirInFolder = ['couplers','full_car','trucks','units']

                for s in subDirInFolder:

                    fullPath = mosaicPath + s
                    
                    try:
                        allFiles = [entry for entry in os.listdir(fullPath) if os.path.isfile(os.path.join(fullPath, entry))]
                        allFiles = [entry for entry in allFiles if entry != 'Thumbs.db']
                        fileCnt = len(allFiles)
                    except:
                        fileCnt = -1

                    #Folder Count Summary
                    folderArray.append(f)
                    viewArray.append(view)
                    countArray.append(fileCnt)
                    daypartArray.append(self.get_daypart(f))
                    mTypeArray.append(s)


        #Print Stats
        d={
            'view': viewArray,
            'folder': folderArray,
            'fileCnt': countArray,
            'daypart': daypartArray,
            'mosaicType': mTypeArray
        }

        #TODO: Set data types

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        #print(df.reset_index().pivot('folder', 'view', 'fileCnt'))
        pd.options.display.float_format = '{:.0f}'.format
        print(df.pivot_table(columns=['view','mosaicType'],values='fileCnt',index=['daypart','folder']))

    def get_dropped_frame_stats(self,dfCounts):
        
        #pd.set_option("display.precision", 0)
        print('\n------------------------ DROPPED FRAME STATS (Line Scans) ------------------------')

        #print(dfCounts.head(10))

        select = (dfCounts['view'] == 'side_bottom') | (dfCounts['view'] == 'side_top')
        dfCountsLS = dfCounts[select]

        pd.options.display.float_format = '{:.0f}'.format
        pvtCnt = dfCountsLS.pivot_table(columns='view',values='fileCnt',index=['daypart','folder'])
        pvtMax = dfCountsLS.pivot_table(values='fileCnt',index=['daypart','folder'],aggfunc={np.max})
        pvtTable = pd.merge(pvtCnt, pvtMax,right_on=None,left_index=True,right_index=True)
        
        #pvtCnt['max']=np.max(pvtCnt['side_bottom'],pvtCnt['side_top'])
        pvtTable['side_bottom_drops'] = pvtTable['amax'] - pvtTable['side_bottom']
        pvtTable['side_top_drops'] = pvtTable['amax'] - pvtTable['side_top']

        print(pvtTable)

        #print(np.sum(pvtTable['side_bottom_drops']),np.sum(pvtTable['amax']))
        #isNan = pvtTable['side_bottom_drops'] == NaN
        #isNan = ~np.isnan(pvtTable['side_top_drops'])
        #print(isNan)
        #print(pvtTable[isNan]['side_top_drops'])

        isNan = ~np.isnan(pvtTable['side_top_drops'])
        sbDrops = 10000 * np.sum(pvtTable[isNan]['side_bottom_drops'])/np.sum(pvtTable[isNan]['amax'])
        isNan = ~np.isnan(pvtTable['side_top_drops'])
        stDrops = 10000 * np.sum(pvtTable[isNan]['side_top_drops'])/np.sum(pvtTable[isNan]['amax'])
        print()
        print("side_bottom dropped frames: ", "{:.1f}".format(sbDrops), " per 10k")
        print("side_top dropped frames: ", "{:.1f}".format(stDrops), " per 10k")

        print('\n------------------------ DROPPED FRAME STATS (Area Scans) ------------------------')

        select = (dfCounts['view'] == 'Brake_lower') | (dfCounts['view'] == 'Brake_upper')
        dfCountsLS = dfCounts[select]

        pd.options.display.float_format = '{:.0f}'.format
        pvtCnt = dfCountsLS.pivot_table(columns='view',values='fileCnt',index=['daypart','folder'])
        pvtMax = dfCountsLS.pivot_table(values='fileCnt',index=['daypart','folder'],aggfunc={np.max})
        pvtTable = pd.merge(pvtCnt, pvtMax,right_on=None,left_index=True,right_index=True)
        
        #pvtCnt['max']=np.max(pvtCnt['side_bottom'],pvtCnt['side_top'])
        pvtTable['Brake_lower_drops'] = pvtTable['amax'] - pvtTable['Brake_lower']
        pvtTable['Brake_upper_drops'] = pvtTable['amax'] - pvtTable['Brake_upper']

        print(pvtTable)

        isNan = ~np.isnan(pvtTable['Brake_lower_drops'])
        sbDrops = 10000 * np.sum(pvtTable[isNan]['Brake_lower_drops'])/np.sum(pvtTable[isNan]['amax'])
        isNan = ~np.isnan(pvtTable['Brake_upper_drops'])
        stDrops = 10000 * np.sum(pvtTable[isNan]['Brake_upper_drops'])/np.sum(pvtTable[isNan]['amax'])
        print()
        print("Brake_lower dropped frames: ", "{:.1f}".format(sbDrops), " per 10k")
        print("Brake_upper dropped frames: ", "{:.1f}".format(stDrops), " per 10k")

        print('\n------------------------ DROPPED FRAME STATS (Area Scans) ------------------------')

        select = (dfCounts['view'] == 'UC_isometric') | (dfCounts['view'] == 'under_up')
        dfCountsLS = dfCounts[select]

        pd.options.display.float_format = '{:.0f}'.format
        pvtCnt = dfCountsLS.pivot_table(columns='view',values='fileCnt',index=['daypart','folder'])
        pvtMax = dfCountsLS.pivot_table(values='fileCnt',index=['daypart','folder'],aggfunc={np.max})
        pvtTable = pd.merge(pvtCnt, pvtMax,right_on=None,left_index=True,right_index=True)
        
        #pvtCnt['max']=np.max(pvtCnt['side_bottom'],pvtCnt['side_top'])
        pvtTable['UC_isometric_drops'] = pvtTable['amax'] - pvtTable['UC_isometric']
        pvtTable['under_up_drops'] = pvtTable['amax'] - pvtTable['under_up']

        print(pvtTable)

        isNan = ~np.isnan(pvtTable['UC_isometric_drops'])
        sbDrops = 10000 * np.sum(pvtTable[isNan]['UC_isometric_drops'])/np.sum(pvtTable[isNan]['amax'])
        isNan = ~np.isnan(pvtTable['under_up_drops'])
        stDrops = 10000 * np.sum(pvtTable[isNan]['under_up_drops'])/np.sum(pvtTable[isNan]['amax'])
        print()
        print("UC_isometric dropped frames: ", "{:.1f}".format(sbDrops), " per 10k")
        print("under_up dropped frames: ", "{:.1f}".format(stDrops), " per 10k")        

    def get_image_stats_per_car(self,aeiDf,dfCounts):
        
        #pd.set_option("display.precision", 0)
        print('\n------------------------ VIEW COUNTS PER RAILCAR ------------------------')

        aeiTable = aeiDf.pivot_table(index=['daypart','folder'],values='carSpeed',aggfunc='count')
        #print(aeiTable)

        cntTable = dfCounts.pivot_table(index=['daypart','folder'],columns='view',values='fileCnt',aggfunc='sum')
        #print(cntTable)
        portionTable = cntTable

        for uView in np.unique(dfCounts['view']):
            portionTable[uView] = portionTable[uView]/aeiTable['carSpeed']

        #portionTable = cntTable.div(aeiTable['carSpeed'])
        pd.options.display.float_format = '{:.1f}'.format
        print(portionTable)

        #print(dfCounts.head(10))

    def sample_images(self, path_to_view, n_samples, dst = ''):

        #Get Source Files
        allFiles = [item for item in os.listdir(path_to_view) if os.path.isfile(os.path.join(path_to_view, item))]
        allFiles = [item for item in allFiles if item != 'Thumbs.db']
        
        #Get Sample Size
        sample_size = min(len(allFiles),n_samples)
        
        #Sample
        sampleFiles = sample(allFiles,sample_size)
        print(len(sampleFiles))

        if dst:
            
            srcFiles = [path_to_view + item for item in sampleFiles]
            dstFiles = [dst + item for item in sampleFiles]

            for s,d in zip(srcFiles,dstFiles):
                shutil.copyfile(s, d)

    def get_trigger_counts(self):
        
        pd.set_option("display.precision", 0)
        print('\n------------------------ Trigger Stats ------------------------')

        folderArray = []
        viewArray = []
        countArray = []
        daypartArray = []
     
        dirsInFolder = [name for name in os.listdir(MASTER_FOLDER) if os.path.isdir(os.path.join(MASTER_FOLDER, name))]
        dirsInFolder = [d for d in dirsInFolder if not d[0] == '.']
        dirsInFolder = [name for name in dirsInFolder if name not in FOLDER_IGNORE_LIST]

        #Loop Through All folders, if aei file, then add to array

        for f in dirsInFolder:

            if f in FOLDER_IGNORE_LIST:
                continue

            triggerLogPath = MASTER_FOLDER + f + '/channelTriggerCounts.txt'
            #print(triggerLogPath)

            try:
                trigCounts = np.loadtxt(triggerLogPath)
            except:
                continue

            #print(trigCounts)

            for key in trigger_dict:
                
                folderArray.append(f)
                viewArray.append(key)
                countArray.append(trigCounts[trigger_dict[key]])
                daypartArray.append(self.get_daypart(f))
                
                #print(key)
                #Print Stats
        
        d={
            'view': viewArray,
            'folder': folderArray,
            'triggers': countArray,
            'daypart': daypartArray,
        }

        #TODO: Set data types

        df = pd.DataFrame.from_dict(d,orient='index').transpose()
        #print(df.reset_index().pivot('folder', 'view', 'fileCnt'))
        pd.options.display.float_format = '{:.0f}'.format
        print(df.pivot_table(columns=['view'],values='triggers',index=['daypart','folder']))

    def run(self):

        #self.sample_images('C:/Users/KSH06/Desktop/2022-09-07/Train_2022_08_29_22_31/side_bottom/',10)
        #self.sample_images('C:/Users/KSH06/Desktop/2022-09-07/Train_2022_08_29_22_31/side_bottom/',10,'C:/Users/KSH06/Desktop/test2/')
        #print(trigger_dict['Isometric'])

        self.get_trigger_counts()

        # self.get_mosiacs_stats()
        # aeiDf = self.aei_stats()
        # dfSpeed = self.axel_trigger_proc()
        # dfFPS, dfCounts = self.file_count_stats()
        
        # self.get_dropped_frame_stats(dfCounts)
        # self.get_image_stats_per_car(aeiDf,dfCounts)

        #-------- Plotting --------
        # #self.plot_FPS(aeiDf,dfFPS,dfSpeed)
        # self.buildPDF(aeiDf,dfFPS,dfSpeed,dfCounts)

        #-------- Depricated Stats --------
        #dfDirection = self.get_direction_summary()
        #self.image_stats_multi() #Overexposure
        #self.process_intensity(dfDirection)
        #self.get_intensity_ts(False,'C:/Users/KSH06/Desktop/2022-09-07/Train_2022_08_30_00_21/Isometric/')
        #self.estimate_stretch('C:/Users/KSH06/Desktop/2022-09-07/Train_2022_09_02_05_26/side_bottom/')




if __name__ == "__main__":
    
    #Check for command line argument specifying date to process
    if len(sys.argv) > 1:
        startDate = sys.argv[-1]
    else:
        startDate = date.today().strftime("%Y-%m-%d")
    
    gtri_stats = GTRI_stats(startDate)
    gtri_stats.run()





