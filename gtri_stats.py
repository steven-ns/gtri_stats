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


pd.set_option("display.precision", 1)
pd.set_option('display.max_rows', None)

MASTER_FOLDER = 'C:/Users/KSH06/Desktop/2022-09-07/'
#MASTER_FOLDER = 'X:/'
#MASTER_FOLDER = 'X:/Left to Right Trains/'
#MASTER_FOLDER = 'X:/Right to Left Trains/'

N_SAMPLES = 200

#FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information']
#FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information','Left to Right Trains','Right to Left Trains']
FOLDER_IGNORE_LIST = ['$RECYCLE.BIN','System Volume Information','2022-09-30-1058_omron','2022-09-30-1058','Left to Right Trains','Right to Left Trains','2022-09-30-0847','2022-09-30-0855','2022-09-30-1025','2022-09-30-1025_omron']


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
        print('------------------------ AEI DATA ------------------------')
        #print(df.groupby(['folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp]}))
        print(df.groupby(['daypart','folder']).agg({'carSpeed' : [np.size, np.mean, np.max, np.min, np.ptp],'axelCnt' : [np.sum],'EOC' : [np.max],'EOT' : [np.max]}))
        #umTable = df.pivot_table(index=['daypart','folder'],columns='view',values='exp_pass',aggfunc='sum')

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

        maxSpeed = aeiDf['carSpeed'].max() + 5

        #f = 'Train_2022_08_29_22_31'
        axCnt = 0
        for f in aeiDf['folder'].unique()[0:15]:
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
        fig, ax = plt.subplots(figsize = (14,7))
        plt.plot(loc,signal_ts)

        for i in range(0,len(peaks)):
            plt.text(loc[peaks[i]]-15, signal_ts[peaks[i]]-1, str(i+1), fontsize = 10,color = 'red')
        plt.show()

        #print(peaks)

        #Plot Collage of Images

        W = 10*400
        H = 8*346

        collage = Image.new("RGB", (W,H))
        cnt = -1
        maxImgCnt = min(len(peaks),80)-1

        for i in range(0,W,400):
            for j in range(0,H,346):
                cnt = cnt + 1

                if cnt >= maxImgCnt:
                    break
                
                if isLeft:
                    fullImgPath = folder + file_names[peaks[cnt]-3]
                else:
                    fullImgPath = folder + file_names[peaks[cnt]+3]
                img = Image.open(fullImgPath)
                img = img.resize((400,346))

                collage.paste(img, (i,j))
        
        collage.show()

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


    def run(self):
        
        #aeiDf = self.aei_stats()
        #self.get_intensity_ts('C:/Users/KSH06/Desktop/2022-09-07/Train_2022_08_30_00_21/Isometric/')
        #aeiDf = self.aei_stats()
        #dfFPS = self.file_count_stats()
        dfDirection = self.get_direction_summary()
        self.process_intensity(dfDirection)
        #self.get_intensity_ts(False,'C:/Users/KSH06/Desktop/2022-09-07/Train_2022_08_30_00_21/Isometric/')
        #self.image_stats_multi()
        #self.plot_FPS(aeiDf,dfFPS)




if __name__ == "__main__":
    
    #Check for command line argument specifying date to process
    if len(sys.argv) > 1:
        startDate = sys.argv[-1]
    else:
        startDate = date.today().strftime("%Y-%m-%d")
    
    gtri_stats = GTRI_stats(startDate)
    gtri_stats.run()





