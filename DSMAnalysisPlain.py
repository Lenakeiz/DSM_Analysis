# Remeber to activate the virtual environment first by typing the following in the terminal
# py -3 -m venv .venv
# .venv\scripts\activate

# open a dialog box to insert the file we are going to process
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# Importing packages for data loading
import pandas as pd

# Importing all the packages for plotting
# importing seabord just to set the color palette
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

# Importing numpy for manipulatring the data
import numpy as np

# Importing distance package for calculating distances 
from scipy.spatial import distance

#For debugging
#filename = 'C:/Users/AndreaCastegnaro/AppData/LocalLow/MXTreality/driving-sim-data-collection/MotorWaySimulation/Data/driving-sim-data-collection_DSMAnalyticsSystem_202201151627_Run_1.csv';
#filename = 'C:/Users/AndreaCastegnaro/AppData/LocalLow/MXTreality/driving-sim-data-collection/MotorWaySimulation/Data/driving-sim-data-collection_DSMAnalyticsSystem_202201281321_Run_1.csv';
#filename = 'C:/Users/AndreaCastegnaro/AppData/LocalLow/MXTreality/driving-sim-data-collection/MotorWaySimulation/Data/driving-sim-data-collection_DSMAnalyticsSystem_202201281610_Run_1.csv';
print('Open dialog box to DSM analytics file...')

filename = fd.askopenfilename(title='Open a DSM generated file',defaultextension='.csv',initialdir='%appdata%')

print('Loading file...')

dsmData = pd.read_csv(filename);
# Dropping the last column as the csv has always a trailing comma
dsmData = dsmData.iloc[:,:-1];
# Setting the time as a current index for the data
#dsmData.head()

del filename

# let s convert the time in milliseconds to second  
dsmData['time_s'] = dsmData.time_ms /1000
dsmData.set_index('time_s',inplace=True)

print('Calculating SMA for speed...')

# let s calculate the moving average using a time window of 2.5 s. Data is captured every 50ms so we need to average across 50 data points
# This will make the graph smoother
dsmData['SMA_50'] = dsmData.EgoSpeed.rolling(50, min_periods=1).mean()

# Resetting the previous saved configuration
sns.reset_defaults()

# Increase the sharpness of the display
plt.rcParams['figure.dpi'] = 360

# Increase the figure size
#sns.set_theme(style="whitegrid")
sns.set(style='white')
# let's also take the colors from the color palette
colPalette = sns.color_palette('Dark2')
# create a new figure
fig, ax = plt.subplots(figsize=(12,4));

# plotting the full time series
# plt.plot(dsmData.index,dsmData.loc[:,"EgoSpeed"])
plt.plot(dsmData.index,dsmData.SMA_50, c=colPalette[2])
# plotting an area where the speed is over the limit
plt.axhline(y=70, color=colPalette[3],linestyle=(5, (4, 3)))

# setting the labels and an horizontal grid 
plt.grid(axis='y')
plt.xlabel('Time (s)',size=18)
plt.ylabel('Speed (mph)',size=18)
plt.xticks(size=15)
plt.yticks(size=15)
plt.title('Ego vehicle speed over time', size=22)

print('Speed over time: plotting and saving in Results folder...')

# Removing top and right border
sns.despine();
plt.savefig('Results/EgoSpeedOverTime.png', bbox_inches='tight')
# Showing the plot
#plt.show()

del fig, ax, colPalette

# Calculating the amount of time spent in each of the lane
# We need to do some filtering. For each of the LaneID column we need to extract the last number after the last '_' character

# Let's create a function that will operate on the elements of the column
def extractLaneNumber(laneRoadID):
    if not pd.isna(laneRoadID):
        laneRoadIDpartitions = laneRoadID.split('_')
        extractedValue  = laneRoadIDpartitions[len(laneRoadIDpartitions)-1]
        fourLanesRoad = ['116','392']
        # In this way all the lanes will have the same index for the right most lane (for taking over)
        if not any(l in laneRoadID for l in fourLanesRoad):
            extractedInt = int(extractedValue) + 1
            extractedValue = str(extractedInt)
    else:
        extractedValue = str(-1)
    return extractedValue

# Let's create a new column containing only the lane information for the egovehicle
# First we duplicate the ego vehicle full lane
dsmData['ExtractedLaneNumber'] = dsmData['EgoLaneID']

# Second we explicity set the type for that string
dsmData['ExtractedLaneNumber'] = dsmData['ExtractedLaneNumber'].astype("string")

# Third we apply a transformation to this column by using the function we defined at the beginning of the block
dsmData['ExtractedLaneNumber'] = dsmData['ExtractedLaneNumber'].transform(extractLaneNumber)


# Let s get unique lane ids
uniqueLanes = pd.unique(dsmData.ExtractedLaneNumber)

# Let's count the occurrence of that lane for the ego vehicle
# We are going to reset the index for this operation
if not dsmData.index.isnull:
    dsmData.reset_index(inplace=True)
secsPerLane = dsmData.groupby('ExtractedLaneNumber')['time_ms'].count()
# print(secsPerLane)

# Trasforming the count in seconds according to the granularityu of the data
secsPerLane = secsPerLane*0.05
# print(secsPerLane)

# Resetting the previous saved configuration
sns.reset_defaults()

# Increase the sharpness of the display
plt.rcParams['figure.dpi'] = 360

# Increase the figure size
#sns.set_theme(style="whitegrid")
sns.set(style='white')

# Increase the figure size
#sns.set_theme(style="whitegrid")
sns.set(style='white')
# let's also take the colors from the color palette
colPalette = sns.color_palette('Dark2')
# plotting the histogram
# create a new figure
fig, ax = plt.subplots(figsize=(12,4));

# Unfortunately this does not work as counting the time is not enough if we can't multiply each count by 0.05 (time interval between data points)
#plt.hist(dsmData.ExtractedLaneNumber)

#plt.bar([-1, 0, 1, 2, 3], secsPerLane, width=0.5)
plt.bar([0, 1, 2, 3], secsPerLane, width=0.5)

# setting the labels and an horizontal grid 
plt.grid(axis='y')
plt.xlabel('Lane',size=18)
plt.ylabel('Cumulative time (s)',size=18)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xticks([0, 1, 2, 3], ['Tiger tail', 'Left', 'Center', 'Right'])
#plt.xticks([-1, 0, 1, 2, 3], ['Hard Shoulder', 'Tiger tail', 'Left', 'Center', 'Right'])
plt.title('Cumulative time spent on lanes', size=22)

print('Cumulative time per lanes: plotting and saving in Results folder...')

sns.despine();
plt.savefig("Results/CumulativeTimePerLanes.png", bbox_inches="tight")

#plt.show()

del ax, fig, colPalette, uniqueLanes, secsPerLane

# Calculating the number of transitions between lanes to calculate how many time the participants switched lane

# Possible transitions are 
# HS -> E ; E -> HS ; E -> L ; L -> E ; L -> C ; C -> L; C -> R; R -> C;

# Resetting the index for precautions in case we launched previous code blocks
if dsmData.index.name == 'time_s':
    dsmData.reset_index(inplace=True)

print('Calculating frequency of lane changes...')

# Let's create an array holding the information of the lanes. We are going to label the transitions based on these labels
laneLabels = ['HS', 'TT', 'L', 'C', 'R']
#laneLabels = ['TT', 'L', 'C', 'R']
laneChanges = []
cnt = 0

#Let s loop through the Extracted Lane Number and count the number of time two following items are different
ni = len(dsmData.ExtractedLaneNumber)-1
for i in range(2,ni):
    if not pd.isna(dsmData.ExtractedLaneNumber[i]) and not pd.isna(dsmData.ExtractedLaneNumber[i-1]) and dsmData.ExtractedLaneNumber[i-1] != dsmData.ExtractedLaneNumber[i]:
        # We have detected a change
        labelOne = laneLabels[int(dsmData.ExtractedLaneNumber[i-1]) + 1]
        labelTwo = laneLabels[int(dsmData.ExtractedLaneNumber[i]) + 1]
        laneChanges.extend([labelOne + '->' + labelTwo])
        cnt = cnt + 1

# Resetting the previous saved configuration
sns.reset_defaults()

# Increase the sharpness of the display
plt.rcParams['figure.dpi'] = 360

# Increase the figure size
#sns.set_theme(style="whitegrid")
sns.set(style='white')
# let's also take the colors from the color palette
colPalette = sns.color_palette('Dark2')
# create a new figure
fig, ax = plt.subplots(figsize=(12,4));

plt.hist(laneChanges)

# setting the labels and an horizontal grid 
plt.grid(axis='y')
plt.xlabel('Lane changes',size=18)
plt.ylabel('Frequency',size=18)
plt.xticks(size=15)
plt.yticks(size=15)
plt.title('Frequency of lane changes', size=22)

# adding a legend to the graph
# plt.legend(['HS = Hard Shoulder', 'TT = Tiger Tail', 'L = Left', 'C = Center', 'R = Right'], loc='upper right')
#text = 'HS = Hard Shoulder\nTT = Tiger Tail\nL = Left\nC = Center\nR = Right'
text = 'TT = Tiger Tail\nL = Left\nC = Center\nR = Right'

boundbox = {'facecolor':'white',
        'edgecolor':'gray',
        'boxstyle':'round,pad=0.5'}

#plt.text(7.25, 4, text, fontsize = 15,bbox=boundbox)

print('Frequency of lane changes: plotting and saving in Results folder...')

sns.despine();
plt.savefig("Results/FrequencyOfLangeChanges.png", bbox_inches="tight")

#plt.show()        

del ax, fig, cnt, i, colPalette, labelOne, labelTwo, laneChanges, laneLabels, ni, text, boundbox

# Resetting the index for precautions in case we launched previous code blocks
if dsmData.index.name == 'time_s':
    dsmData.reset_index(inplace=True)

print('Calculating distance to front car on the same lane...')

# Retrieving the position of nearby cars
laneIDlocations = [ dsmData.columns.get_loc('SV_0_laneID'), 
                    dsmData.columns.get_loc('SV_1_laneID'), 
                    dsmData.columns.get_loc('SV_2_laneID'), 
                    dsmData.columns.get_loc('SV_3_laneID'),
                    dsmData.columns.get_loc('SV_4_laneID'),
                    dsmData.columns.get_loc('SV_5_laneID'),
                    dsmData.columns.get_loc('SV_6_laneID'),
                    dsmData.columns.get_loc('SV_7_laneID')]

# Getting index from the SV_N_laneID where the position is placed for that car (to subtract to laneIDlocations)
posOffsetFromSVLane = 6
idOffsetfromSVLane = 7

# Allocating empty matrix results
distanceToFrontCar = np.zeros([len(dsmData.index),3], dtype='float64')
distanceToFrontCarCheck = np.zeros([len(dsmData.index),8], dtype='bool')

calculatedDist = float('inf')
foundCar = False

for i, row in dsmData.iterrows():

    foundCar = False
    calculatedDist = float('inf')
    distanceToFrontCar[i,0] = dsmData.time_s[i]
    distanceToFrontCar[i,2] = 0

    for j in range(0,7):
        # Checking we are on the same lane
        if(dsmData.EgoLaneID[i]) == dsmData.iloc[i,laneIDlocations[j]]:

            # For debug purposes
            distanceToFrontCarCheck[i,j] = True

            foundCar = True
            # Calculating distance from this car
            egoPos = np.array(dsmData.iloc[i,2:5])
            otherPos = np.array(dsmData.iloc[i,laneIDlocations[j]-posOffsetFromSVLane:laneIDlocations[j]-posOffsetFromSVLane+3])
            currDist = distance.euclidean(egoPos,otherPos)
            if(currDist < calculatedDist):
                    calculatedDist = currDist
                    distanceToFrontCar[i,2] = dsmData.iloc[i,laneIDlocations[j]-idOffsetfromSVLane]

    if not foundCar:
        distanceToFrontCar[i,1] = 0
    else:
        distanceToFrontCar[i,1] = calculatedDist

del egoPos, otherPos, currDist, foundCar, calculatedDist, i , j, posOffsetFromSVLane, laneIDlocations, row

# Calculating moving median to compensate for spurious empty values
movingMedian = pd.Series(distanceToFrontCar[:,1])
movingMedian = movingMedian.rolling(50, min_periods=1).median()

# Resetting the previous saved configuration
sns.reset_defaults()

# Increase the sharpness of the display
plt.rcParams['figure.dpi'] = 360

# Increase the figure size
#sns.set_theme(style="whitegrid")
sns.set(style='white')
# let's also take the colors from the color palette
colPalette = sns.color_palette('Dark2')
# create a new figure
fig, ax = plt.subplots(figsize=(12,4));

# plotting the full time series
# plt.plot(dsmData.index,dsmData.loc[:,"EgoSpeed"])
plt.plot(distanceToFrontCar[:,0],movingMedian, c=colPalette[2])

# setting the labels and an horizontal grid 
plt.grid(axis='y')
plt.xlabel('Time (s)',size=18)
plt.ylabel('Distance (m)',size=18)
plt.xticks(size=15)
plt.yticks(size=15)
plt.title('Distance from front car on the same lane over time', size=22)

print('Distance from front car on same lane: plotting and saving in Results folder...')

# Removing top and right border
sns.despine();
plt.savefig("Results/DistanceFromFrontCarSameLane.png", bbox_inches="tight")

#plt.show()

del colPalette, ax, fig, movingMedian