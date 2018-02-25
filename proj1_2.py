#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:13:01 2018

@author: JC
"""

import cv2
import numpy as np
import sys
import math
from collections import OrderedDict

#Constans setting
RGBTrans = [[3.240479, -1.53715, -0.498535],
            [-0.969256, 1.875991, 0.041556],
            [0.055648, -0.204043, 1.057311]]

XYZTrans = [[0.412453, 0.35758, 0.180423],
            [0.212671, 0.71516, 0.072169],
            [0.019334, 0.119193, 0.950227]]
Xw=0.95; Yw=1; Zw=1.09
uw = 4*Xw/(Xw+15*Yw+3*Zw)
vw = 9*Yw/(Xw+15*Yw+3*Zw)
#Output Luminance range setting
ranH = 100
ranL = 0

#Argument storage
if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

#Input Setting
inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

#Setting
rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

#Log Collector
Log = []

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels
itemNum = (H2-H1)*(W2-W1)
table = OrderedDict()
for j in range(H1,H2):
    for i in range(W1,W2):    
        b, g, r= inputImage[j,i]
        # Truncate b,g,r range from 0 ~ 255
        if b > 255:
            b = 255
            Log.append("@("+str(j)+","+str(i)+") input b > 255")
        if b < 0:
            b = 0
            Log.append("@("+str(j)+","+str(i)+") input b < 0")
        if g > 255:
            g = 255
            Log.append("@("+str(j)+","+str(i)+") input g > 255")
        if g < 0:
            g = 0
            Log.append("@("+str(j)+","+str(i)+") input g < 0")
        if r > 255:
            r = 255
            Log.append("@("+str(j)+","+str(i)+") input r > 255")
        if r < 0:
            r = 0
            Log.append("@("+str(j)+","+str(i)+") input r < 0")
        #
        RGB = [r/255,g/255,b/255]
        # Gamma correction
        for c in range(len(RGB)):
            if RGB[c] < 0.03928:
                RGB[c] = RGB[c] / 12.92    
            else:
                RGB[c] = pow((RGB[c]+0.055) / 1.055 , 2.4)
        # Transfomr from RGB to XYZ
        XYZ = np.matmul(np.asarray(XYZTrans), np.asarray([[RGB[0]],[RGB[1]],[RGB[2]]]))
        X, Y, Z = XYZ[0][0],XYZ[1][0],XYZ[2][0]
        # Get Lumincance
        t = Y/Yw
        if t > 0.008856:
            L = round(116 * pow(t,1/3) - 16)
        else:
            L = round(903.3 * t)
        if L not in table.keys():
            table[L] = [1]
        else:
            table[L][0] += 1

table = OrderedDict(sorted(table.items(), key = lambda x: x[0]))
accum = 0
for key in table.keys():
    accum += table[key][0]
    temp = table[key][0]
    table[key] = [temp, accum]
keys = list(table.keys())
for i in range(len(keys)):
    h = table[keys[i]][0]
    if i < 1:
        fminus = 0
    else:
        fminus = table[keys[i-1]][1]
    f = table[keys[i]][1]
    floor = int((fminus+f) * (ranH-ranL+1) / (2 * itemNum) + ranL)
    table[keys[i]] = [h,f,floor]
# end of getting Max Min Luminance over the window

#Use Mapping Table from the window to do equalizing over the whole picture
newMatrix = np.copy(inputImage)
for j in range(rows):
    for i in range(cols):
        b, g, r= inputImage[j,i]
        # Truncate b,g,r range from 0 ~ 255
        if b > 255:
            b = 255
            Log.append("@("+str(j)+","+str(i)+") input b > 255")
        if b < 0:
            b = 0
            Log.append("@("+str(j)+","+str(i)+") input b < 0")
        if g > 255:
            g = 255
            Log.append("@("+str(j)+","+str(i)+") input g > 255")
        if g < 0:
            g = 0
            Log.append("@("+str(j)+","+str(i)+") input g < 0")
        if r > 255:
            r = 255
            Log.append("@("+str(j)+","+str(i)+") input r > 255")
        if r < 0:
            r = 0
            Log.append("@("+str(j)+","+str(i)+") input r < 0")
        #
        RGB = [r/255,g/255,b/255]
        # Gamma correction
        for c in range(len(RGB)):
            if RGB[c] < 0.03928:
                RGB[c] = RGB[c] / 12.92    
            else:
                RGB[c] = pow((RGB[c]+0.055) / 1.055 , 2.4)
        # Transfomr from RGB to XYZ
        XYZ = np.matmul(np.asarray(XYZTrans), np.asarray([[RGB[0]],[RGB[1]],[RGB[2]]]))
        X, Y, Z = XYZ[0][0],XYZ[1][0],XYZ[2][0]
        # Get Lumincance
        t = Y/Yw
        if t > 0.008856:
            L = round(116 * pow(t,1/3) - 16)
        else:
            L = round(903.3 * t)
        #get new L form the mapping table
        if L in keys:
            nL = table[L][2]
        else:
            if L < keys[0]:
                nL = table[keys[0]][2]
            elif L > keys[-1]:
                nL = table[keys[-1]][2]
            else:
                kprev=0
                kcurr=1
                while( True ):
                    if L > keys[kcurr] and kcurr < len(keys) :
                        kcurr+=1
                        kprev+=1
                    else:
                        nL = table[keys[kprev]][2]
                        #nL = (L - keys[kprev]) / (keys[kcurr] - keys[kprev]) * (table[keys[kcurr]][2] - table[keys[kprev]][2]) + table[keys[kprev]][2]
                        break
                    
        d = X + 15*Y +3*Z
        if d == 0:
            Log.append("@("+str(j)+","+str(i)+") d = 0")
            u_ = 4
            v_ = 9
        else:
            u_ = 4*X/d
            v_ = 9*Y/d
        
        u = 13*nL*(u_ - uw)
        v = 13*nL*(v_ - vw)
   
        if nL ==0:
            Log.append("@("+str(j)+","+str(i)+") nL = 0")
            u__ = uw
            v__ = vw
        else:
            u__ = (u + 13*uw*nL) / (13*nL)
            v__ = (v + 13*vw*nL) / (13*nL)

        if nL > 7.9996:
            nY = pow((nL+16)/116, 3)*Yw
        else:
            nY = nL*Yw/903.3
            
        if v__ == 0:
            Log.append("@("+str(j)+","+str(i)+") v__ = 0")
            nX=0; nZ=0
        else:
            nX = nY*2.25*u__/v__
            nZ = nY*(3 - 0.75*u__ - 5*v__)/v__
        RGB = np.matmul(np.asarray(RGBTrans),np.asarray([[nX],[nY],[nZ]]))
        # Inverse Gamma correction
        for c in range(len(RGB)):
            if RGB[c] < 0.00304:
                RGB[c] = RGB[c]*12.92    
            else:
                RGB[c] = 1.055*pow(RGB[c], 1/2.4) - 0.055
                if RGB[c] > 1:
                    Log.append("@("+str(j)+","+str(i)+") RGB["+str(c)+"] > 1")
                    RGB[c] = 1
        
        nr,ng,nb = RGB[0][0],RGB[1][0],RGB[2][0]
 
        if math.isnan(nr):
            Log.append("@("+str(j)+","+str(i)+") nr is NaN")
            nr = 1
        if math.isnan(ng):
            Log.append("@("+str(j)+","+str(i)+") ng is NaN")
            ng = 1
        if math.isnan(nb):
            Log.append("@("+str(j)+","+str(i)+") nb is NaN")
            nb = 1

        newMatrix[j,i]=[int(nb*255+0.5),int(ng*255+0.5),int(nr*255+0.5)]


cv2.imwrite(name_output, newMatrix)
cv2.imshow("output:", newMatrix)

file = open(name_output[:-4]+str(".txt"),"w") 
for item in Log:
    print(item)
    file.write(item+"\n")
file.close() 


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
