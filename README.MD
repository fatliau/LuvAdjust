# Image Luminance Adjustment

## Description
This is my project 1 from CS6384 Computer Vision in the University of Texas at Dallas

Process: 
RGB —> Gamma Correction RGB -> XYZ -> Luv -> L adjusted Luv -> XYZ -> RGB -> Inverse Gamma Correction -> output RGB

## set up
1. environment: Python3
2. package used: cv2
3. put the image and python file in the same folder

## execute command
```
python proj1_1.py 0.2 0.1 0.8 0.5 fruits.jpg out_2.png
python proj1_2.py 0.2 0.1 0.8 0.5 fruits.jpg out_2.png
```
(format: .py left_range up_range right_range down_range input_image output_image)

## assumption and criteria decision
1. Input RGB out of [0,255]
truncate to [0,255]
2. d = 0 (X=0 and Y=0 and Z=0)
formula change to u_ = 4 and v_ = 9 (Original: u_ = 4*X/d and v_ = 9*Y/d)
3. adjusted L = 0 (new L = 0)
formula change to u__ = uw and v__ = vw (Original: u__ = (u + 13*uw*nL) / (13*nL) and v__ = (v + 13*vw*nL) / (13*nL)  )
4. Output RGB out of [0,1]
truncate to [0,1]
5. Output RGB is NaN
set the value to 1

## all the decision documented in Log
format: @(j,i) brief explaination



