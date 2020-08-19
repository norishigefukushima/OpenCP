Metrics.hpp
================

# countNaN
```cpp
int countNaN(cv::InputArray src)
```
## Usage
NaNの数を返します．  
CV_32FかCV_64F専用です．  

## Optimization
* OpenCV

# countInf
```cpp
int countInf(cv::InputArray src)
```
## Usage
Infの数を返します．  
CV_32FかCV_64F専用です．  

## Optimization
* OpenCV

# countDenormalizedNumber
```cpp
int countDenormalizedNumber(cv::InputArray src)
```
## Usage
非正規化数（subnormal number or denomlized number）の数を返します．  
CV_32FかCV_64F専用です．  
非正規化数が浮動小数点に入っていると演算が非常に重たくなります．  

## Optimization
* OpenCV

# countDenormalizedNumberRatio
```cpp
double countDenormalizedNumberRatio(cv::InputArray src)
```
## Usage
非正規化数（subnormal number or denomlized number）が全体に占める割合(0～1)を返します．  
CV_32FかCV_64F専用です．  

## Optimization
* OpenCV