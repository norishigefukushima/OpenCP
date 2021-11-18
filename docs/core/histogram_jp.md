histogram.hpp
================

# drawHistogramImage
```cpp
void drawHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true, const int normalize_value = 0)
```
## Usage
src画像のヒストグラムを計算し，可視化します．  
`meancolor`で，平均値のbinの色を指定できます．  
`isDrawGrid`フラグでグリッドを表示するかどうか，`isDrawStats`フラグで，平均値，分散，最大値，最小値を表示するか指定します．  
また，`normalize_value`で，ヒストグラムをいくつで除算するか指定できます．  
デフォルトは0で，0の場合のみ，ヒストグラムのbinの最大値で除算します．  
なお，この関数は，下記の`drawHistogramImageGray`をカラーの場合，グレイの場合それぞれにおいて，適切に色付けして表示するためのラッパー関数です．  

## Optimization
* OpenCV

# drawHistogramImageGray
```cpp
void drawHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true, const int normalize_value = 0);
```
## Usage
グレイスケール専用の，ヒストグラム描画関数です．  
ただし，cv::Scalar colorの引数が，追加されており，binの色が指定できます．
より具体的な色の設定やマルチスペクトル画像用の設定に．

## Optimization
* OpenCV

# drawAccumulateHistogramImage
```cpp
void drawAccumulateHistogramImage(cv::InputArray src, cv::OutputArray histogram, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true);
```
## Usage
drawHistogramImage関数に対応する累積ヒストグラムを可視化する関数です．  
ただし，`normalize_value`を引数に取らず，常に最大値で正規化します．  

## Optimization
* OpenCV

# drawAccumulateHistogramImageGray
```cpp
void drawAccumulateHistogramImageGray(cv::InputArray src, cv::OutputArray histogram, cv::Scalar color, cv::Scalar meancolor, const bool isDrawGrid = true, const bool isDrawStats = true);
```
## Usage
drawHistogramImageGray関数に対応する累積ヒストグラムを可視化する関数です．  
ただし，`normalize_value`を引数に取らず，常に最大値で正規化します．  

## Optimization
* OpenCV