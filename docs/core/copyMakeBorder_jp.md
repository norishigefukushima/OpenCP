copyMakeBorder.hpp
==================
`cv::copyMakeBorder`を高速化した関数群です．
内部で並列化されています．  

# copyMakeBorderReplicate
```cpp
void copyMakeBorderReplicate(cv::InputArray src, cv::OutputArray dest, const int top, const int bottom, const int left, const int right);
```
## Usage
`cv::copyMakeBorder`の`borderType == cv::BORDER_REPLICATE`時の処理を高速化します．  

cv::の名前とかぶらないように，Replicateが末尾についています．  
名前空間で区切れますが，`using namespace cv` `using namespace cp`としたときに困らないように．  
以下のタイプのみサポートします．

* CV_8UC1
* CV_8UC3
* CV_32FC1
* CV_32FC3

# splitCopyMakeBorder
```cpp
void copyMakeBorderReplicate(cv::InputArray src, cv::OutputArray dest, const int top, const int bottom, const int left, const int right);	
```
## Usage
splitをしてからcopyMakeBorderをする処理を連結した関数です．  
キャッシュ効率が上がりパフォーマンスが向上しています．  
また，内部で並列化されています．
現在は，`cv::BORDER_REPLICATE`のみをサポートしており，別のタイプを指定するとAssertが呼ばれます．

*CV_8UC3
*CV_32FC3

# Test function
`testCopyMakeBorder.cpp`にパフォーマンステストコードがあります．