checkSameImage.hpp
==========
ランダムサンプルにより，同一画像か確認します．

# class CheckSameImage
```cpp
		class CP_EXPORT CheckSameImage
	{
	private:
		std::vector<cv::Point> positions;
		std::vector<cv::Scalar> samples;

		bool checkSamplePoints(cv::Mat& src);
		void generateRandomSamplePoints(cv::Mat& src, const int num_ckeck_points);
	public:

		bool isSameImage(cv::Mat& src, const int num_ckeck_points = 10);
		bool isSameImage(cv::Mat& src, cv::Mat& ref,const int num_ckeck_points = 10);
	};
```
## Usage
```cpp
bool isSameImage(cv::Mat& src, const int num_ckeck_points = 10);
```
前回呼び出した画像と同一かチェックします．  
`num_ckeck_points`の数だけランダムサンプルします．

```cpp
bool isSameImage(cv::Mat& src, cv::Mat& ref,const int num_ckeck_points = 10);
```
２つの画像が同一かチェックします．

# checkSameImage
```cpp
bool ckeckSameImage(cv::Mat& src, cv::Mat& ref, const int num_ckeck_points=10);
```

classの引数2つの場合のラッパー関数．
