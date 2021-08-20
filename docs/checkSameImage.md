checkSameImage.hpp
==========
2つの入力が同じかどうかだけを見て，同じなら`true`違うなら`false`を返します．
`isShowMessage`が`true`の場合，コンソールに状態をprintします．
PSNRなどの指標よりもただ差分を取ってcountNonZeroしているだけなのでPSNRやMSEを図るよりは高速です．

ランダムサンプルにより，同一画像か確認します．

# class CheckSameImage
```cpp
	class CP_EXPORT CheckSameImage
	{
	private:
		bool isUsePrev = true;
		cv::Mat prev;
		std::vector<cv::Point> positions;
		std::vector<cv::Scalar> samples;

		bool checkSamplePoints(cv::Mat& src);
		void generateRandomSamplePoints(cv::Mat& src, const int num_check_points);
		bool isSameFull(cv::InputArray src, cv::InputArray ref);

	public:
		/// <summary>
		/// set flag for using previous buffer in isSame(cv::InputArray, const int)
		/// </summary>
		/// <param name="flag">flags</param>
		void setUsePrev(const bool flag);

		/// <summary>
		/// check same image with the previous called image
		/// </summary>
		/// <param name="src">src image</param>
		/// <param name="num_check_points">number of random samples. if <=0, check full samples</param>
		/// <returns>true: same, false: not same</returns>
		bool isSame(cv::InputArray src, const int num_check_points = 10);

		/// <summary>
		/// check same image with the previous called image
		/// </summary>
		/// <param name="src">src image</param>
		/// <param name="ref">reference image, the image is pushed.</param>
		/// <param name="num_check_points">number of random samples. if <=0, check full samples</param>
		/// <param name="isShowMessage">flags for show console message or not.</param>
		/// <param name="ok_mes">message if(true)</param>
		/// <param name="ng_mes">message if(false)</param>
		/// <returns>true: same, false: not same</returns>
		bool isSame(cv::InputArray src, cv::InputArray ref, const int num_check_points = 0, const bool isShowMessage = true, const std::string ok_mes = "OK", const std::string ng_mes = "NG");
	};
```
## Usage
```cpp
bool CheckSameImage::isSameImage(cv::Mat& src, const int num_ckeck_points = 0);
```
前回呼び出した画像と同一かチェックします．  
`num_ckeck_points`の数だけランダムサンプルします．
この値が0以下の時，フルサンプルでチェックします．


```cpp
bool CheckSameImage::isSameImage(cv::InputArray src, cv::InputArray ref, const int num_check_points = 0, const bool isShowMessage = true, const std::string ok_mes = "OK", const std::string ng_mes = "NG");
```
２つの画像が同一かチェックします．
refが前回の画像として登録されます．

# checkSameImage
```cpp
bool isSame(cv::InputArray src, cv::InputArray ref, const int num_check_points = 0, const bool isShowMessage = true, const std::string ok_mes = "OK", const std::string ng_mes = "NG");
```

classの引数2つの場合のラッパー関数です．
2引数で1度しか呼ばない場合クラスで呼び出しても関数で呼び出してもコストはほぼ変わりません．
