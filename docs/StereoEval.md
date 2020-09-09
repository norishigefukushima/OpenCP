StereoEval.hpp
================
ステレオマッチングを評価する関数群．

# calcBadPixel
```cpp
double calcBadPixel(cv::InputArray groundtruth, cv::InputArray disparityImage, cv::InputArray mask, double th, double amp);

double calcBadPixel(cv::InputArray groundtruth, cv::InputArray disparityImage, cv::InputArray mask, double th, double amp, cv::OutputArray outErr);
```
[middleburyステレオベンチマーク](https://vision.middlebury.edu/stereo/)のバッドピクセルレートを計算します．
thが閾値，ampは正解の視差画像が実際の視差値から何倍されているかを入力します．
1倍ではサブピクセルの閾値を評価できません．

2つ目の関数はエラーのマップも共に出力します．

# createDisparityALLMask
```cpp
void createDisparityALLMask(cv::Mat& src, cv::Mat& dest);
```
視差画像から，ALLに相当するマスクを生成します．

# createDisparityNonOcclusionMask
```cpp
void createDisparityNonOcclusionMask(cv::Mat& src, double amp, double thresh, cv::Mat& dest);
```
視差画像からNonoccに相当するマスクを生成します．

# class CP_EXPORT StereoEval
```cpp
class CP_EXPORT StereoEval
{
	void threshmap_init();
	bool skip_disc = false;
public:
	bool isInit = false;
	std::string message;
	cv::Mat state_all;
	cv::Mat state_nonocc;
	cv::Mat state_disc;
	cv::Mat ground_truth;
	cv::Mat mask_all;
	cv::Mat all_th;
	cv::Mat mask_nonocc;
	cv::Mat nonocc_th;
	cv::Mat mask_disc;
	cv::Mat disc_th;
	double amp;
	double all;
	double nonocc;
	double disc;
	double allMSE;
	double nonoccMSE;
	double discMSE;
	void init(cv::Mat& groundtruth, cv::Mat& maskNonocc, cv::Mat&maskAll, cv::Mat& maskDisc, double amp);
	void init(cv::Mat& groundtruth, const double amp, const intignoreLeftBoundary);
	StereoEval();
	StereoEval(std::string groundtruthPath, std::stringmaskNonoccPath, std::string maskAllPath, std::string maskDiscPath,double amp);
	StereoEval(cv::Mat& groundtruth, cv::Mat& maskNonocc, cv::Mat&maskAll, cv::Mat& maskDisc, double amp);
	StereoEval(cv::Mat& groundtruth, const double amp, const intignoreLeftBoundary = 0);
	std::string getBadPixel(cv::Mat& src, double threshold = 1.0, boolisPrint = true);
	std::string getMSE(cv::Mat& src, const int disparity_scale = 1,const bool isPrint = true);
	std::string operator() (cv::InputArray src, const double threshold= 1.0, const int disparity_scale = 1, const bool isPrint = true);
	void compare(cv::Mat& before, cv::Mat& after, double threshold = 10, bool isPrint = true);
};
```

上記関数群をクラスにより操作します．
使い方は以下を参照のこと．
```cpp
//正解が2倍の真値の視差画像disp_GTを入力．最後の引数は省略可能だが，左サイド何画素無視するかを指定可能．最大視差値で指定すればマッチングが不安定な領域を無視可能．
cp::StereoEval eval(disp_GT, 2, disp_max);
Mat disp;//output disparity
...
//計算した視差値，閾値，計算した視差値が何倍か？，を出力する．
//戻り値はstring．
//最後の`false`は，trueにしたらcoutしなくても戻り値をコンソール出力する．
cout<< eval(disp, 0.5, 16, false));
cout<< eval(disp, 1.0, 16, false));
cout<< eval(disp, 2.0, 16, false));
//MSEを求める場合は，以下．
eval.getMSE(disp, 16, false);
//値が必要な時は，下記変数が計算後publicになっているため直接取得する．
cout<<eval.all;
cout<<eval.nonocc;
cout<<eval.disc
cout<<eval.allMSE;
cout<<eval.nonoccMSE;
cout<<eval.discMSE;
```