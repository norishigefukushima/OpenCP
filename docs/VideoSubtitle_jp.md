VideoSubtitle.hpp
===================
画像に字幕をつけて，一定時間後に消すクラスです．
論文などのデモンストレーション動画用に使います．

# class VideoSubtitle
```cpp
class CP_EXPORT VideoSubtitle
	{
	private:
		cp::Timer tscript;
		double time_dissolve_start = 500.0;
		double time_dissolve_end = 1000.0;
		double time_dissolve = time_dissolve_end - time_dissolve_start;
		std::string font = "Segoe UI";
		//string font = "Consolas";
		int vspace = 20;
		cv::Mat title;
		cv::Mat show;

		std::vector<std::string> text;
		std::vector<int> fontSize;
		cv::Rect textROI = cv::Rect(0, 0, 0, 0);
		cv::Point textPoint = cv::Point(0, 0);
		int getAlpha();
		cv::Rect getRectText(std::vector<std::string>& text, std::vector<int>& fontSize);
		void addVText(cv::Mat& image, std::vector<std::string>& text, cv::Point point, std::vector<int>& fontSize, cv::Scalar color);
	public:
		enum class POSITION
		{
			CENTER,
			TOP,
			BOTTOM
		};
		VideoSubtitle();

		void restart();
		void setDisolveTime(const double start_msec, const double end_msec);//from 0-start 100%, t/(start-end end-start)*100%
		void setFontType(std::string font = "Segoe UI");
		void setVSpace(const int vspace);//vertical space for multi-line text

		//single-line text case
		void setTitle(const cv::Size size, std::string text, int fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor = cv::Scalar::all(0), POSITION pos = POSITION::CENTER);
		//multi-line text case
		void setTitle(const cv::Size size, std::vector<std::string>& text, std::vector<int>& fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor = cv::Scalar::all(0), POSITION pos = POSITION::CENTER);
		//alpha blending title and image
		void showTitleDissolve(std::string wname, const cv::Mat& image);
		//overlay subscript
		void showScriptDissolve(std::string wname, const cv::Mat& image, const cv::Scalar textColor = cv::Scalar(255, 255, 255));

		//setTitle and then imshow (multi-line)
		void showTitle(std::string wname, const cv::Size size, std::vector<std::string>& text, std::vector<int>& fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor = cv::Scalar::all(0));
		//setTitle and then imshow (single-line)
		void showTitle(std::string wname, const cv::Size size, std::string text, const int fontSize, const cv::Scalar textcolor, const cv::Scalar backgroundcolor = cv::Scalar::all(0));
	};
```
## Usage
* インスタンスを作ります（コンストラクタで自動的にタイマーが起動）
* setTitleで字幕をセットします．
* 必要に応じて`restart()`でタイマーを初期化します．
* 無限ループ内で，showTitleDissolveかshowScriptDissolveを呼ぶと字幕と画像を合成します．

メソッド：
```cpp
void setDisolveTime(const double start_msec, const double end_msec);
```
ブレンド開始までの経過時間をミリ秒指定します．
`start_msec`までは１００％で表示します．
`end_msec`までに０％になるようにディゾルブ表示します．

## サンプル

```cpp
void testVideoSubtitle()
{
	Mat src = imread("img/lenna.png");

	string wname = "testVideoSubtitle";
	namedWindow(wname);
	int sw = 0; createTrackbar("sw", wname, &sw, 1);//subtitle rendering mode
	int pos = 1; createTrackbar("pow", wname, &pos, 2);//subtitle position

	VideoSubtitle vs;
	vector<string> vstring = { "testVideoSubtitle", "press r key to restart" };
	vector<int> vfsize = { 30,20 };
	vs.setFontType("Times New Roman");
	vs.setVSpace(10);

	vs.setDisolveTime(1000, 2000);

	cp::UpdateCheck uc(sw, pos);
	int key = 0;
	while (key != 'q')
	{	
		if (sw == 0)vs.showScriptDissolve(wname, src);
		if (sw == 1)vs.showTitleDissolve(wname, src);

		if (key == 'r' || uc.isUpdate(sw, pos))
		{
			vs.restart();
			vs.setTitle(src.size(), vstring, vfsize, Scalar(255, 255, 255), Scalar::all(0), VideoSubtitle::POSITION(pos));
		}

		key = waitKey(1);
	}
}
```


