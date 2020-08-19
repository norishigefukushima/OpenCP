crop.hpp
================

# cropZoom
```cpp
void cropZoom(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Rect roi, const int zoom_factor = 1);
void cropZoom(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Point center, const int window_size, const int zoom_factor = 1);
```
## Usage
画像をクロップして，その画像を拡大します．  
クロップする範囲は`Rect`でROIを指定するか，`Point`と`window_size`で中心と窓幅を指定してください．  
拡大の量は，`zoom_factor`で指定可能です．  
アップサンプリングは最近傍法で固定です．  

## Optimization
* OpenCV

## Example
See [guiCropZoom](#guiCropZoom)

# cropZoomWithBoundingBox
```cpp
void cropZoomWithBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Rect roi, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);
```
```cpp
void cropZoomWithBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, const cv::Point center, const int window_size, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);
```
## Usage
`cropZoom`に加えて，出力画像にバウンディングボックスとして，矩形を描画します．  
`color`と`thickness`で色と線の幅を指定してください．

論文等で用いる図で重要な領域をマークするのに使います．
## Optimization
* OpenCV

## example
See [guiCropZoom](#guiCropZoom)

# cropZoomWithSrcMarkAndBoundingBox
```cpp
void cropZoomWithSrcMarkAndBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, cv::OutputArray src_mark const cv::Rect roi, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);
```
```cpp
void cropZoomWithSrcMarkAndBoundingBox(cv::InputArray src, cv::OutputArray crop_zoom, cv::OutputArray src_mark, const cv::Point center, const int window_size, const int zoom_factor = 1, const cv::Scalar color = COLOR_RED, const int thickness = 1);
```
## Usage
`cropZoomWithBoundingBox`にさらに加えて，クロップする入力画像にもバウンディングボックスを描画します．  

論文等で用いる図で重要な領域をマークし，さらに，入力画像との対応関係を取るために使います．

## Optimization
* OpenCV

## example
See [guiCropZoom](#guiCropZoom)

# cropCenter
```cpp
void cropCenter(cv::InputArray src, cv::OutputArray crop, const int window_size);
```
# Usage
cropZoomの切り抜き中央点を画像の真ん中にし，zoomの倍率を1にした別名関数．
内部で下記のように呼び出しているだけ．
```cpp
void cropCenter(InputArray src, OutputArray crop, const int window_size)
	{
		Mat s = src.getMat();
		cropZoom(src, crop, Point(s.cols / 2, s.rows / 2), window_size);
	}
```

# guiCropZoom
```cpp
cv::Mat guiCropZoom(cv::InputArray src, const cv::Scalar color = COLOR_RED, const int thickness = 1, const std::string wname = "crop");
```
```cpp
cv::Mat guiCropZoom(cv::InputArray src, cv::Rect& dest_roi, int& dest_zoom_factor, const cv::Scalar color = COLOR_RED, const int thickness = 1, const std::string wname = "crop");
```
## Usage
上記の関数を，GUIから操作します．
また，`dest_roi`, `dest_zoom_factor`の引数を渡すことで，guiCropZoomで設定した拡大領域と拡大量を保持し，のちに上記のクロップズーム処理を複数の画像にバッチ処理することで，論文用の画像を生成できます．

また，戻り値は，クロップした画像です，再帰呼び出しにより様々な用途に使えます．  
例えば，下記のように使えます．  
```cpp
Mat a, b;
guiAlphaBlend(guiCropZoom(a),guiCropZoom(b));
```

**キーボードショートカット**
* i,j,k,l：vimライクな移動
* z:拡大率++
* x:拡大率--
* p:拡大した画像の表示位置を移動
* s:クロップズーム画像と，マーク画像を保存．wname+index.pngとwname+index_mark.pngの2つが保存される．indexはstaticで単調増加する．
* ?:ヘルプ
* q:終了

## Optimization
* OpenCV

## example
```cpp
void testCropZoom()
{
	Mat im1 = imread("img/stereo/Dolls/view1.png");
	Mat im2 = imread("img/stereo/Dolls/view5.png");

	cout << "call guiCropZoom" << endl;
	guiCropZoom(im1);
	cout << "check zoom factor and rectangle at same position" << endl;
	Rect roi;
	int factor;
	guiCropZoom(im2, roi, factor);

	Mat crop1, crop2, srcmark1, srcmark2;
	cropZoomWithSrcMarkAndBoundingBox(im1, crop1, srcmark1, roi, factor);
	cropZoomWithSrcMarkAndBoundingBox(im2, crop2, srcmark2, roi, factor);
	imshow("crop0", crop1);
	imshow("crop1", crop2);
	imshow("srcmark", srcmark1);
	imshow("srcmark", srcmark2);
	waitKey(0);
	/*imwrite("crop1.png", crop1);
	imwrite("crop2.png", crop2);
	imwrite("srcmark1.png", srcmark2);
	imwrite("srcmark2.png", srcmark2);*/
}
```
この例では，初めに`guiCropZoom`でGUI操作できることを確認しています．  
そして次の`guiCropZoom`で，前回起動したパラメータが保持できていることを確認しています．  
加えて，内部で設定したパラメータを外に取り出すために引数としてROIや拡大量を渡しています．  
最後にバッチ処理で，得られたパラメータを使ってクロップと拡大を繰り返します．
