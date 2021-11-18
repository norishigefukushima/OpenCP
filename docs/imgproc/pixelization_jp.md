pixelization.hpp
===================
画像をピクセル化する関数群


# pixelization
画像をピクセル化（ダウンサンプルして，最近傍補間して，境界に線を引く）する関数です．
論文やパワーポイントで表示する図にグリッドが必要な時に使います．
画像サイズがpixelサイズで割り切れないときは，画像サイズを割り切れるサイズに切り上げしてからピクセル化します．そのため，右端，下端以外は常にpixelSizeの大きさになります．

```cpp
	CP_EXPORT void pixelization(cv::InputArray src, cv::OutputArray dest, const cv::Size pixelSize, const cv::Scalar color = cv::Scalar::all(255), const int thichness = 0);
	CP_EXPORT void guiPixelization(std::string wname, cv::Mat& src);
```

## Usage
* src: 入力画像
* dest: 出力画像
* pixelSize: ピクセル化するサイズ
* color: 境界線の色
* thichness: 境界線の太さ

# guiPixelization
pixelization関数をgui付きで呼び出します．

```cpp
	CP_EXPORT void guiPixelization(std::string wname, cv::Mat& src);
```

やっていることは以下の単純なことです．
```cpp
void guiPixelization(std::string wname, cv::Mat& src)
	{
		namedWindow(wname);
		static int pixel_guiPixelization = 16; createTrackbar("pixel", wname, &pixel_guiPixelization, 50); setTrackbarMin("pixel", wname, 1);
		static int thickness_guiPixelization = 1; createTrackbar("thickness", wname, &thickness_guiPixelization, 10);
		static int gray_guiPixelization = 100; createTrackbar("gray", wname, &gray_guiPixelization, 255);
		Mat dest;
		int key = 0;
		while (key != 'q')
		{
			pixelization(src, dest, Size(pixel_guiPixelization, pixel_guiPixelization), Scalar(gray_guiPixelization, gray_guiPixelization, gray_guiPixelization), thickness_guiPixelization);
			imshow(wname, dest);
			key = waitKey(1);
		}
	}
```
