concat.hpp
==========
画像を結合する関数である`cv::hconcat`と`cv::vconcat`の拡張関数群です．
それぞれ，横方向に連結する関数と，縦方向に連結する関数です．
これを縦横に連結したり，連結後の画像から特定のindexの画像を取り出したり，`vector<Mat>`に戻したりできるようになります．

# concat/concatMerge
```cpp
	void concat(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_cols, const int tile_rows = 1);
	void concatMerge(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_cols, const int tile_rows = 1);
```
## Usage
縦横にconcatします．
`concatMerge`と`concat`は全く同じ関数のためどちらを呼び出しても構いません．  
下記にあるconcatSplitとの名前の整合性を取るために作成しています．

# concatSplit
```cpp
	void concatSplit(cv::InputArray src, cv::OutputArrayOfArrays dest, const cv::Size image_size);
	void concatSplit(cv::InputArray src, cv::OutputArrayOfArrays dest, const int tile_cols, const int tile_rows);
```
## Usage
concatの逆の処理です．  
連結画像から`vector<Mat>`を生成します．  
2つの関数は引数が違うだけであり，入力画像のサイズか，分割数のどちらかが分かれば戻せます．

# concatExtract
```cpp
	void concatExtract(cv::InputArrayOfArrays src, cv::OutputArray dest, const int tile_col_index, const int tile_row_index = 0);
	void concatExtract(cv::InputArrayOfArrays src, cv::Size size, cv::OutputArray dest, const int tile_col_index, const int tile_row_index = 0);
```
## Usage
連結画像から縦横のインデックスを指定して1つだけ抽出する関数です．  
インデックスは，分割数と違って`[0:width-1]`で指定する0スタートのものであることに注意してください．  
また，rowのインデックスが0の場合は，先頭から何番目の画像かを指定します．  
二つの関数は，`dest`のメモリが確保されているかどうかで使い分けます．  

# Test function
`testConcat.cpp`にサンプルコードがあります．以下コードの概要です．  
1. Kodakの24の画像を`concat`で6x4の画像として結合してリサイズ表示．  
2. その後，`concatExtract`で1，1番目を取得．  
3. また，`concatExtract`の相対位置指定（yを省略）で12番目の画像を取得．  
4. `concatSplit`で分解して，`vector`の要素の3番目を取得
5. `concatSplit`で分解する引数指定の方法2で3番目を取得

```cpp
void testConcat()
{
	vector<Mat> kodak24;
	for (int i = 1; i < 25; i++)
	{
		Mat temp = imread(format("img/kodak/kodim%02d.png", i), 1);
		if (temp.cols < temp.rows)transpose(temp, temp);
		//imshow("a", temp); waitKey();
		kodak24.push_back(temp);
	}

	Mat img;
	cp::concat(kodak24, img, 4, 6);
	cp::imshowResize("kodak24", img, Size(), 0.25, 0.25);

	Mat temp;
	cp::concatExtract(img, Size(768, 512), temp, 1, 1);
	imshow("extract (1,1)", temp);

	cp::concatExtract(img, Size(768, 512), temp, 12);
	imshow("extract 12", temp);

	vector<Mat> v;
	cp::concatSplit(img, v, Size(768, 512));
	imshow("concatSplit[3]", v[3]);

	vector<Mat> v2;
	cp::concatSplit(img, v2, 4, 6);
	imshow("concatSplit[3] v2", v2[3]);

	waitKey();
}
```
