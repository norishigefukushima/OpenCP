UpdateCheck.hpp
===============
`cv::copyMakeBorder`を高速化した関数群です．
内部で並列化されています．  

# class UpdateCheck
```cpp
class CP_EXPORT UpdateCheck
	{
		bool isSkip = true;
		std::vector<double> parameters;
		bool firstTimeCheck(const bool flag);
	public:
		bool isFourceRetTrueFirstTime = true;
		void setIsFourceRetTrueFirstTime(const bool flag);
		UpdateCheck(double p0);
		UpdateCheck(double p0, double p1);
		UpdateCheck(double p0, double p1, double p2);
		UpdateCheck(double p0, double p1, double p2, double p3);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9);
		UpdateCheck(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10);


		bool isUpdate(double p0);
		bool isUpdate(double p0, double p1);
		bool isUpdate(double p0, double p1, double p2);
		bool isUpdate(double p0, double p1, double p2, double p3);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9);
		bool isUpdate(double p0, double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10);
	};
```
## Usage
変数が変更されたかを確認するためのクラス．チェックできる変数はdoubleにキャストして代入できるものだけ．

whileループ内で変更の有無に応じて実行するかどうかを判断することなどに使う．

setIsFourceRetTrueFirstTime(bool flag)は，初回1回目に強制的に`true`を返すかどうかを設定する関数．
デフォルトはtrueになっている．
初回だけは無視できない設定が多いため，デフォルトの`true`で困ることは少ないはず．

使い方：
初回は変数が変わっていなくても一度は表示される．
```cpp
int a = 0; createTrackBar("a", "window",&a, 100);
int b = 0; createTrackBar("b", "window",&b, 100);
cp::UpdateCheck uc(a,b);
//初回も表示しない場合．
//uc.setIsFourceRetTrueFirstTime(false);
for(;;)
{
    uc.isUpdate(a,b)
    {
        cout<<"update: "<<a<<","<<b<<endl;
    }
    waitKey(1);
}
```




