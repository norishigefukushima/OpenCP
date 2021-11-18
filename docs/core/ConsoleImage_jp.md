Consolemage.hpp
===============
使い方，`consoleImageTest()`を参照の事．

# classConsoleImage
使い方は下記．
```cpp
//宣言．
ConsoleImage ci;
//デフォルト引数はこれ．画面のサイズと，ウィンドウ名が指定できる．
//ConsoleImage ci(Size(640, 480), "console")

ci("a");//文字列が入力できる．強制的に改行される．
ci("aa %d", i);//printfスタイルで文字も入力できる．
string a="aaaa";
ci(a);//stringも入力できる．
ci(Scalar(0, 255, 0), "=");//色を最初に指定すると色が出る．

if (key == 'l')
{
	isLine = (isLine) ? false : true;
	ci.setIsLineNumber(isLine);//行番号を表示するか否か．
}
if (key == 'p')
{
	ci.push();//表示状態を記憶する．（トラックバーで選べるようになる．）
	//.showメソッドよりも前に呼び出さないと，clearされた画像を表示してしまう．
}

ci.printData();//console窓に，printfする．表示内容をコピペしたいときなどに必須．

ci.show();
//ci.show(true);//デフォルト動作．画像表示後，表示画像をクリア
//ci.show(false);//表示画像を消さない．

ci.clear();//表示画像を消去．ただしこの呼び出しは，showでクリアされているため，不必要．show(false)だった場合はどこかでクリアが必要．
```

**発展**
* setFont(フォント名)で，表示するフォントを選べる．"Times New Roman"など，システムに入っているフォントなら表示可．
* setFontSize(size)でフォントサイズを指定可
* setLineSpaceSize(size)で行間を指定可

```cpp
class C_EXPORT ConsoleImage
{
priate:
	int count;
	std::string windowName;
	StackImage si;
	std::vector<std::string> strings;
	bool isLineNumber;
	std::string fontName;
	int fontSize;
	int lineSpaceSize;
pubic:
	void setFont(std::string fontName);
	void setFontSize(int size);
	void setLineSpaceSize(int size);
	void setIsLineNumber(bool isLine = true);
	bool getIsLineNumber();
	cv::Mat image;
	void init(cv::Size size, std::string wname);
	ConsoleImage();
	ConsoleImage(cv::Size size, std::string wname = "console");
	~ConsoleImage();
	void printData();
	void clear();
	void operator()(std::string str);
	void operator()(const char *format, ...);
	void operator()(cv::Scalar color, std::string str);
	void operator()(cv::Scalar color, const char *format, ...);
	
	void show(bool isClear = true);
	void push();
}
```