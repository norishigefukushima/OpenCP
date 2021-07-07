Timer.hpp
===================
時間計測用class

# class Timer
```cpp
class CP_EXPORT Timer
	{
		int64 pre = 0;
		std::string mes = "";
		std::string unit = "";
		int timeMode = 0;

		double cTime = 0.0;
		bool isShow = true;

		int autoMode = 0;
		int getAutoTimeMode();
		cp::Stat stat;

		int countIgnoringThreshold = 1;
		int countMax = 0;
		int countIndex = 0;

		void convertTime(bool isShow, std::string message);
	public:

		void init(std::string message, int mode, bool isShow);

		void setMode(int mode);
		void setMessage(std::string& src);
		void setIsShow(const bool flag);

		void start();//call getTickCount();
		void clearStat();//clear Stat

		void setCountMax(const int value);//set ring buffer max (loop value max) for Stat. Default is infinity.
		void setIgnoringThreshold(const int value);//if(sample number < value) does not push the value into Stat for ignure cache effect
		double getTime(bool isPrint = false, std::string message = "");//only getTickCount()
		double getpushLapTime(bool isPrint = false, std::string message = "");//getTickCount() and push the value to Stat
		double getLapTimeMedian(bool isPrint = false, std::string message = "");//get median value from Stat
		double getLapTimeMean(bool isPrint = false, std::string message = "");//get mean value from Stat
		std::string getUnit();//return string unit
		int getStatSize();//get the size of Stat
		void drawDistribution(std::string wname = "Stat distribution", int div = 100);
		void drawDistribution(std::string wname, int div, double minv, double maxv);

		Timer(std::string message, int mode = TIME_AUTO, bool isShow = true);
		Timer(char* message, int mode = TIME_AUTO, bool isShow = true);
		Timer();

		~Timer();
	};
```

## Usage
### 初期化
コンストラクタは以下の３つ．

```cpp
Timer(std::string message, int mode = TIME_AUTO, bool isShow = true);
Timer(char* message, int mode = TIME_AUTO, bool isShow = true);
Timer();
```

* messageにcoutするときの先頭メッセージを記述
* modeは表示時間のミリ秒や秒などの設定方法．
* isShowのフラグでコンソールにprintfするかどうかを制御．

表示時間を`mode`に指定することで以下のように選べる．
コンストラクタのデフォルトは`TIME_AUTO`で，表示の桁がちょうどよくなるように自動的に選ぶ．
引数なしはmessageに"time "を入れて，あとはデフォルト引数と同じ．

```cpp
	enum
	{
		TIME_AUTO = 0,
		TIME_NSEC,
		TIME_MICROSEC,
		TIME_MSEC,
		TIME_SEC,
		TIME_MIN,
		TIME_HOUR,
		TIME_DAY
	};
```

なお後からセットも可能
```cpp
void setMode(int mode);
void setMessage(std::string& src);
void setIsShow(const bool flag);
```

### 実行（４パターン）
#### パターン１
1. スコープにはさんで，オブジェクトが消滅するまでの時間を図る．{Timer t; hoge...}

#### パターン２
1. `start()`メソッドでタイマーのスタート，(オブジェクトの生成と同時に自動的にスタート)
2. `getTime`メソッドで時刻を取得

#### パターン３
1. `getTime`メソッドで時刻を取得(オブジェクトの生成と同時にタイマーは自動的にスタート)

#### パターン４
1. `start()`メソッドでタイマーのスタート，(オブジェクトの生成と同時に自動的にスタート)
2.`getpushLapTime`メソッドで時刻を取得し，内部のStatクラスにpush
3. `getLapTimeMean()`か`getLapTimeMedian()`メソッドで，Statにpushされた値の平均か中央値を出力
4. Statをクリア（必要であれば）

なお，メディアン値や平均値を取るの値がどれだけpushされたかは，`int getStatSize();`で取得可能．
また，Statに入れる値は以下メソッドで制御できる．

```cpp
void setCountMax(const int value);//set ring buffer max (loop value max) for Stat. Default is infinity.
void setIgnoringThreshold(const int value);//if(sample number < value) does not push the value into Stat for ignure cache 
```

* `setCountMax`で有限の値をリングバッファで確保し，一定区間の統計情報を取得可能．デフォルトは無限区間
* `setIgnoringThreshold`ではじめvalue個分だけpushせずに無視するダミーループを設定可能．
例えば，setCountMax(100);とたら，直近１００個のpushされた値だけから統計値を取得する．  
またsetIgnoringThreshold（10);としたら，先頭１０個の値は統計値としてカウントしない．  

なお，下記メソッドで計算時間の分布が表示可能．安定してない場合に可視化して確認するとよい．
```cpp
void drawDistribution(std::string wname = "Stat distribution", int div = 100);//compute min max and then divide interval by div
void drawDistribution(std::string wname, int div, double minv, double maxv);//set min max and then divide interval by div
```

### サンプル
```cpp
	//Sample 1: compute time by using constructor and destructor for scope
	{	//must use scope
	Timer t;
	//some function
	}

	//Sample 2: manually start timer
	Timer t;
	t.start()
	//some function
	t.getTime()

	//Sample 3: skip start timer (start function is called by constructor) 
	Timer t;
	//some function
	t.getTime()

	//Sample 4: compute mean or median value of trails
	Timer t;
	for(int i=0;i<loop;i++)
	{
		t.start()
		//some function
		t.getpushLapTime()
	}
	t.getLapTimeMean();
	t.getLapTimeMedian();
	t.clearStat();//clear stat and then re-compute computing time
	for(int i=0;i<loop;i++)
	{
		t.start()
		//some function
		t.getpushLapTime()
	}
	t.getLapTimeMean();
	t.getLapTimeMedian();
```

# class DestinationTimePrediction
```cpp
```