Stat.hpp
===================
統計情報を計算するclass

# class Timer
doubleでキャスト可能な数値をpush_backで入れて，その合計値，平均値，中央値，最大値，最小値，分散，標準偏差を計算するクラス．

```cpp
class CP_EXPORT Stat
	{
	public:
		std::vector<double> data;
		int getSize();//get data size
		double getSum();
		double getMin();
		double getMax();
		double getMean();
		double getVar();
		double getStd();
		double getMedian();

		void pop_back();
		void push_back(double val);

		void clear();//clear data
		void print();//print all stat
		void drawDistribution(std::string wname = "Stat distribution", int div = 100);
		void drawDistribution(std::string wname, int div, double minv, double maxv);
	};
```

## Usage
以下のように．データをpush_backして，値を取得する．
最後に入れたものはpop_backで削除も可能．
```cpp
Stat st;
st.push_back(1);
st.push_back(2);
st.push_back(3);
st.pop_back();//pop

cout<<st.getMean()<<endl;
cout<<st.getMedian()<<endl;
```

取れるデータは以下のもの．
```cpp
		int getSize();//get data size
		double getSum();
		double getMin();
		double getMax();
		double getMean();
		double getVar();
		double getStd();
		double getMedian();
```

* `drawDistribution`メソッドで，中の統計情報のヒストグラムを描画可能．
* `print`メソッドは，すべての統計情報をprintfする．
* `clear`によりすべての情報をクリア