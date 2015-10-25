Documents: Utility functions  
----------------------------

**void addNoise(Mat&src, Mat& dest, double sigma, double solt_peppar_rate)**
* Mat& src: input image  
* Mat& dest: output image  
* double sigma: sigma for Gaussian noise  
* double solt_peppar_rate: rate of solt-and-peppar noise  

**class  CalcTime**
timer class  

    class CalcTime
    {
	int64 pre;
	string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode();
	vector<string> lap_mes;
    public:
	void start();
	void setMode(int mode);
	void setMessage(string src);
	void restart();
	double getTime();
	void show();
	void show(string message);
	void lap(string message);
	void init(string message, int mode, bool isShow);

	CalcTime(string message, int mode=TIME_AUTO, bool isShow=true);
	CalcTime();
	~CalcTime();
    };

How to use  


**class Stat **
show stats 

    class Stat 
    {
    public:
	Vector<double> data;
	int num_data;
	Stat();
	~Stat();
	double getMin();
	double getMax();
	double getMean();
	double getStd();
	double getMedian();

	void push_back(double val);

	void clear();
	void show();
    };
