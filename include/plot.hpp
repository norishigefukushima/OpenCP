#pragma once

#include "common.hpp"

namespace cp
{
	class CP_EXPORT gnuplot
	{
		FILE* fp;
	public:
		gnuplot(std::string gnuplotpath);
		void cmd(std::string name);
		~gnuplot();
	};

	class CP_EXPORT Plot
	{
	protected:
		std::string font = "Times New Roman";//"Consolas"
		int fontSize = 20;
		int fontSize2 = 18;
		int foregroundIndex = 0;
		struct PlotInfo
		{
			std::vector<cv::Point2d> data;
			cv::Scalar color;
			int symbolType;
			int lineType;
			int lineWidth;

			std::string title;
		};
		std::vector<PlotInfo> pinfo;

		std::string xlabel = "x";
		std::string ylabel = "y";

		int data_max = 1;

		cv::Scalar background_color = COLOR_WHITE;
		int gridLevel = 0;
		bool isDrawMousePosition = true;

		cv::Size plotsize;
		cv::Point origin;


		bool isSetXRange = false;
		bool isSetYRange = false;
		double xmin_plotwindow;//x max of plot window
		double xmax_plotwindow;//x min of plot window
		double ymin_plotwindow;//y max of plot window
		double ymax_plotwindow;//y min of plot window
		double xmax_data;//x max of data
		double xmin_data;//x min of data
		double ymax_data;//y max of data
		double ymin_data;//y min of data

		int keyPosition = RIGHT_TOP;

		bool isZeroCross;
		bool isXYMAXMIN;
		bool isXYCenter;

		bool isLogScaleX = false;
		bool isLogScaleY = false;

		cv::Mat plotImage;
		cv::Mat keyImage;

		void init(cv::Size imsize);
		void point2val(cv::Point pt, double* valx, double* valy);
		cv::Scalar getPseudoColor(uchar val);
		void plotGrid(int level);
		void renderingOutsideInformation(bool isFont);
		void computeDataMaxMin();
		void computeWindowXRangeMAXMIN(bool isCenter = false, double margin_rate = 0.9, int rounding_value = 0);
		void computeWindowYRangeMAXMIN(bool isCenter = false, double margin_rate = 0.9, int rounding_value = 0);
		
	public:
		void recomputeXYRangeMAXMIN(bool isCenter = false, double margin_rate = 0.9, int rounding_balue = 0);
		//symbolType
		enum SYMBOL
		{
			NOPOINT = 0,
			PLUS,
			TIMES,
			ASTERISK,
			CIRCLE,
			RECTANGLE,
			CIRCLE_FILL,
			RECTANGLE_FILL,
			TRIANGLE,
			TRIANGLE_FILL,
			TRIANGLE_INV,
			TRIANGLE_INV_FILL,
			DIAMOND,
			DIAMOND_FILL,
			PENTAGON,
			PENTAGON_FILL,
		};

		enum LINE
		{
			NOLINE,
			LINEAR,
			H2V,
			V2H,

			LINE_METHOD_SIZE
		};

		enum KEY
		{
			NOKEY,
			RIGHT_TOP,
			LEFT_TOP,
			LEFT_BOTTOM,
			RIGHT_BOTTOM,
			FLOATING,

			KEY_METHOD_SIZE
		};

		cv::Mat render;
		cv::Mat graphImage;

		Plot(cv::Size window_size);//cv::Size(1024, 768)
		Plot();
		~Plot();

		void setXYOriginZERO();
		void setXOriginZERO();
		void setYOriginZERO();


		void setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_);
		void setImageSize(cv::Size s);
		void setXYRange(double xmin, double xmax, double ymin, double ymax);
		void setXRange(double xmin, double xmax);
		void setYRange(double ymin, double ymax);
		void setLogScaleX(const bool flag);
		void setLogScaleY(const bool flag);

		
		//NOKEY,
		//RIGHT_TOP,
		//LEFT_TOP,
		//LEFT_BOTTOM,
		//RIGHT_BOTTOM,
		//FLOATING,
		void setKey(int key_method);
		void setXLabel(std::string xlabel);
		void setYLabel(std::string ylabel);
		void setGrid(int level = 0);//0: no grid, 1: div 4, 2: div 16
		void setBackGoundColor(cv::Scalar cl);

		void setPlot(int plotnum, cv::Scalar color = COLOR_RED, int symbol_type = PLUS, int line_type = LINEAR, int line_width = 1);
		void setPlotLineWidth(int plotnum, int line_width);
		void setPlotLineWidthALL(int line_width);
		void setPlotColor(int plotnum, cv::Scalar color);

		//NOPOINT = 0,
		//PLUS,
		//TIMES,
		//ASTERISK,
		//CIRCLE,
		//RECTANGLE,
		//CIRCLE_FILL,
		//RECTANGLE_FILL,
		//TRIANGLE,
		//TRIANGLE_FILL,
		//TRIANGLE_INV,
		//TRIANGLE_INV_FILL,
		//DIAMOND,
		//DIAMOND_FILL,
		//PENTAGON,
		//PENTAGON_FILL,
		void setPlotSymbol(int plotnum, int symbol_type);
		void setPlotSymbolALL(int symbol_type);

		//Plot::LINE::NONE,
		//Plot::LINE::LINEAR,
		//Plot::LINE::H2V,
		//Plot::LINE::V2H
		void setPlotLineType(int plotnum, int line_type);
		void setPlotLineTypeALL(int line_type);

		void setPlotTitle(int plotnum, std::string name);
		void setPlotForeground(int plotnum);

		void setIsDrawMousePosition(const bool flag);
		

		void plotPoint(cv::Point2d = cv::Point2d(0.0, 0.0), cv::Scalar color = COLOR_BLACK, int thickness_ = 1, int linetype = LINEAR);

		void plotData(int gridlevel = 0);

		void plotMat(cv::InputArray src, std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");
		void plot(std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe", std::string message = "");

		void generateKeyImage(int num);

		void save(std::string name);

		void push_back(std::vector<cv::Point> point, int plotIndex = 0);
		void push_back(std::vector<cv::Point2d> point, int plotIndex = 0);
		void push_back(double x, double y, int plotIndex = 0);

		void erase(int sampleIndex, int plotIndex = 0);
		void insert(cv::Point2d v, int sampleIndex, int plotIndex = 0);
		void insert(cv::Point v, int sampleIndex, int plotIndex = 0);
		void insert(double x, double y, int sampleIndex, int plotIndex = 0);

		void clear(int datanum = -1);

		void swapPlot(int plotIndex1, int plotIndex2);
	};

	enum
	{
		PLOT_ARG_MAX = 1,
		PLOT_ARG_MIN = -1
	};
	class CP_EXPORT Plot2D
	{
		cv::Scalar background_color = cv::Scalar(255, 255, 255, 0);
		std::string font = "Times New Roman";//"Consolas"
		int fontSize = 20;
		int fontSize2 = 18;

		std::vector<std::vector<double>> data;
		cv::Mat labelxImage;
		cv::Mat labelyImage;
		cv::Mat gridData;
		int w;
		int h;
		void createPlot();
		void setMinMaxX(double minv, double maxv, double interval);
		void setMinMaxY(double minv, double maxv, double interval);
		void addLabelToGraph();
	public:
		std::string labelx="";
		std::string labely="";
		cv::Mat show;
		cv::Mat graph;
		cv::Size plotImageSize;
		double x_min;
		double x_max;
		double x_interval;
		int x_size;

		double y_min;
		double y_max;
		double y_interval;
		int y_size;

		Plot2D(cv::Size graph_size, double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);

		void setFont(std::string font);
		void setFontSize(const int size);
		void setFontSize2(const int size);
		void setMinMax(double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);
		void add(int x, int y, double val);
		void writeGraph(bool isColor, int arg_min_max, double minvalue = 0, double maxvalue = 0, bool isMinMaxSet = false);
		void setLabel(std::string namex, std::string namey);
		
		//void plot(CSV& result);
		void plot(std::string wname = "plot2D");
	};

	CP_EXPORT void plotGraph(cv::OutputArray graphImage, std::vector<cv::Point2d>& data, double xmin, double xmax, double ymin, double ymax,
		cv::Scalar color = COLOR_RED, int lt = Plot::PLUS, int isLine = Plot::LINEAR, int thickness = 1, int ps = 4, bool isLogX = false, bool isLogY = false);

	class CP_EXPORT RGBHistogram
	{
	private:
		cv::Size size = cv::Size(512, 512);
		cv::Mat k = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);

		void projectPointsParallel(const cv::Mat& xyz, const cv::Mat& R, const cv::Mat& t, const cv::Mat& K, std::vector<cv::Point2f>& dest, const bool isRotationThenTranspose);
		void projectPoints(const cv::Mat& xyz, const cv::Mat& R, const cv::Mat& t, const cv::Mat& K, std::vector<cv::Point2f>& dest, const bool isRotationThenTranspose);
		void projectPoint(cv::Point3d& xyz, const cv::Mat& R, const cv::Mat& t, const cv::Mat& K, cv::Point2d& dest);


		void convertRGBto3D(cv::Mat& src, cv::Mat& rgb);
		cv::Mat additionalPoints;
		cv::Mat additionalPointsDest;
		cv::Mat center;
	public:
		RGBHistogram();
		void setCenter(cv::Mat& src);
		void push_back(cv::Mat& src);
		void push_back(cv::Vec3f src);
		void push_back(const float b, const float g, const float r);


		void clear();
		void plot(cv::Mat& src, bool isWait = true, std::string wname = "RGB histogram");
	};
}
