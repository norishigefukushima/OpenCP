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
		struct PlotInfo
		{
			std::vector<cv::Point2d> data;
			cv::Scalar color;
			int symbolType;
			int lineType;
			int thickness;

			std::string keyname;
		};
		std::vector<PlotInfo> pinfo;

		std::string xlabel;
		std::string ylabel;

		int data_max;

		cv::Scalar background_color;

		cv::Size plotsize;
		cv::Point origin;

		double xmin;
		double xmax;
		double ymin;
		double ymax;
		double xmax_no_margin;
		double xmin_no_margin;
		double ymax_no_margin;
		double ymin_no_margin;

		void init();
		void point2val(cv::Point pt, double* valx, double* valy);

		bool isZeroCross;
		bool isXYMAXMIN;
		bool isXYCenter;

		bool isPosition;
		cv::Scalar getPseudoColor(uchar val);
		cv::Mat plotImage;
		cv::Mat keyImage;
	public:
		//symbolType
		enum
		{
			SYMBOL_NOPOINT = 0,
			SYMBOL_PLUS,
			SYMBOL_TIMES,
			SYMBOL_ASTERRISK,
			SYMBOL_CIRCLE,
			SYMBOL_RECTANGLE,
			SYMBOL_CIRCLE_FILL,
			SYMBOL_RECTANGLE_FILL,
			SYMBOL_TRIANGLE,
			SYMBOL_TRIANGLE_FILL,
			SYMBOL_TRIANGLE_INV,
			SYMBOL_TRIANGLE_INV_FILL,
		};

		//lineType
		enum
		{
			LINE_NONE,
			LINE_LINEAR,
			LINE_H2V,
			LINE_V2H
		};

		cv::Mat render;
		cv::Mat graphImage;

		Plot(cv::Size window_size = cv::Size(1024, 768));
		~Plot();

		void setXYOriginZERO();
		void setXOriginZERO();
		void setYOriginZERO();

		void recomputeXYMAXMIN(bool isCenter = false, double marginrate = 0.9);
		void setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_);
		void setPlotImageSize(cv::Size s);
		void setXYMinMax(double xmin_, double xmax_, double ymin_, double ymax_);
		void setXMinMax(double xmin_, double xmax_);
		void setYMinMax(double ymin_, double ymax_);
		void setBackGoundColor(cv::Scalar cl);

		void makeBB(bool isFont);

		void setPlot(int plotnum, cv::Scalar color = COLOR_RED, int symboltype = SYMBOL_PLUS, int linetype = LINE_LINEAR, int thickness = 1);
		void setPlotThickness(int plotnum, int thickness_);
		void setPlotColor(int plotnum, cv::Scalar color);
		void setPlotSymbol(int plotnum, int symboltype);
		void setPlotLineType(int plotnum, int linetype);
		void setPlotKeyName(int plotnum, std::string name);

		void setPlotSymbolALL(int symboltype);
		void setPlotLineTypeALL(int linetype);

		void plotPoint(cv::Point2d = cv::Point2d(0.0, 0.0), cv::Scalar color = COLOR_BLACK, int thickness_ = 1, int linetype = LINE_LINEAR);
		void plotGrid(int level);
		void plotData(int gridlevel = 0, int isKey = 0);

		void plotMat(cv::InputArray src, std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");
		void plot(std::string name = "Plot", bool isWait = true, std::string gnuplotpath = "pgnuplot.exe");

		void makeKey(int num);

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
		std::vector<std::vector<double>> data;
		cv::Mat graphBase;
		int w;
		int h;
		void createPlot();
		void setMinMaxX(double minv, double maxv, int count);
		void setMinMaxY(double minv, double maxv, int count);
	public:
		cv::Mat show;
		cv::Mat graph;
		cv::Size size;
		double minx;
		double maxx;
		int countx;

		double miny;
		double maxy;
		int county;

		Plot2D(cv::Size graph_size, double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);

		void setMinMax(double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);
		void add(int x, int y, double val);
		void writeGraph(bool isColor, int arg_min_max, double minvalue = 0, double maxvalue = 0, bool isMinMaxSet = false);
		void setLabel(std::string namex, std::string namey);
		//void plot(CSV& result, vector<ExperimentalParameters>& parameters);
	};

	CP_EXPORT void plotGraph(cv::OutputArray graph, std::vector<cv::Point2d>& data, double xmin, double xmax, double ymin, double ymax,
		cv::Scalar color = COLOR_RED, int lt = Plot::SYMBOL_PLUS, int isLine = Plot::LINE_LINEAR, int thickness = 1, int ps = 4);
}
