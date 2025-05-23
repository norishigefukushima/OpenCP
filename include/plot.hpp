#pragma once

#include "common.hpp"

namespace cp
{
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
			int parametricLine;//0: default, 1: horizontal line, 2: vertical line
		};
		std::vector<PlotInfo> pinfo;
		std::vector<cv::Point2d> ymaxpt;
		std::vector<cv::Point2d> yminpt;

		std::string xlabel = "x";
		std::string ylabel = "y";
		std::string xlabel_subscript = "";
		std::string ylabel_subscript = "";
		bool isLabelXGreekLetter = false;
		bool isLabelYGreekLetter = false;

		int data_labelmax = 1;

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

		bool isWideKey = false;
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
		void unsetXYRange();
		void setXRange(double xmin, double xmax);
		void unsetXRange();
		void setYRange(double ymin, double ymax);
		void unsetYRange();
		void setLogScaleX(const bool flag);
		void setLogScaleY(const bool flag);

		/*enum KEY
		{
			cp::Plot::NOKEY,
			cp::Plot::RIGHT_TOP,
			cp::Plot::LEFT_TOP,
			cp::Plot::LEFT_BOTTOM,
			cp::Plot::RIGHT_BOTTOM,
			cp::Plot::FLOATING,

			cp::Plot::KEY_METHOD_SIZE
		};
		*/
		void setKey(const KEY key_method);
		void setWideKey(bool flag);
		void setXLabel(std::string xlabel);
		void setYLabel(std::string ylabel);
		void setXLabelGreekLetter(std::string greeksymbol, std::string subscript);
		void setYLabelGreekLetter(std::string greeksymbol, std::string subscript);

		void setGrid(int level = 0);//0: no grid, 1: div 4, 2: div 16
		void setBackGoundColor(cv::Scalar cl);

		void setPlot(int plotnum, cv::Scalar color = COLOR_RED, int symbol_type = PLUS, int line_type = LINEAR, int line_width = 1);
		void setPlotLineWidth(int plotnum, int line_width);
		void setPlotLineWidthALL(int line_width);
		void setPlotColor(int plotnum, cv::Scalar color);

		
		//Plot::SYMBOL::NOPOINT = 0,
		//Plot::SYMBOL::PLUS,
		//Plot::SYMBOL::TIMES,
		//Plot::SYMBOL::ASTERISK,
		//Plot::SYMBOL::CIRCLE,
		//Plot::SYMBOL::RECTANGLE,
		//Plot::SYMBOL::CIRCLE_FILL,
		//Plot::SYMBOL::RECTANGLE_FILL,
		//Plot::SYMBOL::TRIANGLE,
		//Plot::SYMBOL::TRIANGLE_FILL,
		//Plot::SYMBOL::TRIANGLE_INV,
		//Plot::SYMBOL::TRIANGLE_INV_FILL,
		//Plot::SYMBOL::DIAMOND,
		//Plot::SYMBOL::DIAMOND_FILL,
		//Plot::SYMBOL::PENTAGON,
		//Plot::SYMBOL::PENTAGON_FILL,
		void setPlotSymbol(int plotnum, int symbol_type);
		//Plot::SYMBOL::NOPOINT = 0,
		//Plot::SYMBOL::PLUS,
		//Plot::SYMBOL::TIMES,
		//Plot::SYMBOL::ASTERISK,
		//Plot::SYMBOL::CIRCLE,
		//Plot::SYMBOL::RECTANGLE,
		//Plot::SYMBOL::CIRCLE_FILL,
		//Plot::SYMBOL::RECTANGLE_FILL,
		//Plot::SYMBOL::TRIANGLE,
		//Plot::SYMBOL::TRIANGLE_FILL,
		//Plot::SYMBOL::TRIANGLE_INV,
		//Plot::SYMBOL::TRIANGLE_INV_FILL,
		//Plot::SYMBOL::DIAMOND,
		//Plot::SYMBOL::DIAMOND_FILL,
		//Plot::SYMBOL::PENTAGON,
		//Plot::SYMBOL::PENTAGON_FILL,
		void setPlotSymbolALL(int symbol_type);

		//Plot::LINE::NOLINE,
		//Plot::LINE::LINEAR,
		//Plot::LINE::H2V,
		//Plot::LINE::V2H
		void setPlotLineType(int plotnum, int line_type);
		void setPlotLineTypeALL(int line_type);

		void setPlotTitle(int plotnum, std::string name);
		void setPlotForeground(int plotnum);
		void setFontSize(int fontSize = 20);
		void setFontSize2(int fontSize2 = 18);
		void setIsDrawMousePosition(const bool flag);


		void plotPoint(cv::Point2d = cv::Point2d(0.0, 0.0), cv::Scalar color = COLOR_BLACK, int thickness_ = 1, int linetype = LINEAR);

		void plotData(int gridlevel = 0);

		void plotMat(cv::InputArray src, const std::string name = "Plot", bool isWait = true, const std::string gnuplotpath = "gnuplot.exe");
		void plot(const std::string name = "Plot", bool isWait = true, const std::string gnuplotpath = "C:/bin/gnuplot/bin/gnuplot.exe", const std::string message = "");

		void generateKeyImage(int num, bool isWideKey);

		void saveDatFile(const std::string name, const bool isPrint = true);

		void push_back(std::vector<float>& x, std::vector<float>& y, int plotIndex = 0);
		void push_back(std::vector<double>& x, std::vector<double>& y, int plotIndex = 0);
		void push_back(std::vector<cv::Point>& points, int plotIndex = 0);
		void push_back(std::vector<cv::Point2f>& points, int plotIndex = 0);
		void push_back(std::vector<cv::Point2d>& points, int plotIndex = 0);
		void push_back(cv::Mat& points, int plotIndex = 0);
		void push_back(double x, double y, int plotIndex = 0);
		void push_back_HLine(double y, int plotIndex);
		void push_back_VLine(double x, int plotIndex);

		void erase(int sampleIndex, int plotIndex = 0);
		void insert(cv::Point2d v, int sampleIndex, int plotIndex = 0);
		void insert(cv::Point v, int sampleIndex, int plotIndex = 0);
		void insert(double x, double y, int sampleIndex, int plotIndex = 0);

		void clear(int datanum = -1);

		void swapPlot(int plotIndex1, int plotIndex2);
	};

	class CP_EXPORT GNUPlot
	{
		bool isShowCMD;
		FILE* fp;
		enum class PlotSize
		{
			TEXT_WIDTH,
			COLUMN_WIDTH,
			COLUMN_WIDTH_HALF
		};
	public:
		GNUPlot(const std::string gnuplotpath, const bool isShowCommand = true);
		void setKey(Plot::KEY pos, const bool align_right = false, const double spacing = 1.0, const int width_offset = 0, const int height_offset = 0);
		void setXLabel(const std::string label);
		void setYLabel(const std::string label);

		void plotPDF(std::string plot_command, PlotSize mode = PlotSize::COLUMN_WIDTH, const std::string font = "CMU Serif", const int fontSize = 10, bool isCrop = true);
		void cmd(const std::string name);
		~GNUPlot();
	};

	class CP_EXPORT Plot2D
	{
		cv::Size plotImageSize;
		cv::Scalar background_color = cv::Scalar(255, 255, 255, 0);
		std::vector<cv::Scalar> colorIndex;

		std::string font = "Times New Roman";
		//std::string font = "Computer Modern";
		//std::string font = "Consolas";
		int fontSize = 20;
		int fontSize2 = 18;

		void createPlot();
		void setYMinMax(double minv, double maxv, double interval);
		void setXMinMax(double minv, double maxv, double interval);

		double x_min;
		double x_max;
		double x_interval;
		int x_size;

		double y_min;
		double y_max;
		double y_interval;
		int y_size;

		double z_min = 0.0;
		double z_max = 0.0;

		bool isSetZMinMax = false;
		bool isLabelXGreekLetter = false;
		bool isLabelYGreekLetter = false;
		bool isLabelZGreekLetter = false;

		cv::Point z_min_point;
		cv::Point z_max_point;
		cv::Mat gridData;
		cv::Mat gridDataRes;
		int colormap = 20;
		cv::Mat graph;
		void plotGraph(bool isColor, cv::Mat& graph);

		std::vector<std::string> contourLabels;
		std::vector<double> contourThresh;
		void drawContoursZ(double threth, cv::Scalar color, int lineWidth);

		cv::Mat barImage;
		int barWidth = 25;
		int barSpace = 5;

		int keyState = 1;
		cv::Mat keyImage;
		void generateKeyImage(int lineWidthBB = 1, int lineWidthKey = 2);

		std::string labelx = "x";
		std::string labelx_subscript = "";
		std::string labely = "y";
		std::string labely_subscript = "";
		std::string labelz = "z";
		std::string labelz_subscript = "";
		cv::Mat labelxImage;
		cv::Mat labelyImage;
		cv::Mat labelzImage;
		void addLabelToGraph(bool isDrawingContour, int keyState, bool isPlotMin, bool isPlotMax);
		bool isPlotMax = false;
		bool isPlotMin = false;
		//bool isPlotMin = true;
		int maxColorIndex = 0;
		int minColorIndex = 0;
		int plot_x = 0;
		int plot_y = 0;
	public:

		Plot2D(cv::Size graph_size, double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);
		void add(double x, double y, double val);
		void addIndex(int x, int y, double val);
		cv::Mat show;
		void plot(std::string wname = "plot2D");

		void setFont(std::string font);
		void setFontSize(const int size);
		void setFontSize2(const int size);
		void setLabel(std::string namex, std::string namey, std::string namez);
		void setZMinMax(double minv, double maxv);
		void setPlotContours(std::string label, double thresh, int index);
		void setPlotMaxMin(bool plot_max, bool plot_min);
		void setLabelXGreekLetter(std::string greeksymbol, std::string subscript);
		void setLabelYGreekLetter(std::string greeksymbol, std::string subscript);
		void setLabelZGreekLetter(std::string greeksymbol, std::string subscript);
		void setMinMax(double xmin, double xmax, double xstep, double ymin, double ymax, double ystep);
		cv::Mat getGridData();
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
		cv::Mat additionalStartLines;
		cv::Mat additionalStartLinesDest;
		cv::Mat additionalEndLines;
		cv::Mat additionalEndLinesDest;
		cv::Mat additionalPoints;
		cv::Mat additionalPointsDest;
		cv::Mat center;
	public:
		RGBHistogram();
		void setCenter(cv::Mat& src);
		void push_back(cv::Mat& src);
		void push_back(cv::Vec3f src);
		void push_back(const float b, const float g, const float r);
		void push_back_line(cv::Mat& src, cv::Mat& dest);
		void push_back_line(cv::Vec3f src, cv::Vec3f dest);
		void push_back_line(const float b_s, const float g_s, const float r_s, const float b_d, const float g_d, const float r_d);

		void clear();
		void plot(cv::Mat& src, bool isWait = true, std::string wname = "RGB histogram");
	};
}
