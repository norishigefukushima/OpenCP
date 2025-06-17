#include "plot.hpp"
#include "color.hpp"
#include "draw.hpp"
#include "stereo_core.hpp" //for RGB histogram
#include "pointcloud.hpp" //for RGB histogram
#include "highguiex.hpp"
#include "contrast.hpp"

#include "debugcp.hpp"

using namespace std;
using namespace cv;

namespace cp
{
#pragma region gnuplot
	GNUPlot::GNUPlot(string gnuplotpath, const bool isShowCommand)
	{
		isShowCMD = isShowCommand;
		if ((fp = _popen(gnuplotpath.c_str(), "w")) == NULL)
		{
			fprintf(stderr, "Cannot open gnuplot @ %s\n", gnuplotpath.c_str());
			exit(1);
		}
	}

	void GNUPlot::setKey(Plot::KEY pos, const bool align_right, const double spacing, const int width_offset, const int height_offset)
	{
		string command = "";

		string set_key = "set key ";
		string space = " spacing " + to_string(spacing);
		string align = (align_right) ? " Right " : " Left ";
		string wh_offset = " width " + to_string(width_offset) + " height " + to_string(height_offset);
		switch (pos)
		{
		case cp::Plot::KEY::NOKEY:
			command = "unset key";
			break;
		case cp::Plot::KEY::RIGHT_TOP:
			command = set_key + "right top" + align + space + wh_offset;
			break;
		case cp::Plot::KEY::LEFT_TOP:
			command = set_key + "left top" + align + space + wh_offset;
			break;
		case cp::Plot::KEY::LEFT_BOTTOM:
			command = set_key + "left bottom" + align + space + wh_offset;
			break;
		case cp::Plot::KEY::RIGHT_BOTTOM:
			command = set_key + "right bottom" + align + space + wh_offset;
			break;
		case cp::Plot::KEY::FLOATING:
			break;
		default:
			break;
		}
		cmd(command);
	}

	void GNUPlot::setXLabel(const string label)
	{
		cmd("set xlabel '" + label + "'");
	}

	void GNUPlot::setYLabel(const string label)
	{
		cmd("set ylabel '" + label + "'");
	}

	void GNUPlot::plotPDF(string plot_command, PlotSize mode, string font, const int fontSize, bool isCrop)
	{
		string command = "";
		string terminal = "set terminal pdf enhanced font '" + font + "' size ";

		switch (mode)
		{
		case cp::GNUPlot::PlotSize::TEXT_WIDTH:
			command = terminal + "1.85,1.3";
			break;
		case cp::GNUPlot::PlotSize::COLUMN_WIDTH:
			command = terminal + "3.6,2.6";
			break;
		case cp::GNUPlot::PlotSize::COLUMN_WIDTH_HALF:
			command = terminal + "1.85,1.3";
			break;
		default:
			break;
		}
		cmd(command);
		cmd("set output 'out.pdf'");
		cmd(plot_command);
		cmd("set output");
		//system()
	}

	void GNUPlot::cmd(string name)
	{
		if (isShowCMD)cout << name << endl;
		fprintf(fp, name.c_str());
		fprintf(fp, "\n");
		fflush(fp);
	}

	GNUPlot::~GNUPlot()
	{
		cmd("exit");
		_pclose(fp);
	}

#pragma endregion

	template<int lt>
	void plotGraph_(OutputArray graphImage_, vector<Point2d>& data, double xmin, double xmax, double ymin, double ymax,
		Scalar color, int isLine, int thickness, int pointSize, bool isLogX, bool isLogY)
	{
		CV_Assert(!graphImage_.empty());
		const int ps = pointSize;

		Mat graph = graphImage_.getMat();
		if (isLogX)
		{
			xmax = log10(max(1.0, xmax));
			xmin = log10(max(1.0, xmin));
		}
		if (isLogY)
		{
			ymax = log10(max(1.0, ymax));
			ymin = log10(max(1.0, ymin));
		}
		//cout << xmin << "," << xmax << endl;
		const double x_step = (double)(graph.cols - 1) / (xmax - xmin);
		const double y_step = (double)(graph.rows - 1) / (ymax - ymin);

		int H = graph.rows - 1;
		const int size = (int)data.size();

#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			const double x = isLogX ? log10(max(1.0, data[i].x)) : data[i].x;
			const double y = isLogY ? log10(max(1.0, data[i].y)) : data[i].y;
			cv::Point p = Point(cvRound(x_step * (x - xmin)), H - cvRound(y_step * (y - ymin)));

			if (isLine == Plot::NOLINE)
			{
				;
			}
			else if (i != size - 1)
			{
				double xn = data[i + 1].x;
				double yn = data[i + 1].y;

				if (isLogX) xn = log10(max(1.0, xn));
				if (isLogY) yn = log10(max(1.0, yn));
				if (isLine == Plot::LINEAR)
				{
					line(graph, p, Point(cvRound(x_step * (xn - xmin)), H - cvRound(y_step * (yn - ymin))), color, thickness);
				}
				else if (isLine == Plot::H2V)
				{
					line(graph, p, Point(cvRound(x_step * (xn - xmin)), p.y), color, thickness);
					line(graph, Point(cvRound(x_step * (xn - xmin)), p.y), Point(cvRound(x_step * (xn - xmin)), H - cvRound(y_step * (yn - ymin))), color, thickness);
				}
				else if (isLine == Plot::V2H)
				{
					line(graph, p, Point(p.x, H - cvRound(y_step * (yn - ymin))), color, thickness);
					line(graph, Point(p.x, H - cvRound(y_step * (yn - ymin))), Point(cvRound(x_step * (xn - xmin)), H - cvRound(y_step * (yn - ymin))), color, thickness);
				}
			}


			if constexpr (lt == Plot::NOPOINT)
			{
				;
			}
			else if constexpr (lt == Plot::PLUS)
			{
				drawPlus(graph, p, 2 * ps + 1, color, thickness);
			}
			else if constexpr (lt == Plot::TIMES)
			{
				drawTimes(graph, p, 2 * ps + 1, color, thickness);
			}
			else if constexpr (lt == Plot::ASTERISK)
			{
				drawAsterisk(graph, p, 2 * ps + 1, color, thickness);
			}
			else if constexpr (lt == Plot::CIRCLE)
			{
				circle(graph, p, ps, color, thickness);
			}
			else if constexpr (lt == Plot::RECTANGLE)
			{
				rectangle(graph, Point(p.x - ps, p.y - ps), Point(p.x + ps, p.y + ps),
					color, thickness);
			}
			else if constexpr (lt == Plot::CIRCLE_FILL)
			{
				circle(graph, p, ps, color, FILLED);
			}
			else if constexpr (lt == Plot::RECTANGLE_FILL)
			{
				rectangle(graph, Point(p.x - ps, p.y - ps), Point(p.x + ps, p.y + ps),
					color, FILLED);
			}
			else if constexpr (lt == Plot::TRIANGLE)
			{
				triangle(graph, p, 2 * ps, color, thickness);
			}
			else if constexpr (lt == Plot::TRIANGLE_FILL)
			{
				triangle(graph, p, 2 * ps, color, FILLED);
			}
			else if constexpr (lt == Plot::TRIANGLE_INV)
			{
				triangleinv(graph, p, 2 * ps, color, thickness);
			}
			else if constexpr (lt == Plot::TRIANGLE_INV_FILL)
			{
				triangleinv(graph, p, 2 * ps, color, FILLED);
			}
			else if constexpr (lt == Plot::DIAMOND)
			{
				diamond(graph, p, 2 * ps, color, thickness);
			}
			else if constexpr (lt == Plot::DIAMOND_FILL)
			{
				diamond(graph, p, 2 * ps, color, FILLED);
			}
			else if constexpr (lt == Plot::PENTAGON)
			{
				pentagon(graph, p, 2 * ps, color, thickness);
			}
			else if constexpr (lt == Plot::PENTAGON_FILL)
			{
				pentagon(graph, p, 2 * ps, color, FILLED);
			}
		}
	}

	void plotGraph(OutputArray graphImage, vector<Point2d>& data, double xmin, double xmax, double ymin, double ymax,
		Scalar color, int lt, int isLine, int thickness, int pointSize, bool isLogX, bool isLogY)
	{
		switch (lt)
		{
		case Plot::NOPOINT: plotGraph_<Plot::NOPOINT>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::PLUS: plotGraph_<Plot::PLUS>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::TIMES: plotGraph_<Plot::TIMES>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::ASTERISK: plotGraph_<Plot::ASTERISK>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::CIRCLE: plotGraph_<Plot::CIRCLE>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::RECTANGLE: plotGraph_<Plot::RECTANGLE>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::CIRCLE_FILL: plotGraph_<Plot::CIRCLE_FILL>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::RECTANGLE_FILL: plotGraph_<Plot::RECTANGLE_FILL>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::TRIANGLE: plotGraph_<Plot::TRIANGLE>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::TRIANGLE_FILL: plotGraph_<Plot::TRIANGLE_FILL>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::TRIANGLE_INV: plotGraph_<Plot::TRIANGLE_INV>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::TRIANGLE_INV_FILL: plotGraph_<Plot::TRIANGLE_INV_FILL>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::DIAMOND: plotGraph_<Plot::DIAMOND>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::DIAMOND_FILL: plotGraph_<Plot::DIAMOND_FILL>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::PENTAGON: plotGraph_<Plot::PENTAGON>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		case Plot::PENTAGON_FILL: plotGraph_<Plot::PENTAGON_FILL>(graphImage, data, xmin, xmax, ymin, ymax, color, isLine, thickness, pointSize, isLogX, isLogY); break;
		default:
			break;
		}
	}

	void plotHLine(OutputArray graphImage_, Point2d data, double xmin, double xmax, double ymin, double ymax,
		Scalar color, int lt, int isLine, int thickness, int pointSize, bool isLogX, bool isLogY)
	{
		CV_Assert(!graphImage_.empty());
		const int ps = pointSize;

		Mat graph = graphImage_.getMat();
		if (isLogX)
		{
			xmax = log10(max(1.0, xmax));
			xmin = log10(max(1.0, xmin));
		}
		if (isLogY)
		{
			ymax = log10(max(1.0, ymax));
			ymin = log10(max(1.0, ymin));
		}
		//cout << xmin << "," << xmax << endl;
		const double x_step = (double)(graph.cols - 1) / (xmax - xmin);
		const double y_step = (double)(graph.rows - 1) / (ymax - ymin);

		int H = graph.rows - 1;

		double x = data.x;
		double y = data.y;
		if (isLogX) x = log10(max(1.0, x));
		if (isLogY) y = log10(max(1.0, y));

		cv::Point point_st = Point(0, H - cvRound(y_step * (y - ymin)));
		cv::Point point_ed = Point(graph.cols - 1, H - cvRound(y_step * (y - ymin)));
		{
			if (isLine == Plot::LINEAR)
			{
				line(graph, point_st, point_ed, color, thickness);
			}
		}

		if (lt == Plot::NOPOINT)
		{
			;
		}
		else if (lt == Plot::PLUS)
		{
			drawPlus(graph, point_st, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::TIMES)
		{
			drawTimes(graph, point_st, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::ASTERISK)
		{
			drawAsterisk(graph, point_st, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::CIRCLE)
		{
			circle(graph, point_st, ps, color, thickness);
		}
		else if (lt == Plot::RECTANGLE)
		{
			rectangle(graph, Point(point_st.x - ps, point_st.y - ps), Point(point_st.x + ps, point_st.y + ps),
				color, thickness);
		}
		else if (lt == Plot::CIRCLE_FILL)
		{
			circle(graph, point_st, ps, color, FILLED);
		}
		else if (lt == Plot::RECTANGLE_FILL)
		{
			rectangle(graph, Point(point_st.x - ps, point_st.y - ps), Point(point_st.x + ps, point_st.y + ps),
				color, FILLED);
		}
		else if (lt == Plot::TRIANGLE)
		{
			triangle(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::TRIANGLE_FILL)
		{
			triangle(graph, point_st, 2 * ps, color, FILLED);
		}
		else if (lt == Plot::TRIANGLE_INV)
		{
			triangleinv(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::TRIANGLE_INV_FILL)
		{
			triangleinv(graph, point_st, 2 * ps, color, FILLED);
		}
		else if (lt == Plot::DIAMOND)
		{
			diamond(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::DIAMOND_FILL)
		{
			diamond(graph, point_st, 2 * ps, color, FILLED);
		}
		else if (lt == Plot::PENTAGON)
		{
			pentagon(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::PENTAGON_FILL)
		{
			pentagon(graph, point_st, 2 * ps, color, FILLED);
		}
	}

	void plotVLine(OutputArray graphImage_, Point2d data, double xmin, double xmax, double ymin, double ymax,
		Scalar color, int lt, int isLine, int thickness, int pointSize, bool isLogX, bool isLogY)
	{
		CV_Assert(!graphImage_.empty());
		const int ps = pointSize;

		Mat graph = graphImage_.getMat();
		if (isLogX)
		{
			xmax = log10(max(1.0, xmax));
			xmin = log10(max(1.0, xmin));
		}
		if (isLogY)
		{
			ymax = log10(max(1.0, ymax));
			ymin = log10(max(1.0, ymin));
		}
		//cout << xmin << "," << xmax << endl;
		const double x_step = (double)(graph.cols - 1) / (xmax - xmin);
		const double y_step = (double)(graph.rows - 1) / (ymax - ymin);

		int H = graph.rows - 1;

		double x = data.x;
		double y = data.y;
		if (isLogX) x = log10(max(1.0, x));
		if (isLogY) y = log10(max(1.0, y));

		double xn = x;
		double yn = y;
		if (isLogX) xn = log10(max(1.0, xn));
		if (isLogY) yn = log10(max(1.0, yn));
		cv::Point point_st = Point(cvRound(x_step * (xn - xmin)), 0);
		cv::Point point_ed = Point(cvRound(x_step * (xn - xmin)), graph.cols - 1);
		{
			if (isLine == Plot::LINEAR)
			{
				line(graph, point_st, point_ed, color, thickness);
			}
		}

		if (lt == Plot::NOPOINT)
		{
			;
		}
		else if (lt == Plot::PLUS)
		{
			drawPlus(graph, point_st, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::TIMES)
		{
			drawTimes(graph, point_st, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::ASTERISK)
		{
			drawAsterisk(graph, point_st, 2 * ps + 1, color, thickness);
		}
		else if (lt == Plot::CIRCLE)
		{
			circle(graph, point_st, ps, color, thickness);
		}
		else if (lt == Plot::RECTANGLE)
		{
			rectangle(graph, Point(point_st.x - ps, point_st.y - ps), Point(point_st.x + ps, point_st.y + ps),
				color, thickness);
		}
		else if (lt == Plot::CIRCLE_FILL)
		{
			circle(graph, point_st, ps, color, FILLED);
		}
		else if (lt == Plot::RECTANGLE_FILL)
		{
			rectangle(graph, Point(point_st.x - ps, point_st.y - ps), Point(point_st.x + ps, point_st.y + ps),
				color, FILLED);
		}
		else if (lt == Plot::TRIANGLE)
		{
			triangle(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::TRIANGLE_FILL)
		{
			triangle(graph, point_st, 2 * ps, color, FILLED);
		}
		else if (lt == Plot::TRIANGLE_INV)
		{
			triangleinv(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::TRIANGLE_INV_FILL)
		{
			triangleinv(graph, point_st, 2 * ps, color, FILLED);
		}
		else if (lt == Plot::DIAMOND)
		{
			diamond(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::DIAMOND_FILL)
		{
			diamond(graph, point_st, 2 * ps, color, FILLED);
		}
		else if (lt == Plot::PENTAGON)
		{
			pentagon(graph, point_st, 2 * ps, color, thickness);
		}
		else if (lt == Plot::PENTAGON_FILL)
		{
			pentagon(graph, point_st, 2 * ps, color, FILLED);
		}
	}

#pragma region Plot
	Plot::Plot()
	{
		init(cv::Size(1024, 768));
	}

	Plot::Plot(Size plotsize_)
	{
		init(plotsize_);
	}

	Plot::~Plot()
	{
		;
	}

	void Plot::init(cv::Size imsize)
	{
		origin = Point(64, 64);//default
		setImageSize(imsize);

		keyImage.create(Size(256, 256), CV_8UC3);
		keyImage.setTo(background_color);

		//setXYMinMax(0, plotsize.width, 0, plotsize.height);

		const int DefaultPlotInfoSize = 64;
		pinfo.resize(DefaultPlotInfoSize);

		vector<Scalar> c;
		//default gnuplot5.0
		c.push_back(CV_RGB(148, 0, 211));
		c.push_back(CV_RGB(0, 158, 115));
		c.push_back(CV_RGB(86, 180, 233));
		c.push_back(CV_RGB(230, 159, 0));
		c.push_back(CV_RGB(240, 228, 66));
		c.push_back(CV_RGB(0, 114, 178));
		c.push_back(CV_RGB(229, 30, 16));
		c.push_back(COLOR_BLACK);
		/*
		//classic
		c.push_back(COLOR_RED);
		c.push_back(COLOR_GREEN);
		c.push_back(COLOR_BLUE);
		c.push_back(COLOR_MAGENDA);
		c.push_back(COLOR_CYAN);
		c.push_back(COLOR_YELLOW);
		c.push_back(COLOR_BLACK);
		c.push_back(CV_RGB(0, 76, 255));
		c.push_back(COLOR_GRAY128);
		*/
		vector<int> s;
		s.push_back(PLUS);
		s.push_back(TIMES);
		s.push_back(ASTERISK);
		s.push_back(RECTANGLE);
		s.push_back(RECTANGLE_FILL);
		s.push_back(CIRCLE);
		s.push_back(CIRCLE_FILL);
		s.push_back(TRIANGLE);
		s.push_back(TRIANGLE_FILL);
		s.push_back(TRIANGLE_INV);
		s.push_back(TRIANGLE_INV_FILL);
		s.push_back(DIAMOND);
		s.push_back(DIAMOND_FILL);
		s.push_back(PENTAGON);
		s.push_back(PENTAGON_FILL);

		for (int i = 0; i < pinfo.size(); i++)
		{
			pinfo[i].symbolType = s[i % s.size()];
			pinfo[i].lineType = Plot::LINEAR;
			pinfo[i].lineWidth = 1;

			double v = (double)i / DefaultPlotInfoSize * 255.0;
			pinfo[i].color = c[i % c.size()];

			pinfo[i].title = format("data %02d", i);
			pinfo[i].parametricLine = 0;
		}

		setPlotProfile(false, true, false);
		graphImage = render;
	}

	void Plot::point2val(cv::Point pt, double* valx, double* valy)
	{
		double x = (double)plotImage.cols / (xmax_plotwindow - xmin_plotwindow);
		double y = (double)plotImage.rows / (ymax_plotwindow - ymin_plotwindow);
		int H = plotImage.rows - 1;

		*valx = (pt.x - (origin.x) * 2) / x + xmin_plotwindow;
		*valy = (H - (pt.y - origin.y)) / y + ymin_plotwindow;
	}

	void Plot::computeDataMaxMin()
	{
		ymax_data = -DBL_MAX;
		ymin_data = DBL_MAX;
		xmax_data = -DBL_MAX;
		xmin_data = DBL_MAX;

		ymaxpt.resize(data_labelmax);
		yminpt.resize(data_labelmax);
		double y_max_val = -DBL_MAX;
		double y_min_val = DBL_MAX;
		for (int i = 0; i < data_labelmax; i++)
		{
			for (int j = 0; j < pinfo[i].data.size(); j++)
			{
				const double y = pinfo[i].data[j].y;
				if (y_max_val <= y)
				{
					y_max_val = y;
					ymax_data = y;
					ymaxpt[i] = pinfo[i].data[j];
				}
				if (y_min_val >= y)
				{
					y_min_val = y;
					ymin_data = y;
					yminpt[i] = pinfo[i].data[j];
				}

				const double x = pinfo[i].data[j].x;
				xmax_data = max(xmax_data, x);
				xmin_data = min(xmin_data, x);
			}
		}
		//print_debug4(xmin_data, xmax_data, ymin_data, ymax_data);
	}

	void Plot::computeWindowXRangeMAXMIN(bool isCenter, double margin_rate, int rounding_value)
	{
		if (margin_rate < 0.0 || margin_rate>1.0) margin_rate = 1.0;

		xmax_plotwindow = xmax_data;
		xmin_plotwindow = xmin_data;

		double xmargin = (xmax_plotwindow - xmin_plotwindow) * (1.0 - margin_rate) * 0.5;

		xmax_plotwindow += xmargin;
		if (xmax_plotwindow >= 0)
		{
			if (rounding_value != 0) xmax_plotwindow = (double)cp::ceilToMultiple((int)xmax_plotwindow, rounding_value);
		}
		else
		{
			if (rounding_value != 0) xmax_plotwindow = (double)-cp::floorToMultiple((int)(-xmax_plotwindow), rounding_value);
		}

		xmin_plotwindow -= xmargin;
		if (xmin_plotwindow >= 0)
		{
			if (rounding_value != 0) xmin_plotwindow = (double)cp::floorToMultiple((int)xmin_plotwindow, rounding_value);
		}
		else
		{
			if (rounding_value != 0) xmin_plotwindow = (double)-cp::ceilToMultiple((int)-xmin_plotwindow, rounding_value);
		}


		if (isCenter)
		{
			double xxx = abs(xmax_plotwindow);
			xxx = (xxx < abs(xmin_plotwindow)) ? abs(xmin_plotwindow) : xxx;

			xmax_plotwindow = xxx;
			xmin_plotwindow = -xxx;
		}
	}

	void Plot::computeWindowYRangeMAXMIN(bool isCenter, double margin_rate, int rounding_value)
	{
		if (margin_rate < 0.0 || margin_rate>1.0)margin_rate = 1.0;

		ymax_plotwindow = ymax_data;
		ymin_plotwindow = ymin_data;

		double ymargin = (ymax_plotwindow - ymin_plotwindow) * (1.0 - margin_rate) * 0.5;

		ymax_plotwindow += ymargin;
		if (ymax_plotwindow >= 0.0)
		{
			if (rounding_value != 0) ymax_plotwindow = (double)cp::ceilToMultiple((int)ymax_plotwindow, rounding_value);
		}
		else
		{
			if (rounding_value != 0) ymax_plotwindow = (double)-cp::floorToMultiple((int)-ymax_plotwindow, rounding_value);
		}

		ymin_plotwindow -= ymargin;
		if (ymin_plotwindow >= 0.0)
		{
			if (rounding_value != 0) ymin_plotwindow = (double)cp::floorToMultiple((int)ymin_plotwindow, rounding_value);
		}
		else
		{
			if (rounding_value != 0) ymin_plotwindow = (double)-cp::ceilToMultiple((int)-ymin_plotwindow, rounding_value);
		}

		if (isCenter)
		{
			double yyy = abs(ymax_plotwindow);
			yyy = (yyy < abs(ymin_plotwindow)) ? abs(ymin_plotwindow) : yyy;

			ymax_plotwindow = yyy;
			ymin_plotwindow = -yyy;
		}
	}

	void Plot::recomputeXYRangeMAXMIN(bool isCenter, double margin_rate, int rounding_value)
	{
		computeWindowXRangeMAXMIN(isCenter, margin_rate, rounding_value);
		computeWindowYRangeMAXMIN(isCenter, margin_rate, rounding_value);
	}

	void Plot::setXYOriginZERO()
	{
		recomputeXYRangeMAXMIN(false);
		xmin_plotwindow = 0;
		ymin_plotwindow = 0;
	}

	void Plot::setYOriginZERO()
	{
		recomputeXYRangeMAXMIN(false);
		ymin_plotwindow = 0;
	}

	void Plot::setXOriginZERO()
	{
		recomputeXYRangeMAXMIN(false);
		xmin_plotwindow = 0;
	}

	void Plot::setPlotProfile(bool isXYCenter_, bool isXYMAXMIN_, bool isZeroCross_)
	{
		isZeroCross = isZeroCross_;
		isXYMAXMIN = isXYMAXMIN_;
		isXYCenter = isXYCenter_;
	}

	void Plot::setImageSize(Size s)
	{
		plotsize = s;
		plotImage.create(s, CV_8UC3);
		render.create(Size(plotsize.width + 4 * origin.x, plotsize.height + 2 * origin.y), CV_8UC3);
	}

	void Plot::setXYRange(double xmin_, double xmax_, double ymin_, double ymax_)
	{
		isSetXRange = true;
		isSetYRange = true;
		xmin_plotwindow = xmin_;
		xmax_plotwindow = xmax_;
		ymin_plotwindow = ymin_;
		ymax_plotwindow = ymax_;

		xmax_data = xmax_plotwindow;
		xmin_data = xmin_plotwindow;
		ymax_data = ymax_plotwindow;
		ymin_data = ymin_plotwindow;
	}

	void Plot::setXRange(double xmin, double xmax)
	{
		isSetXRange = true;
		xmin_plotwindow = xmin;
		xmax_plotwindow = xmax;
	}

	void Plot::setYRange(double ymin, double ymax)
	{
		isSetYRange = true;
		ymin_plotwindow = ymin;
		ymax_plotwindow = ymax;
	}

	void Plot::unsetXRange()
	{
		isSetXRange = false;
	}

	void Plot::unsetYRange()
	{
		isSetYRange = false;
	}

	void Plot::unsetXYRange()
	{
		isSetXRange = false;
		isSetYRange = false;
	}

	void Plot::setLogScaleX(const bool flag)
	{
		isLogScaleX = flag;
	}

	void Plot::setLogScaleY(const bool flag)
	{
		isLogScaleY = flag;
	}


	void Plot::setKey(const KEY key)
	{
		keyPosition = (int)key;
	}

	void Plot::setWideKey(bool flag)
	{
		this->isWideKey = flag;
	}

	void Plot::setXLabel(string xlabel)
	{
		isLabelXGreekLetter = false;
		this->xlabel = xlabel;
	}

	void Plot::setYLabel(string ylabel)
	{
		isLabelYGreekLetter = false;
		this->ylabel = ylabel;
	}

	void Plot::setXLabelGreekLetter(std::string greeksymbol, std::string subscript)
	{
		isLabelXGreekLetter = true;
		this->xlabel = greeksymbol;
		this->xlabel_subscript = subscript;
	}

	void Plot::setYLabelGreekLetter(std::string greeksymbol, std::string subscript)
	{
		isLabelXGreekLetter = true;
		this->ylabel = greeksymbol;
		this->ylabel_subscript = subscript;
	}

	void Plot::setGrid(int grid_level)
	{
		this->gridLevel = grid_level;
	}

	void Plot::setBackGoundColor(Scalar cl)
	{
		background_color = cl;
	}


	void Plot::setPlot(int plotnum, Scalar color, int symboltype, int linetype, int line_width)
	{
		setPlotColor(plotnum, color);
		setPlotSymbol(plotnum, symboltype);
		setPlotLineType(plotnum, linetype);
		setPlotLineWidth(plotnum, line_width);
	}

	void Plot::setPlotColor(int plotnum, Scalar color_)
	{
		pinfo[plotnum].color = color_;
	}

	void Plot::setPlotSymbol(int plotnum, int symboltype)
	{
		pinfo[plotnum].symbolType = symboltype;
	}

	void Plot::setPlotSymbolALL(int symboltype)
	{
		for (int i = 0; i < pinfo.size(); i++)
		{
			pinfo[i].symbolType = symboltype;
		}
	}

	void Plot::setPlotLineType(int plotnum, int lineType)
	{
		pinfo[plotnum].lineType = lineType;
	}

	void Plot::setPlotLineTypeALL(int linetype)
	{
		for (int i = 0; i < pinfo.size(); i++)
		{
			pinfo[i].lineType = linetype;
		}
	}

	void Plot::setPlotLineWidth(int plotnum, int lineWidth)
	{
		pinfo[plotnum].lineWidth = lineWidth;
	}

	void Plot::setPlotLineWidthALL(int lineWidth)
	{
		for (int i = 0; i < pinfo.size(); i++)
		{
			pinfo[i].lineWidth = lineWidth;
		}
	}

	void Plot::setPlotTitle(int plotnum, string name)
	{
		pinfo[plotnum].title = name;
	}

	void Plot::setPlotForeground(int plotnum)
	{
		foregroundIndex = plotnum;
	}

	void Plot::setFontSize(int fontSiz)
	{
		this->fontSize = fontSize;
	}

	void Plot::setFontSize2(int fontSize2)
	{
		this->fontSize2 = fontSize2;
	}

	void Plot::push_back_HLine(double y, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		pinfo[plotIndex].data.push_back(Point2d(0, y));
		pinfo[plotIndex].parametricLine = 1;
	}

	void Plot::push_back_VLine(double x, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		pinfo[plotIndex].data.push_back(Point2d(x, 0));
		pinfo[plotIndex].parametricLine = 2;
	}

	void Plot::push_back(double x, double y, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		pinfo[plotIndex].data.push_back(Point2d(x, y));
	}

	void Plot::push_back(std::vector<float>& x, std::vector<float>& y, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		for (int i = 0; i < (int)x.size(); i++)
		{
			push_back(x[i], y[i], plotIndex);
		}
	}

	void Plot::push_back(std::vector<double>& x, std::vector<double>& y, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		for (int i = 0; i < (int)x.size(); i++)
		{
			push_back(x[i], y[i], plotIndex);
		}
	}

	void Plot::push_back(vector<cv::Point>& point, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		const int size = (int)point.size();
		if (pinfo[plotIndex].data.size() == 0)
		{
			pinfo[plotIndex].data.resize(size);
			for (int i = 0; i < size; i++)
			{
				pinfo[plotIndex].data[i].x = point[i].x;
				pinfo[plotIndex].data[i].y = point[i].y;
			}
		}
		else
		{
			for (int i = 0; i < (int)point.size(); i++)
			{
				push_back(point[i].x, point[i].y, plotIndex);
			}
		}
	}

	void Plot::push_back(vector<cv::Point2d>& point, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		const int size = (int)point.size();
		if (pinfo[plotIndex].data.size() == 0)
		{
			pinfo[plotIndex].data.resize(size);
			for (int i = 0; i < size; i++)
			{
				pinfo[plotIndex].data[i].x = point[i].x;
				pinfo[plotIndex].data[i].y = point[i].y;
			}
		}
		else
		{
			for (int i = 0; i < size; i++)
			{
				push_back(point[i].x, point[i].y, plotIndex);
			}
		}
	}

	void Plot::push_back(vector<cv::Point2f>& point, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		const int size = (int)point.size();
		if (pinfo[plotIndex].data.size() == 0)
		{
			pinfo[plotIndex].data.resize(size);
			for (int i = 0; i < size; i++)
			{
				pinfo[plotIndex].data[i].x = (double)point[i].x;
				pinfo[plotIndex].data[i].y = (double)point[i].y;
			}
		}
		else
		{
			for (int i = 0; i < size; i++)
			{
				push_back((double)point[i].x, (double)point[i].y, plotIndex);
			}
		}
	}

	void Plot::push_back(Mat& point, int plotIndex)
	{
		data_labelmax = max(data_labelmax, plotIndex + 1);
		const int size = (int)point.size().area();
		if (pinfo[plotIndex].data.size() == 0)
		{
			pinfo[plotIndex].data.resize(size);
			if (point.depth() == CV_8U)
			{
				for (int i = 0; i < size; i++)
				{
					pinfo[plotIndex].data[i].x = (double)i;
					pinfo[plotIndex].data[i].y = (double)point.at<uchar>(0, i);
				}
			}
			else if (point.depth() == CV_32F)
			{
				for (int i = 0; i < size; i++)
				{
					pinfo[plotIndex].data[i].x = (double)i;
					pinfo[plotIndex].data[i].y = (double)point.at<float>(0, i);
				}
			}
			else if (point.depth() == CV_64F)
			{
				for (int i = 0; i < size; i++)
				{
					pinfo[plotIndex].data[i].x = (double)i;
					pinfo[plotIndex].data[i].y = point.at<double>(0, i);
				}
			}
		}
		else
		{
			if (point.depth() == CV_8U)
			{
				for (int i = 0; i < size; i++)
				{
					push_back((double)i, (double)point.at<uchar>(0, i), plotIndex);
				}
			}
			else if (point.depth() == CV_32F)
			{
				for (int i = 0; i < size; i++)
				{
					push_back((double)i, (double)point.at<float>(0, i), plotIndex);
				}
			}
			else if (point.depth() == CV_64F)
			{
				for (int i = 0; i < size; i++)
				{
					push_back((double)i, point.at<double>(0, i), plotIndex);
				}
			}

		}
	}


	void Plot::erase(int sampleIndex, int plotIndex)
	{
		pinfo[plotIndex].data.erase(pinfo[plotIndex].data.begin() + sampleIndex);
	}

	void Plot::insert(Point2d v, int sampleIndex, int plotIndex)
	{
		pinfo[plotIndex].data.insert(pinfo[plotIndex].data.begin() + sampleIndex, v);
	}

	void Plot::insert(Point v, int sampleIndex, int plotIndex)
	{
		insert(Point2d((double)v.x, (double)v.y), sampleIndex, plotIndex);
	}

	void Plot::insert(double x, double y, int sampleIndex, int plotIndex)
	{
		insert(Point2d(x, y), sampleIndex, plotIndex);
	}

	void Plot::clear(int datanum)
	{
		if (datanum < 0)
		{
			for (int i = 0; i < data_labelmax; i++)
				pinfo[i].data.clear();

		}
		else
			pinfo[datanum].data.clear();
	}

	cv::Point2d Plot::getPlotPoint(const int pointIndex, const int plotIndex)
	{
		return pinfo[max(0, min((int)pinfo.size() - 1, plotIndex))].data[max(0, min((int)pinfo[0].data.size() - 1, pointIndex))];
	}

	void Plot::swapPlot(int plotIndex1, int plotIndex2)
	{
		swap(pinfo[plotIndex1].data, pinfo[plotIndex2].data);
	}

	void boundingRectImage(InputArray src_, OutputArray dest)
	{
		Mat src = src_.getMat();
		CV_Assert(src.depth() == CV_8U);

		vector<Point> v;
		const int ch = src.channels();
		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				bool flag = false;
				for (int c = 0; c < ch; c++)
				{
					if (src.at<uchar>(j, ch * i + c) != 0)
					{
						flag = true;
					}
				}
				if (flag) v.push_back(Point(i, j));
			}
		}

		Rect r = boundingRect(v);
		src(r).copyTo(dest);
	}

	void Plot::renderingOutsideInformation(bool isPrintLabel)
	{
		render.setTo(background_color);
		Mat roi = render(Rect(origin.x * 2, origin.y, plotsize.width, plotsize.height));
		rectangle(plotImage, Point(0, 0), Point(plotImage.cols - 1, plotImage.rows - 1), COLOR_BLACK, 1);
		plotImage.copyTo(roi);

		Mat label = Mat::zeros(plotImage.size(), CV_8UC3);
		Mat labelximage;
		Mat labelyimage;
		if (isPrintLabel)
		{
			if (isLabelXGreekLetter)
			{
				Mat a = getTextImageQt(xlabel, "Symbol", fontSize, Scalar::all(0), background_color, true);
				if (xlabel_subscript.size() != 0)
				{
					Mat b = getTextImageQt(xlabel_subscript, font, int(fontSize / 1.8), Scalar::all(0), background_color, true);
					copyMakeBorder(b, b, a.rows - b.rows, 0, 0, 0, BORDER_CONSTANT, background_color);
					hconcat(a, b, label);
				}
				else
				{
					a.copyTo(label);
				}
				label = ~label;
			}
			else
			{
				cv::addText(label, xlabel, Point(0, 100), font, fontSize, COLOR_WHITE);
			}
			boundingRectImage(label, labelximage);

			label.create(plotImage.size(), CV_8UC3);
			label.setTo(0);
			cv::addText(label, ylabel, Point(0, 100), font, fontSize, COLOR_WHITE);
			boundingRectImage(label, labelyimage);

			Mat(~labelximage).copyTo(render(Rect(render.cols / 2 - labelximage.cols / 2, (int)(render.rows - labelximage.rows - 2), labelximage.cols, labelximage.rows)));

			labelyimage = labelyimage.t();
			flip(labelyimage, labelyimage, 0);
			Mat(~labelyimage).copyTo(render(Rect(2, (int)(origin.y * 0.25 + plotImage.rows / 2 - labelyimage.rows / 2), labelyimage.cols, labelyimage.rows)));

			string buff;
			//x coordinate
			if (isLogScaleX)
			{
				if (xmax_data - xmin_data < 10)
				{
					buff = format("1", saturate_cast<int>(xmin_plotwindow));
					cv::addText(render, buff, Point(origin.x + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10^0.25", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.25 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.25 + 40), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10^0.5", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.5 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.5 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10^0.75");
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.75 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%d", 10);
					cv::addText(render, buff, Point(origin.x + plotImage.cols + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				}
				else if (xmax_data - xmin_data < 100)
				{
					buff = format("1", saturate_cast<int>(xmin_plotwindow));
					cv::addText(render, buff, Point(origin.x + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10^0.25", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.25 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.25 + 40), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10^0.5", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.5 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.5 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10");
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.75 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("100");
					cv::addText(render, buff, Point(origin.x + plotImage.cols + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				}
				else if (xmax_data - xmin_data < 1000)
				{
					buff = format("1", saturate_cast<int>(xmin_plotwindow));
					cv::addText(render, buff, Point(origin.x + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10^0.5", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.25 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.25 + 40), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.5 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.5 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("100");
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.75 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("1000");
					cv::addText(render, buff, Point(origin.x + plotImage.cols + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				}
				else if (xmax_data - xmin_data < 10000)
				{
					buff = format("1", saturate_cast<int>(xmin_plotwindow));
					cv::addText(render, buff, Point(origin.x + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.25 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.25 + 40), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("100", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.5 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.5 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("1000");
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.75 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("10000");
					cv::addText(render, buff, Point(origin.x + plotImage.cols + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				}
			}
			else
			{
				if (xmax_plotwindow - xmin_plotwindow < 5)
				{
					buff = format("%.2f", xmin_plotwindow);
					cv::addText(render, buff, Point(origin.x + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%.2f", (xmax_plotwindow - xmin_plotwindow) * 0.25 + xmin_plotwindow);
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.25 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%.2f", (xmax_plotwindow - xmin_plotwindow) * 0.5 + xmin_plotwindow);
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.5 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%.2f", (xmax_plotwindow - xmin_plotwindow) * 0.75 + xmin_plotwindow);
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.75 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%.2f", xmax_plotwindow);
					cv::addText(render, buff, Point(origin.x + plotImage.cols + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				}
				else
				{
					buff = format("%d", saturate_cast<int>(xmin_plotwindow));
					cv::addText(render, buff, Point(origin.x + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%d", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.25 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.25 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%d", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.5 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.5 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%d", saturate_cast<int>((xmax_plotwindow - xmin_plotwindow) * 0.75 + xmin_plotwindow));
					cv::addText(render, buff, Point((int)(origin.x + plotImage.cols * 0.75 + 45), (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
					buff = format("%d", saturate_cast<int>(xmax_plotwindow));
					cv::addText(render, buff, Point(origin.x + plotImage.cols + 30, (int)(origin.y * 1.35 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				}
			}
			//y coordinate			
			if (ymax_plotwindow - ymin_plotwindow < 0.01)
			{
				buff = format("%g", ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				buff = format("%g", (ymax_plotwindow - ymin_plotwindow) * 0.5 + ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.5)), font, fontSize2, COLOR_BLACK);
				buff = format("%g", (ymax_plotwindow - ymin_plotwindow) * 0.25 + ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.75)), font, fontSize2, COLOR_BLACK);
				buff = format("%g", (ymax_plotwindow - ymin_plotwindow) * 0.75 + ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.25)), font, fontSize2, COLOR_BLACK);
				buff = format("%g", ymax_plotwindow);
				cv::addText(render, buff, Point(origin.x, origin.y), font, fontSize2, COLOR_BLACK);
			}
			else if (ymax_plotwindow - ymin_plotwindow < 5)
			{
				buff = format("%.2f", ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				buff = format("%.2f", (ymax_plotwindow - ymin_plotwindow) * 0.5 + ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.5)), font, fontSize2, COLOR_BLACK);
				buff = format("%.2f", (ymax_plotwindow - ymin_plotwindow) * 0.25 + ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.75)), font, fontSize2, COLOR_BLACK);
				buff = format("%.2f", (ymax_plotwindow - ymin_plotwindow) * 0.75 + ymin_plotwindow);
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.25)), font, fontSize2, COLOR_BLACK);
				buff = format("%.2f", ymax_plotwindow);
				cv::addText(render, buff, Point(origin.x, origin.y), font, fontSize2, COLOR_BLACK);
			}
			else
			{
				buff = format("%d", saturate_cast<int>(ymin_plotwindow));
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows)), font, fontSize2, COLOR_BLACK);
				buff = format("%d", saturate_cast<int>((ymax_plotwindow - ymin_plotwindow) * 0.5 + ymin_plotwindow));
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.5)), font, fontSize2, COLOR_BLACK);
				buff = format("%d", saturate_cast<int>((ymax_plotwindow - ymin_plotwindow) * 0.25 + ymin_plotwindow));
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.75)), font, fontSize2, COLOR_BLACK);
				buff = format("%d", saturate_cast<int>((ymax_plotwindow - ymin_plotwindow) * 0.75 + ymin_plotwindow));
				cv::addText(render, buff, Point(origin.x, (int)(origin.y * 1.0 + plotImage.rows * 0.25)), font, fontSize2, COLOR_BLACK);
				buff = format("%d", saturate_cast<int>(ymax_plotwindow));
				cv::addText(render, buff, Point(origin.x, origin.y), font, fontSize2, COLOR_BLACK);
			}
		}
	}

	void Plot::plotPoint(Point2d point, Scalar color_, int thickness_, int linetype)
	{
		vector<Point2d> data;

		data.push_back(Point2d(point.x, ymin_plotwindow));
		data.push_back(Point2d(point.x, ymax_plotwindow));
		data.push_back(Point2d(point.x, point.y));
		data.push_back(Point2d(xmax_plotwindow, point.y));
		data.push_back(Point2d(xmin_plotwindow, point.y));

		plotGraph(plotImage, data, xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow, color_, NOPOINT, linetype, thickness_);
	}

	void Plot::plotGrid(int level)
	{
		if (level > 0)
		{
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) / 2.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) / 2.0 + ymin_plotwindow), COLOR_GRAY150, 1);
		}
		if (level > 1)
		{
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 1.0 / 4.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 1.0 / 4.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 3.0 / 4.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 1.0 / 4.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 1.0 / 4.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 3.0 / 4.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 3.0 / 4.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 3.0 / 4.0 + ymin_plotwindow), COLOR_GRAY200, 1);
		}
		if (level > 2)
		{
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 1.0 / 8.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 1.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 3.0 / 8.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 1.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 1.0 / 8.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 3.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * 3.0 / 8.0 + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 3.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);

			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (1.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 1.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (3.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 1.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (1.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 3.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (3.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * 3.0 / 8.0 + ymin_plotwindow), COLOR_GRAY200, 1);

			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (1.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (1.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (3.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (1.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (1.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (3.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (3.0 / 8.0 + 0.5) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (3.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);

			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (1.0 / 8.0) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (1.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (3.0 / 8.0) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (1.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (1.0 / 8.0) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (3.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);
			plotPoint(Point2d((xmax_plotwindow - xmin_plotwindow) * (3.0 / 8.0) + xmin_plotwindow, (ymax_plotwindow - ymin_plotwindow) * (3.0 / 8.0 + 0.5) + ymin_plotwindow), COLOR_GRAY200, 1);
		}
	}

	void Plot::generateKeyImage(int num, bool isWideKey)
	{
		const int step = 24;
		const int offset = 3;

		keyImage.release();
		if (isWideKey) keyImage.create(Size(256 * 2, step * (num + 2) + offset), CV_8UC3);
		else keyImage.create(Size(256, step * (num + 2) + offset), CV_8UC3);
		keyImage.setTo(background_color);

		int height = (int)(0.8 * keyImage.rows);
		for (int i = 0; i < num; i++)
		{
			if (pinfo[i].data.size() == 0) continue;

			vector<Point2d> data;
			data.push_back(Point2d(keyImage.cols / 3.0 * 4.0, keyImage.rows - (i + 1) * step));
			data.push_back(Point2d(keyImage.cols - step, keyImage.rows - (i + 1) * step));

			plotGraph(keyImage, data, 0, keyImage.cols, 0, keyImage.rows, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].lineWidth);
			//putText(keyImage, pinfo[i].keyname, Point(0, (i + 1) * step + 3), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, pinfo[i].color);
			if (foregroundIndex != 0)
			{
				if (foregroundIndex == i + 1)
				{
					cv::addText(keyImage, "*" + pinfo[i].title, Point(0, (i + 1) * step + offset), font, fontSize + 2, pinfo[i].color);
				}
				else
				{
					cv::addText(keyImage, pinfo[i].title, Point(0, (i + 1) * step + offset), font, fontSize, pinfo[i].color);
				}
			}
			else
			{
				cv::addText(keyImage, pinfo[i].title, Point(0, (i + 1) * step + offset), font, fontSize, pinfo[i].color);
			}
		}
	}

	void Plot::plotData(int gridlevel)
	{
		plotImage.setTo(background_color);
		plotGrid(gridlevel);

		const int symbolSize = 4;
		if (isZeroCross)	plotPoint(Point2d(0.0, 0.0), COLOR_ORANGE, 1);

		if (foregroundIndex == 0)
		{
			for (int i = 0; i < data_labelmax; i++)
			{
				if (pinfo[i].parametricLine == 0) plotGraph(plotImage, pinfo[i].data, xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].lineWidth, symbolSize, isLogScaleX, isLogScaleY);
				else if (pinfo[i].parametricLine == 1)plotHLine(plotImage, pinfo[i].data[0], xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].lineWidth, symbolSize, isLogScaleX, isLogScaleY);
				else if (pinfo[i].parametricLine == 2)plotVLine(plotImage, pinfo[i].data[0], xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].lineWidth, symbolSize, isLogScaleX, isLogScaleY);
			}
		}
		else
		{
			for (int i = 0; i < data_labelmax; i++)
			{
				if (i + 1 != foregroundIndex)
				{
					plotGraph(plotImage, pinfo[i].data, xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow, pinfo[i].color, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].lineWidth, symbolSize, isLogScaleX, isLogScaleY);
				}
			}
			plotGraph(plotImage, pinfo[foregroundIndex - 1].data, xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow, pinfo[foregroundIndex - 1].color, pinfo[foregroundIndex - 1].symbolType, pinfo[foregroundIndex - 1].lineType, pinfo[foregroundIndex - 1].lineWidth, symbolSize, isLogScaleX, isLogScaleY);
		}

		//for (int i = 0; i < data_max; i++)
			//plotLine(plotImage, ymaxpt[i], xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow, COLOR_RED, pinfo[i].symbolType, pinfo[i].lineType, pinfo[i].lineWidth, symbolSize, isLogScaleX, isLogScaleY);

		renderingOutsideInformation(true);

		Mat temp = render.clone();

		if (keyPosition == NOKEY)
		{
			if (cv::getWindowProperty("key", WND_PROP_VISIBLE))
				destroyWindow("key");
		}
		else
		{
			Mat roi;
			if (keyPosition == RIGHT_TOP)
			{
				roi = render(Rect(render.cols - keyImage.cols - 150, 80, keyImage.cols, keyImage.rows));
			}
			else if (keyPosition == LEFT_TOP)
			{
				roi = render(Rect(160, 80, keyImage.cols, keyImage.rows));
			}
			else if (keyPosition == LEFT_BOTTOM)
			{
				roi = render(Rect(160, render.rows - keyImage.rows - 150, keyImage.cols, keyImage.rows));
			}
			else if (keyPosition == RIGHT_BOTTOM)
			{
				roi = render(Rect(render.cols - keyImage.cols - 150, render.rows - keyImage.rows - 150, keyImage.cols, keyImage.rows));
			}
			else if (keyPosition == FLOATING)
			{
				imshow("key", keyImage);
			}

			if (keyPosition != FLOATING)
			{
				keyImage.copyTo(roi);

				if (cv::getWindowProperty("key", WND_PROP_VISIBLE))
					destroyWindow("key");
			}
		}
		addWeighted(render, 0.8, temp, 0.2, 0.0, render);
	}

	void Plot::saveDatFile(const string name, const bool isPrint)
	{
		FILE* fp = fopen(name.c_str(), "w");

		int dmax = (int)pinfo[0].data.size();
		for (int i = 1; i < data_labelmax; i++)
		{
			dmax = max((int)pinfo[i].data.size(), dmax);
		}

		for (int n = 0; n < dmax; n++)
		{
			for (int i = 0; i < data_labelmax; i++)
			{
				if (n < pinfo[i].data.size())
				{
					double x = pinfo[i].data[n].x;
					double y = pinfo[i].data[n].y;
					fprintf(fp, "%f %f ", x, y);
				}
				else if (pinfo[i].data.size() != 0)
				{
					double x = pinfo[i].data[pinfo[i].data.size() - 1].x;
					double y = pinfo[i].data[pinfo[i].data.size() - 1].y;
					fprintf(fp, "%f %f ", x, y);
				}
				else
				{
					fprintf(fp, "%f %f ", 0.0, 0.0);
				}
			}
			fprintf(fp, "\n");
		}
		if (isPrint)cout << "p ";
		for (int i = 0; i < data_labelmax; i++)
		{
			if (isPrint) cout << "'" << name << "'" << " u " << 2 * i + 1 << ":" << 2 * i + 2 << " w lp" << "lt " << i + 1 << " t \"" << pinfo[i].title << "\",";
		}
		if (isPrint)cout << endl;
		fclose(fp);
	}

	Scalar Plot::getPseudoColor(uchar val)
	{
		int i = val;
		double d = 255.0 / 63.0;
		Scalar ret;

		{//g
			uchar lr[256];
			for (int i = 0; i < 64; i++)
				lr[i] = cvRound(d * i);
			for (int i = 64; i < 192; i++)
				lr[i] = 255;
			for (int i = 192; i < 256; i++)
				lr[i] = cvRound(255 - d * (i - 192));

			ret.val[1] = lr[val];
		}
		{//r
			uchar lr[256];
			for (int i = 0; i < 128; i++)
				lr[i] = 0;
			for (int i = 128; i < 192; i++)
				lr[i] = cvRound(d * (i - 128));
			for (int i = 192; i < 256; i++)
				lr[i] = 255;

			ret.val[0] = lr[val];
		}
		{//b
			uchar lr[256];
			for (int i = 0; i < 64; i++)
				lr[i] = 255;
			for (int i = 64; i < 128; i++)
				lr[i] = cvRound(255 - d * (i - 64));
			for (int i = 128; i < 256; i++)
				lr[i] = 0;
			ret.val[2] = lr[val];
		}
		return ret;
	}

	void Plot::setIsDrawMousePosition(const bool flag)
	{
		this->isDrawMousePosition = flag;
	}

	static void guiPreviewMousePlot(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;

		if (flags == EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

	void Plot::plot(const string wname, bool isWait, const string gnuplotpath, const string message)
	{
		Point pt = Point(0, 0);
		namedWindow(wname);

		setMouseCallback(wname, (MouseCallback)guiPreviewMousePlot, (void*)&pt);

		generateKeyImage(data_labelmax, isWideKey);

		computeDataMaxMin();
		if (!isWait) setIsDrawMousePosition(false);
		bool isUseMinmaxBar = (xmax_data > 1 && ymax_data > 1);
		//bool isUseMinmaxBar = false;
		const int xmin = (int)xmin_data;
		const int xmax = (int)ceil(xmax_data);
		const int ymin = (int)ymin_data;
		const int ymax = (int)ceil(ymax_data);

		int xminbar = xmin;
		int xmaxbar = xmax;
		int yminbar = ymin;
		int ymaxbar = ymax;
		if (isUseMinmaxBar && isWait)
		{
			if (xmax > 0 && xmax - xmin > 0) { createTrackbar("xmin", wname, &xminbar, xmax); setTrackbarMin("xmin", wname, xmin); }
			if (xmax > 0 && xmax - xmin > 0) { createTrackbar("xmax", wname, &xmaxbar, xmax); setTrackbarMin("xmax", wname, xmin); }
			if (ymax > 0 && ymax - ymin > 0) { createTrackbar("ymin", wname, &yminbar, ymax); setTrackbarMin("ymin", wname, ymin); }
			if (ymax > 0 && ymax - ymin > 0) { createTrackbar("ymax", wname, &ymaxbar, ymax); setTrackbarMin("ymax", wname, ymin); }
		}
		const double margin_ratio = 1.0;//0.9
		if (!isSetXRange)
		{
			if (isLogScaleX)
			{
				xmin_plotwindow = 1;
				if (xmax_data - xmin_data < 10)xmax_plotwindow = 10;
				else if (xmax_data - xmin_data < 100)xmax_plotwindow = 100;
				else if (xmax_data - xmin_data < 1000)xmax_plotwindow = 1000;
				else if (xmax_data - xmin_data < 10000)xmax_plotwindow = 10000;
			}
			else
			{
				if (xmax_data - xmin_data > 50)
					computeWindowXRangeMAXMIN(false, margin_ratio, 10);
				else if (xmax_data - xmin_data > 10)
					computeWindowXRangeMAXMIN(false, margin_ratio, 1);
				else
					computeWindowXRangeMAXMIN(false, margin_ratio, 0);
			}
		}
		if (!isSetYRange)
		{
			if (ymax_data - ymin_data > 50)
				computeWindowYRangeMAXMIN(false, margin_ratio, 10);
			else if (ymax_data - ymin_data > 10)
				computeWindowYRangeMAXMIN(false, margin_ratio, 1);
			else
				computeWindowYRangeMAXMIN(false, margin_ratio, 0);
		}

		int keyboard = 0;

		//xmin_plotwindow, xmax_plotwindow, ymin_plotwindow, ymax_plotwindow

		while (keyboard != 'q')
		{
			if (isUseMinmaxBar)
			{
				xmin_data = xminbar;
				xmax_data = xmaxbar;
				ymin_data = yminbar;
				ymax_data = ymaxbar;
			}

			if (!isSetXRange)
			{
				if (isLogScaleX)
				{
					xmin_plotwindow = 1;
					if (xmax_data - xmin_data < 10)xmax_plotwindow = 10;
					else if (xmax_data - xmin_data < 100)xmax_plotwindow = 100;
					else if (xmax_data - xmin_data < 1000)xmax_plotwindow = 1000;
					else if (xmax_data - xmin_data < 10000)xmax_plotwindow = 10000;
				}
				else
				{
					/*if (xmax_data - xmin_data > 50)
						computeWindowXRangeMAXMIN(false, margin_ratio, 10);
					else if (xmax_data - xmin_data > 10)
						computeWindowXRangeMAXMIN(false, margin_ratio, 1);
					else
						computeWindowXRangeMAXMIN(false, margin_ratio, 0);*/
				}
			}
			if (!isSetYRange)
			{
				if (ymax_data - ymin_data > 50)
					computeWindowYRangeMAXMIN(false, margin_ratio, 10);
				else if (ymax_data - ymin_data > 10)
					computeWindowYRangeMAXMIN(false, margin_ratio, 1);
				else
					computeWindowYRangeMAXMIN(false, margin_ratio, 0);
			}

			plotData(gridLevel);

			if (isDrawMousePosition)
			{
				double xx = 0.0;
				double yy = 0.0;
				point2val(pt, &xx, &yy);
				if (pt.x < 0 || pt.y < 0 || pt.x >= render.cols || pt.y >= render.rows)
				{
					pt = Point(0, 0);
					xx = 0.0;
					yy = 0.0;
				}
				string text = format("(%.02f,%.02f) ", xx, yy) + message;
				//putText(render, text, Point(100, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, COLOR_BLACK);
				cv::addText(render, text, Point(100, 30), font, fontSize, COLOR_BLACK);

				drawGrid(render, pt, Scalar(180, 180, 255), 1, 4, 0);
			}
			else
			{
				cv::addText(render, message, Point(100, 30), font, fontSize, COLOR_BLACK);
			}
			imshow(wname, render);

#pragma region keyborad
			if (keyboard == '?')
			{
				cout << "*** Help message ***" << endl;
				cout << "m: " << "show mouseover position and grid" << endl;
				cout << "c: " << "(0,0)point must posit center" << endl;
				cout << "g: " << "Show grid" << endl;

				cout << "k: " << "Show key" << endl;

				cout << "x: " << "Set X origin zero " << endl;
				cout << "y: " << "Set Y origin zero " << endl;
				cout << "z: " << "Set XY origin zero " << endl;
				cout << "r: " << "Reset XY max min" << endl;

				cout << "s: " << "Save image (plot.png)" << endl;
				cout << "q: " << "Quit" << endl;

				cout << "********************" << endl;
				cout << endl;
			}
			if (keyboard == 'm')
			{
				isDrawMousePosition = (isDrawMousePosition) ? false : true;
			}
			if (keyboard == 'r')
			{
				recomputeXYRangeMAXMIN(false);
			}
			if (keyboard == 'c')
			{
				recomputeXYRangeMAXMIN(true);
			}
			if (keyboard == 'x')
			{
				setXOriginZERO();
			}
			if (keyboard == 'y')
			{
				setYOriginZERO();
			}
			if (keyboard == 'z')
			{
				setXYOriginZERO();
			}
			if (keyboard == 'k')
			{
				keyPosition++;
				if (keyPosition == KEY::KEY_METHOD_SIZE)
					keyPosition = 0;
			}
			if (keyboard == 'g')
			{
				gridLevel++;
				if (gridLevel > 3)gridLevel = 0;
			}
			if (keyboard == 'p')
			{
				saveDatFile("plot", false);
				std::string a("plot ");
				GNUPlot gplot(gnuplotpath);
				gplot.setXLabel(xlabel);
				gplot.setYLabel(ylabel);
				gplot.setKey(KEY(keyPosition));
				for (int i = 0; i < data_labelmax; i++)
				{
					char name[256];
					if (i != data_labelmax - 1)
					{
						sprintf(name, "'plot' u %d:%d w lp ps 0.5 lt %d t \"%s\",", 2 * i + 1, 2 * i + 2, i + 1, pinfo[i].title.c_str());
					}
					else
					{
						sprintf(name, "'plot' u %d:%d w lp ps 0.5 lt %d t \"%s\"", 2 * i + 1, 2 * i + 2, i + 1, pinfo[i].title.c_str());
					}
					a += name;
				}
				gplot.plotPDF(a);
				//gplot.cmd(a.c_str());
			}
			if (keyboard == 's')
			{
				saveDatFile("plot");
				imwrite("plotim.png", render);
				//imwrite("plot.png", save);
			}
			if (keyboard == '0')
			{
				for (int i = 0; i < pinfo[0].data.size(); i++)
				{
					cout << i << pinfo[0].data[i] << endl;
				}
			}
			if (keyboard == 'w')
			{
				isWait = false;
			}
#pragma endregion

			if (!isWait) break;
			keyboard = waitKey(1);
		}

		if (isWait) destroyWindow(wname);
	}

	void Plot::plotMat(InputArray src_, const string wname, const bool isWait, const string gnuplotpath)
	{
		Mat src = src_.getMat();
		clear();

		if (src.depth() == CV_32F)
			for (int i = 0; i < src.size().area(); i++) push_back(i, src.at<float>(i));
		else if (src.depth() == CV_64F)
			for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<double>(i));
		else if (src.depth() == CV_8U)
			for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<uchar>(i));
		else if (src.depth() == CV_16U)
			for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<ushort>(i));
		else if (src.depth() == CV_16S)
			for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<short>(i));
		else if (src.depth() == CV_32S)
			for (int i = 0; i < src.size().area(); i++)push_back(i, src.at<int>(i));

		plot(wname, isWait, gnuplotpath);
	}
#pragma endregion

#pragma region Plot2D
	void Plot2D::createPlot()
	{
		gridData = Mat::zeros(Size(x_size, y_size), CV_64F);
	}

	void Plot2D::setYMinMax(double minv, double maxv, double interval)
	{
		x_min = minv;
		x_max = maxv;
		x_interval = interval;
		x_size = (int)ceil((maxv - minv) / interval) + 1;
	}

	void Plot2D::setXMinMax(double minv, double maxv, double interval)
	{
		y_min = minv;
		y_max = maxv;
		y_interval = interval;
		y_size = (int)ceil((maxv - minv) / interval) + 1;
	}

	void Plot2D::setMinMax(double xmin, double xmax, double xinterval, double ymin, double ymax, double yinterval)
	{
		setYMinMax(xmin, xmax, xinterval);
		setXMinMax(ymin, ymax, yinterval);
		createPlot();
	}

	Plot2D::Plot2D(Size graph_size, double xmin, double xmax, double xinterval, double ymin, double ymax, double yinterval)
	{
		plotImageSize = graph_size;
		setMinMax(xmin, xmax, xinterval, ymin, ymax, yinterval);

		//default gnuplot5.0
		colorIndex.push_back(CV_RGB(148, 0, 211));
		colorIndex.push_back(CV_RGB(0, 158, 115));
		colorIndex.push_back(CV_RGB(86, 180, 233));
		colorIndex.push_back(CV_RGB(230, 159, 0));
		colorIndex.push_back(CV_RGB(240, 228, 66));
		colorIndex.push_back(CV_RGB(0, 114, 178));
		colorIndex.push_back(CV_RGB(229, 30, 16));
		colorIndex.push_back(COLOR_BLACK);
	}

	void Plot2D::setFont(string font)
	{
		this->font = font;
	}

	void Plot2D::setFontSize(const int size)
	{
		fontSize = size;
	}

	void Plot2D::setFontSize2(const int size)
	{
		fontSize2 = size;
	}

	void Plot2D::setZMinMax(double minv, double maxv)
	{
		z_min = minv;
		z_max = maxv;
		isSetZMinMax = true;
	}

	void Plot2D::setLabel(string namex, string namey, string namez)
	{
		labelx = namex;
		labely = namey;
		labelz = namez;
	}

	void Plot2D::setLabelXGreekLetter(std::string greeksymbol, std::string subscript)
	{
		labelx = greeksymbol;
		labelx_subscript = subscript;
		isLabelXGreekLetter = true;
	}

	void Plot2D::setLabelYGreekLetter(std::string greeksymbol, std::string subscript)
	{
		labely = greeksymbol;
		labely_subscript = subscript;
		isLabelYGreekLetter = true;
	}

	void Plot2D::setLabelZGreekLetter(std::string greeksymbol, std::string subscript)
	{
		labelz = greeksymbol;
		labelz_subscript = subscript;
		isLabelZGreekLetter = true;
	}

	void Plot2D::setPlotContours(std::string label, double thresh, int index)
	{
		if (contourLabels.size() < index + 1)
		{
			contourLabels.resize(index + 1);
			contourThresh.resize(index + 1);
		}
		contourLabels[index] = label;
		contourThresh[index] = thresh;
	}

	void Plot2D::setPlotMaxMin(bool plot_max, bool plot_min)
	{
		isPlotMax = plot_max;
		isPlotMin = plot_min;
	}

	cv::Mat Plot2D::getGridData()
	{
		return this->gridData;
	}


	void Plot2D::plotGraph(bool isColor, Mat& graph)
	{
		Mat dataflip;
		flip(gridData, dataflip, 0);
		int rate_x = plotImageSize.width / dataflip.cols;
		int rate_y = plotImageSize.height / dataflip.rows;
		plotImageSize = Size(dataflip.cols * rate_x, dataflip.rows * rate_y);
		resize(dataflip, gridDataRes, plotImageSize, 0, 0, cv::INTER_LINEAR);
		Mat barImageGray(Size(barWidth, plotImageSize.height), CV_8U);
		for (int j = 0; j < plotImageSize.height; j++)
		{
			uchar* d = barImageGray.ptr<uchar>(j);
			for (int i = 0; i < barImageGray.cols; i++)
			{
				d[i] = saturate_cast<uchar>(255.0 * (plotImageSize.height - 1 - j) / (plotImageSize.height - 1));
			}
		}

		double z_min_temp, z_max_temp;
		minMaxLoc(gridDataRes, &z_min_temp, &z_max_temp, &z_min_point, &z_max_point);
		if (!isSetZMinMax)
		{
			z_min = z_min_temp;
			z_max = z_max_temp;
		}

		Mat graphG(gridDataRes.size(), CV_8U);
		for (int i = 0; i < gridDataRes.size().area(); i++)
		{
			graphG.at<uchar>(i) = cv::saturate_cast<uchar>(max(0.0, gridDataRes.at<double>(i) - z_min) / (z_max - z_min) * 255.0);
		}

		if (isColor)
		{
			applyColorMap(barImageGray, barImage, colormap);
			applyColorMap(graphG, graph, colormap);
		}
		else
		{
			cvtColor(barImageGray, barImage, COLOR_GRAY2BGR);
			cvtColor(graphG, graph, COLOR_GRAY2BGR);
		}
	}

	inline int getNumTicks(int length, int fontSize, double fontSpace)
	{
		return int(length / (fontSize * fontSpace)) + 1;
	}

	//1�ȉ��͕��������_
	//ticks��2�Ȃ�min max
	//ticks��3�Ȃ�min max
	class GraphTicksGenerator
	{
		double minv;
		double maxv;
		int length;
		int fontSize;
		double fontSpace;
		int num_ticks_max;

		bool isFloatTicks()
		{
			bool ret = false;
			if (maxv - minv > 10)return false;

			if (maxv - minv <= 1)return true;

			if (num_ticks_max == 2)return false;

			if (num_ticks_max == 3)
			{
				if (maxv == 2) return false;
				else return true;
			}
			if (num_ticks_max == 4)
			{
				if (maxv == 3) return false;
				else return true;
			}

			//num_ticks_max>5
			{
				double eps = DBL_EPSILON;
				int subi = (int)(maxv - minv + eps);
				if (abs(subi) >= 4)return false;
				else return true;
			}

			return ret;
		}

		void generateTicks()
		{
			double sub = maxv - minv;
			int subi = (int)ceil(sub);

			if (subi >= 0)
			{
				int step = 1;
				if (subi / 1 < num_ticks_max)
				{
					step = 1;
				}
				else if (subi / 2 < num_ticks_max)
				{
					step = 2;
				}
				else if (subi / 5 < num_ticks_max)
				{
					step = 5;
				}
				else if (subi / 10 < num_ticks_max)
				{
					step = 10;
				}
				else if (subi / 20 < num_ticks_max)
				{
					step = 20;
				}
				else if (subi / 25 < num_ticks_max)
				{
					step = 25;
				}
				else if (subi / 50 < num_ticks_max)
				{
					step = 50;
				}
				else if (subi / 100 < num_ticks_max)
				{
					step = 100;
				}
				else if (subi / 200 < num_ticks_max)
				{
					step = 200;
				}
				else if (subi / 250 < num_ticks_max)
				{
					step = 250;
				}

				num_ticks = subi / step + 1;

				if (ceilToMultiple(minv, step) == minv)
				{
					impos.resize(num_ticks);
					tick_val.resize(num_ticks);
					double dl = (length - 1) / (maxv - minv);

					for (int i = 0; i < num_ticks; i++)
					{
						float v = float(step * i);
						impos[i] = int(dl * v);
						tick_val[i] = int(v + minv);
					}
				}
				else
				{
					//num_ticks--;
					impos.resize(num_ticks);
					tick_val.resize(num_ticks);
					int ming = ceilToMultiple(minv, step);
					double dl = (length - 1) / (maxv - minv);
					double fastl = (ming - minv) * dl;
					for (int i = 0; i < num_ticks; i++)
					{
						int v = step * i;
						if (length - 1 > dl * v + fastl)
						{
							impos[i] = int(dl * v + fastl);
							tick_val[i] = v + ming;
						}
						else
						{
							num_ticks--;
							impos.erase(impos.begin() + num_ticks);
							tick_val.erase(tick_val.begin() + num_ticks);
						}
					}
				}
			}
		}

		int float_state = 0;
		bool isMultiple(double val, double target)
		{
			int v = int(val / target);
			return(val == double(v * target));
		}
		double ceilToMultiple(double val, double target)
		{
			return ceil(val / target) * target;
			double ret = 0.0;
			if (isMultiple(val, target))
			{
				ret = val;
			}
			else
			{
				int v = int(ceil(val / target));
				ret = (v * target);
			}
			return ret;
		}

		void generateTicksFloat()
		{
			float_state = 1;

			if (0 <= maxv && maxv <= 1)
			{
				double step = 0.0;
				int maxi = (int)ceil(maxv * 10);
				if (maxi / 1 < num_ticks_max)
				{
					step = 0.1;
				}
				else if (maxi / 2 < num_ticks_max)
				{
					step = 0.2;
				}
				else if (maxi / 2.5 < num_ticks_max)
				{
					step = 0.25;
					float_state = 2;
				}
				else if (maxi / 5 < num_ticks_max)
				{
					step = 0.5;
				}
				else if (maxi / 10 < num_ticks_max)
				{
					step = 1.0;
				}

				num_ticks = (int)ceil(maxv / step) + 1;
				//print_debug3(ceil(maxv / step) + 1,anum, step);
				impos.resize(num_ticks);
				tick_val32f.resize(num_ticks);

				for (int i = 0; i < num_ticks; i++)
				{
					float v = float(step * i + minv);
					if (v <= 1.0)
					{
						impos[i] = int(length / maxv * v);
						tick_val32f[i] = v;
					}
				}
			}
			else
			{
				double step = 0.0;
				int maxi = (int)ceil(maxv * 10);

				if (maxi / 1 < num_ticks_max)
				{
					step = 0.1;
				}
				else if (maxi / 2 < num_ticks_max)
				{
					step = 0.2;
				}
				else if (maxi / 2.5 < num_ticks_max)
				{
					step = 0.25;
					float_state = 2;
				}
				else if (maxi / 5 < num_ticks_max)
				{
					step = 0.5;
				}
				else if (maxi / 10 < num_ticks_max)
				{
					step = 1.0;
				}

				num_ticks = (int)ceil(maxv / step);

				if (isMultiple(minv, step))
				{
					impos.resize(num_ticks);
					tick_val32f.resize(num_ticks);
					double dl = (length - 1) / (maxv - minv);

					for (int i = 0; i < num_ticks; i++)
					{
						float v = float(step * i);
						{
							impos[i] = int(dl * v);
							tick_val32f[i] = float(v + minv);
						}
					}
				}
				else
				{
					impos.resize(num_ticks);
					tick_val32f.resize(num_ticks);
					double ming = ceilToMultiple(minv, step);
					double dl = (length - 1) / (maxv - minv);
					double fastl = (ming - minv) * dl;
					for (int i = 0; i < num_ticks; i++)
					{
						float v = float(step * i);
						if (length - 1 > dl * v + fastl)
						{
							impos[i] = int(dl * v + fastl);
							tick_val32f[i] = float(v + ming);
						}
						else
						{
							num_ticks--;
							impos.erase(impos.begin() + num_ticks);
							tick_val32f.erase(tick_val32f.begin() + num_ticks);
						}
					}
				}
			}
		}
	public:
		int num_ticks = 0;
		vector<int> impos;
		vector<float> tick_val32f;
		vector<int> tick_val;
		bool isFloat = false;
		bool isMinZero = false;
		GraphTicksGenerator(double minv, double maxv, int length, int fontSize, double fontSpace) :
			minv(minv), maxv(maxv), length(length), fontSize(fontSize), fontSpace(fontSpace), num_ticks_max(min(11, getNumTicks(length, fontSize, fontSpace)))
		{
			if (minv < 0.001) isMinZero = true;

			isFloat = isFloatTicks();
			if (isFloat)generateTicksFloat();
			else generateTicks();
		}

		void printTicks()
		{
			if (isFloat) cout << "float" << endl;
			else cout << "int" << endl;

			for (int i = 0; i < num_ticks; i++)
			{
				if (isFloat) cout << i << ": " << (float)tick_val32f[i] << ": " << impos[i] << endl;
				else cout << i << ": " << tick_val[i] << ": " << impos[i] << endl;
			}
		}
		Mat generateImage(int index, std::string font, cv::Scalar color, cv::Scalar background_color)
		{
			if (isFloat)
			{
				if (float_state == 1)
					return getTextImageQt(format("%5.1f", tick_val32f[index]), font, fontSize, color, background_color);
				else //if (float_state == 2)
					return getTextImageQt(format("%5.2f", tick_val32f[index]), font, fontSize, color, background_color);
			}
			else
			{
				return getTextImageQt(format("%d", tick_val[index]), font, fontSize, color, background_color);
			}
		}

		int getTicksCharactors()
		{
			if (isFloat)
			{
				return float_state + 1;
			}
			else
			{
				if (maxv < 10)return 1;
				if (maxv < 100)return 2;
				if (maxv < 1000)return 3;
				if (maxv < 10000)return 4;
				if (maxv < 100000)return 5;
				if (maxv < 1000000)return 6;
				if (maxv < 10000000)return 7;
			}

			return 0;
		}
	};

	void embed(Mat& src, Mat& target, Point pt, const bool isCenteringX, const bool isCenteringY)
	{
		CV_Assert(!src.empty());
		CV_Assert(!target.empty());
		const int offset_x = (isCenteringX) ? src.cols / 2 : 0;
		const int offset_y = (isCenteringY) ? src.rows / 2 : 0;
		bool isOutOfRange = false;
		Rect roi(pt.x - offset_x, pt.y - offset_y, src.cols, src.rows);
		if (0 <= roi.x && roi.x < target.cols && 0 <= roi.y && roi.y < target.rows)
		{
			int subx = 0;
			int suby = 0;
			if (roi.x + src.cols > target.cols - 1)
			{
				subx = (roi.x + src.cols) - (target.cols - 1);
				isOutOfRange = true;
			}
			if (roi.y + src.rows > target.rows - 1)
			{
				suby = (roi.y + src.rows) - (target.rows - 1);
				isOutOfRange = true;
			}

			if (isOutOfRange)
			{
				src(Rect(0, 0, src.cols - subx, src.rows - suby)).copyTo(target(Rect(roi.x, roi.y, src.cols - subx, src.rows - suby)));
			}
			else
			{
				src.copyTo(target(roi));
			}
		}
	}

	void Plot2D::addLabelToGraph(bool isDrawingContour, int keyState, bool isPlotMin, bool isPlotMax)
	{
		if (isDrawingContour)
		{
			for (int i = 0; i < contourLabels.size(); i++)
			{
				drawContoursZ(contourThresh[i], colorIndex[i], 2);
			}

			int distance_from_boundary_for_key = 10;
			generateKeyImage(1);
			if (keyState == 0)
			{
				destroyWindow("key");
			}
			else if (keyState == 1)
			{
				keyImage.copyTo(graph(Rect(graph.cols - keyImage.cols - distance_from_boundary_for_key, distance_from_boundary_for_key, keyImage.cols, keyImage.rows)));
				destroyWindow("key");
			}
			else if (keyState == 2)
			{
				keyImage.copyTo(graph(Rect(distance_from_boundary_for_key, distance_from_boundary_for_key, keyImage.cols, keyImage.rows)));
				destroyWindow("key");
			}
			else if (keyState == 3)
			{
				keyImage.copyTo(graph(Rect(distance_from_boundary_for_key, graph.rows - keyImage.rows - distance_from_boundary_for_key, keyImage.cols, keyImage.rows)));
				destroyWindow("key");
			}
			else if (keyState == 4)
			{
				keyImage.copyTo(graph(Rect(graph.cols - keyImage.cols - distance_from_boundary_for_key, graph.rows - keyImage.rows - distance_from_boundary_for_key, keyImage.cols, keyImage.rows)));
				destroyWindow("key");
			}
			else if (keyState == 5)
			{
				imshow("key", keyImage);
			}
		}

		if (isPlotMax)
		{
			circle(graph, z_max_point, 5, colorIndex[maxColorIndex], 2);
		}
		if (isPlotMin)
		{
			drawPlus(graph, z_min_point, 8 * 3, colorIndex[minColorIndex]);
		}

		GraphTicksGenerator tickx(x_min, x_max, graph.cols, fontSize2, 4);
		GraphTicksGenerator ticky(y_min, y_max, graph.rows, fontSize2, 1.5);
		GraphTicksGenerator tickz(z_min, z_max, graph.rows, fontSize2, 1.5);

		const int offset_left = fontSize + fontSize2 * ticky.getTicksCharactors();
		const int offset_right = fontSize2 * tickz.getTicksCharactors();
		const int xlabel_vspace = 5;
		const int offset_bottom = fontSize + fontSize2 + xlabel_vspace + 10;
		const int offset_top = fontSize + fontSize2;

		int lineWidth = 2;
		int tickLength = 7;

		if (isLabelXGreekLetter)
		{
			Mat a = getTextImageQt(labelx, "Symbol", fontSize, Scalar::all(0), background_color, true);
			if (labelx_subscript.size() != 0)
			{
				Mat b = getTextImageQt(labelx_subscript, font, int(fontSize / 1.8), Scalar::all(0), background_color, true);
				copyMakeBorder(b, b, a.rows - b.rows, 0, 0, 0, BORDER_CONSTANT, background_color);
				hconcat(a, b, labelxImage);
			}
			else
			{
				a.copyTo(labelxImage);
			}
		}
		else
		{
			labelxImage = getTextImageQt(labelx, font, fontSize, Scalar::all(0), background_color);
		}
		labelzImage = getTextImageQt(labelz, font, fontSize, Scalar::all(0), background_color);

		if (isLabelYGreekLetter)
		{
			Mat a = getTextImageQt(labely, "Symbol", fontSize, Scalar::all(0), background_color, true);
			if (labely_subscript.size() != 0)
			{
				Mat b = getTextImageQt(labely_subscript, font, int(fontSize / 1.8), Scalar::all(0), background_color, true);
				copyMakeBorder(b, b, a.rows - b.rows, 0, 0, 0, BORDER_CONSTANT, background_color);
				hconcat(a, b, labelyImage);
			}
			else
			{
				a.copyTo(labelyImage);
			}
		}
		else
		{
			labelyImage = getTextImageQt(labely, font, fontSize, Scalar::all(0), background_color);
		}
		rotate(labelyImage, labelyImage, ROTATE_90_COUNTERCLOCKWISE);

		Mat graph2;

		rectangle(barImage, Rect(0, 0, barImage.cols, barImage.rows), Scalar::all(0), lineWidth);
		copyMakeBorder(barImage, barImage, 0, 0, barSpace, 0, BORDER_CONSTANT, background_color);
		rectangle(graph, Rect(0, 0, graph.cols, graph.rows), Scalar::all(0), lineWidth);

		hconcat(graph, barImage, graph2);

		copyMakeBorder(graph2, show, offset_top, offset_bottom, offset_left, offset_right, BORDER_CONSTANT, Scalar(255, 255, 255));

		int graph_centerx = graph.cols / 2 + offset_left;
		Rect rx = Rect(graph_centerx - labelxImage.cols / 2, offset_top + graph.rows + fontSize2 + xlabel_vspace, labelxImage.cols, labelxImage.rows);
		labelxImage.copyTo(show(rx));

		int graph_centery = graph.rows / 2 + offset_top;
		Rect ry = Rect(1, graph_centery - labelyImage.rows / 2, labelyImage.cols, labelyImage.rows);
		labelyImage.copyTo(show(ry));

		Rect rz = Rect(show.cols - labelzImage.cols - 2, 1, labelzImage.cols, labelzImage.rows);
		labelzImage.copyTo(show(rz));


		/*Mat infoImage = getTextImageQt(format("min:%5.2f,%5.2f,%5.2f, max:%5.2f,%5.2f,%5.2f",
			gridData.at<double>(z_min_point), (z_min_point.x - x_min) * x_interval, (z_min_point.y - y_min) * y_interval,
			gridData.at<double>(z_max_point), (z_max_point.x - x_min) * x_interval, (z_max_point.y - y_min) * y_interval),
			font, fontSize, Scalar::all(0), background_color);*/
			//print_debug(z_min_point);
			//print_debug(y_min);
			//print_debug(y_interval);
			/*const int y = gridDataRes.rows - 1 - z_min_point.y;
			const double ratex = gridDataRes.cols / gridData.cols;
			const double ratey = gridDataRes.rows / gridData.rows;
			Mat infoImage = getTextImageQt(format("min:%5.2f=(%5.2f,%5.2f)",
				gridDataRes.at<double>(Size(z_min_point.x, y)), z_min_point.x* x_interval/ratex + x_min, y* y_interval/ratey + y_min),
				font, fontSize, Scalar::all(0), background_color);
			Rect ri = Rect(1, 1, infoImage.cols, infoImage.rows);
			infoImage.copyTo(show(ri));*/

			//x
		for (int i = 0; i < tickx.num_ticks; i++)
		{
			line(show, Point(tickx.impos[i] + offset_left, offset_top + graph.rows - 1), Point(tickx.impos[i] + offset_left, offset_top + graph.rows - 1 - tickLength), Scalar::all(0));
			Mat label = tickx.generateImage(i, font, Scalar::all(0), background_color);
			if (i == 0 && tickx.isMinZero)
			{
				embed(label, show, Point(offset_left + tickx.impos[i], offset_top + graph.rows + 2), false, false);
			}
			else
			{
				embed(label, show, Point(offset_left + tickx.impos[i], offset_top + graph.rows + 2), true, false);
			}
		}

		//y
		for (int i = 0; i < ticky.num_ticks; i++)
		{
			line(show, Point(offset_left, offset_top + graph.rows - 1 - ticky.impos[i]), Point(offset_left + tickLength, offset_top + graph.rows - 1 - ticky.impos[i]), Scalar::all(0));
			Mat label = ticky.generateImage(i, font, Scalar::all(0), background_color);
			if (i == 0 && ticky.isMinZero)
			{
				embed(label, show, Point(offset_left - 2 - label.cols, offset_top + graph.rows - ticky.impos[i] - label.rows), false, false);
			}
			else
			{
				embed(label, show, Point(offset_left - 2 - label.cols, offset_top + graph.rows - ticky.impos[i]), false, true);
			}
		}

		//z
		int bar_stx = offset_left + graph.cols + barSpace;
		int bar_edx = offset_left + graph.cols - 1 + barSpace + barWidth;
		int bar_sty = offset_top;
		for (int i = 0; i < tickz.num_ticks; i++)
		{
			Mat label = tickz.generateImage(i, font, Scalar::all(0), background_color);
			Rect r;
			if (i == 0 && tickz.isMinZero)
			{
				r = Rect(bar_edx + 2, offset_top + graph.rows - tickz.impos[i] - label.rows, label.cols, label.rows);
				embed(label, show, Point(bar_edx + 2, offset_top + graph.rows - tickz.impos[i] - label.rows), false, false);
			}
			else
			{
				line(show, Point(bar_stx, bar_sty + graph.rows - 1 - tickz.impos[i]), Point(bar_stx + tickLength, bar_sty + graph.rows - 1 - tickz.impos[i]), Scalar::all(0));
				line(show, Point(bar_edx, bar_sty + graph.rows - 1 - tickz.impos[i]), Point(bar_edx - tickLength, bar_sty + graph.rows - 1 - tickz.impos[i]), Scalar::all(0));
				embed(label, show, Point(bar_edx + 2, offset_top + graph.rows - tickz.impos[i]), false, true);
			}
		}
	}

	void Plot2D::drawContoursZ(double thresh, cv::Scalar color, int lineWidth)
	{
		Mat mask;
		threshold(gridDataRes, mask, thresh, 255, THRESH_BINARY);
		mask.convertTo(mask, CV_8U);

		vector<vector<Point>> contours;
		findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		drawContours(graph, contours, 0, color, lineWidth);
	}

	void Plot2D::generateKeyImage(int lineWidthBB, int lineWidthKey)
	{
		int additional = 0;
		if (isPlotMax)additional++;
		if (isPlotMin)additional++;

		int step = fontSize2 + 3;
		int v = step * (int)(contourThresh.size() + additional) + 5;

		int charactor_max = 0;
		for (int i = 0; i < contourLabels.size(); i++)
		{
			charactor_max = max(charactor_max, (int)contourLabels[i].size());
		}
		charactor_max--;
		int lst = 10;
		int led = 30;
		int space = 10;
		keyImage.create(Size(fontSize2 * charactor_max + led + space, v), CV_8UC3);
		keyImage.setTo(background_color);
		if (lineWidthBB > 0)
			rectangle(keyImage, Rect(0, 0, keyImage.cols, keyImage.rows), Scalar::all(0), lineWidthBB);

		for (int i = 0; i < contourLabels.size(); i++)
		{
			line(keyImage, Point(lst, (i + 1) * step - fontSize2 / 2), Point(led, (i + 1) * step - fontSize2 / 2), colorIndex[i], lineWidthKey);
			cv::addText(keyImage, contourLabels[i], Point(led + space, (i + 1) * step), font, fontSize2);
		}

		int i = (int)contourLabels.size();
		if (isPlotMax)
		{
			circle(keyImage, Point((lst + led) / 2, (i + 1) * step - fontSize2 / 2), 5, colorIndex[i], 2);
			cv::addText(keyImage, "MAX", Point(led + space, (i + 1) * step), font, fontSize2);
			maxColorIndex = i;
			i++;
		}
		if (isPlotMin)
		{
			drawPlus(keyImage, Point((lst + led) / 2, (i + 1) * step - fontSize2 / 2), 8, Scalar(0, 0, 0), 2);
			cv::addText(keyImage, "MIN", Point(led + space, (i + 1) * step), font, fontSize2);
			minColorIndex = i;
		}
	}

	void Plot2D::plot(string wname)
	{
		namedWindow(wname);
		string wname2 = "";
		createTrackbar("font size", wname2, &fontSize, 128);
		createTrackbar("font size2", wname2, &fontSize2, 128);
		createTrackbar("width", wname2, &plotImageSize.width, 1920);
		setTrackbarMin("width", wname2, 256);
		createTrackbar("height", wname2, &plotImageSize.height, 1080);
		setTrackbarMin("height", wname2, 256);
		createTrackbar("colormap", wname2, &colormap, 20);
		plot_x = x_size / 2; createTrackbar("plot x", wname2, &plot_x, x_size - 1);
		plot_y = y_size / 2; createTrackbar("plot y", wname2, &plot_y, y_size - 1);
		//int minvalue = 30; createTrackbar("min", wname, &minvalue, 100);
		//int maxvalue = 50; createTrackbar("max", wname, &maxvalue, 100);

		//int xc = 0; createTrackbar("x", wname, &xc, result.width);
		//int yc = 1; createTrackbar("y", wname, &yc, result.width);
		//int zc = 3; createTrackbar("z", wname, &zc, result.width);

		bool isColorFlag = true;
		bool isMinMaxSet = false;
		int key = 0;
		Plot ptx(plotImageSize);
		Plot pty(plotImageSize);
		ptx.setKey(cp::Plot::NOKEY);
		pty.setKey(cp::Plot::NOKEY);
		ptx.setYLabel(labelz);
		pty.setYLabel(labelz);

		if (isLabelXGreekLetter)
		{
			ptx.setXLabelGreekLetter(labelx, labelx_subscript);
		}
		else
		{
			ptx.setXLabel(labelx);
		}
		if (isLabelYGreekLetter)
		{
			pty.setXLabelGreekLetter(labely, labely_subscript);
		}
		else
		{
			pty.setXLabel(labely);
		}

		ptx.setIsDrawMousePosition(false);
		pty.setIsDrawMousePosition(false);
		bool isGrid = false;
		bool isDrawingContour = false;
		while (key != 'q')
		{
			//writeGraph(isColor, PLOT_ARG_MAX, minvalue, maxvalue, isMinMaxSet);
			plotGraph(isColorFlag, graph);
			int resx = graph.cols / gridData.cols;
			int resy = graph.rows / gridData.rows;
			if (isGrid) drawGrid(graph, Point(plot_x * resx, plot_y * resy), COLOR_RED);
			addLabelToGraph(isDrawingContour, keyState, isPlotMin, isPlotMax);

			for (int i = 0; i < gridData.cols; i++)
			{
				ptx.push_back((i - x_min) * x_interval, gridData.at<double>(plot_y, i));
			}
			for (int i = 0; i < gridData.rows; i++)
			{
				pty.push_back((i - y_min) * y_interval, gridData.at<double>(i, plot_x));
			}
			imshow(wname, show);
			ptx.plot(wname + "x", false);
			pty.plot(wname + "y", false);


			//imshow(wname, graphBase);
			key = waitKey(1);
			if (key == 'c')
			{
				isColorFlag = (isColorFlag) ? false : true;
			}
			if (key == 'g')
			{
				isGrid = isGrid ? false : true;
			}
			if (key == 'k')
			{
				keyState++;
				if (keyState > 5) keyState = 0;
			}
			if (key == 'n')
			{
				isPlotMin = (isPlotMin) ? false : true;
			}
			if (key == 's')
			{
				imwrite("plot2d.png", show);
				imwrite("plotx.png", ptx.render);
				imwrite("ploty.png", pty.render);
				ptx.saveDatFile("plotx");
				pty.saveDatFile("ploty");
			}
			if (key == 'x')
			{
				isPlotMax = (isPlotMax) ? false : true;
			}
			if (key == 'z')
			{
				isDrawingContour = (isDrawingContour) ? false : true;
			}
			if (key == '?' || key == 'h')
			{
				cout << "c: flip isColor" << endl;
				cout << "g: flip drawGrid" << endl;
				cout << "k: key position" << endl;
				cout << "n: flip draw cross for min point" << endl;
				cout << "x: flip draw circle for max point" << endl;
				cout << "z: flip isDrawContour" << endl;
			}

			ptx.clear();
			pty.clear();
		}
	}

	void Plot2D::addIndex(int x, int y, double val)
	{
		if (
			x < gridData.cols && x >= 0 &&
			y < gridData.rows && y >= 0
			)
		{
			gridData.at<double>(y, x) = val;
		}
	}

	void Plot2D::add(double x, double y, double val)
	{
		const int X = int((x - x_min) / x_interval);
		const int Y = int((y - y_min) / y_interval);
		addIndex(X, Y, val);
	}

#pragma endregion

#pragma region RGBHistogram
	static void onMouseHistogram3D(int event, int x, int y, int flags, void* param)
	{
		Point* ret = (Point*)param;

		if (flags == EVENT_FLAG_LBUTTON)
		{
			ret->x = x;
			ret->y = y;
		}
	}

	void RGBHistogram::projectPointsParallel(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest, const bool isRotationThenTranspose)
	{
		float* data = (float*)xyz.ptr<float>(0);
		Point2f* dst = &dest[0];
		int size2 = xyz.size().area();
		int i;
		float tt[3];
		tt[0] = (float)t.at<double>(0, 0);
		tt[1] = (float)t.at<double>(1, 0);
		tt[2] = (float)t.at<double>(2, 0);

		float r[3][3];
		if (isRotationThenTranspose)
		{
			const float f00 = (float)K.at<double>(0, 0);
			const float xc = (float)K.at<double>(0, 2);
			const float f11 = (float)K.at<double>(1, 1);
			const float yc = (float)K.at<double>(1, 2);

			r[0][0] = (float)R.at<double>(0, 0);
			r[0][1] = (float)R.at<double>(0, 1);
			r[0][2] = (float)R.at<double>(0, 2);

			r[1][0] = (float)R.at<double>(1, 0);
			r[1][1] = (float)R.at<double>(1, 1);
			r[1][2] = (float)R.at<double>(1, 2);

			r[2][0] = (float)R.at<double>(2, 0);
			r[2][1] = (float)R.at<double>(2, 1);
			r[2][2] = (float)R.at<double>(2, 2);

			for (i = 0; i < size2; i++)
			{
				const float x = data[0];
				const float y = data[1];
				//const float z = data[2];
				const float z = 0.f;

				const float px = r[0][0] * x + r[0][1] * y + r[0][2] * z + tt[0];
				const float py = r[1][0] * x + r[1][1] * y + r[1][2] * z + tt[1];
				const float pz = r[2][0] * x + r[2][1] * y + r[2][2] * z + tt[2];

				const float div = 1.f / pz;

				dst->x = (f00 * px + xc * pz) * div;
				dst->y = (f11 * py + yc * pz) * div;

				data += 3;
				dst++;
			}
		}
		else
		{
			Mat kr = K * R;

			r[0][0] = (float)kr.at<double>(0, 0);
			r[0][1] = (float)kr.at<double>(0, 1);
			r[0][2] = (float)kr.at<double>(0, 2);

			r[1][0] = (float)kr.at<double>(1, 0);
			r[1][1] = (float)kr.at<double>(1, 1);
			r[1][2] = (float)kr.at<double>(1, 2);

			r[2][0] = (float)kr.at<double>(2, 0);
			r[2][1] = (float)kr.at<double>(2, 1);
			r[2][2] = (float)kr.at<double>(2, 2);

			for (i = 0; i < size2; i++)
			{
				const float x = data[0] + tt[0];
				const float y = data[1] + tt[1];
				const float z = data[2] + tt[2];

				const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);

				dst->x = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
				dst->y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;

				data += 3;
				dst++;
			}
		}
	}

	void RGBHistogram::projectPoints(const Mat& xyz, const Mat& R, const Mat& t, const Mat& K, vector<Point2f>& dest, const bool isRotationThenTranspose)
	{
		float* data = (float*)xyz.ptr<float>(0);
		Point2f* dst = &dest[0];
		int size2 = xyz.size().area();
		int i;
		float tt[3];
		tt[0] = (float)t.at<double>(0, 0);
		tt[1] = (float)t.at<double>(1, 0);
		tt[2] = (float)t.at<double>(2, 0);

		float r[3][3];
		if (isRotationThenTranspose)
		{
			const float f00 = (float)K.at<double>(0, 0);
			const float xc = (float)K.at<double>(0, 2);
			const float f11 = (float)K.at<double>(1, 1);
			const float yc = (float)K.at<double>(1, 2);

			r[0][0] = (float)R.at<double>(0, 0);
			r[0][1] = (float)R.at<double>(0, 1);
			r[0][2] = (float)R.at<double>(0, 2);

			r[1][0] = (float)R.at<double>(1, 0);
			r[1][1] = (float)R.at<double>(1, 1);
			r[1][2] = (float)R.at<double>(1, 2);

			r[2][0] = (float)R.at<double>(2, 0);
			r[2][1] = (float)R.at<double>(2, 1);
			r[2][2] = (float)R.at<double>(2, 2);

			for (i = 0; i < size2; i++)
			{
				const float x = data[0];
				const float y = data[1];
				const float z = data[2];
				//const float z = 0.f;

				const float px = r[0][0] * x + r[0][1] * y + r[0][2] * z + tt[0];
				const float py = r[1][0] * x + r[1][1] * y + r[1][2] * z + tt[1];
				const float pz = r[2][0] * x + r[2][1] * y + r[2][2] * z + tt[2];

				const float div = 1.f / pz;

				dst->x = (f00 * px + xc * pz) * div;
				dst->y = (f11 * py + yc * pz) * div;

				data += 3;
				dst++;
			}
		}
		else
		{
			Mat kr = K * R;

			r[0][0] = (float)kr.at<double>(0, 0);
			r[0][1] = (float)kr.at<double>(0, 1);
			r[0][2] = (float)kr.at<double>(0, 2);

			r[1][0] = (float)kr.at<double>(1, 0);
			r[1][1] = (float)kr.at<double>(1, 1);
			r[1][2] = (float)kr.at<double>(1, 2);

			r[2][0] = (float)kr.at<double>(2, 0);
			r[2][1] = (float)kr.at<double>(2, 1);
			r[2][2] = (float)kr.at<double>(2, 2);

			for (i = 0; i < size2; i++)
			{
				const float x = data[0] + tt[0];
				const float y = data[1] + tt[1];
				const float z = data[2] + tt[2];

				const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);

				dst->x = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
				dst->y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;

				data += 3;
				dst++;
			}
		}
	}

	void RGBHistogram::projectPoint(Point3d& xyz, const Mat& R, const Mat& t, const Mat& K, Point2d& dest)
	{
		float r[3][3];
		Mat kr = K * R;
		r[0][0] = (float)kr.at<double>(0);
		r[0][1] = (float)kr.at<double>(1);
		r[0][2] = (float)kr.at<double>(2);

		r[1][0] = (float)kr.at<double>(3);
		r[1][1] = (float)kr.at<double>(4);
		r[1][2] = (float)kr.at<double>(5);

		r[2][0] = (float)kr.at<double>(6);
		r[2][1] = (float)kr.at<double>(7);
		r[2][2] = (float)kr.at<double>(8);

		float tt[3];
		tt[0] = (float)t.at<double>(0);
		tt[1] = (float)t.at<double>(1);
		tt[2] = (float)t.at<double>(2);

		const float x = (float)xyz.x + tt[0];
		const float y = (float)xyz.y + tt[1];
		const float z = (float)xyz.z + tt[2];

		const float div = 1.f / (r[2][0] * x + r[2][1] * y + r[2][2] * z);
		dest.x = (r[0][0] * x + r[0][1] * y + r[0][2] * z) * div;
		dest.y = (r[1][0] * x + r[1][1] * y + r[1][2] * z) * div;
	}

	void RGBHistogram::convertRGBto3D(Mat& src, Mat& rgb)
	{
		if (rgb.size().area() != src.size().area()) rgb.create(src.size().area(), 1, CV_32FC3);

		const Vec3f centerv = Vec3f(center.at<float>(0), center.at<float>(1), center.at<float>(2));
		if (src.depth() == CV_8U)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				rgb.at<Vec3f>(i) = Vec3f(src.at<Vec3b>(i)) - centerv;
			}
		}
		else if (src.depth() == CV_32F)
		{
			for (int i = 0; i < src.size().area(); i++)
			{
				rgb.at<Vec3f>(i) = src.at<Vec3f>(i) - centerv;
			}
		}
	}

	RGBHistogram::RGBHistogram()
	{
		center.create(1, 3, CV_32F);
		center.at<float>(0) = 127.5f;
		center.at<float>(1) = 127.5f;
		center.at<float>(2) = 127.5f;
	}

	void RGBHistogram::setCenter(Mat& src)
	{
		src.copyTo(center);
	}

	void RGBHistogram::push_back(Mat& src)
	{
		const Vec3f centerv = Vec3f(center.at<float>(0), center.at<float>(1), center.at<float>(2));
		for (int i = 0; i < src.rows; i++)
		{
			additionalPoints.push_back(Vec3f(src.at<float>(i, 0), src.at<float>(i, 1), src.at<float>(i, 2)) - centerv);
		}
	}

	void RGBHistogram::push_back(Vec3f src)
	{
		additionalPoints.push_back(src - Vec3f(127.5f, 127.5f, 127.5f));
	}

	void RGBHistogram::push_back(const float b, const float g, const float r)
	{
		push_back(Vec3f(b, g, r));
	}

	void RGBHistogram::push_back_line(Mat& src, Mat& dest)
	{
		const Vec3f centerv = Vec3f(center.at<float>(0), center.at<float>(1), center.at<float>(2));
		for (int i = 0; i < src.rows; i++)
		{
			additionalStartLines.push_back(Vec3f(src.at<float>(i, 0), src.at<float>(i, 1), src.at<float>(i, 2)) - centerv);
			additionalEndLines.push_back(Vec3f(dest.at<float>(i, 0), dest.at<float>(i, 1), dest.at<float>(i, 2)) - centerv);
		}
	}

	void RGBHistogram::push_back_line(Vec3f src, Vec3f dest)
	{
		additionalStartLines.push_back(src - Vec3f(center.at<float>(0), center.at<float>(1), center.at<float>(2)));
		additionalEndLines.push_back(dest - Vec3f(center.at<float>(0), center.at<float>(1), center.at<float>(2)));
	}

	void RGBHistogram::push_back_line(const float b_s, const float g_s, const float r_s, const float b_d, const float g_d, const float r_d)
	{
		push_back_line(Vec3f(b_s, g_s, r_s), Vec3f(b_d, g_d, r_d));
	}

	void RGBHistogram::clear()
	{
		additionalPoints.release();
		additionalStartLines.release();
		additionalEndLines.release();
	}

	void RGBHistogram::plot(Mat& src, bool isWait, string wname)
	{
		//setup gui
		namedWindow(wname);
		static int sw = 0; createTrackbar("sw", wname, &sw, 2);
		static int sw_projection = 0; createTrackbar("projection", wname, &sw_projection, 1);
		static int f = 1000;  createTrackbar("f", wname, &f, 2000);
		static int z = 1250; createTrackbar("z", wname, &z, 2000);
		static int pitch = 90; createTrackbar("pitch", wname, &pitch, 360);
		static int roll = 0; createTrackbar("roll", wname, &roll, 180);
		static int yaw = 90; createTrackbar("yaw", wname, &yaw, 360);
		static int isDrawPCA = 1; createTrackbar("PCA", wname, &isDrawPCA, 1);
		static Point ptMouse = Point(cvRound((size.width - 1) * 0.75), cvRound((size.height - 1) * 0.25));

		cv::setMouseCallback(wname, (MouseCallback)onMouseHistogram3D, (void*)&ptMouse);

		//project RGB2XYZ
		Mat rgb;
		convertRGBto3D(src, rgb);

		Mat evec, eval, mean, dest;
		cvtColorPCA(rgb, dest, 1, evec, eval, mean);

		Mat xyz;
		vector<Point2f> pt(rgb.size().area());

		//set up etc plots
		Mat guide, guideDest;
		guide.push_back(Point3f(-127.5f, -127.5f, -127.5f)); //rgbzerof;
		guide.push_back(Point3f(-127.5, -127.5, 127.5)); //rmax
		guide.push_back(Point3f(-127.5, 127.5, -127.5)); //gmax
		guide.push_back(Point3f(127.5, -127.5, -127.5)); //rmax
		guide.push_back(Point3f(127.5, 127.5, 127.5)); //rgbmax
		guide.push_back(Point3f(-127.5, 127.5, 127.5)); //grmax
		guide.push_back(Point3f(127.5, -127.5, 127.5)); //brmax
		guide.push_back(Point3f(127.5, 127.5, -127.5)); //bgmax

		//guide.push_back(Point3f(0.f, 0.f, 0.f)); //eigen 0
		Point3f cnt = Point3f((float)mean.at<double>(0), (float)mean.at<double>(1), (float)mean.at<double>(2));
		float eveclen = 127.5;
		guide.push_back(cnt); //eigen 0
		guide.push_back(cnt + Point3f((float)evec.at<double>(0, 0), (float)evec.at<double>(0, 1), (float)evec.at<double>(0, 2)) * eveclen); //eigen 0
		guide.push_back(cnt + Point3f((float)evec.at<double>(1, 0), (float)evec.at<double>(1, 1), (float)evec.at<double>(1, 2)) * eveclen); //eigen 1
		guide.push_back(cnt + Point3f((float)evec.at<double>(2, 0), (float)evec.at<double>(2, 1), (float)evec.at<double>(2, 2)) * eveclen); //eigen 2

		vector<Point2f> guidept(guide.rows);
		vector<Point2f> additionalpt(additionalPoints.rows);
		vector<Point2f> additional_start_line(additionalStartLines.rows);
		vector<Point2f> additional_end_line(additionalEndLines.rows);

		int key = 0;
		Mat show = Mat::zeros(size, CV_8UC3);
		while (key != 'q')
		{
			pitch = (int)(180 * (double)ptMouse.y / (double)(size.height - 1) + 0.5);
			yaw = (int)(180 * (double)ptMouse.x / (double)(size.width - 1) + 0.5);
			setTrackbarPos("pitch", wname, pitch);
			setTrackbarPos("yaw", wname, yaw);

			//intrinsic
			k.at<double>(0, 2) = (size.width - 1) * 0.5;
			k.at<double>(1, 2) = (size.height - 1) * 0.5;
			k.at<double>(0, 0) = show.cols * 0.001 * f;
			k.at<double>(1, 1) = show.cols * 0.001 * f;
			t.at<double>(2) = z - 800;

			//rotate & plot RGB plots
			Mat rot;
			cp::Eular2Rotation(pitch - 90.0, roll - 90, yaw - 90, rot);
			cp::moveXYZ(rgb, xyz, rot, Mat::zeros(3, 1, CV_64F), true);
			if (sw_projection == 0) projectPointsParallel(xyz, R, t, k, pt, true);
			if (sw_projection == 1) projectPoints(xyz, R, t, k, pt, true);

			//rotate & plot guide information
			cp::moveXYZ(guide, guideDest, rot, Mat::zeros(3, 1, CV_64F), true);
			if (sw_projection == 0) projectPointsParallel(guideDest, R, t, k, guidept, true);
			if (sw_projection == 1) projectPoints(guideDest, R, t, k, guidept, true);

			//rotate & plot additionalPoint
			if (!additionalPoints.empty())
			{
				cp::moveXYZ(additionalPoints, additionalPointsDest, rot, Mat::zeros(3, 1, CV_64F), true);
				if (sw_projection == 0) projectPointsParallel(additionalPointsDest, R, t, k, additionalpt, true);
				if (sw_projection == 1) projectPoints(additionalPointsDest, R, t, k, additionalpt, true);
			}

			if (!additionalStartLines.empty())
			{
				cp::moveXYZ(additionalStartLines, additionalStartLinesDest, rot, Mat::zeros(3, 1, CV_64F), true);
				cp::moveXYZ(additionalEndLines, additionalEndLinesDest, rot, Mat::zeros(3, 1, CV_64F), true);
				if (sw_projection == 0) projectPointsParallel(additionalStartLinesDest, R, t, k, additional_start_line, true);
				if (sw_projection == 1) projectPoints(additionalStartLinesDest, R, t, k, additional_start_line, true);
				if (sw_projection == 0) projectPointsParallel(additionalEndLinesDest, R, t, k, additional_end_line, true);
				if (sw_projection == 1) projectPoints(additionalEndLinesDest, R, t, k, additional_end_line, true);
			}

			//draw lines for etc points
			Point rgbzero = Point(cvRound(guidept[0].x), cvRound(guidept[0].y));
			Point rmax = Point(cvRound(guidept[1].x), cvRound(guidept[1].y));
			Point gmax = Point(cvRound(guidept[2].x), cvRound(guidept[2].y));
			Point bmax = Point(cvRound(guidept[3].x), cvRound(guidept[3].y));
			Point bgrmax = Point(cvRound(guidept[4].x), cvRound(guidept[4].y));
			Point grmax = Point(cvRound(guidept[5].x), cvRound(guidept[5].y));
			Point brmax = Point(cvRound(guidept[6].x), cvRound(guidept[6].y));
			Point bgmax = Point(cvRound(guidept[7].x), cvRound(guidept[7].y));
			Point ezero = Point(cvRound(guidept[8].x), cvRound(guidept[8].y));
			Point eigenx = Point(cvRound(guidept[9].x), cvRound(guidept[9].y));
			Point eigeny = Point(cvRound(guidept[10].x), cvRound(guidept[10].y));
			Point eigenz = Point(cvRound(guidept[11].x), cvRound(guidept[11].y));
			line(show, bgmax, bgrmax, COLOR_WHITE);
			line(show, brmax, bgrmax, COLOR_WHITE);
			line(show, grmax, bgrmax, COLOR_WHITE);
			line(show, bgmax, bmax, COLOR_WHITE);
			line(show, brmax, bmax, COLOR_WHITE);
			line(show, brmax, rmax, COLOR_WHITE);
			line(show, grmax, rmax, COLOR_WHITE);
			line(show, grmax, gmax, COLOR_WHITE);
			line(show, bgmax, gmax, COLOR_WHITE);
			circle(show, rgbzero, 3, COLOR_WHITE, cv::FILLED);
			arrowedLine(show, rgbzero, rmax, COLOR_RED, 2);
			arrowedLine(show, rgbzero, gmax, COLOR_GREEN, 2);
			arrowedLine(show, rgbzero, bmax, COLOR_BLUE, 2);
			if (isDrawPCA == 1)
			{
				arrowedLine(show, ezero, eigenx, COLOR_WHITE, 3);
				arrowedLine(show, ezero, eigeny, COLOR_GRAY200, 2);
				arrowedLine(show, ezero, eigenz, COLOR_GRAY100, 2);
			}
			//arrowedLine(show, rgbzero, bgrmax, Scalar::all(50), 1);


			//rendering RGB plots
			if (sw == 0)
			{
				for (int i = 0; i < src.size().area(); i++)
				{
					const int x = cvRound(pt[i].x);
					const int y = cvRound(pt[i].y);
					int inc = 5;
					if (x >= 0 && x < show.cols && y >= 0 && y < show.rows)
					{
						show.at<Vec3b>(Point(x, y)) += Vec3b(inc, inc, inc);
					}
				}
			}
			else if (sw == 1)
			{
				for (int i = 0; i < src.size().area(); i++)
				{
					const int x = cvRound(pt[i].x);
					const int y = cvRound(pt[i].y);
					if (x >= 0 && x < show.cols && y >= 0 && y < show.rows)
					{
						Vec3f v = rgb.at<Vec3f>(i) + Vec3f(center);
						show.at<Vec3b>(Point(x, y)) = v;
					}
				}
			}
			else if (sw == 2)
			{
				for (int i = 0; i < src.size().area(); i++)
				{
					const int x = cvRound(pt[i].x);
					const int y = cvRound(pt[i].y);
					int inc = 2;
					if (x >= 0 && x < show.cols && y >= 0 && y < show.rows)
					{
						show.at<Vec3b>(Point(x, y)) = Vec3b(128, 128, 128);
					}
				}
			}

			//rendering additional points
			if (!additionalPoints.empty())
			{
				for (int i = 0; i < additionalPoints.size().area(); i++)
				{
					const int x = cvRound(additionalpt[i].x);
					const int y = cvRound(additionalpt[i].y);

					if (x >= 0 && x < show.cols && y >= 0 && y < show.rows)
					{
						circle(show, Point(x, y), 2, COLOR_WHITE);
					}
				}
			}

			if (!additionalStartLines.empty())
			{
				for (int i = 0; i < additionalStartLines.size().area(); i++)
				{
					const int sx = cvRound(additional_start_line[i].x);
					const int sy = cvRound(additional_start_line[i].y);
					const int ex = cvRound(additional_end_line[i].x);
					const int ey = cvRound(additional_end_line[i].y);
					//if (sx >= 0 && sx < show.cols && sy >= 0 && sy < show.rows)
					{
						line(show, Point(sx, sy), Point(ex, ey), COLOR_WHITE, 2);
					}
				}
			}

			imshow(wname, show);
			key = waitKeyEx(1);
			if (key == 'v')
			{
				sw++;
				if (sw > 2)sw = 0;
				setTrackbarPos("sw", wname, sw);
			}
			if (key == 's')
			{
				cout << "write rgb_histogram.png" << endl;
				imwrite("rgb_histogram.png", show);
			}
			if (key == 't')
			{
				ptMouse.x = cvRound((size.width - 1) * 0.5);
				ptMouse.y = cvRound((size.height - 1) * 0.5);
			}
			if (key == 'r')
			{
				ptMouse.x = cvRound((size.width - 1) * 0.75);
				ptMouse.y = cvRound((size.height - 1) * 0.25);
			}
			if (key == '?')
			{
				cout << "v: switching rendering method" << endl;
				cout << "r: reset viewing direction for parallel view" << endl;
				cout << "t: reset viewing direction for paspective view" << endl;
				cout << "s: save 3D RGB plot" << endl;
			}
			show.setTo(0);
			if (!isWait)break;
		}
	}
#pragma endregion

}