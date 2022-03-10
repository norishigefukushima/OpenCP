#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void testPlot()
{
	//if (false)
	{
		Plot test;
		test.setPlotTitle(0, "sin(x)");
		test.setPlotTitle(1, "y=100");
		test.setPlotTitle(2, "x=0.5");
		test.setXLabel("x label");
		test.setYLabel("y label");
		test.setYRange(-1, 1);
		test.setXRange(0, 320);
		const int amp = 100;
		double scale = 0.95;
		for (int i = 0; i < int(CV_PI * amp); i++)
		{
			test.push_back(i, scale * sin(CV_PI * i * 1.0 / amp));
		}
		test.push_back_HLine(0.5, 1);
		test.push_back_VLine(100, 2);
		test.plot("test1");
	}

	//if (false)
	{
		Plot test;
		test.setPlotTitle(0, "sin(x)");
		test.setXLabel("x label");
		test.setYLabel("y label");
		test.setYRange(-1, 1);
		test.setXRange(0, 320);
		const int amp = 100;

		for (int i = 0; i < int(CV_PI * amp); i++)
		{
			test.push_back(i, sin(CV_PI * i * 1.0 / amp));
		}

		test.plot("test2");
	}

	//if (false)
	{
		Plot test;
		test.setXLabel("x label");
		test.setYLabel("y label");
		for (int i = 0; i < 10; i++)
		{
			test.push_back(i, 2 * i);
			test.push_back(i, 3 * i, 1);
		}

		test.erase(3, 0);
		test.insert(3, 10, 3, 0);

		test.plot("test3");
	}

	//if (false)
	{
		Plot test;
		test.setXLabel("x label");
		test.setYLabel("y label");
		for (int i = 0; i < 10; i++)
		{
			test.push_back(i, 2 * i);
			test.push_back(i, 3 * i, 1);
		}

		test.erase(3, 0);
		test.insert(3, 10, 3, 0);

		test.plot("test4");
	}

	//if (false)
	{
		Plot test;
		test.setXLabel("log(x) label");
		test.setLogScaleX(true);
		test.setYLabel("y label");

		for (int i = 1; i < 10000; i++)
		{
			test.push_back(i, i, 0);
		}
		test.plot("test5");
	}

	Plot test;
	RNG rng;
	for (int j = 0; j < 100; j++)
	{
		for (int i = 0; i < 10; i++)
		{
			test.push_back(i, rng.uniform(0, i), 0);
			test.push_back(i, rng.uniform(0, i), 1);
		}

		test.plot("test6", false);
	}
}

void testPlot2D()
{
	double xmin = 6;
	double xmax = 32;
	double xstep = 1;
	double ymin = 19;
	double ymax = 128;
	double ystep = 1;

	Plot2D p(Size(512, 512), xmin, xmax, xstep, ymin, ymax, ystep);
	p.setLabel("sigma space", "sigma range", "PSNR [dB]");
	RNG rng(getTickCount());

	double rx = rng.uniform(xmin, xmax);
	double ry = rng.uniform(ymin, ymax);
	double sigma = (ymax - ymin) / (2.0 * 3.0);

	double zmin = 30;
	double zmax = 55;
	p.setZMinMax(zmin, zmax);
	p.setLabelXGreekLetter("s", "s");
	p.setLabelYGreekLetter("s", "r");
	p.setPlotContours("40 dB", 40, 0);
	p.setPlotContours("45 dB", 45, 1);
	p.setPlotContours("50 dB", 50, 2);
	for (double y = ymin; y <= ymax; y += ystep)
	{
		for (double x = xmin; x <= xmax; x += xstep)
		{
			//p.add(j, i, rand());
			//double v = rng.uniform(0.0, 5.0);
			double v = exp(((y - ry) * (y - ry) + (x - rx) * (x - rx)) / (-2.0 * sigma * sigma)) * (zmax - zmin) + zmin;
			p.add(x, y, v);
		}
	}

	p.plot("a");
}