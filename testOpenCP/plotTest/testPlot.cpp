#include <opencp.hpp>
using namespace std;
using namespace cv;
using namespace cp;

void testPlot()
{
	//if (false)
	{
		Plot test;
		test.setYRange(-1, 1);
		test.setXRange(0, 320);
		const int amp = 100;

		for (int i = 0; i < int(CV_PI * amp); i++)
		{
			test.push_back(i, sin(CV_PI * i * 1.0 / amp));
		}

		test.plot();
	}

	if (false)
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

		test.plot();
	}

	if (false)
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

		test.plot();
	}

	if (false)
	{
		Plot test;
		test.setXLabel("log(x) label");
		test.setLogScaleX(true);
		test.setYLabel("y label");

		for (int i = 1; i < 10000; i++)
		{
			test.push_back(i, i, 0);
		}
		test.plot();
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

		test.plot("loop", false);
	}
}

void testPlot2D()
{
	int xmin = 3;
	int xmax = 13;
	int xstep = 2;
	int ymin = 1;
	int ymax = 64;
	int ystep = 5;

	Plot2D p(Size(256, 256), xmin, xmax, xstep, ymin, ymax, ystep);
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
	for (int y = ymin; y <= ymax; y += ystep)
	{
		for (int x = xmin; x <= xmax; x += xstep)
		{
			//p.add(j, i, rand());
			//double v = rng.uniform(0.0, 5.0);
			double v = exp(((y - ry) * (y - ry) + (x - rx) * (x - rx)) / (-2.0 * sigma * sigma)) * (zmax - zmin) + zmin;
			p.add(x, y, v);
		}
	}

	p.plot("a");
}