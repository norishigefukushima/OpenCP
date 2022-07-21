#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void testCorrelationCoefficient()
{
	cp::SpearmanRankOrderCorrelationCoefficient srcc;
	cp::KendallRankOrderCorrelationCoefficient krcc;
	RNG rng;
	for (int s = 5000; s < 10000; s += 1000)
	{
		vector<double> v1d(s);
		vector<double> v2d(s);
		vector<float> v1f(s);
		vector<float> v2f(s);
		vector<int> v1i(s);
		vector<int> v2i(s);
		for (int i = 0; i < s; i++)
		{
			const double v = rng.uniform(0.0, 100.0);
			v1f[i] = v1d[i] = v;
			v1i[i] = (int)v;
			//double v2 = rng.uniform(-50.0, 50.0);
			double v2 = rng.uniform(-10, 10);
			v2f[i] = v2d[i] = v + v2;
			v2i[i] = (int)v + (int)v2;
		}

		bool isComputeTime = true;

		bool isIgnoringTie = true;
		bool isUsingTie = true;
		bool isReference = true;
		bool isKROCC = true;
		const int iteration = 1;

		if (isComputeTime)
		{
			if (isIgnoringTie)
			{
				{
					Timer t(format("ig_i %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.compute(v1i, v2i, true);
				}
				{
					Timer t(format("ig_f %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.compute(v1f, v2f, true);
				}
				{
					Timer t(format("ig_d %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.compute(v1d, v2d, true);
				}
			}

			if (isUsingTie)
			{
				{
					Timer t(format("fl_i %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.compute(v1i, v2i);
				}
				{
					Timer t(format("fl_f %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.compute(v1f, v2f);
				}
				{
					Timer t(format("fl_d %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.compute(v1d, v2d);
				}
			}

			if (isReference)
			{
				srcc.setReference(v2i);
				{
					Timer t(format("rf_i %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.computeUsingReference(v1i);
				}
				srcc.setReference(v2f);
				{
					Timer t(format("rf_f %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.computeUsingReference(v1f);
				}
				srcc.setReference(v2d);
				{
					Timer t(format("rf_d %05d: ", s));
					for (int i = 0; i < iteration; i++)
						srcc.computeUsingReference(v1d);
				}
			}

			if (isKROCC)
			{
				{
					Timer t(format("kr_i %05d: ", s));
					for (int i = 0; i < iteration; i++)
						krcc.compute(v1i, v2i);
				}
				{
					Timer t(format("kr_f %05d: ", s));
					for (int i = 0; i < iteration; i++)
						krcc.compute(v1f, v2f);
				}
				{
					Timer t(format("kr_d %05d: ", s));
					for (int i = 0; i < iteration; i++)
						krcc.compute(v1d, v2d);
				}
			}
		}

		srcc.setReference(v2i);
		cout << s << "(int)   : " << srcc.compute(v1i, v2i, true) << "," << srcc.compute(v1i, v2i) << "," << srcc.computeUsingReference(v1i) << endl;
		srcc.setReference(v2f);
		cout << s << "(float) : " << srcc.compute(v1f, v2f, true) << "," << srcc.compute(v1f, v2f) << "," << srcc.computeUsingReference(v1f) << endl;
		srcc.setReference(v2d);
		cout << s << "(double): " << srcc.compute(v1d, v2d, true) << "," << srcc.compute(v1d, v2d) << "," << srcc.computeUsingReference(v1d) << endl;
	}
}