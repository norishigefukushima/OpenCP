#include "csv.hpp"
#include <fstream>

using namespace std;

namespace cp
{

	void CSV::findMinMax(int result_index, bool isUseFilter, double minValue, double maxValue)
	{
		argMin.resize(data[0].size());
		argMax.resize(data[0].size());

		argMin[result_index] = DBL_MAX;
		argMax[result_index] = DBL_MIN;

		const int dsize = (int)argMin.size();
		if (isUseFilter)
		{
			for (int i = 0; i < data.size(); i++)
			{
				if (filter[i])
				{
					if (argMin[result_index] < data[i][result_index])
					{
						for (int j = 0; j < dsize; j++)
						{
							argMin[j] = data[i][j];
						}
					}
					if (argMax[result_index] > data[i][result_index])
					{
						for (int j = 0; j < dsize; j++)
						{
							argMax[j] = data[i][j];
						}
					}
				}
			}
		}
		else
		{
			for (int i = 0; i < data.size(); i++)
			{
				if (argMin[result_index] < data[i][result_index])
				{
					for (int j = 0; j < dsize; j++)
					{
						argMin[j] = data[i][j];
					}
				}
				if (argMax[result_index] > data[i][result_index])
				{
					for (int j = 0; j < dsize; j++)
					{
						argMax[j] = data[i][j];
					}
				}
			}
		}
		minValue = argMin[result_index];
		maxValue = argMax[result_index];
	}

	void CSV::initFilter()
	{
		filter.clear();
		for (int i = 0; i < data.size(); i++)
		{
			filter.push_back(true);
		}
	}

	void CSV::filterClear()
	{
		for (int i = 0; i < data.size(); i++)
		{
			filter[i] = true;
		}
	}

	void CSV::makeFilter(int index, double val, double emax)
	{
		for (int i = 0; i < data.size(); i++)
		{
			double diff = abs(data[i][index] - val);
			if (diff > emax)
			{
				filter[i] = false;
			}
		}
	}

	void CSV::readHeader()
	{
		fseek(fp, 0, SEEK_END);
		fileSize = ftell(fp);
		fseek(fp, 0, SEEK_SET);

		int countSep = 0;
		char str[1000];
		fgets(str, 1000, fp);
		for (int i = 0; i < strlen(str); i++)
		{
			switch (str[i])
			{
			case ',': countSep++;  break;
			}
		}
		width = countSep + 1;
	}

	void CSV::readData()
	{
		char vv[100];
		char* str = new char[fileSize];
		fileSize = (long)fread(str, sizeof(char), fileSize, fp);

		int c = 0;
		vector<double> v;
		for (int i = 0; i < fileSize; i++)
		{
			if (str[i] == ',')
			{
				vv[c] = '\0';
				double d = atof(vv);
				c = 0;
				v.push_back(d);
			}
			else if (str[i] == '\n')
			{
				vv[c] = '\0';
				double d = atof(vv);
				c = 0;
				v.push_back(d);

				/*for(int n=0;n<v.size();n++)
				cout<<v[n]<<",";
				cout<<endl;*/

				data.push_back(v);
				v.clear();
			}
			else
			{
				vv[c] = str[i];
				c++;
			}
		}

		delete[] str;
	}

	void CSV::init(string name, bool isWrite, bool isClear)
	{
		isTop = true;
		if (isWrite)
		{
			if (isClear)
			{
				fp = fopen(name.c_str(), "w");
			}
			else
			{
				fp = fopen(name.c_str(), "w+");
			}
			if (fp == NULL)
			{
				string n = name + "(1)";
				filename = n;
				fp = fopen(n.c_str(), "w");
			}
		}
		else
		{
			if (isClear)
			{
				fp = fopen(name.c_str(), "r");
			}
			else
			{
				fp = fopen(name.c_str(), "r+");
			}
			filename = name;
			if (fp == NULL)
			{
				cout << "file open error " << name << endl;
			}
			else
			{
				readHeader();
			}
		}
	}

	CSV::CSV()
	{
		fp = nullptr;
	}

	CSV::CSV(string name, bool isWrite, bool isClear)
	{
		fp = nullptr;
		init(name, isWrite, isClear);
	}

	CSV::~CSV()
	{
		if (fp != NULL) fclose(fp);
		//ifstream file1(filename);
		//ofstream file2("backup"+filename);
		//char ch;
		//while(file1 && file1.get(ch))
		//{
		//	file2.put(ch);
		//}
	}

	void CSV::write(string v)
	{
		if (isTop)
			fprintf(fp, "%s", v.c_str());
		else
		{
			if (isCommaSeparator)
				fprintf(fp, ",%s", v.c_str());
			else
				fprintf(fp, " %s", v.c_str());
		}

		isTop = false;
	}

	void CSV::write(int v)
	{
		if (isTop)
			fprintf(fp, "%d", v);
		else
		{
			if (isCommaSeparator)
				fprintf(fp, ",%d", v);
			else
				fprintf(fp, " %d", v);
		}

		isTop = false;
	}

	void CSV::write(double v)
	{
		if (isTop)
			fprintf(fp, "%f", v);
		else
		{
			if (isCommaSeparator)
			{
				fprintf(fp, ",%f", v);
			}
			else
			{
				fprintf(fp, " %f", v);
			}
		}
		isTop = false;
	}

	void CSV::end()
	{
		fprintf(fp, "\n");
		isTop = true;
	}

	void CSV::setSeparator(bool flag)
	{
		isCommaSeparator = flag;
	}

	template<typename srcType>
	void writeCSV_(string name, cv::Mat& src)
	{
		CSV csv(name, true, true);
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				csv.write(src.at<srcType>(i, j));
			}
			csv.end();
		}
	}

	void writeCSV(string name, cv::InputArray src_)
	{
		cv::Mat src = src_.getMat();
		CV_Assert(src.channels() == 1);

		switch (src.depth())
		{
		case CV_8U: writeCSV_<uchar>(name, src); break;
		case CV_16S: writeCSV_<short>(name, src); break;
		case CV_16U: writeCSV_<ushort>(name, src); break;
		case CV_32S: writeCSV_<int>(name, src); break;
		case CV_32F: writeCSV_<float>(name, src); break;
		case CV_64F: writeCSV_<double>(name, src); break;
		default:
			break;
		}
	}
}