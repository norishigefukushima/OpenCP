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
		int ret = 0;
		//ret = fseek(fp, 0, SEEK_END);
		ret = _fseeki64(fp, 0, SEEK_END);
		if (ret != 0)cout << "fseek error: readHeader" << endl;
		//fileSize = ftell(fp);
		fileSize = _ftelli64(fp);
		if (fileSize == -1)cout << "ftell error: readHeader" << endl;
		//ret = fseek(fp, 0, SEEK_SET);
		ret = _fseeki64(fp, 0, SEEK_SET);
		if (ret != 0)cout << "fseek error: readHeader" << endl;

		/*char buff[1000];
		size_t line = 0;
		while (fgets(buff, 1000, fp) != NULL)
		{
			line++;
		}
		cout << line << endl;
		ret = _fseeki64(fp, 0, SEEK_SET);*/

		int countSep = 0;
		cv::AutoBuffer<char> str(INT_MAX);
		fgets(str, INT_MAX, fp);
		size_t size = strlen(str);

		for (int i = 0; i < size; i++)
		{
			switch (str[i])
			{
			case ',': countSep++;  break;
			}
		}
		width = countSep + 1;
	}

	void CSV::readDataLineByLine()
	{
		char vv[1024];
		char str[1024];
		size_t line = 0;
		vector<double> v;
		while (fgets(str, 1024, fp) != NULL)
		{
			size_t size = strlen(str);
			int c = 0;
			for (size_t i = 0; i < size; i++)
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

					data.push_back(v);
					v.clear();
					break;
				}
				else
				{
					vv[c] = str[i];
					c++;
				}
			}
			line++;
		}
		cout << "line " << line << endl;
	}

	void CSV::readData()
	{
		/*size_t line = 1;
		{
			//line count
			cv::AutoBuffer<char> buff(8);
			while (fgets(buff, 8, fp) != NULL)
			{
				line++;
			}
		}*/

		int code = _fseeki64(fp, 0, SEEK_SET);
		if (code != 0)cout << "fseek error: readData Init" << endl;

		char localBuffer[1024];
		char* str = nullptr;
		try
		{
			str = new char[fileSize];
		}
		catch (bad_alloc)
		{
			cerr << "bad_alloc " << fileSize << endl;
		}

		size_t ret = fread(str, sizeof(char), fileSize, fp);
		cout << "ret fread/fileSize: "<<ret << "/"<<fileSize<<"| "<<fileSize-ret << endl;
		/*for (int i = 0; i < 20; i++)
		{
			cout << str[i] << " ";
		}
		 cout<< endl;
		*/
		int c = 0;
		vector<double> v;
		//int start = 3;//with BOM
		int start = 0;//without BOM
		for (size_t i = start; i < ret; i++)
		{
			if (str[i] == ',')
			{
				localBuffer[c] = '\0';
				const double d = atof(localBuffer);
				/*if (d < 2)
				{
					cout << i << endl;
					cout << d << endl; getchar();
				}*/
				c = 0;
				v.push_back(d);
			}
			else if (str[i] == '\n')
			{
				//if (data.size() % 1000000 == 0)cout << data.size() << "/" << line << endl;
				localBuffer[c] = '\0';
				const double d = atof(localBuffer);
				c = 0;
				v.push_back(d);

				/*for(int n=0;n<v.size();n++)
				cout<<v[n]<<",";
				cout<<endl;*/

				data.push_back(v);
				v.clear();
			}
			else if (i == ret - 1)
			{
				localBuffer[c] = '\0';
				const double d = atof(localBuffer);
				v.push_back(d);
				data.push_back(v);
			}
			else
			{
				localBuffer[c] = str[i];
				c++;
			}
		}
		
		delete[] str;
	}

	cv::Mat CSV::getMat()
	{
		readData();
		cv::Mat_<double> ret(data.size(), data[0].size());
		for (int j = 0; j < data.size(); j++)
		{
			for (int i = 0; i < data[0].size(); i++)
			{
				ret(j, i) = data[j][i];
			}
		}
		return ret;
	}

	void CSV::init(string name, bool isWrite, bool isClear, bool isHeader)
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
				//if(isHeader) 
				readHeader();
			}
		}
	}

	CSV::CSV()
	{
		fp = nullptr;
	}

	CSV::CSV(string name, bool isWrite, bool isClear, bool isHeader)
	{
		fp = nullptr;
		init(name, isWrite, isClear, isHeader);
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