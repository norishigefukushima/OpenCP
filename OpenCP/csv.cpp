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
		cv::AutoBuffer<char> str(CSV_READLINE_MAX);
		fgets(str, CSV_READLINE_MAX, fp);
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

	int CSV::readDataLineByLine(const bool isFirstString)
	{
		char* cell = (char*)_mm_malloc(sizeof(char) * CSV_READLINE_MAX, 32);
		char* str = (char*)_mm_malloc(sizeof(char) * CSV_READLINE_MAX, 32);

		size_t line = 0;
		vector<double> localvec;
		while (fgets(str, CSV_READLINE_MAX, fp) != NULL)
		{
			bool isFirst = true;
			size_t size = strlen(str);
			int c = 0;
			for (size_t i = 0; i < size; i++)
			{
				if (str[i] == ',')
				{
					cell[c] = '\0';
					c = 0;
					if (isFirst && isFirstString)
					{
						firstlabel.push_back(string(cell));
					}
					else
					{
						double d = atof(cell);
						localvec.push_back(d);
					}
					isFirst = false;
				}
				else if (str[i] == '\n')
				{
					cell[c] = '\0';
					double d = atof(cell);
					c = 0;
					localvec.push_back(d);

					data.push_back(localvec);
					localvec.clear();
					break;
				}
				else
				{
					cell[c] = str[i];
					c++;
				}
			}
			line++;
		}
		//cout << "num lines " << line << endl;
		_mm_free(cell);
		_mm_free(str);
		return (int)line;
	}

	int CSV::readDataLineByLineAsString()
	{
		char* cell = (char*)_mm_malloc(sizeof(char) * CSV_READLINE_MAX, 32);
		char* str = (char*)_mm_malloc(sizeof(char) * CSV_READLINE_MAX, 32);

		size_t line = 0;
		while (fgets(str, CSV_READLINE_MAX, fp) != NULL)
		{
			vector<string> localstr;
			size_t size = strlen(str);
			int c = 0;
			for (size_t i = 0; i < size; i++)
			{
				if (str[i] == ',')
				{
					cell[c] = '\0';
					c = 0;
					localstr.push_back(string(cell));
				}
				else if (str[i] == '\n')
				{
					cell[c] = '\0';
					localstr.push_back(string(cell));
					c = 0;

					datastr.push_back(localstr);
					break;
				}
				else
				{
					cell[c] = str[i];
					c++;
				}
			}
			line++;
		}
		//cout << "num lines " << line << endl;
		_mm_free(cell);
		_mm_free(str);
		return (int)line;
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
		cout << "ret fread/fileSize: " << ret << "/" << fileSize << "| " << fileSize - ret << endl;
		/*for (int i = 0; i < 20; i++)
		{
			cout << str[i] << " ";
		}
		 cout<< endl;
		*/
		int c = 0;
		vector<double> localvec;
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
				localvec.push_back(d);
			}
			else if (str[i] == '\n')
			{
				//if (data.size() % 1000000 == 0)cout << data.size() << "/" << line << endl;
				localBuffer[c] = '\0';
				const double d = atof(localBuffer);
				c = 0;
				localvec.push_back(d);

				/*for(int n=0;n<v.size();n++)
				cout<<v[n]<<",";
				cout<<endl;*/

				data.push_back(localvec);
				localvec.clear();
			}
			else if (i == ret - 1)
			{
				localBuffer[c] = '\0';
				const double d = atof(localBuffer);
				localvec.push_back(d);
				data.push_back(localvec);
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
		cv::Mat_<double> ret((int)data.size(), (int)data[0].size());
		for (int j = 0; j < data.size(); j++)
		{
			for (int i = 0; i < data[0].size(); i++)
			{
				ret(j, i) = data[j][i];
			}
		}
		return ret;
	}

	//file open then read header if true
	// if(isClear==true) clear csv files
	void CSV::init(string name, bool isWrite, bool isClear, bool isHeader, bool isBinary)
	{
		isTop = true;

		if (isWrite)
		{
			filename = name;
			string mode = (isBinary) ? "wb" : "w";
			if (isClear)
			{
				fp = fopen(name.c_str(), mode.c_str());
			}
			else
			{
				string m = "a+";
				fp = fopen(name.c_str(), m.c_str());
			}

			if (fp == nullptr)
			{
				string n = name + "(1)";
				filename = n;
				if (isClear)
				{
					fp = fopen(filename.c_str(), mode.c_str());
				}
				else
				{
					string m = "a+";
					fp = fopen(filename.c_str(), m.c_str());
				}
				if (fp == NULL)
				{
					cout << "file open error " << name << endl;
				}
			}
		}
		else
		{
			string mode = isBinary ? "rb" : "r";
			if (isClear)
			{
				fp = fopen(name.c_str(), mode.c_str());
			}
			else
			{
				string m = mode + "+";
				fp = fopen(name.c_str(), m.c_str());
			}
			filename = name;
			if (fp == nullptr)
			{
				cout << "file open error " << name << endl;
			}
			else
			{
				if (isHeader) readHeader();
			}
		}
	}

	CSV::CSV()
	{
		fp = nullptr;
	}

	CSV::CSV(string name, bool isWrite, bool isClear, bool isHeader, bool isBinary)
	{
		fp = nullptr;
		init(name, isWrite, isClear, isHeader, isBinary);
	}

	CSV::~CSV()
	{
		if (fp != nullptr) fclose(fp);
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
		if (isTop) fprintf(fp, "%s", v.c_str());
		else
		{
			if (isCommaSeparator) fprintf(fp, ",%s", v.c_str());
			else fprintf(fp, " %s", v.c_str());
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
		if (isTop) fprintf(fp, "%f", v);
		else
		{
			if (isCommaSeparator) fprintf(fp, ",%f", v);
			else fprintf(fp, " %f", v);
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

	void CSV::dumpDataAsBinary(string name, int depth)
	{
		FILE* file = fopen(name.c_str(), "wb");
		if (file == nullptr)
		{
			cout << "file open error " << name << endl;
			return;
		}

		if (depth == CV_32F)
		{
			for (int j = 0; j < data.size(); j++)
			{
				cv::AutoBuffer<float> buff(data[j].size());
				for (int i = 0; i < data[j].size(); i++)
				{
					buff[i] = (float)data[j][i];
				}
				fwrite(buff, sizeof(float), data[j].size(), file);
			}
		}
		else
		{
			for (int j = 0; j < data.size(); j++)
			{
				cv::AutoBuffer<double> buff(data[j].size());
				for (int i = 0; i < data[j].size(); i++)
				{
					buff[i] = data[j][i];
				}
				fwrite(buff, sizeof(double), data[j].size(), file);
			}
		}
		fclose(fp);
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