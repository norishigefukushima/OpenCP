#include <opencp.hpp>

using namespace std;
using namespace cv;
using namespace cp;

void jpegtest()
{
	vector<int> iindex = { 3,6,7,148 };
	vector<string> t1v;
	t1v.push_back("inter_1.png");
	t1v.push_back("inter_3.png");
	t1v.push_back("inter_4.png");
	t1v.push_back("intra_1.png");
	t1v.push_back("intra_2.png");
	t1v.push_back("intra_4.png");

	vector<string> t2v;
	t2v.push_back("inter_2.png");
	t2v.push_back("inter_3.png");
	t2v.push_back("inter_4.png");
	t2v.push_back("inter_5.png");
	t2v.push_back("intra_1.png");
	t2v.push_back("intra_2.png");
	t2v.push_back("intra_4.png");

	vector<string> t3v;
	t3v.push_back("inter_1.png");
	t3v.push_back("inter_2.png");
	t3v.push_back("inter_4.png");
	t3v.push_back("intra_1.png");
	t3v.push_back("intra_2.png");
	t3v.push_back("intra_4.png");

	Mat rd;
	for (int n = 0; n < iindex.size(); n++)
	{
		const int num = iindex[n];
		Mat ref = imread(format("./csv/img/ref_%d.png", num));

		vector<string> tt;
		if (n == 1)tt = t2v;
		else if (n == 2)tt = t3v;
		else tt = t1v;

		for (int t = 0; t < tt.size(); t++)
		{
			//Mat dist = imread("./csv/img/distort_3_jpegCompression_" + t1v[t]);
			Mat dist = imread(format("./csv/img/distort_%d_jpegCompression_", num) + tt[t]);
			double maxval = 0.0;
			int argq = 0;
			for (int q = 1; q < 100; q++)
			{
				cp::addJPEGNoise(ref, rd, q);
				double val = getPSNR(rd, dist);
				if (val > maxval)
				{
					maxval = val;
					argq = q;
				}
				//cout << q << ": " << getPSNR(rd, dist) << endl;
			}
			cout << num << ": " << tt[t] << ": " << argq << endl;
		}
	}
	getchar();
}
void anaCSV()
{
	string oname = "./csv/out.csv";
	CSV ana(oname, false, false, false, false);
	ana.readDataLineByLineAsString();

	//1, train, ref_1.png, jpegCompression, inter, 5, medianDenoisesaltPepper, inter, 5, , 0.881
	vector<vector<int>> stat(200);
	for (int i = 0; i < 200; i++)
		stat[i].resize(5);
	for (int i = 0; i < ana.datastr.size(); i++)
	{
		
		if (ana.datastr[i][3] == "jpegCompression")
		{
			int idx = atoi(ana.datastr[i][0].c_str()) - 1;
			int ldx = atoi(ana.datastr[i][5].c_str()) - 1;
			stat[idx][ldx]++;
			stat[idx][ldx] = min(1, stat[idx][ldx]);
		}
		if (ana.datastr[i][6] == "jpegCompression")
		{
			int idx = atoi(ana.datastr[i][0].c_str()) - 1;
			int ldx = atoi(ana.datastr[i][8].c_str()) - 1;
			stat[idx][ldx]++;
			stat[idx][ldx] = min(1, stat[idx][ldx]);
		}
		//0: index
		//3: type1
		//5: level1
		//6: type2
		//8: level2
	}
	string jname = "./csv/jpeg.csv";
	CSV out(jname, true, true, false, false);
	for (int i = 0; i < 200; i++)
	{
		out.write(i+1);
		for (int j = 0; j < 5; j++)
		{
			out.write(stat[i][j]);
		}
		out.end();
	}
}
void testCSV()
{
	jpegtest(); return;
	anaCSV(); return;
	string oname = "./csv/out.csv";
	CSV out(oname, true, true, false, false);

	for (int i = 0; i < 140; i++)
	{
		const int index = i + 1;
		string name = format("./csv/train/ref_%d_pairwise_labels.csv", index);
		CSV csv(name, false, false, true, false);
		csv.readDataLineByLineAsString();
		for (int n = 0; n < csv.datastr.size(); n++)
		{
			out.write(index);
			out.write("train");
			for (int m = 0; m < csv.datastr[0].size(); m++)
			{
				if (m == 1 || m == 2)
				{
					//out.write(csv.datastr[n][m]);
					vector<string> v = string_split(csv.datastr[n][m], '_');

					if (v.size() == 5)
					{
						out.write(v[2]);
						out.write(v[3]);
						vector<string> vv = string_split(v[4], '.');
						out.write(vv[0]);
					}
					else if (v.size() == 6)
					{
						out.write(v[2] + v[3]);
						out.write(v[4]);
						vector<string> vv = string_split(v[5], '.');
						out.write(vv[0]);
					}
					else
					{
						out.write(v[2] + v[3] + v[4]);
						out.write(v[5]);
						vector<string> vv = string_split(v[6], '.');
						out.write(vv[0]);
					}
				}
				else
				{
					out.write(csv.datastr[n][m]);
				}
			}
			out.end();
		}
	}
	for (int i = 140; i < 160; i++)
	{
		const int index = i + 1;
		string name = format("./csv/val/ref_%d_pairwise_labels.csv", index);
		CSV csv(name, false, false, true, false);
		csv.readDataLineByLineAsString();
		for (int n = 0; n < csv.datastr.size(); n++)
		{
			out.write(index);
			out.write("val");
			for (int m = 0; m < csv.datastr[0].size(); m++)
			{
				if (m == 1 || m == 2)
				{
					//out.write(csv.datastr[n][m]);
					vector<string> v = string_split(csv.datastr[n][m], '_');

					if (v.size() == 5)
					{
						out.write(v[2]);
						out.write(v[3]);
						vector<string> vv = string_split(v[4], '.');
						out.write(vv[0]);
					}
					else if (v.size() == 6)
					{
						out.write(v[2] + v[3]);
						out.write(v[4]);
						vector<string> vv = string_split(v[5], '.');
						out.write(vv[0]);
					}
					else
					{
						out.write(v[2] + v[3] + v[4]);
						out.write(v[5]);
						vector<string> vv = string_split(v[6], '.');
						out.write(vv[0]);
					}
				}
				else
				{
					out.write(csv.datastr[n][m]);
				}
			}
			out.end();
		}
	}

	//ref.image, distorted image A, distorted image B, raw preference for A, processed preference for A
		//ref.image, distorted image A, distorted image B, preference for A
	for (int i = 160; i < 200; i++)
	{
		const int index = i + 1;
		string name = format("./csv/test/ref_%d_pairwise_labels.csv", index);
		CSV csv(name, false, false, true, false);
		csv.readDataLineByLineAsString();
		for (int n = 0; n < csv.datastr.size(); n++)
		{
			out.write(index);
			out.write("test");
			for (int m = 0; m < csv.datastr[0].size(); m++)
			{
				if (m == 1 || m == 2)
				{
					//out.write(csv.datastr[n][m]);
					vector<string> v = string_split(csv.datastr[n][m], '_');
					if (v.size() == 2)
					{
						vector<string> vv = string_split(v[1], '.');
						out.write("ref");
						out.write("");
						out.write("");
					}
					else if (v.size() == 4)
					{
						out.write(v[2]);
						out.write("");
						vector<string> vv = string_split(v[3], '.');
						out.write(vv[0]);
					}
					else if (v.size() == 5)
					{
						out.write(v[2] + v[3]);
						out.write("");
						vector<string> vv = string_split(v[4], '.');
						out.write(vv[0]);
					}
					else if (v.size() == 6)
					{
						out.write(v[2] + v[3] + v[4]);
						out.write("");
						vector<string> vv = string_split(v[5], '.');
						out.write(vv[0]);
					}
					else
					{
						cout << v.size() << endl;
					}
				}
				else
				{
					out.write(csv.datastr[n][m]);
				}
			}
			out.end();
		}
	}
}