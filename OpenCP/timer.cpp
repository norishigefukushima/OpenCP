#include "opencp.hpp"
using namespace std;

void CalcTime::start()
{
	pre = getTickCount();
}

void CalcTime::restart()
{
	start();
}

void CalcTime::lap(string message)
{
	string v = message + format(" %f",getTime());
	switch(timeMode)
	{
	case TIME_NSEC:
		v += " NSEC";
		break;
	case TIME_SEC:
		v += " SEC";
		break;
	case TIME_MIN:
		v += " MIN";
		break;
	case TIME_HOUR:
		v += " HOUR";
		break;

	case TIME_MSEC:
	default:
		v += " msec";
		break;
	}


	lap_mes.push_back(v);
	restart();
}
void CalcTime:: show()
{
	getTime();

	int mode = timeMode;
	if(timeMode==TIME_AUTO)
	{
		mode = autoMode;
	}

	switch(mode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;

	case TIME_MSEC:
	default:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	}
}

void CalcTime:: show(string mes)
{
	getTime();

	int mode = timeMode;
	if(timeMode==TIME_AUTO)
	{
		mode = autoMode;
	}

	switch(mode)
	{
	case TIME_NSEC:
		cout<< mes<< ": "<<cTime<<" nsec"<<endl;
		break;
	case TIME_SEC:
		cout<< mes<< ": "<<cTime<<" sec"<<endl;
		break;
	case TIME_MIN:
		cout<< mes<< ": "<<cTime<<" minute"<<endl;
		break;
	case TIME_HOUR:
		cout<< mes<< ": "<<cTime<<" hour"<<endl;
		break;
	case TIME_DAY:
		cout<< mes<< ": "<<cTime<<" day"<<endl;
	case TIME_MSEC:
		cout<<mes<< ": "<<cTime<<" msec"<<endl;
		break;
	default:
		cout<<mes<< ": error"<<endl;
		break;
	}
}

int CalcTime::autoTimeMode()
{
	if(cTime>60.0*60.0*24.0)
	{
		return TIME_DAY;
	}
	else if(cTime>60.0*60.0)
	{
		return TIME_HOUR;
	}
	else if(cTime>60.0)
	{
		return TIME_MIN;
	}
	else if(cTime>1.0)
	{
		return TIME_SEC;
	}
	else if(cTime>1.0/1000.0)
	{
		return TIME_MSEC;
	}
	else
	{

		return TIME_NSEC;
	}
}
double CalcTime:: getTime()
{
	cTime = (getTickCount()-pre)/(getTickFrequency());

	int mode=timeMode;
	if(mode==TIME_AUTO)
	{
		mode = autoTimeMode();
		autoMode=mode;
	}

	switch(mode)
	{
	case TIME_NSEC:
		cTime*=1000000.0;
		break;
	case TIME_SEC:
		cTime*=1.0;
		break;
	case TIME_MIN:
		cTime /=(60.0);
		break;
	case TIME_HOUR:
		cTime /=(60*60);
		break;
	case TIME_DAY:
		cTime /=(60*60*24);
		break;
	case TIME_MSEC:
	default:
		cTime *=1000.0;
		break;
	}
	return cTime;
}
void CalcTime::setMessage(string src)
{
	mes=src;
}
void CalcTime:: setMode(int mode)
{
	timeMode = mode;
}

void CalcTime::init(string message, int mode, bool isShow)
{
	_isShow = isShow;
	timeMode = mode;

	setMessage(message);
	start();
}


CalcTime::CalcTime()
{
	init("time ", TIME_AUTO, true);
}

CalcTime::CalcTime(string message,int mode,bool isShow)
{
	init(message,mode,isShow);
}
CalcTime::~CalcTime()
{
	getTime();
	if(_isShow)	show();
	if(lap_mes.size()!=0)
	{
		for(int i=0;i<lap_mes.size();i++)
		{
			cout<<lap_mes[i]<<endl;
		}
	}
}