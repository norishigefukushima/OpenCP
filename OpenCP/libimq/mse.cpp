
#include "imq.h"

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double MSE16bit(_UINT16 *orig_img, _UINT16 *comp_img, _INT32 PX, _INT32 PY)
{

double MSE = 0.0;
_INT64 tmp = 0;
_INT32 size = PX * PY;

for(_INT32 pos = 0; pos < size; ++pos)  tmp += ((_INT64)orig_img[pos] - (_INT64)comp_img[pos])*((_INT64)orig_img[pos] - (_INT64)comp_img[pos]);
	
MSE = double(tmp) / double(size);

return MSE;
}


//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double MSE8bit(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY)
{

double MSE = 0.0;
_INT64 tmp = 0;
_INT32 size = PX * PY;

for(_INT32 pos = 0; pos < size; ++pos)  tmp += ((_INT64)orig_img[pos] - (_INT64)comp_img[pos])*((_INT64)orig_img[pos] - (_INT64)comp_img[pos]);
	
MSE = double(tmp) / double(size);

return MSE;
}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double DoMSE(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP) {

_INT32 size = PX * PY;
double result = 0.0;

switch(BPP) {


	case 8: {

		_BYTE *orig_imgb = NULL;
		_BYTE *comp_imgb = NULL;
		try {
		      orig_imgb = new _BYTE [ size ];
		      comp_imgb = new _BYTE [ size ];
		} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }

		for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (_BYTE)(orig_img[i]&_MBYTE); comp_imgb[i] = (_BYTE)(comp_img[i]&_MBYTE); }	
		
		result = MSE8bit(orig_imgb,comp_imgb,PX,PY);
		delete [] orig_imgb;
		delete [] comp_imgb;


	};
	break;

	case 16: {


		_UINT16 *orig_imgb = NULL;
		_UINT16 *comp_imgb = NULL;
		try {
		      orig_imgb = new _UINT16 [ size ];
		      comp_imgb = new _UINT16 [ size ];
		} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }

		for(_INT32 i = 0; i < size; ++i) { orig_imgb[i] = (_UINT16)(orig_img[i]&_MINT16); comp_imgb[i] = (_UINT16)(comp_img[i]&_MINT16); }	
		
		result = MSE16bit(orig_imgb,comp_imgb,PX,PY);
		delete [] orig_imgb;
		delete [] comp_imgb;
                    

	};
	break;


	default: break;


}

return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------


double DoMSEY(_INT32 *orig_img, _INT32 *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP)
{

	_INT32 size = PX * PY;
	double result = 0.0;
	
	switch (BPP) {

	case 8: case 16:
				result = DoMSE(orig_img, comp_img, PX,PY,BPP);
				break;

	case 24:  {
			
			_UINT16 *orig_imgb = NULL;
			_UINT16 *comp_imgb = NULL;
			try {
		      		orig_imgb = new _UINT16 [ size ];
		      		comp_imgb = new _UINT16 [ size ];
			} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }

			for(_INT32 i=0; i < size; ++i) {
			double Y1 = rgB * (double)((orig_img[i]>>16)&_MBYTE) + rGb * (double)((orig_img[i]>>8)&_MBYTE) + Rgb * (double)(orig_img[i]&_MBYTE) + Crgb;
			double Y2 = rgB * (double)((comp_img[i]>>16)&_MBYTE) + rGb * (double)((comp_img[i]>>8)&_MBYTE) + Rgb * (double)(comp_img[i]&_MBYTE) + Crgb;
			comp_imgb[i] = (_UINT16)Y2;
			orig_imgb[i] = (_UINT16)Y1;
			}

			result = MSE16bit(orig_imgb, comp_imgb, PX,PY);
			delete [] orig_imgb;
			delete [] comp_imgb;
		};
		break;

	default: break;

	}

return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

double DoMSEY(_BYTE *orig_img, _BYTE *comp_img, _INT32 PX, _INT32 PY, _INT32 BPP)
{

	_INT32 size = PX * PY;
	_INT32 bsize = size * 3;
	double result = 0.0;
	
	switch (BPP) {

	case 24:  {
			
			_UINT16 *orig_imgb = NULL;
			_UINT16 *comp_imgb = NULL;
			try {
		      		orig_imgb = new _UINT16 [ size ];
		      		comp_imgb = new _UINT16 [ size ];
			} catch (...) { if (orig_imgb) delete orig_imgb; if (comp_imgb) delete [] comp_imgb; return result; }

			for(_INT32 i=0,j=0; i < bsize; i+=3,++j) {
			if ((i < bsize) && (i + 1 < bsize) && (i + 2 < bsize) && (j < size)) {
				double Y1 = rgB * (double)(orig_img[i]) + rGb * (double)(orig_img[i+1]) + Rgb * (double)(orig_img[i+2]) + Crgb;
				double Y2 = rgB * (double)(comp_img[i]) + rGb * (double)(comp_img[i+1]) + Rgb * (double)(comp_img[i+2]) + Crgb;
				comp_imgb[j] = (_UINT16)Y2;
				orig_imgb[j] = (_UINT16)Y1;
			} 			
			}

			result = MSE16bit(orig_imgb, comp_imgb, PX,PY);
			delete [] orig_imgb;
			delete [] comp_imgb;
		};
		break;

	default: break;

	}

return result;

}

//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
