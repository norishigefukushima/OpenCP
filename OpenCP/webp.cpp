#include "webp.hpp"
#include "./webp/encode.h"
#include "./webp/mux.h"
#pragma comment(lib,"./webp/libwebp.lib")
#pragma comment(lib,"./webp/libwebpmux.lib")
using namespace cv;
using namespace std;

namespace cp
{
	void imwriteAnimationWebp(std::string name, vector<Mat>& src)
	{
		int	quality = 10;
		int loop_count = 0;//0: infinit
		int timestamp_ms = 1000;
		const int width = src[0].cols;
		const int height = src[0].rows;

		WebPPicture pic;
		if (!WebPPictureInit(&pic))
		{
			cerr << "picture init" << endl;
			return;  // version error
		}
		pic.width = width;
		pic.height = height;
		pic.use_argb = 1;
		//pic.colorspace = WEBP_YUV420

		// allocated picture of dimension width x height
		if (!WebPPictureAlloc(&pic))
		{
			cerr << "picture alloc" << endl;
			return;   // memory error
		}

		WebPAnimEncoderOptions enc_options = { {0} };
		WebPAnimEncoderOptionsInit(&enc_options);
		WebPAnimEncoder* enc = WebPAnimEncoderNew(width, height, &enc_options);

		WebPConfig config;
		WebPConfigInit(&config);
		config.quality = quality;

		Mat temp;
		for (int i = 0; i < src.size(); i++)
		{
			cvtColor(src[i], temp, COLOR_BGR2RGB);
			if (!WebPPictureImportRGB(&pic, temp.data, width * src[i].channels()))
			{
				cerr << "WebPPictureImportRGB" << endl;
			}
			WebPAnimEncoderAdd(enc, &pic, timestamp_ms * i, &config);
		}
		WebPAnimEncoderAdd(enc, NULL, timestamp_ms, NULL);

		WebPData webp_data = { 0 };
		WebPDataInit(&webp_data);

		//write data
		WebPAnimEncoderAssemble(enc, &webp_data);

		//Mux assemble
		WebPMux* mux = WebPMuxCreate(&webp_data, 1);
		WebPMuxAnimParams anim_params;
		anim_params.loop_count = loop_count;
		WebPMuxSetAnimationParams(mux, &anim_params);
		WebPMuxAssemble(mux, &webp_data);


		FILE* fp = fopen(name.c_str(), "wb");
		fwrite(webp_data.bytes, sizeof(char), webp_data.size, fp);
		fclose(fp);

		WebPPictureFree(&pic);
		WebPAnimEncoderDelete(enc);
		WebPDataClear(&webp_data);
		WebPMuxDelete(mux);
	}
}
