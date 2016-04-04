#ifndef INTRINSIC_CUDA_HEADER
#define INTRINSIC_CUDA_HEADER
#include "DllExportDef.h"

using namespace cv;
using namespace std;
using namespace cv::cuda;
namespace btl{
	namespace device{


		class DLL_EXPORT CIntrinsicsAnalysisSingle{
		public:
			typedef CIntrinsicsAnalysisSingle* tp_ptr;
			typedef boost::shared_ptr< CIntrinsicsAnalysisSingle > tp_shared_ptr;
			vector<GpuMat> _vRadianceBGR;

			double _dM[3];
			double _dS[3];

			GpuMat _normalized_bgr[4];
			GpuMat _error_bgr[4];

			int _nRows, _nCols;

			GpuMat _inv_crf; //camera response function
			GpuMat _exp_crf; //camera response function
			GpuMat _rad_re; //CRF x
			GpuMat _inten; //CRF y
			Mat _ln_sample;
			Mat _nlf_sqr;//noise level function
			GpuMat _derivCRF;
			Mat _normalize_factor;

			CIntrinsicsAnalysisSingle(const string& pathFileName, int rows, int cols);

			void analysis(const GpuMat& BGR_);
			void apply(CIntrinsicsAnalysisSingle& ia_);

			void store(string& fileName);
			void clear();
			void copyTo(CIntrinsicsAnalysisSingle& ia_);
			void loadInvCRF(const string& pathFileName);
			void calcDeriv();
			void cRF(const GpuMat& radiance_, GpuMat *pRGB_);
			void cRFNoisy(const GpuMat& radiance_, GpuMat *pRGB_);
		};

		class DLL_EXPORT CIntrinsicsAnalysisMult : public CIntrinsicsAnalysisSingle{
		public:
			typedef CIntrinsicsAnalysisMult* tp_ptr;
			typedef boost::shared_ptr< CIntrinsicsAnalysisMult > tp_shared_ptr;
			short _patch_radius;
			float _ratio;
			GpuMat _thumb;
			GpuMat _mean[3];
			GpuMat _std[3];
		public:
			CIntrinsicsAnalysisMult(const string& pathFileName, int rows, int cols, float ratio, int win_radius);

			void analysis(const GpuMat& BGR_);
			void store(string& fileName);
			void apply(CIntrinsicsAnalysisMult& ia_);
			void clear();
			void copyTo(CIntrinsicsAnalysisMult& ia_);
			void normalize(const GpuMat& radiance_, short ch);
			void calc_mean_std(const GpuMat& radiance_, short ch);
		};

	}//device
}//btl
#endif
