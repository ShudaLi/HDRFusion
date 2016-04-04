//Copyright(c) 2016 Shuda Li[lishuda1980@gmail.com]
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//FOR A PARTICULAR PURPOSE AND NON - INFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
//COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//

#ifndef BTL_UTILITY_HEADER
#define BTL_UTILITY_HEADER


#include "OtherUtil.hpp"
#include "EigenUtil.hpp"
#include "CVUtil.hpp"
#include "Converters.hpp"
#include <se3.hpp>

#define _USE_MATH_DEFINES
#include <math.h>


namespace btl
{
namespace utility
{
using namespace Eigen;
using namespace cv;
using namespace Sophus;


template<typename T>
SE3Group<T> extractRTFromBuffer(const T* aHostTmp_)
{
	//declare A and b
	Eigen::Matrix<T, 6, 6, Eigen::RowMajor> A;
	Eigen::Matrix<T, 6, 1> b;
	//retrieve A and b from cvmSumBuf
	short sShift = 0;
	for (int i = 0; i < 6; ++i){   // rows
		for (int j = i; j < 7; ++j) { // cols + b
			T value = aHostTmp_[sShift++];
			if (j == 6)       // vector b
				b.data()[i] = value;
			else
				A.data()[j * 6 + i] = A.data()[i * 6 + j] = value;
		}//for each col
	}//for each row
	//checking nullspace
	T dDet = A.determinant();
	if (fabs(dDet) < 1e-15 || dDet != dDet){
		if (dDet != dDet)
			PRINTSTR("Failure -- dDet cannot be qnan. ");
		//reset ();
		 return SE3Group<T>();
	}//if dDet is rational

    Eigen::Matrix<T, 6, 1> result = A.llt().solve(b);

	SE3Group<T> T_rl(SO3Group<T>::exp(SE3Group<T>::Point(result(0), result(1), result(2))),
									  SE3Group<T>::Point(result(3), result(4), result(5)));

	//Mat rvec(3, 1, CV_64FC1, result.data());
	//Matrix<T,3,3> Rinc;
	//Mat rot(3, 3, CV_64FC1, Rinc.data());
	//cv::Rodrigues(rvec, rot);
	//Rinc.transposeInPlace();
	//Matrix3d Rinc = (Matrix3d)AngleAxisd(result(2), Vector3d::UnitZ()) * AngleAxisd(result(1), Vector3d::UnitY()) * AngleAxisd(result(0), Vector3d::UnitX());

	//Transform<T, 3, Affine> Tnc; 
	//Tnc.linear()= T_rl.so3().matrix(); //transform from current to next coordinate
	//Tnc.translation() = T_rl.translation();
	//Tnc.linear() = Rinc;
	//Tnc.translation() = result.template tail<3>();
	return T_rl;
}

template<typename T>
void extractRTFromBuffer(const T* aHostTmp_, Matrix3f* pRw_, Vector3f* pTw_) 
{
	Transform<T, 3, Affine> Tnc = extractRTFromBuffer<T>(aHostTmp_);
	//compose
	Eigen::Vector3f Cw = -pRw_->transpose() * (*pTw_);
	Eigen::Matrix3f Rcw_t = pRw_->transpose();
        Rcw_t = Tnc.linear().template cast<float>() * Rcw_t;
        Cw = Tnc.linear().template cast<float>() * Cw + Tnc.translation().template cast<float>();
	*pRw_ = Rcw_t.transpose();
	*pTw_ = -Rcw_t.transpose() * Cw;

	return;
}

template< class T >
T norm3( const T* pPtCur_, const T* pPtRef_ ){
	T tTmp0 = pPtCur_[0]-pPtRef_[0];
	T tTmp1 = pPtCur_[1]-pPtRef_[1];
	T tTmp2 = pPtCur_[2]-pPtRef_[2];
	return std::sqrt(tTmp0*tTmp0 + tTmp1*tTmp1 + tTmp2*tTmp2);
}

template< class T >
T norm3( const T* pPtCurCam_, const T* pPtRefWor_, const T* pCurRw_/*col major*/, const T* pCurTw_ ){
	//C = CurRw*W + T
	T tRefCam[3];
	tRefCam[0] = pCurRw_[0]*pPtRefWor_[0] + pCurRw_[3]*pPtRefWor_[1] + pCurRw_[6]*pPtRefWor_[2] + pCurTw_[0];
	tRefCam[1] = pCurRw_[1]*pPtRefWor_[0] + pCurRw_[4]*pPtRefWor_[1] + pCurRw_[7]*pPtRefWor_[2] + pCurTw_[1];
	tRefCam[2] = pCurRw_[2]*pPtRefWor_[0] + pCurRw_[5]*pPtRefWor_[1] + pCurRw_[8]*pPtRefWor_[2] + pCurTw_[2];

	return norm3< T >( pPtCurCam_,tRefCam);
}

template< class T >
void normalVotes( const T* pNormal_, const double& dS_, int* pR_, int* pC_, btl::utility::tp_coordinate_convention eCon_ = btl::utility::BTL_GL)
{
	//pNormal[3] is a normal defined in a right-hand reference
	//system with positive-z the elevation, and counter-clockwise from positive-x is
	//the azimuth, 
	//dS_ is the step length in radian
	//*pR_ is the discretized elevation 
	//*pC_ is the discretized azimuth

	//normal follows GL-convention
	const double dNx = pNormal_[0];
	const double dNy = pNormal_[1];
	double dNz = pNormal_[2];
	if(btl::utility::BTL_CV == eCon_) {dNz = -dNz;}

	double dA = atan2(dNy,dNx); //atan2 ranges from -pi to pi
	dA = dA <0 ? dA+2*M_PI :dA; // this makes sure that dA ranging from 0 to 2pi
	double dyx= sqrt( dNx*dNx + dNy*dNy );
	double dE = atan2(dNz,dyx);

	*pC_ = floor(dA/dS_);
	*pR_ = floor(dE/dS_);

}

template< class T >
void avgNormals(const cv::Mat& cvmNls_,const std::vector<unsigned int>& vNlIdx_, Eigen::Vector3d* peivAvgNl_)
{
	//note that not all normals in vNormals_ will be averaged
	*peivAvgNl_ << 0,0,0;
	const float* const pNl = (float*)cvmNls_.data;
	for(std::vector<unsigned int>::const_iterator cit_vNlIdx = vNlIdx_.begin();
		cit_vNlIdx!=vNlIdx_.end(); cit_vNlIdx++)
	{
		unsigned int nOffset = (*cit_vNlIdx)*3;
		*peivAvgNl_+=Eigen::Vector3d(pNl[nOffset],pNl[nOffset+1],pNl[nOffset+2]);
	}
	peivAvgNl_->normalize();
}

template< class T >
bool isNormalSimilar( const T* pNormal1_, const Eigen::Vector3d& eivNormal2_, const T& dCosThreshold_)
{
	//if the angle between eivNormal1_ and eivNormal2_ is larger than dCosThreshold_
	//the two normal is not similar and return false
	T dCos = pNormal1_[0]*eivNormal2_[0] + pNormal1_[1]*eivNormal2_[1] + pNormal1_[2]*eivNormal2_[2];
	if(dCos>dCosThreshold_)
		return true;
	else
		return false;
}

template< class T >
void normalCluster( const cv::Mat& cvmNls_,const std::vector< unsigned int >& vNlIdx_, 
	const Eigen::Vector3d& eivClusterCenter_, 
	const double& dCosThreshold_, const short& sLabel_, cv::Mat* pcvmLabel_, std::vector< unsigned int >* pvNlIdx_ )
{
	//the pvLabel_ must be same length as vNormal_ 
	//with each element assigned with a NEGATIVE value
	const float* pNl = (const float*)cvmNls_.data; 
	short* pLabel = (short*) pcvmLabel_->data;
	for( std::vector< unsigned int >::const_iterator cit_vNlIdx_ = vNlIdx_.begin();
		cit_vNlIdx_!= vNlIdx_.end(); cit_vNlIdx_++ ){
		int nOffset = (*cit_vNlIdx_)*3;
		if( pLabel[*cit_vNlIdx_]<0 && btl::utility::isNormalSimilar< float >(pNl+nOffset,eivClusterCenter_,dCosThreshold_) )	{
			pLabel[*cit_vNlIdx_] = sLabel_;
			pvNlIdx_->push_back(*cit_vNlIdx_);
		}//if
	}
	return;
}

template< typename T >
void convert(const Eigen::Transform<T, 3, Affine>& eiM_, Mat_<T>* pM_){
	pM_->create(4, 4);
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
			pM_->at<T>(r, c) = eiM_(r, c);
		}

	return;
}

template< typename T >
void convert(const Eigen::Matrix<T,4,4>& eiM_, Mat_<T>* pM_){
	pM_->create(4, 4);
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
			pM_->at<T>(r, c) = eiM_(r, c);
		}

	return;
}

template< typename T >
void convert(const Mat_<T>& M_, Eigen::Transform<T,3,Affine>* peiM_){

	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
			(*peiM_)(r, c) = M_.at<T>(r, c);
		}

	return;
}

template< typename T >
void convert(const Mat_<T>& M_, Eigen::Matrix<T, 4,4>* peiM_){

	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
			(*peiM_)(r, c) = M_.at<T>(r, c);
		}

	return;
}

}//utility
}//btl
#endif
