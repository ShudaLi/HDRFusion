#ifndef BTL_Eigen_UTILITY_HEADER
#define BTL_Eigen_UTILITY_HEADER

//eigen-based helpers
#include "OtherUtil.hpp"
#include <Eigen/Dense>
#include <se3.hpp>
#include <opencv2/core.hpp>

namespace btl
{
namespace utility
{

using namespace Eigen;
using namespace Sophus;
using namespace btl::utility;

template< class T >
Eigen::Matrix< T, 4, 4 > inver(const Eigen::Matrix< T, 4, 4 >& F_ctw_){
	using namespace Eigen;
	Matrix< T, 4, 4 > F_wtc; F_wtc.setIdentity();

	Matrix< T, 3, 3 > R_trans = F_ctw_.block(0,0,3,3);
	Matrix< T, 3, 1 > Cw = F_ctw_.block(0,3,3,1);

	F_wtc.block(0,0,3,3) = R_trans.transpose();
	F_wtc.block(0,3,3,1) = -R_trans*Cw;
	return F_wtc;
}

template< class T >
void getCwVwFromPrj_Cam2World(const Eigen::Matrix< T, 4, 4 >&  Prj_ctw_,Eigen::Matrix< T, 3, 1 >* pCw_,Eigen::Matrix< T, 3, 1 >* pVw_){
	*pCw_ = Prj_ctw_.template block<3,1>(0,3); //4th column is camera centre
	*pVw_ = Prj_ctw_.template block<3,1>(0,0); //1st column is viewing direction
}

template< class T >
void getRTCVfromModelViewGL ( const Eigen::Matrix< T, 4, 4 >&  mMat_, Eigen::Matrix< T, 3, 3 >* pmR_, Eigen::Matrix< T, 3, 1 >* pvT_ )
{
    (* pmR_) ( 0, 0 ) =  mMat_ ( 0, 0 );   (* pmR_) ( 0, 1 ) =   mMat_ ( 0, 1 );  (* pmR_) ( 0, 2 ) = mMat_ ( 0, 2 );
    (* pmR_) ( 1, 0 ) = -mMat_ ( 1, 0 );   (* pmR_) ( 1, 1 ) = - mMat_ ( 1, 1 );  (* pmR_) ( 1, 2 ) = -mMat_ ( 1, 2 );
    (* pmR_) ( 2, 0 ) = -mMat_ ( 2, 0 );   (* pmR_) ( 2, 1 ) = - mMat_ ( 2, 1 );  (* pmR_) ( 2, 2 ) = -mMat_ ( 2, 2 );
    
	(*pvT_) ( 0 ) = mMat_ ( 0, 3 );
    (*pvT_) ( 1 ) = -mMat_ ( 1, 3 );
    (*pvT_) ( 2 ) = -mMat_ ( 2, 3 );

    return;
}

template< class T >
Eigen::Matrix< T, 4, 4 > setModelViewGLfromPrj(const Eigen::Transform<T, 3, Eigen::Affine> & Prj_)
{
	// column first for pGLMat_[16];
	// row first for Matrix3d;
	// pGLMat_[ 0] =  mR_(0,0); pGLMat_[ 4] =  mR_(0,1); pGLMat_[ 8] =  mR_(0,2); pGLMat_[12] =  vT_(0);
	// pGLMat_[ 1] = -mR_(1,0); pGLMat_[ 5] = -mR_(1,1); pGLMat_[ 9] = -mR_(1,2); pGLMat_[13] = -vT_(1);
	// pGLMat_[ 2] = -mR_(2,0); pGLMat_[ 6] = -mR_(2,1); pGLMat_[10] = -mR_(2,2); pGLMat_[14] = -vT_(2);
	// pGLMat_[ 3] =  0;        pGLMat_[ 7] =  0;        pGLMat_[11] =  0;        pGLMat_[15] = 1;

	Eigen::Matrix< T , 4, 4 > mMat;
	mMat.row( 0 ) =  Prj_.matrix().row( 0 );
	mMat.row( 1 ) = -Prj_.matrix().row( 1 );
	mMat.row( 2 ) = -Prj_.matrix().row( 2 );
	mMat.row( 3 ) =  Prj_.matrix().row( 3 );

	return mMat;
}

template< class T >
Eigen::Matrix< T , 4, 4 > setModelViewGLfromRTCV ( const SO3Group<T>& mR_, const Eigen::Matrix< T, 3, 1 >& vT_ )
{
    // column first for pGLMat_[16];
    // row first for Matrix3d;
    // pGLMat_[ 0] =  mR_(0,0); pGLMat_[ 4] =  mR_(0,1); pGLMat_[ 8] =  mR_(0,2); pGLMat_[12] =  vT_(0);
    // pGLMat_[ 1] = -mR_(1,0); pGLMat_[ 5] = -mR_(1,1); pGLMat_[ 9] = -mR_(1,2); pGLMat_[13] = -vT_(1);
    // pGLMat_[ 2] = -mR_(2,0); pGLMat_[ 6] = -mR_(2,1); pGLMat_[10] = -mR_(2,2); pGLMat_[14] = -vT_(2);
    // pGLMat_[ 3] =  0;        pGLMat_[ 7] =  0;        pGLMat_[11] =  0;        pGLMat_[15] = 1;

    Eigen::Matrix< T , 4, 4 > mMat;
    mMat ( 0, 0 ) =  mR_.matrix() ( 0, 0 ); mMat ( 0, 1 ) =  mR_.matrix() ( 0, 1 ); mMat ( 0, 2 ) =  mR_.matrix() ( 0, 2 ); mMat ( 0, 3 ) =  vT_ ( 0 );
    mMat ( 1, 0 ) = -mR_.matrix() ( 1, 0 ); mMat ( 1, 1 ) = -mR_.matrix() ( 1, 1 ); mMat ( 1, 2 ) = -mR_.matrix() ( 1, 2 ); mMat ( 1, 3 ) = -vT_ ( 1 );
    mMat ( 2, 0 ) = -mR_.matrix() ( 2, 0 ); mMat ( 2, 1 ) = -mR_.matrix() ( 2, 1 ); mMat ( 2, 2 ) = -mR_.matrix() ( 2, 2 ); mMat ( 2, 3 ) = -vT_ ( 2 );
    mMat ( 3, 0 ) =  0;            mMat ( 3, 1 ) =  0;            mMat ( 3, 2 ) =  0;            mMat ( 3, 3 ) =  1;
    
    return mMat;
}

template< class T >
Eigen::Matrix< T , 4, 4 > setModelViewGLfromRCCV ( const Eigen::Matrix< T, 3, 3 >& mR_, const Eigen::Matrix< T, 3, 1 >& vC_ )
{
	Eigen::Matrix< T, 3,1> eivT = -mR_.transpose()*vC_;
	return setModelViewGLfromRTCV(mR_,vC_);
}

template< class T1, class T2 >
void unprojectCamera2World ( const int& nX_, const int& nY_, const unsigned short& nD_, const Eigen::Matrix< T1, 3, 3 >& mK_, Eigen::Matrix< T2, 3, 1 >* pVec_ )
{
	//the pixel coordinate is defined w.r.t. opencv camera reference, which is defined as x-right, y-downward and z-forward. It's
	//a right hand system.
	//when rendering the point using opengl's camera reference which is defined as x-right, y-upward and z-backward. the
	//	glVertex3d ( Pt(0), -Pt(1), -Pt(2) );
	if ( nD_ > 400 ) {
		T2 dZ = nD_ / 1000.; //convert to meter
		T2 dX = ( nX_ - mK_ ( 0, 2 ) ) / mK_ ( 0, 0 ) * dZ;
		T2 dY = ( nY_ - mK_ ( 1, 2 ) ) / mK_ ( 1, 1 ) * dZ;
		( *pVec_ ) << dX + 0.0025, dY, dZ + 0.00499814; // the value is esimated using CCalibrateKinectExtrinsics::calibDepth()
		// 0.0025 by experience.
	}
	else {
		( *pVec_ ) << 0, 0, 0;
	}
}

template< class T >
void projectWorld2Camera ( const Eigen::Matrix< T, 3, 1 >& vPt_, const Eigen::Matrix3d& mK_, Eigen::Matrix< short, 2, 1>* pVec_  )
{
	// this is much faster than the function
	// eiv2DPt = mK * vPt; eiv2DPt /= eiv2DPt(2);
	( *pVec_ ) ( 0 ) = short ( mK_ ( 0, 0 ) * vPt_ ( 0 ) / vPt_ ( 2 ) + mK_ ( 0, 2 ) + 0.5 );
	( *pVec_ ) ( 1 ) = short ( mK_ ( 1, 1 ) * vPt_ ( 1 ) / vPt_ ( 2 ) + mK_ ( 1, 2 ) + 0.5 );
}

template< class T >
void convertPrj2Rnt(const Eigen::Transform< T, 3, Eigen::Affine >& Prj_, SO3Group< T >* pR_, Eigen::Matrix< T, 3, 1 >* pT_)
{
	*pR_ = SO3Group<T>(Prj_.linear());
	*pT_ = Prj_.translation();
	return;
}
template< class T >
Eigen::Transform< T, 3, Eigen::Affine > convertRnt2Prj(const SO3Group< T >& R_, const Eigen::Matrix< T, 3, 1 >& T_)
{
	Eigen::Transform< T, 3, Eigen::Affine > prj;
	prj.setIdentity();
	prj.linear() = R_.matrix();
	prj.translation() = T_;
	return prj;
}
template< class T >
void convertPrjInv2RpnC( const Eigen::Matrix< T, 4, 4 >& Prj_, Eigen::Matrix< T, 3, 3 >* pR_trans_, Eigen::Matrix< T, 3, 1 >* pT_)
{
	*pR_trans_ = Prj_.template block<3,3>(0,0);
	*pT_ = Prj_.template block<3,1>(0,3);
	return;
}
template< class T >
Eigen::Matrix< T, 4, 4 > convertRpnC2PrjInv(  const Eigen::Matrix< T, 3, 3 >& R_trans_, const Eigen::Matrix< T, 3, 1 >& C_ )
{
	Eigen::Matrix< T, 4, 4 > prj;
	prj.setIdentity();
	prj.template block<3,3>(0,0) = R_trans_;
	prj.template block<3,1>(0,3) = C_;
	return prj;
}

template< class T, int ROW, int COL >
T matNormL1 ( const Eigen::Matrix< T, ROW, COL >& eimMat1_, const Eigen::Matrix< T, ROW, COL >& eimMat2_ )
{
	Eigen::Matrix< T, ROW, COL > eimTmp = eimMat1_ - eimMat2_;
	Eigen::Matrix< T, ROW, COL > eimAbs = eimTmp.cwiseAbs();
	return (T) eimAbs.sum();
}

template< class T >
void setSkew( T x_, T y_, T z_, Eigen::Matrix< T, 3,3 >* peimMat_){
	*peimMat_ << 0, -z_, y_, z_, 0, -x_, -y_, x_, 0 ;
}

template< class T >
void setRotMatrixUsingExponentialMap( T x_, T y_, T z_, Eigen::Matrix< T, 3,3 >* peimR_ ){
	//http://opencv.itseez.com/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=rodrigues#void Rodrigues(InputArray src, OutputArray dst, OutputArray jacobian)
	T theta = sqrt( x_*x_ + y_*y_ + z_*z_ );
	if(	theta < std::numeric_limits<T>::epsilon() ){
		*peimR_ = Eigen::Matrix< T, 3,3 >::Identity();
		return;
	}
	T sinTheta = sin(theta);
	T cosTheta = cos(theta);
	Eigen::Matrix< T, 3,3 > eimSkew; 
	setSkew< T >(x_/theta,y_/theta,z_/theta,&eimSkew);
	*peimR_ = Eigen::Matrix< T, 3,3 >::Identity() + eimSkew*sinTheta + eimSkew*eimSkew*(1-cosTheta);
}

}//utility
}//btl
#endif
