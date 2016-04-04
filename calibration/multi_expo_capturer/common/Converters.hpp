#ifndef BTL_CONVERTER_HEADER
#define BTL_CONVERTER_HEADER
/**
* @file helper.hpp
* @brief helpers developed consistent with btl2 format, it contains a group of useful when developing btl together with
* opencv and Eigen.
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* 1. << and >> converter from standard std::vector to cv::Mat and Eigen::Matrix<T, ROW, COL>
* 2. PRINT() for debugging
* 3. << to output std::vector using std::cout
* 4. exception handling and exception related macro including CHECK( condition, "error message") and THROW ("error message")
* @date 2011-03-15
*/
#define CV_SSE2 1

#include "OtherUtil.hpp"

#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <Eigen/Dense>

// ====================================================================
// === Implementation
namespace btl
{
namespace utility
{

//for print
	/*template <class T>
	std::ostream& operator << ( std::ostream& os, const cv::Size_< T >& s )
	{
		os << "[ " << s.width << ", " << s.height << " ]";
		return os;
	}*/

template < class T, int ROW >
std::vector< T >& operator << ( std::vector< T >& vVec_, const Eigen::Matrix< T, ROW, 1 >& eiVec_ )
{
    vVec_.clear();

    for ( int r = 0; r < ROW; r++ )
    {
        vVec_.push_back ( eiVec_ ( r, 0 ) );
    }

    return vVec_;
}

template < class T, int ROW >
const Eigen::Matrix< T, ROW, 1 >&  operator >> ( const Eigen::Matrix< T, ROW, 1 >& eiVec_, std::vector< T >& vVec_ )
{
    vVec_ << eiVec_;
}

template < class T >
Eigen::Matrix< T, Eigen::Dynamic, 1 >& operator << ( Eigen::Matrix< T, Eigen::Dynamic, 1 >& eiVec_, const std::vector< T >& vVec_ )
{
    if ( vVec_.empty() )
    {
        eiVec_.resize ( 0, 0 );
    }
    else
    {
        eiVec_.resize ( vVec_.size(), 1 );
    }

    for ( int r = 0; r < vVec_.size(); r++ )
    {
        eiVec_ ( r, 0 ) = vVec_[r];
    }

    return eiVec_;
}

template < class T >
Eigen::Matrix< T, Eigen::Dynamic, 1 >& operator << ( Eigen::Matrix< T, Eigen::Dynamic, 1 >& eiVec_, const cv::Mat& cvmVec_ )
{
	if ( cvmVec_.empty() ){
		eiVec_.resize ( 0, 0 );
	}
	else{
		eiVec_.resize ( cvmVec_.rows, 1 );
	}

	const T* p = (const T*)cvmVec_.data;
	for ( int r = 0; r < cvmVec_.rows; r++ ){
		eiVec_ ( r, 0 ) = *p;
		p ++;
	}

	return eiVec_;
}

template < class T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& operator << ( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eiMat_, const cv::Mat& cvmMat_ )
{
	if ( cvmMat_.empty() ){
		eiMat_.resize ( 0, 0 );
	}
	else{
		eiMat_.resize ( cvmMat_.rows, cvmMat_.cols );
	}

	for ( int r = 0; r < cvmMat_.rows; r++ ){
		for ( int c = 0; c < cvmMat_.cols; c++ ){
			eiMat_ ( r, c ) = cvmMat_.template at< T > ( r , c );
		}
	}

	return eiMat_;
}

template < class T >
const std::vector< T >&  operator >> ( const std::vector< T >& vVec_, Eigen::Matrix< T, Eigen::Dynamic, 1 >& eiVec_ )
{
    eiVec_ << vVec_;
}

template < class T, int ROW >
Eigen::Matrix< T, ROW, 1 >& operator << ( Eigen::Matrix< T, ROW, 1 >& eiVec_, const std::vector< T >& vVec_ )
{
    CHECK ( eiVec_.rows() == vVec_.size(), "Eigen::Vector << std::vector wrong!" );

    for ( int r = 0; r < vVec_.size(); r++ )
    {
        eiVec_ ( r, 0 ) = vVec_[r];
    }

    return eiVec_;
}

template < class T, int ROW >
void  operator >> ( const std::vector< T >& vVec_, Eigen::Matrix< T, ROW, 1 >& eiVec_ )
{
    eiVec_ << vVec_;
}


/*template < class T>
std::*/

template < class T, int ROW, int COL  >
std::vector< std::vector< T > >& operator << ( std::vector< std::vector< T > >& vvVec_, const Eigen::Matrix< T, ROW, COL >& eiMat_ )
{
    vvVec_.clear();

    for ( int r = 0; r < ROW; r++ )
    {
        std::vector< T > v;

        for ( int c = 0; c < COL; c++ )
        {
            v.push_back ( eiMat_ ( r, c ) );
        }

        vvVec_.push_back ( v );
    }

    return vvVec_;
}

template < class T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& operator << ( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eiMat_, const std::vector< std::vector< T > >& vvVec_ )
{
    if ( vvVec_.empty() )
    {
        eiMat_.resize ( 0, 0 );
    }
    else
    {
        eiMat_.resize ( vvVec_.size(), vvVec_[0].size() );
    }

    for ( int r = 0; r < vvVec_.size(); r++ )
        for ( int c = 0; c < vvVec_[r].size(); c++ )
        {
            eiMat_ ( r, c ) = vvVec_[r][c];
        }

    return eiMat_;
}

template < class T , int ROW, int COL>
Eigen::Matrix< T, ROW, COL >& operator << ( Eigen::Matrix< T, ROW, COL >& eiMat_, const std::vector< std::vector< T > >& vvVec_ )
{
    if ( ROW != vvVec_.size() )
    {
        CError cE;
        cE << CErrorInfo ( " std::vector< std::vector<> > is inconsistent with ROW of Matrix. \n" );
        throw cE;
    }
    else if ( COL != vvVec_[0].size() )
    {
        CError cE;
        cE << CErrorInfo ( " std::vector<> is inconsistent with COL of Matrix. \n" );
        throw cE;
    }

    for ( int r = 0; r < vvVec_.size(); r++ )
        for ( int c = 0; c < vvVec_[r].size(); c++ )
        {
            eiMat_ ( r, c ) = vvVec_[r][c];
        }

    return eiMat_;
}

template <class T>
std::vector< T >& operator << ( std::vector< T >& vVec_, const cv::Point3_< T >& cvPt_ )
{
    vVec_.clear();
    vVec_.push_back ( cvPt_.x );
    vVec_.push_back ( cvPt_.y );
    vVec_.push_back ( cvPt_.z );
    return vVec_;
}

template <class T>
cv::Point3_< T >& operator << ( cv::Point3_< T >& cvPt_, const std::vector< T >& vVec_ )
{
    if ( vVec_.empty() )
    {
        cvPt_.x = cvPt_.y = cvPt_.z = 0;
    }
    else if ( 3 != vVec_.size() )
    {
        CError cE;
        cE << CErrorInfo ( " std::vector<> is inconsistent with cv::Point3_. \n" );
        throw cE;
    }
    else
    {
        cvPt_.x = vVec_[0];
        cvPt_.y = vVec_[1];
        cvPt_.z = vVec_[2];
    }

    return cvPt_;
}
//vector -> cv::Point_
template <class T>
cv::Point_< T >& operator << ( cv::Point_< T >& cvPt_, const std::vector< T >& vVec_ )
{
    if ( vVec_.empty() )
    {
        cvPt_.x = cvPt_.y = 0;
    }
    else if ( 2 != vVec_.size() )
    {
        CError cE;
        cE << CErrorInfo ( " std::vector<> is inconsistent with cv::Point3_. \n" );
        throw cE;
    }
    else
    {
        cvPt_.x = vVec_[0];
        cvPt_.y = vVec_[1];
    }

    return cvPt_;
}
//Point_ -> std::vector
template <class T>
std::vector< T >& operator << ( std::vector< T >& vVec_, const cv::Point_< T >& cvPt_ )
{
    vVec_.clear();
    vVec_.push_back ( cvPt_.x );
    vVec_.push_back ( cvPt_.y );
    return vVec_;
}



template < class T, int ROW, int COL  >
Eigen::Matrix< T, ROW, COL >& operator << ( Eigen::Matrix< T, ROW, COL >& eiVec_, const cv::Mat_< T >& cvVec_ )
{
    if ( ROW != cvVec_.rows || COL != cvVec_.cols )
    {
        CError cE;
        cE << CErrorInfo ( " cv::Mat dimension is inconsistent with Vector3d . \n" );
        //PRINT( cvVec_.cols ); PRINT( cvVec_.rows );
        throw cE;
    }

    for ( int r = 0; r < ROW; r++ )
        for ( int c = 0; c < COL; c++ )
        {
            eiVec_ ( r, c ) = cvVec_.template at< T > ( r, c );
        }

    return eiVec_;
}
// 4.1 std::vector -> cv::Mat_ -> std::vector
template < class T >
std::vector< std::vector< T > >& operator << ( std::vector< std::vector< T > >& vvVec_,  const cv::Mat_< T >& cvMat_ )
{
    vvVec_.clear();

    for ( int r = 0; r < cvMat_.rows; r++ )
    {
        std::vector< T > v;

        for ( int c = 0; c < cvMat_.cols; c++ )
        {
            v.push_back ( cvMat_.template at< T > ( r, c ) );
        }

        vvVec_.push_back ( v );
    }

    return vvVec_;
}

template < class T >
cv::Mat_< T >& operator << ( cv::Mat_< T >& cvMat_,  const  std::vector< T >& vVec_ )
{

	if ( vVec_.empty() )
	{
		CError cE;
		cE << CErrorInfo ( " the input std::vector<> cannot be empty.\n" );
		throw cE;
	}

	cvMat_.create ( ( int ) vVec_.size(), 1 );

	for ( int r = 0; r < ( int ) vVec_.size(); r++ )
	{
		cvMat_.template at< T > ( r, 0 ) = vVec_[r];
	}

	return cvMat_;
}

template < class T >
void operator >> (  const  std::vector< T >& vVec_, cv::Mat_< T >& cvMat_ )
{
	cvMat_ << vVec_;
}

template < class T >
void operator << (  std::vector< T >& vVec_, const  cv::Mat_< T >& cvMat_ )
{
	vVec_.clear();
	for ( int r = 0; r < ( int ) cvMat_.rows*cvMat_.cols; r++ )
	{
		vVec_.push_back( cvMat_.template at< T > ( r, 0 ) );
	}
}

template < class T >
void operator >> (   const  cv::Mat_< T >& cvMat_, std::vector< T >& vVec_  )
{
	vVec_ << cvMat_;
}

template < class T >
cv::Mat_< T >& operator << ( cv::Mat_< T >& cvMat_,  const  std::vector< std::vector< T > >& vvVec_ )
{

    if ( vvVec_.empty() || vvVec_[0].empty() )
    {
        CError cE;
        cE << CErrorInfo ( " the input std::vector<> cannot be empty.\n" );
        throw cE;
    }

    cvMat_.create ( ( int ) vvVec_.size(), ( int ) vvVec_[0].size() );

    for ( int r = 0; r < ( int ) vvVec_.size(); r++ )
    {
        for ( int c = 0; c < vvVec_[r].size(); c++ )
        {
            cvMat_.template at< T > ( r, c ) = vvVec_[r][c];
        }
    }

    return cvMat_;
}

template < class T >
std::vector< std::vector< std::vector< T > > >& operator << ( std::vector< std::vector< std::vector< T > > >& vvvVec_,  const std::vector< cv::Mat_< T > >& vmMat_ )
{
    vvvVec_.clear();
    typename std::vector< cv::Mat_< T > >::const_iterator constItr = vmMat_.begin();

    for ( ; constItr != vmMat_.end(); ++constItr )
    {
        std::vector< std::vector< T > > vv;
        vv << ( *constItr );
        vvvVec_.push_back ( vv );
    }

    return vvvVec_;
}

template < class T >
std::vector< std::vector< std::vector< T > > >& operator << ( std::vector< std::vector< std::vector< T > > >& vvvVec_,  const std::vector< cv::Mat >& vmMat_ )
{
    std::vector< cv::Mat_< T > > vmTmp;

    typename std::vector< cv::Mat >::const_iterator constItr = vmMat_.begin();

    for ( ; constItr != vmMat_.end(); ++constItr )
    {
        vmTmp.push_back ( cv::Mat_< T > ( *constItr ) );
    }

    vvvVec_ << vmTmp;
    return vvvVec_;
}


template < class T >
std::vector< cv::Mat_< T > >& operator << ( std::vector< cv::Mat_< T > >& vmMat_ ,  const std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    vmMat_.clear();
    typename std::vector< std::vector< std::vector< T > > >::const_iterator constVectorVectorItr = vvvVec_.begin();

    for ( ; constVectorVectorItr != vvvVec_.end(); ++ constVectorVectorItr )
    {
        cv::Mat_< T > mMat;
        mMat << ( *constVectorVectorItr );
        vmMat_.push_back ( mMat );
    }

    return vmMat_;
}

template < class T >
std::vector< cv::Mat >& operator << ( std::vector< cv::Mat >& vmMat_ ,  const std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    std::vector< cv::Mat_< T > > vmTmp;
    vmTmp << vvvVec_;
    typename std::vector< cv::Mat_< T > >::const_iterator constItr = vmTmp.begin();

    for ( ; constItr != vmTmp.end(); ++constItr )
    {
        vmMat_.push_back ( cv::Mat ( *constItr ) );
    }

    return vmMat_;
}

//vector< cv::Point_ > -> std::vector< < > >
template <class T>
std::vector< std::vector< T > >& operator << ( std::vector< std::vector< T > >& vvVec_, const std::vector< cv::Point_ < T > >& cvPt_ )
{
    vvVec_.clear();
    typename std::vector< cv::Point_< T > >::const_iterator constItr = cvPt_.begin();

    for ( ; constItr != cvPt_.end(); ++constItr )
    {
        std::vector < T > v;
        v << *constItr;
        vvVec_.push_back ( v );
    }

    return vvVec_;
}

//vector< cv::Point3_ > -> std::vector< < > >
template <class T>
std::vector< std::vector< T > >& operator << ( std::vector< std::vector< T > >& vvVec_, const std::vector< cv::Point3_ < T > >& cvPt3_ )
{
    vvVec_.clear();
    typename std::vector< cv::Point3_< T > >::const_iterator constItr = cvPt3_.begin();

    for ( ; constItr != cvPt3_.end(); ++constItr )
    {
        std::vector < T > v;
        v << *constItr;
        vvVec_.push_back ( v );
    }

    return vvVec_;
}


//vector < <> > -> std::vector< cv::Point_ >
template <class T>
std::vector< cv::Point_< T > >& operator << ( std::vector< cv::Point_< T > >& cvPt_, const std::vector< std::vector< T > >& vvVec_ )
{
    cvPt_.clear();
    typename std::vector< std::vector< T > >::const_iterator constItr = vvVec_.begin();

    for ( ; constItr != vvVec_.end(); ++constItr )
    {
        cv::Point_< T > Pt;
        Pt << *constItr;
        cvPt_.push_back ( Pt );
    }

    return cvPt_;
}

// 2.3 std::vector < < < > > > -> std::vector< < cv::Point_ > >
template <class T>
std::vector< std::vector< cv::Point_< T > > >& operator << ( std::vector< std::vector< cv::Point_< T > > >& vvPt_, const std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    vvPt_.clear();
    typename std::vector< std::vector< std::vector< T > > >::const_iterator constItr = vvvVec_.begin();

    for ( ; constItr != vvvVec_.end(); ++constItr )
    {
        std::vector< cv::Point_< T > > vPt;
        vPt << *constItr;
        vvPt_.push_back ( vPt );
    }

    return vvPt_;
}


//vector < <> > -> std::vector< cv::Point3_ >
template <class T>
std::vector< cv::Point3_< T > >& operator << ( std::vector< cv::Point3_< T > >& cvPt_, const std::vector< std::vector< T > >& vvVec_ )
{
    cvPt_.clear();
    typename std::vector< std::vector< T > >::const_iterator constItr = vvVec_.begin();

    for ( ; constItr != vvVec_.end(); ++constItr )
    {
        cv::Point3_< T > Pt3;
        Pt3 << *constItr;
        cvPt_.push_back ( Pt3 );
    }

    return cvPt_;
}

// 1.3 std::vector < < < > > > -> std::vector < < cv::Point3_ > >
template <class T>
std::vector< std::vector< cv::Point3_< T > > >& operator << ( std::vector< std::vector< cv::Point3_< T > > >& vvPt_, const std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    vvPt_.clear();
    typename std::vector< std::vector< std::vector< T > > >::const_iterator constItr = vvvVec_.begin();

    for ( ; constItr != vvvVec_.end(); ++constItr )
    {
        std::vector< cv::Point3_< T > > vPt3;
        vPt3 << *constItr;
        vvPt_.push_back ( vPt3 );
    }

    return vvPt_;
}



//vector< std::vector< cv::Point3_ > > -> std::vector< < < > > >
template <class T>
std::vector< std::vector< std::vector< T > > >& operator << ( std::vector< std::vector< std::vector< T > > >& vvvVec_, const std::vector< std::vector< cv::Point3_ < T > > >& vvPt3_ )
{
    typename std::vector< std::vector< cv::Point3_ < T > > >::const_iterator constItr = vvPt3_.begin();

    for ( ; constItr != vvPt3_.end(); ++ constItr )
    {
        std::vector< std::vector < T > > vv;
        vv << *constItr;
        vvvVec_.push_back ( vv );
    }

    return vvvVec_;
}

template <class T>
std::vector< std::vector< std::vector< T > > >& operator << ( std::vector< std::vector< std::vector< T > > >& vvvVec_, const std::vector< std::vector< cv::Point_< T > > >& vvPt_ )
{
    typename std::vector< std::vector< cv::Point_ < T > > >::const_iterator constItr = vvPt_.begin();

    for ( ; constItr != vvPt_.end(); ++ constItr )
    {
        std::vector< std::vector < T > > vv;
        vv << *constItr;
        vvvVec_.push_back ( vv );
    }

    return vvvVec_;
}

// 2.1 cv::Point3_ -> Vector
template < class T >
Eigen::Matrix<T, 3, 1> & operator << ( Eigen::Matrix< T, 3, 1 >& eiVec_, const cv::Point3_< T >& cvVec_ )
{
    eiVec_ ( 0 ) = cvVec_.x;
    eiVec_ ( 1 ) = cvVec_.y;
    eiVec_ ( 2 ) = cvVec_.z;
}

// other -> other
// 1.2 Static Matrix -> cv::Mat_ < >
template < class T, int ROW, int COL  >
cv::Mat_< T >& operator << ( cv::Mat_< T >& cvVec_, const Eigen::Matrix< T, ROW, COL >& eiVec_ )
{
    cvVec_.create ( ROW, COL );

    for ( int r = 0; r < ROW; r++ )
        for ( int c = 0; c < COL; c++ )
        {
            cvVec_.template at<T> ( r, c ) = eiVec_ ( r, c );
        }

    return cvVec_;
}
// other -> other
// 1.2 Static Matrix -> cv::Mat
template < class T, int ROW, int COL  >
cv::Mat& operator << ( cv::Mat& cvVec_, const Eigen::Matrix< T, ROW, COL >& eiVec_ )
{
    cvVec_ = cv::Mat_< T > ( ROW, COL );

    for ( int r = 0; r < ROW; r++ )
        for ( int c = 0; c < COL; c++ )
        {
            cvVec_.template at<T> ( r, c ) = eiVec_ ( r, c );
        }

    return cvVec_;
}

//CvMat -> cv::Mat_
template < class T >
cv::Mat_< T >& operator << ( cv::Mat_< T >& cppM_, const CvMat& cM_ )
{
    CHECK ( cppM_.type() == CV_MAT_TYPE ( cM_.type ) , "operator CvMat << cv::Mat_: the type of cv::Mat_ and CvMat is inconsistent. \n" );
    CHECK ( CV_IS_MAT ( &cM_ ),                       "operator CvMat << cv::Mat_: the data of CvMat must be pre-allocated. \n" );

    cppM_.create ( cM_.rows, cM_.cols );

    for ( int r = 0; r < cM_.rows; r++ )
        for ( int c = 0; c < cM_.cols; c++ )
        {
            cppM_.template at< T > ( r, c ) = CV_MAT_ELEM ( cM_, T, r, c );
        }

    return cppM_;
}

//cv::Mat_ -> CvMat
template < class T >
CvMat& operator << ( CvMat& cM_, const cv::Mat_< T >& cppM_ )
{
    CHECK ( cppM_.type() == CV_MAT_TYPE ( cM_.type ) , "operator CvMat << cv::Mat_: the type of cv::Mat_ and CvMat is inconsistent. \n" );
    CHECK ( CV_IS_MAT ( &cM_ ),                       "operator CvMat << cv::Mat_: the data of CvMat is not allocated. \n" );
    CHECK ( cppM_.rows == cM_.rows ,                 "operator CvMat << cv::Mat_: the # of rows of cv::Mat_ and CvMat is inconsistent. \n" );
    CHECK ( cppM_.cols == cM_.cols,                  "operator CvMat << cv::Mat_: the # of cols of cv::Mat_ and CvMat is inconsistent. \n" );


    for ( int r = 0; r < cppM_.rows; r++ )
        for ( int c = 0; c < cppM_.cols; c++ )
        {
            CV_MAT_ELEM ( cM_, T, r, c ) = cppM_.template at< T > ( r, c );
        }

    return cM_;
}

//cv::Mat_ -> CvMat
template < class T >
void assignPtr ( cv::Mat_< T >* cppM_, CvMat* pcM_ )
{
    for ( int r = 0; r < cppM_->rows; r++ )
        for ( int c = 0; c < cppM_->cols; c++ )
        {
            CV_MAT_ELEM ( *pcM_, T, r, c ) = cppM_->template at< T > ( r, c );
        }

}
//cv::CvMat -> cv::Mat_
template < class T >
void assignPtr ( CvMat* pcM_,  cv::Mat_< T >* cppM_ )
{
    for ( int r = 0; r < pcM_->rows; r++ )
        for ( int c = 0; c < pcM_->cols; c++ )
        {
            cppM_->template at< T > ( r, c ) = CV_MAT_ELEM ( *pcM_, T, r, c );
        }
}

// 1.1 cv::Point3_ -> std::vector
template <class T>
const cv::Point3_< T >& operator >> ( const cv::Point3_< T >& cvPt_, std::vector< T >& vVec_ )
{
    vVec_ << cvPt_;
}

// 1.2 std::vector < cv::Point3_ > -> std::vector< < > >
template <class T>
const std::vector< cv::Point3_ < T > >& operator >> ( const std::vector< cv::Point3_ < T > >& vPt3_, std::vector< std::vector< T > >& vvVec_ )
{
    vvVec_ << vPt3_;
}

// 1.3 std::vector < std::vector < cv::Point3_ > > -> std::vector< < < > > >
template <class T>
const std::vector< std::vector< cv::Point3_ < T > > >& operator >> ( const std::vector< std::vector< cv::Point3_ < T > > >& vvPt3_, std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    vvvVec_ << vvPt3_;
}

// 2.1 cv::Point_ -> std::vector
template <class T>
const cv::Point_< T >& operator >> ( const cv::Point_< T >& cvPt_, std::vector< T >& vVec_ )
{
    vVec_ << cvPt_;
}

// 2.2 std::vector < cv::Point_ > -> std::vector< < > >
template <class T>
const std::vector< cv::Point_< T > >& operator >> ( const std::vector< cv::Point_< T > >& vPt_, std::vector< std::vector< T > >& vvVec_ )
{
    vvVec_ << vPt_;
}

// 2.3 std::vector < std::vector < cv::Point_ > > -> std::vector< < < > > >
template <class T>
const std::vector< cv::Point_< T > >&  operator >> ( const std::vector< std::vector< cv::Point_< T > > >& vvPt_, std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    vvvVec_ << vvPt_;
}

// 3.  Static cv::Matrix -> std::vector < < > >
template < class T , int ROW, int COL >
const Eigen::Matrix< T, ROW, COL >& operator >> ( const Eigen::Matrix< T, ROW, COL >& eiMat_, std::vector< std::vector< T > >& vvVec_ )
{
    vvVec_ << eiMat_;
}

// 4.1 cv::Mat_ -> std::vector
template < class T >
const cv::Mat_< T >& operator >> ( const cv::Mat_< T >& cvMat_, std::vector< std::vector< T > >& vvVec_ )
{
    vvVec_ << cvMat_;
}

// 4.2 std::vector< cv::Mat_<> > -> std::vector
template < class T >
const std::vector< cv::Mat_< T > >& operator >> ( const std::vector< cv::Mat_< T > >& vmMat_, std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    vvvVec_ << vmMat_;
}

// 5.1 std::vector< cv::Mat > -> std::vector
template < class T >
const std::vector< cv::Mat >& operator >> ( const std::vector< cv::Mat >& vmMat_, std::vector< std::vector< std::vector< T > > >& vvvVec_ )
{
    vvvVec_ << vmMat_;
    return vmMat_;
}

// operator >>
// std::vector -> other
// 1.1 std::vector -> cv::Point3_Eigen::Matrix<short int, 2, 1,
template <class T>
const std::vector< T >& operator >> ( const std::vector< T >& vVec_, cv::Point3_< T >& cvPt_ )
{
    cvPt_ << vVec_;
}

// 1.2 std::vector < < > > -> std::vector< cv::Point3_ >
template <class T>
const std::vector< std::vector< T > >& operator >> ( const std::vector< std::vector< T > >& vvVec_ , std::vector< cv::Point3_< T > >& cvPt_ )
{
    cvPt_ << vvVec_;
}

// 1.3 std::vector < < < > > > -> std::vector < < cv::Point3_ > >
template <class T>
const std::vector< std::vector< std::vector< T > > >& operator >> ( const std::vector< std::vector< std::vector< T > > >& vvvVec_, std::vector< std::vector< cv::Point3_< T > > >& vvPt_ )
{
    vvPt_ << vvvVec_;
	return vvvVec_;
}

// 2.1 std::vector -> cv::Point_
template <class T>
const std::vector< T >& operator >> ( const std::vector< T >& vVec_, cv::Point_< T >& cvPt_ )
{
    cvPt_ << vVec_;
}

// 2.2 std::vector < < > > -> std::vector< cv::Point_ >
template <class T>
const std::vector< std::vector< T > >& operator >> ( const std::vector< std::vector< T > >& vvVec_, std::vector< cv::Point_< T > >& cvPt_ )
{
    cvPt_ << vvVec_;
}

// 2.3 std::vector < < < > > > -> std::vector< < cv::Point_ > >
template <class T>
const std::vector< std::vector< std::vector< T > > >& operator >> ( const std::vector< std::vector< std::vector< T > > >& vvvVec_, std::vector< std::vector< cv::Point_< T > > >& vvPt_ )
{
    vvPt_ << vvvVec_;
	return vvvVec_;
}

// 3.1 std::vector < < > > -> Eigen::Dynamic, Matrix
template < class T >
const std::vector< std::vector< T > >& operator >> (  const std::vector< std::vector< T > >& vvVec_, Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eiMat_ )
{
    eiMat_ << vvVec_;
}

// 3.2 std::vector < < > > -> Static, Matrix
template < class T , int ROW, int COL>
const std::vector< std::vector< T > >& operator >> ( const std::vector< std::vector< T > >& vvVec_, Eigen::Matrix< T, ROW, COL >& eiMat_ )
{
    eiMat_ << vvVec_;
	return vvVec_;
}

// 4.1 std::vector -> cv::Mat_
template < class T >
const std::vector< std::vector< T > >& operator >> ( const std::vector< std::vector< T > >& vvVec_,  cv::Mat_< T >& cvMat_ )
{
    cvMat_ << vvVec_;
	return vvVec_;
}

// 4.2 std::vector< < < > > > -> std::vector< cv::Mat_<> >
template < class T >
const std::vector< std::vector< std::vector< T > > >& operator >> ( const std::vector< std::vector< std::vector< T > > >& vvvVec_, std::vector< cv::Mat_< T > >& vmMat_ )
{
    vmMat_ << vvvVec_;
}

// 5.1 std::vector< < < > > > -> std::vector< cv::Mat >
template < class T >
const std::vector< std::vector< std::vector< T > > >& operator >> ( const std::vector< std::vector< std::vector< T > > >& vvvVec_, std::vector< cv::Mat >& vmMat_ )
{
    vmMat_ << vvvVec_;
	return vvvVec_;
}









//template< class T >
//Matrix< T, 3, 3 > skewSymmetric( const Matrix< T, 3, 1>& eivVec_ )
//{
//	Matrix< T, 3, 3 > eimMat;
//	/*0*/                       eimMat(0,1) =  -eivVec_(2); eimMat(0,2) =  eivVec_(1);
//	eimMat(1,0) =   eivVec_(2); /*0*/                       eimMat(1,2) = -eivVec_(0);
//	eimMat(2,0) =  -eivVec_(1); eimMat(2,1) =   eivVec_(0); /*0*/
//	return eimMat;
//}
/*
template< class T >
Matrix< T, 3,3> fundamental(const Matrix< T, 3, 3 >& eimK1_, const Matrix< T, 3, 3 >& eimK2_, const Matrix< T, 3,3>& eimR_, const Matrix< T, 3,1 >& eivT_, cv::Mat_< T >* pcvmDepthNew_ )
{
// compute fundamental matrix that the first camera is on classic pose and the second is on R and T pose, the internal
// parameters of first camera is K1, and the second is K2
// reference Multiple view geometry on computer vision page 244.
//  F = K2^{-T}RK^{T} * skew( K R^{T} t );
	Matrix< T, 3, 3> eimF = eimK2_.inverse().eval().transpose() * eimR_ * eimK1_.transpose() * skewSymmetric( eimK1_ * eimR_.transpose() * eivT_ );
	return eimF;
}
*/
}//utility
}//btl

#endif
