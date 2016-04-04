#ifndef BTL_OTHER_UTILITY_HELPER
#define BTL_OTHER_UTILITY_HELPER

//helpers based-on stl and boost

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <complex>
#include <string>

//#include <Eigen/Dense>

#include <boost/exception/all.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <opencv2/core.hpp>
namespace btl
{
namespace utility
{

#define SMALL 1e-50 // a small value
#define BTL_DOUBLE_MAX 10e20
	enum tp_coordinate_convention { BTL_GL, BTL_CV };
	//exception based on boost
	typedef boost::error_info<struct tag_my_info, std::string> CErrorInfo;
	struct CError: virtual boost::exception, virtual std::exception { };
#define THROW(what)\
	{\
	btl::utility::CError cE;\
	cE << btl::utility::CErrorInfo ( what );\
	throw cE;\
	}
	//exception from btl2
	struct CException : public std::runtime_error
	{
		CException(const std::string& str) : std::runtime_error(str) {}
	};
#define BTL_THROW(what) {throw btl::utility::CException(what);}
	//ASSERT condition to be true; other wise throw
#define CHECK( AssertCondition_, Otherwise_) \
	if ((AssertCondition_) != true)\
	BTL_THROW( Otherwise_ );
	//THROW( Otherwise_ );
	//if condition happen then throw
#define BTL_ERROR( ErrorCondition_, ErrorMessage_ ) CHECK( !(ErrorCondition_), ErrorMessage_) 
#define BTL_ASSERT CHECK
// for print
template <class T>
std::ostream& operator << ( std::ostream& os, const std::vector< T > & v )
{
	os << "[";

	for ( typename std::vector< T >::const_iterator constItr = v.begin(); constItr != v.end(); ++constItr )
	{
		os << " " << ( *constItr ) << " ";
	}

	os << "]";
	return os;
}

template <class T1, class T2>
std::ostream& operator << ( std::ostream& os, const std::map< T1, T2 > & mp )
{
    os << "[";

    for ( typename std::map< T1, T2 >::const_iterator constItr = mp.begin(); constItr != mp.end(); ++constItr )
    {
        os << " " << ( *constItr ).first << ": " << ( *constItr ).second << " ";
    }

    os << "]";
    return os;
}

template <class T>
std::ostream& operator << ( std::ostream& os, const std::list< T >& l_ )
{
    os << "[";
    for ( typename std::list< T >::const_iterator cit_List = l_.begin(); cit_List != l_.end(); cit_List++ )
    {
        os << " " << *cit_List << " ";
    }
    os << "]";
    return os;
}

#ifdef  INFO
// based on boost stringize.hpp
#define PRINT( a ) std::cout << BOOST_PP_STRINGIZE( a ) << " = " << std::endl << (a) << std::flush << std::endl;
#define PRINTSTR( a ) std::cout << a << std::endl << std::flush;
#else
#define PRINT( a ) 
#define PRINTSTR( a ) 
#endif//INFO

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}
//calculate vector<> difference for testing
template< class T>
T matNormL1 ( const std::vector< T >& vMat1_, const std::vector< T >& vMat2_ )
{
	T tAccumDiff = 0;
	for(unsigned int i=0; i < vMat1_.size(); i++ )
	{
		T tDiff = vMat1_[i] - vMat2_[i];
		tDiff = tDiff >= 0? tDiff:-tDiff;
		tAccumDiff += tDiff;
	}
	return tAccumDiff;
}


template< class T >
void getNeighbourIdxCylinder(const unsigned short& usRows, const unsigned short& usCols, const T& i, std::vector< T >* pNeighbours_ )
{
	// get the neighbor 1d index in a cylindrical coordinate system
	int a = usRows*usCols;
	BTL_ASSERT(i>=0 && i<a,"btl::utility::getNeighbourIdx() i is out of range");

	pNeighbours_->clear();
	pNeighbours_->push_back(i);
	T r = i/usCols;
	T c = i%usCols;
	T nL= c==0?        i-1 +usCols : i-1;	
	T nR= c==usCols-1? i+1 -usCols : i+1;
	pNeighbours_->push_back(nL);
	pNeighbours_->push_back(nR);

	if(r>0)//get up
	{
		T nU= i-usCols;
		pNeighbours_->push_back(nU);
		T nUL= nU%usCols == 0? nU-1 +usCols: nU-1;
		pNeighbours_->push_back(nUL);
		T nUR= nU%usCols == usCols-1? nU+1 -usCols : nU+1;
		pNeighbours_->push_back(nUR);
	}
	else if(r==usRows-1)//correspond to polar region
	{
		T t = r*usCols;
		for( T n=0; n<usCols; n++)
			pNeighbours_->push_back(t+n);
	}
	if(r<usRows-1)//get down
	{
		T nD= i+usCols;
		pNeighbours_->push_back(nD);
		T nDL= nD%usCols == 0? nD-1 +usCols: nD-1;
		pNeighbours_->push_back(nDL);
		T nDR= nD%usCols == usCols-1? nD+1 -usCols : nD+1;
		pNeighbours_->push_back(nDR);
	}

	return;
}


using namespace std;
// places randomly selected  element at end of array
// then shrinks array by 1 element. Done when only 1 element is left
// m is the # to be selected from
// n is the total # of elements
template< class T >
class  RandomElements
{
	T* _idx;
	int _n;
public:
	RandomElements(int n):_n(n){
		_idx = new T[n];
	}
	~RandomElements(){
		delete _idx;
	}

	void run( int m, vector< T >* p_v_idx_ )
	{
		p_v_idx_->clear();
		for (T i = 0; i < _n; i++) {
			_idx[i] = i;
		}
		int temp = 0;
		int ridx = _n-1;
		for(int j=(_n-1); j>_n-m-1; j--)// one pass through array
		{
			ridx = rand()%(j+1);// index = 0 to j
			temp = _idx[ridx];// value will be moved to end element
			_idx[ridx] = _idx[j];// end element value in random spot
			_idx[j] = temp;// selected element moved to end. This value is final
			p_v_idx_->push_back(temp);
		}
		return;
	}
};//class RandElement


template< class T >
void rand_sel_2(int nSelections_, int nGroupSize_, cv::Mat* p_rand_idx)
{
	cv::Mat& rand_idx = *p_rand_idx; rand_idx.create(nSelections_, 1, CV_16UC2);
	btl::utility::RandomElements<int> re(nGroupSize_*nGroupSize_);
	vector<T> raw_idx;
	re.run(nSelections_ + 10, &raw_idx);
	int nSelected = 0;
	for (vector<int>::const_iterator cit = raw_idx.begin(); cit != raw_idx.end(); cit++){
		int r = *cit / 100;
		int c = *cit % 100;
		if (r != c){
			
			rand_idx.template ptr<ushort>(nSelected)[0] = r;
			rand_idx.template ptr<ushort>(nSelected)[1] = c;
			nSelected++;
			if (nSelected >= nSelections_) break;
		}
	}
	if (nSelected < nSelections_) rand_idx.pop_back(nSelections_ - nSelected);
	return;
}

}//utility
}//btl

#endif
