#ifndef DLL_EXPORT_DEF_HEADER
#define DLL_EXPORT_DEF_HEADER


#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#	if defined (_MSC_VER)
#		pragma warning(disable: 4251)
#	endif
	#if defined(EXPORT)
#		define  DLL_EXPORT __declspec(dllexport)
#	else
#		define  DLL_EXPORT __declspec(dllimport)
#	endif
#else
#	define DLL_EXPORT
#endif

#endif