/****************************************************************************

 http://www.libqglviewer.com - contact@libqglviewer.com
 
*****************************************************************************/
using namespace std;


class Viewer : public QGLViewer
{
public:
	Viewer(string strName_, DataLive::tp_shared_ptr pData, QWidget* parent, const QGLWidget* shareWidget);
    ~Viewer();
protected :
    virtual void draw();
    virtual void init();
    virtual QString helpString() const;
	virtual void keyPressEvent(QKeyEvent *e);
	virtual void mouseDoubleClickEvent(QMouseEvent* e);
	DataLive::tp_shared_ptr _pData;
	string _strViewer;
	bool _bShowText;
};
