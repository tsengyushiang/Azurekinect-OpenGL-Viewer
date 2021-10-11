#pragma once

#include <thread>
#include <functional>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Video write
using namespace std;

#define INPUT_COLOR_CHANNEL 4

typedef struct intrinsic {
    float fx;
    float fy;
    float ppx;
    float ppy;
    float depth_scale;

}Intrinsic;

class InputBase
{
private:
    void keepTryingSave();

public:

    bool showOpenCVwindow = false;

    // for json data
    int syncTime = 0;
    int currentTime = -1;
    virtual bool setFrameIndex(int) { return false; };

    InputBase(
        int cw,
        int ch,
        int dw,
        int dh
    );

    ~InputBase();

    bool calibrated = false;
    glm::mat4 modelMat;

    bool floorEquationGot = false;
    glm::vec3 esitmatePlaneCenter;
    glm::vec3 esitmatePlaneNormal;

    std::string serial;
    // result resolution of aligned depth/color
    int width;
    int height;

    // resolution for color/depth
    int cwidth, cheight, dwidth, dheight;

    uint16_t* p_depth_frame;    
    unsigned char* p_color_frame;

    // record test
    uint16_t** depth_cache;
    unsigned char** color_cache;
    const int MAXCACHE = 1000;
    int curRecordFrame=0;
    int curSaveFrame = 0;
    int maxSave = 0;
    bool isRecording;
    thread* recordThread;
    FILE* recordColorFile;
    FILE* recordDepthFile;
    void startRecord(int);

    // post process : clip point
    float farPlane = 5.0;
    float point2floorDistance=100;
    bool enableUpdateFrame = true;
    bool frameNeedsUpdate = false;
    virtual bool fetchframes(std::function<void(
        const void* depthRaw, size_t depthSize,
        const void* colorRaw, size_t colorSize)
    >callback);

    // fetch signle pixel-point
    glm::vec3 colorPixel2point(glm::vec2);


    /*
    cuda mem for 3d point:
        x= xy_table_cuda[i*2]*depth
        y= xy_table_cuda[i*2+1]*depth
    */
    Intrinsic intri;
    float* xy_table_cuda;
    float* xy_table;
    bool xy_tableReady = false;
    void setXYtable(float ppx, float ppy, float fx, float fy, bool forceupdate=false);
};

