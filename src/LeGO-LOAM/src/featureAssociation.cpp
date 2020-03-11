// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include "utility.h"


/*
    涉及到的坐标转换关系
    上下两帧之间的关系 Xi = (R_roll * R_pitch * R_yaw) * Xi-1 + T
                       Xi-1 = R_yaw^-1 * R_pitch^-1 * R_roll^-1 * (Xi - T)
                            = R_(-yaw) * R_(-pitch) * R_(-roll) * (Xi - T)

    第i-1帧和全局坐标系关系 X0 = R_sum * Xi-1 + T_sum
    所以第i帧和全局坐标系关系 X0 = R_sum * (R_(-yaw) * R_(-pitch) * R_(-roll) * (Xi - T)) * T_sum
                                 = R_sum * (R_(-yaw) * R_(-pitch) * R_(-roll)) * Xi + (T_sum - R_sum * (R_(-yaw) * R_(-pitch) * R_(-roll)) * T)
                              
                      更新R  R_sum = R_sum * R_(-yaw) * R_(-pitch) * R_(-roll) 
                      更新T  T_sum = T_sum - R_sum * T
*/


//特征提取匹配类
class FeatureAssociation{

private:

	ros::NodeHandle nh;         //句柄

    ros::Subscriber subLaserCloud;      //订阅点云
    ros::Subscriber subLaserCloudInfo;  //订阅点云距离
    ros::Subscriber subOutlierCloud;    //订阅外点句柄
    ros::Subscriber subImu;             //订阅IMU信息

    ros::Publisher pubCornerPointsSharp;        //发布边缘点
    ros::Publisher pubCornerPointsLessSharp;    //发布次边缘点
    ros::Publisher pubSurfPointsFlat;           //发布平面点
    ros::Publisher pubSurfPointsLessFlat;       //发布次平面点

    pcl::PointCloud<PointType>::Ptr segmentedCloud;     // 用于特征提取的点云(分簇的点+部分地面点)
    pcl::PointCloud<PointType>::Ptr outlierCloud;       // 外点

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;      //边缘点信息
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;  //次边缘点信息
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;         //平面点信息
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;     //次平面点信息

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;     // 每一线的次平面点
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;   // 所有的次平面点

    pcl::VoxelGrid<PointType> downSizeFilter;       //采样器

    // 时间戳
    double timeScanCur;
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewOutlierCloud;

    // 新来一帧的标志
    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newOutlierCloud;

   
    cloud_msgs::cloud_info segInfo;     //点云信息
    std_msgs::Header cloudHeader;       //点云消息头

  
    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;   //保存每一个点的曲率
    float cloudCurvature[N_SCAN*Horizon_SCAN];      //协方差
    int cloudNeighborPicked[N_SCAN*Horizon_SCAN];   //点云是否被筛选
    int cloudLabel[N_SCAN*Horizon_SCAN];            //特征点类型
    

    //imu
    int imuPointerFront;    //当前点对应的imu下标
    int imuPointerLast;     //当前imu消息的下标
    int imuPointerLastIteration;    //最新的imu下标 

    //点云中第一个点对应的imu信息
    float imuRollStart, imuPitchStart, imuYawStart;
    float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
    float imuRollCur, imuPitchCur, imuYawCur;   //当前点的imu信息，当所有点云遍历完后，imuXXXCur就指向了最后一个点对应的imu信息

    //第一个点对应的imu计算得到的速度，位移
    float imuVeloXStart, imuVeloYStart, imuVeloZStart;
    float imuShiftXStart, imuShiftYStart, imuShiftZStart;

    //当前点对应的imu计算得到的速度，位移
    float imuVeloXCur, imuVeloYCur, imuVeloZCur;
    float imuShiftXCur, imuShiftYCur, imuShiftZCur;

    //imu匀速运动模型的误差(已经转到第一个点坐标系下)
    float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
    float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

    float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;       //当前帧第一个点对应的imu角速度计算得到的累积角度
    float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;    //上一帧第一个点对应的imu角速度计算得到的累积角度
    float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;         //上下两帧之间imu角速度计算得到的角度差

    double imuTime[imuQueLength];   //imu时间戳
    float imuRoll[imuQueLength];    //imu roll
    float imuPitch[imuQueLength];   //imu pitch
    float imuYaw[imuQueLength];     //imu yaw

    float imuAccX[imuQueLength];    //imu x轴加速度
    float imuAccY[imuQueLength];    //imu y轴加速度
    float imuAccZ[imuQueLength];    //imu z轴加速度

    float imuVeloX[imuQueLength];   //imu x轴速度(匀加速模型)
    float imuVeloY[imuQueLength];   //imu y轴速度
    float imuVeloZ[imuQueLength];   //imu z轴速度

    float imuShiftX[imuQueLength];  //imu x轴位移(匀加速模型)
    float imuShiftY[imuQueLength];  //imu y轴位移
    float imuShiftZ[imuQueLength];  //imu z轴位移

    float imuAngularVeloX[imuQueLength];    //imu x轴角速度
    float imuAngularVeloY[imuQueLength];    //imu y轴角速度
    float imuAngularVeloZ[imuQueLength];    //imu z轴角速度

    float imuAngularRotationX[imuQueLength];    //imu角速度累积得到的角度变化
    float imuAngularRotationY[imuQueLength];    //
    float imuAngularRotationZ[imuQueLength];    //



    ros::Publisher pubLaserCloudCornerLast;     //发布边缘点信息
    ros::Publisher pubLaserCloudSurfLast;       //发布平面点信息
    ros::Publisher pubLaserOdometry;            //发布odometry
    ros::Publisher pubOutlierCloudLast;         //发布outliers点信息

    int skipFrameNum;           //跳帧
    bool systemInitedLM;        //系统初始化

    //点个数
    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    //次边缘点 
    int pointSelCornerInd[N_SCAN*Horizon_SCAN];
    float pointSearchCornerInd1[N_SCAN*Horizon_SCAN];
    float pointSearchCornerInd2[N_SCAN*Horizon_SCAN];

    //次平面点
    int pointSelSurfInd[N_SCAN*Horizon_SCAN];
    float pointSearchSurfInd1[N_SCAN*Horizon_SCAN];
    float pointSearchSurfInd2[N_SCAN*Horizon_SCAN];
    float pointSearchSurfInd3[N_SCAN*Horizon_SCAN];

    // 6个自由度
    float transformCur[6];      //保存上下两帧之间的位姿关系
    float transformSum[6];      //odometry

    // 
    float imuRollLast, imuPitchLast, imuYawLast;        //最后一个点对应的imu
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;    //最后一个点和第一个点利用imu的参数计算得到的位置误差(转到第一个点坐标系下)
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;      //最后一个点和第一个点利用imu的参数计算得到的速度误差(转到第一个点坐标系下)

    //上一帧的次边缘点和次平面点
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    //边缘点和平面点的kd树
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    //下标和距离
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    nav_msgs::Odometry laserOdometry;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    //判断退化场景
    bool isDegenerate;
    cv::Mat matP;

    int frameCount;

public:

    //构造函数
    FeatureAssociation():
        nh("~")
        {

        //用于特征提取的点云(成簇的点云+部分地面点)
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &FeatureAssociation::laserCloudHandler, this);
        
        //点云信息
        subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &FeatureAssociation::laserCloudInfoHandler, this);
        
        //外点
        subOutlierCloud = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlierCloudHandler, this);
        
        //IMU信息
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imuHandler, this);

        //发布边缘点信息
        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        
        //发布次边缘点信息
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        
        //发布平面点信息
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
        
        //发布次平面点信息
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

        //用于地图匹配的次边缘点
        pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
        
        //用于地图匹配的次平面点
        pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
        
        //用于地图匹配的外点
        pubOutlierCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
        
        //发布轨迹
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);
        
        //初始化成员变量
        initializationValue();
    }

    //初始化
    void initializationValue()
    {
        // N_SCANS = 16 Horizon_SCAN = 1800
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);    // 点云曲率16*1800

        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);      // 采样间隔0.2m

        segmentedCloud.reset(new pcl::PointCloud<PointType>());  //用于特征提取的点云
        outlierCloud.reset(new pcl::PointCloud<PointType>());    //外点

        //边缘点和次边缘点
        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        
        //平面点和次平面点
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        //每一线的次平面点
        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        //初始化为0
        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewOutlierCloud = 0;

        //初始化为false
        newSegmentedCloud = false;
        newSegmentedCloudInfo = false;
        newOutlierCloud = false;

        //系统初始化
        systemInitCount = 0;
        systemInited = false;

        //imu信息
        imuPointerFront = 0;
        imuPointerLast = -1;
        imuPointerLastIteration = 0;

        imuRollStart = 0; imuPitchStart = 0; imuYawStart = 0;
        cosImuRollStart = 0; cosImuPitchStart = 0; cosImuYawStart = 0;
        sinImuRollStart = 0; sinImuPitchStart = 0; sinImuYawStart = 0;
        imuRollCur = 0; imuPitchCur = 0; imuYawCur = 0;

        imuVeloXStart = 0; imuVeloYStart = 0; imuVeloZStart = 0;
        imuShiftXStart = 0; imuShiftYStart = 0; imuShiftZStart = 0;

        imuVeloXCur = 0; imuVeloYCur = 0; imuVeloZCur = 0;
        imuShiftXCur = 0; imuShiftYCur = 0; imuShiftZCur = 0;

        imuShiftFromStartXCur = 0; imuShiftFromStartYCur = 0; imuShiftFromStartZCur = 0;
        imuVeloFromStartXCur = 0; imuVeloFromStartYCur = 0; imuVeloFromStartZCur = 0;

        imuAngularRotationXCur = 0; imuAngularRotationYCur = 0; imuAngularRotationZCur = 0;
        imuAngularRotationXLast = 0; imuAngularRotationYLast = 0; imuAngularRotationZLast = 0;
        imuAngularFromStartX = 0; imuAngularFromStartY = 0; imuAngularFromStartZ = 0;

        //全部初始化为0
        for (int i = 0; i < imuQueLength; ++i)
        {
            imuTime[i] = 0;
            imuRoll[i] = 0; imuPitch[i] = 0; imuYaw[i] = 0;
            imuAccX[i] = 0; imuAccY[i] = 0; imuAccZ[i] = 0;
            imuVeloX[i] = 0; imuVeloY[i] = 0; imuVeloZ[i] = 0;
            imuShiftX[i] = 0; imuShiftY[i] = 0; imuShiftZ[i] = 0;
            imuAngularVeloX[i] = 0; imuAngularVeloY[i] = 0; imuAngularVeloZ[i] = 0;
            imuAngularRotationX[i] = 0; imuAngularRotationY[i] = 0; imuAngularRotationZ[i] = 0;
        }


        skipFrameNum = 1;       //跳一帧发一次

        //全部初始化为0
        for (int i = 0; i < 6; ++i){
            transformCur[i] = 0;
            transformSum[i] = 0;
        }

        systemInitedLM = false;

        //初始化为0
        imuRollLast = 0; imuPitchLast = 0; imuYawLast = 0;
        imuShiftFromStartX = 0; imuShiftFromStartY = 0; imuShiftFromStartZ = 0;
        imuVeloFromStartX = 0; imuVeloFromStartY = 0; imuVeloFromStartZ = 0;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/laser_odom";

        laserOdometryTrans.frame_id_ = "/camera_init";
        laserOdometryTrans.child_frame_id_ = "/laser_odom";
        
        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        frameCount = skipFrameNum;
    }

    //
    void updateImuRollPitchYawStartSinCos(){
        cosImuRollStart = cos(imuRollStart);
        cosImuPitchStart = cos(imuPitchStart);
        cosImuYawStart = cos(imuYawStart);
        sinImuRollStart = sin(imuRollStart);
        sinImuPitchStart = sin(imuPitchStart);
        sinImuYawStart = sin(imuYawStart);
    }

    //畸变转换到第一个点对应的imu坐标系下
    void ShiftToStartIMU(float pointTime)
    {
        //匀速运动模型产生的误差
        imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
        imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
        imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

        //绕yaw逆时针旋转
        float x1 = cosImuYawStart * imuShiftFromStartXCur - sinImuYawStart * imuShiftFromStartZCur;
        float y1 = imuShiftFromStartYCur;
        float z1 = sinImuYawStart * imuShiftFromStartXCur + cosImuYawStart * imuShiftFromStartZCur;

        //绕pitch逆时针旋转
        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;

        //绕roll逆时针旋转
        imuShiftFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuShiftFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuShiftFromStartZCur = z2;
    }

    //速度畸变转换到第一个点坐标系下
    void VeloToStartIMU()
    {
        //匀速模型的速度误差
        imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
        imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
        imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

        //绕yaw逆时针旋转
        float x1 = cosImuYawStart * imuVeloFromStartXCur - sinImuYawStart * imuVeloFromStartZCur;
        float y1 = imuVeloFromStartYCur;
        float z1 = sinImuYawStart * imuVeloFromStartXCur + cosImuYawStart * imuVeloFromStartZCur;

        //绕pitch逆时针旋转 
        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;

        //绕roll逆时针旋转
        imuVeloFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuVeloFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuVeloFromStartZCur = z2;
    }

    //转到imuStart坐标系下
    void TransformToStartIMU(PointType *p)
    {
        float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
        float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
        float z1 = p->z;

        float x2 = x1;
        float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
        float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

        float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
        float y3 = y2;
        float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

        float x4 = cosImuYawStart * x3 - sinImuYawStart * z3;
        float y4 = y3;
        float z4 = sinImuYawStart * x3 + cosImuYawStart * z3;

        float x5 = x4;
        float y5 = cosImuPitchStart * y4 + sinImuPitchStart * z4;
        float z5 = -sinImuPitchStart * y4 + cosImuPitchStart * z4;

        p->x = cosImuRollStart * x5 + sinImuRollStart * y5 + imuShiftFromStartXCur;
        p->y = -sinImuRollStart * x5 + cosImuRollStart * y5 + imuShiftFromStartYCur;
        p->z = z5 + imuShiftFromStartZCur;
    }

    // IMU距离和角度累积
    void AccumulateIMUShiftAndRotation()
    {
        // roll, pitch, yaw, accX, accY, accZ
        //当前imu返回的参数
        float roll = imuRoll[imuPointerLast];
        float pitch = imuPitch[imuPointerLast];
        float yaw = imuYaw[imuPointerLast];
        float accX = imuAccX[imuPointerLast];
        float accY = imuAccY[imuPointerLast];
        float accZ = imuAccZ[imuPointerLast];

        // 绕roll(z轴)旋转
        float x1 = cos(roll) * accX - sin(roll) * accY;
        float y1 = sin(roll) * accX + cos(roll) * accY;
        float z1 = accZ;

        // 绕pitch(x轴)旋转
        float x2 = x1;
        float y2 = cos(pitch) * y1 - sin(pitch) * z1;
        float z2 = sin(pitch) * y1 + cos(pitch) * z1;

        // 绕yaw(y轴)旋转
        accX = cos(yaw) * x2 + sin(yaw) * z2;
        accY = y2;
        accZ = -sin(yaw) * x2 + cos(yaw) * z2;

        int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;    //下标
        double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];    //时间差
        if (timeDiff < scanPeriod) {

            //匀加速推算位移
            imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff + accX * timeDiff * timeDiff / 2;
            imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff + accY * timeDiff * timeDiff / 2;
            imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff + accZ * timeDiff * timeDiff / 2;

            //匀加速推算速度
            imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
            imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
            imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;

            //推算角度
            imuAngularRotationX[imuPointerLast] = imuAngularRotationX[imuPointerBack] + imuAngularVeloX[imuPointerBack] * timeDiff;
            imuAngularRotationY[imuPointerLast] = imuAngularRotationY[imuPointerBack] + imuAngularVeloY[imuPointerBack] * timeDiff;
            imuAngularRotationZ[imuPointerLast] = imuAngularRotationZ[imuPointerBack] + imuAngularVeloZ[imuPointerBack] * timeDiff;
        }
    }

    // imu回调函数
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
    {
        //tf转换得到roll,pitch,yaw
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // 剔除重力加速度
        float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
        float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
        float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

        // imu当前下标更新
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;

        //时间戳
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();

        //roll, pitch, yaw
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
        imuYaw[imuPointerLast] = yaw;

        //加速度
        imuAccX[imuPointerLast] = accX;
        imuAccY[imuPointerLast] = accY;
        imuAccZ[imuPointerLast] = accZ;

        //角速度
        imuAngularVeloX[imuPointerLast] = imuIn->angular_velocity.x;
        imuAngularVeloY[imuPointerLast] = imuIn->angular_velocity.y;
        imuAngularVeloZ[imuPointerLast] = imuIn->angular_velocity.z;

        //累积
        AccumulateIMUShiftAndRotation();
    }

    //点云回调函数
    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;    //点云消息头

        timeScanCur = cloudHeader.stamp.toSec();    //时间戳
        timeNewSegmentedCloud = timeScanCur;

        //segmentedCloud赋值
        segmentedCloud->clear();
        pcl::fromROSMsg(*laserCloudMsg, *segmentedCloud);

        newSegmentedCloud = true;   //点云更新标志位置true
    }

    // 外点回调函数
    void outlierCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        timeNewOutlierCloud = msgIn->header.stamp.toSec();  //时间戳

        //outlierCloud赋值
        outlierCloud->clear();
        pcl::fromROSMsg(*msgIn, *outlierCloud);

        newOutlierCloud = true;     //点云更新标志置true
    }

    //点云信息回调函数
    void laserCloudInfoHandler(const cloud_msgs::cloud_infoConstPtr& msgIn)
    {
        timeNewSegmentedCloudInfo = msgIn->header.stamp.toSec();
        segInfo = *msgIn;
        newSegmentedCloudInfo = true;
    }


    //运动补偿,畸变处理
    void adjustDistortion()
    {
        bool halfPassed = false;
        int cloudSize = segmentedCloud->points.size();

        PointType point;

        for (int i = 0; i < cloudSize; i++) {

            //三轴互换(为什么这样做？)
            point.x = segmentedCloud->points[i].y;  //对应pitch
            point.y = segmentedCloud->points[i].z;  //对应yaw
            point.z = segmentedCloud->points[i].x;  //对应roll

            float ori = -atan2(point.x, point.z);   //计算x轴夹角
            if (!halfPassed) {
                if (ori < segInfo.startOrientation - M_PI / 2)
                    ori += 2 * M_PI;
                else if (ori > segInfo.startOrientation + M_PI * 3 / 2)
                    ori -= 2 * M_PI;

                if (ori - segInfo.startOrientation > M_PI)
                    halfPassed = true;
            } else {
                ori += 2 * M_PI;

                if (ori < segInfo.endOrientation - M_PI * 3 / 2)
                    ori += 2 * M_PI;
                else if (ori > segInfo.endOrientation + M_PI / 2)
                    ori -= 2 * M_PI;
            }

            float relTime = (ori - segInfo.startOrientation) / segInfo.orientationDiff; //计算当前点在一个lidar周期内的位置
            point.intensity = int(segmentedCloud->points[i].intensity) + scanPeriod * relTime;  //整数部分表示线号，小数部分表示一个lidar周期内的时间戳

            //判断是否有IMU消息接入
            if (imuPointerLast >= 0) {
                float pointTime = relTime * scanPeriod;
                imuPointerFront = imuPointerLastIteration;
                while (imuPointerFront != imuPointerLast) {
                    if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
                        break;
                    }
                    imuPointerFront = (imuPointerFront + 1) % imuQueLength;
                }

                //imu消息滞后了,那就用最新的IMU信息
                if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
                    imuRollCur = imuRoll[imuPointerFront];
                    imuPitchCur = imuPitch[imuPointerFront];
                    imuYawCur = imuYaw[imuPointerFront];

                    imuVeloXCur = imuVeloX[imuPointerFront];
                    imuVeloYCur = imuVeloY[imuPointerFront];
                    imuVeloZCur = imuVeloZ[imuPointerFront];

                    imuShiftXCur = imuShiftX[imuPointerFront];
                    imuShiftYCur = imuShiftY[imuPointerFront];
                    imuShiftZCur = imuShiftZ[imuPointerFront];   
                } else {
                   //线性插值
                    int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                    float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                     / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                    / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

                    imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                    imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
                    if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
                    } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
                    } else {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
                    }

                    imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
                    imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
                    imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

                    imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
                    imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
                    imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
                }

                if (i == 0) {
                    //点云第一个点对应的IMU信息
                    imuRollStart = imuRollCur;
                    imuPitchStart = imuPitchCur;
                    imuYawStart = imuYawCur;

                    imuVeloXStart = imuVeloXCur;
                    imuVeloYStart = imuVeloYCur;
                    imuVeloZStart = imuVeloZCur;

                    imuShiftXStart = imuShiftXCur;
                    imuShiftYStart = imuShiftYCur;
                    imuShiftZStart = imuShiftZCur;

                    if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront];
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront];
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront];
                    }else{
                        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront] * ratioFront + imuAngularRotationX[imuPointerBack] * ratioBack;
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront] * ratioFront + imuAngularRotationY[imuPointerBack] * ratioBack;
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront] * ratioFront + imuAngularRotationZ[imuPointerBack] * ratioBack;
                    }

                    //第一个点的累积角度和上一帧第一个点累积角度差
                    imuAngularFromStartX = imuAngularRotationXCur - imuAngularRotationXLast;
                    imuAngularFromStartY = imuAngularRotationYCur - imuAngularRotationYLast;
                    imuAngularFromStartZ = imuAngularRotationZCur - imuAngularRotationZLast;

                    imuAngularRotationXLast = imuAngularRotationXCur;
                    imuAngularRotationYLast = imuAngularRotationYCur;
                    imuAngularRotationZLast = imuAngularRotationZCur;

                    updateImuRollPitchYawStartSinCos();
                } else {
                    VeloToStartIMU();
                    TransformToStartIMU(&point);        //转到startImu坐标系
                }
            }
            segmentedCloud->points[i] = point;  //重新赋值 
        }

        imuPointerLastIteration = imuPointerLast;
    }

    //计算曲率
    void calculateSmoothness()
    {
        int cloudSize = segmentedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++) {

            //当前点前后10个点计算距离差
            float diffRange = segInfo.segmentedCloudRange[i-5] + segInfo.segmentedCloudRange[i-4]
                            + segInfo.segmentedCloudRange[i-3] + segInfo.segmentedCloudRange[i-2]
                            + segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i] * 10
                            + segInfo.segmentedCloudRange[i+1] + segInfo.segmentedCloudRange[i+2]
                            + segInfo.segmentedCloudRange[i+3] + segInfo.segmentedCloudRange[i+4]
                            + segInfo.segmentedCloudRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;    //计算协方差

            cloudNeighborPicked[i] = 0;     //初始化
            cloudLabel[i] = 0;              //初始化

            //初始化
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    //遮挡
    void markOccludedPoints()
    {
        int cloudSize = segmentedCloud->points.size();

        for (int i = 5; i < cloudSize - 6; ++i){

            float depth1 = segInfo.segmentedCloudRange[i];
            float depth2 = segInfo.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i+1] - segInfo.segmentedCloudColInd[i]));
            
            //列下标之差
            if (columnDiff < 10){

                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }

            float diff1 = std::abs(segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i]);
            float diff2 = std::abs(segInfo.segmentedCloudRange[i+1] - segInfo.segmentedCloudRange[i]);

            if (diff1 > 0.02 * segInfo.segmentedCloudRange[i] && diff2 > 0.02 * segInfo.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    //特征点选取
    void extractFeatures()
    {
        //清空容器
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();

        for (int i = 0; i < N_SCAN; i++) {

            surfPointsLessFlatScan->clear();

            //每一根线分6份
            for (int j = 0; j < 6; j++) {

                int sp = (segInfo.startRingIndex[i] * (6 - j)    + segInfo.endRingIndex[i] * j) / 6;
                int ep = (segInfo.startRingIndex[i] * (5 - j)    + segInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                //按照曲率从小到大排序
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                //边缘点选取不在地面上
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > edgeThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == false) {
                    
                        largestPickedNum++;
                        if (largestPickedNum <= 2) {        //边缘点选取2个
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else if (largestPickedNum <= 20) {    //次边缘点选取20个
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else {
                            break;
                        }

                        //防止特征点聚集
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                //平面点选取在地面上
                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < surfThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == true) {

                        cloudLabel[ind] = -1;
                        surfPointsFlat->push_back(segmentedCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4) {   //平面点选取4个
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {    //剩下全是次平面点
                    if (cloudLabel[k] <= 0) {
                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
                    }
                }
            }

            //下采样
            surfPointsLessFlatScanDS->clear();
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.filter(*surfPointsLessFlatScanDS);

            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
        }
    }

    void publishCloud()
    {
        sensor_msgs::PointCloud2 laserCloudOutMsg;

        //边缘点
	    if (pubCornerPointsSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsSharp.publish(laserCloudOutMsg);
	    }

        //次边缘点
	    if (pubCornerPointsLessSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsLessSharp.publish(laserCloudOutMsg);
	    }

        //平面点
	    if (pubSurfPointsFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsFlat.publish(laserCloudOutMsg);
	    }

        //次平面点
	    if (pubSurfPointsLessFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsLessFlat.publish(laserCloudOutMsg);
	    }
    }


    /*
        将点pi转换到这一帧第一个点坐标系下
        Xi = R_roll * R_pitch * R_yaw * Xi-1 + T
        Xi-1 = R_yaw^-1 * R_pitch^-1 * R_roll-1 * (Xi - T)
        先按照roll(z轴)逆时针旋转，然后pitch(x轴)逆时针旋转，最后yaw(y轴)逆时针旋转
    */
    void TransformToStart(PointType const * const pi, PointType * const po)
    {
        float s = 10 * (pi->intensity - int(pi->intensity));    //pi这个点在lidar周期内的位置

        //线性插值计算pi这个点对应的位姿
        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];

        //绕roll(z轴)逆时针旋转
        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);

        //绕pitch(x轴)逆时针旋转
        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;

        //绕yaw(y轴)逆时针旋转
        po->x = cos(ry) * x2 - sin(ry) * z2;
        po->y = y2;
        po->z = sin(ry) * x2 + cos(ry) * z2;
        po->intensity = pi->intensity;
    }

    /*
        将点pi转换到这一帧最后一个点坐标系下
        Xi = R_roll * R_pitch * R_yaw * Xi-1 + T
    */
    void TransformToEnd(PointType const * const pi, PointType * const po)
    {
        float s = 10 * (pi->intensity - int(pi->intensity));    //pi这个点在lidar周期内的位置

        //线性插值计算pi点对应的位姿
        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];

        //绕roll(z轴)逆时针旋转
        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);

        //绕pitch(x轴)逆时针旋转
        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;

        //绕yaw(y轴)逆时针旋转
        float x3 = cos(ry) * x2 - sin(ry) * z2;
        float y3 = y2;
        float z3 = sin(ry) * x2 + cos(ry) * z2;

        rx = transformCur[0];
        ry = transformCur[1];
        rz = transformCur[2];
        tx = transformCur[3];
        ty = transformCur[4];
        tz = transformCur[5];

        //绕yaw(y轴)旋转
        float x4 = cos(ry) * x3 + sin(ry) * z3;
        float y4 = y3;
        float z4 = -sin(ry) * x3 + cos(ry) * z3;

        //绕pitch(x轴)旋转
        float x5 = x4;
        float y5 = cos(rx) * y4 - sin(rx) * z4;
        float z5 = sin(rx) * y4 + cos(rx) * z4;

        //绕roll(z轴)旋转
        float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
        float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
        float z6 = z5 + tz;

        float x7 = cosImuRollStart * (x6 - imuShiftFromStartX) 
                 - sinImuRollStart * (y6 - imuShiftFromStartY);
        float y7 = sinImuRollStart * (x6 - imuShiftFromStartX) 
                 + cosImuRollStart * (y6 - imuShiftFromStartY);
        float z7 = z6 - imuShiftFromStartZ;

        float x8 = x7;
        float y8 = cosImuPitchStart * y7 - sinImuPitchStart * z7;
        float z8 = sinImuPitchStart * y7 + cosImuPitchStart * z7;

        float x9 = cosImuYawStart * x8 + sinImuYawStart * z8;
        float y9 = y8;
        float z9 = -sinImuYawStart * x8 + cosImuYawStart * z8;

        float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
        float y10 = y9;
        float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

        float x11 = x10;
        float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
        float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

        po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
        po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
        po->z = z11;
        po->intensity = int(pi->intensity);
    }

    void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                           float alx, float aly, float alz, float &acx, float &acy, float &acz)
    {
        float sbcx = sin(bcx);
        float cbcx = cos(bcx);
        float sbcy = sin(bcy);
        float cbcy = cos(bcy);
        float sbcz = sin(bcz);
        float cbcz = cos(bcz);

        float sblx = sin(blx);
        float cblx = cos(blx);
        float sbly = sin(bly);
        float cbly = cos(bly);
        float sblz = sin(blz);
        float cblz = cos(blz);

        float salx = sin(alx);
        float calx = cos(alx);
        float saly = sin(aly);
        float caly = cos(aly);
        float salz = sin(alz);
        float calz = cos(alz);

        float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
                  - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                  - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                  - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
        acx = -asin(srx);

        float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
        
        float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
                     - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
                     - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
                     + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
                     - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
                     + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
                     + calx*cblx*salz*sblz);
        float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
                     - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
                     + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
                     + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
                     - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
                     - calx*calz*cblx*sblz);
        acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
    }

    //角度累积
    void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                            float &ox, float &oy, float &oz)
    {
        /*
            (1) Xi = R_cur * Xi-1 + T_cur 前后两帧之间的位姿关系
            (2) X0 = R_sum * Xi + T_sum 当前帧与地图坐标系之间的关系

            当(1)式中i=1时，(1)式变为 X1 = R_cur * X0 + T_cur
            当(2)式中i=1时，(2)式变为 X0 = R_sum * X1 + T_sum
            由此推算可得：
                X1 = R_cur * X0 + T_cur ---> X0 = R_cur^-1 * (X1 - T_cur)
                                                = R_cur^-1 * X1 + R_cur^-1 * (-T_cur)
                R_sum = R_cur^-1 = (R_roll * R_pitch * R_yaw)^-1
                      = R_(-yaw) * R_(-pitch) * R_(-roll)
                      = R_(yaw_sum) * R_(pitch_sum) * R_(yaw_sum)

                T_sum = R_cur^-1 * (-T_cur) = R_sum * (-T_cur)

                当i>1时，X0 = R_sum * Xi-1 + T_sum
                            = R_sum * (R_cur^-1 * (Xi - T_cur)) + T_sum
                            = R_sum * R_cur^-1 * Xi + T_sum - R_sum * R_cur^-1 * T_cur
                
                所以更新 R_sum = R_sum * R_cur^-1
                               = R_(yaw_sum) * R_(pitch_sum) * R_(roll_sum) * R_(-yaw_cur) * R_(-pitch_cur) * R_(-roll_cur) 

                cx,cy,cz = pitch_sum，yaw_sum，roll_sum
                lx,ly,lz = -pitch_cur, -yaw_cur, -roll_cur
                ox,oy,oz = cx,cy,cz

                R = R_yaw * R_pitch * R_roll
                  = R_ry * R_rx * R_rz
                    |crycrz+srxsrysrz  srxsrycrz-crysrz  crxsry|
                    |crxsrz  crxcrz  -srx|
                    |srxcrysrz-srycrz  srxcrycrz+srysrz  crxcry|
        */
        float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
        ox = -asin(srx);

        float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
                     + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
        float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
                     - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
        oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

        float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
                     + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
        float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
                     - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
        oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
    }

    //弧度转角度
    double rad2deg(double radians)
    {
        return radians * 180.0 / M_PI;
    }

    //角度转弧度
    double deg2rad(double degrees)
    {
        return degrees * M_PI / 180.0;
    }

    //边缘点匹配
    void findCorrespondingCornerFeatures(int iterCount){

        int cornerPointsSharpNum = cornerPointsSharp->points.size();

        for (int i = 0; i < cornerPointsSharpNum; i++) {

            //第i个点转到第一个点坐标系下
            TransformToStart(&cornerPointsSharp->points[i], &pointSel);

            if (iterCount % 5 == 0) {

                //kdtree查找最近点
                kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1;
                
                //找到最近点,nearestFeatureSearchSqDist=25
                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity); //最近点线号

                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {
                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);

                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {
                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);

                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                }

                pointSearchCornerInd1[i] = closestPointInd;
                pointSearchCornerInd2[i] = minPointInd2;
            }

            if (pointSearchCornerInd2[i] >= 0) {

                tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
                tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                float x1 = tripod1.x;
                float y1 = tripod1.y;
                float z1 = tripod1.z;
                float x2 = tripod2.x;
                float y2 = tripod2.y;
                float z2 = tripod2.z;

                /*
                    点线距计算
                        |(pointSel - tripod1) * (pointSel - tripod2)|
                    d = -------------------------------------------
                                 |tripod1 - tripod2|

                    (m11,m22,m33) = (pointSel - tripod1) * (pointSel - tripod2)
                    a012 = |(m11,m22,m33)|
                    l12 = (tripod1 - tripod2)
                    (la,lb,lc) 三个轴的距离分量
                    ld2 点线距
                */ 

                float m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1));
                float m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1));
                float m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1));

                float a012 = sqrt(m11 * m11  + m22 * m22 + m33 * m33);

                float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                float la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12;

                float lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12;

                float lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12;

                float ld2 = a012 / l12;

                //计算权重
                float s = 1;
                if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(ld2);
                }

                if (s > 0.1 && ld2 != 0) {
                    coeff.x = s * la; 
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;
                  
                    laserCloudOri->push_back(cornerPointsSharp->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    //平面点匹配
    void findCorrespondingSurfFeatures(int iterCount){

        int surfPointsFlatNum = surfPointsFlat->points.size();

        for (int i = 0; i < surfPointsFlatNum; i++) {

            TransformToStart(&surfPointsFlat->points[i], &pointSel);

            if (iterCount % 5 == 0) {

                kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;

                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);

                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist, minPointSqDis3 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
                        if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                              minPointSqDis2 = pointSqDis;
                              minPointInd2 = j;
                            }
                        } else {
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {
                        if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        } else {
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                }

                pointSearchSurfInd1[i] = closestPointInd;
                pointSearchSurfInd2[i] = minPointInd2;
                pointSearchSurfInd3[i] = minPointInd3;
            }

            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {

                tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
                tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
                tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

                /*
                    点面距计算
                    平面法线(pa, pb, pc) = (tripod2 - tripod1) * (tripod3 - tripod1)
                    平面参数方程 pa * X + pb * Y + pc * Z + pd = 0
                    点面距：pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd
                    
                */

                float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                         - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
                float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                         - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
                float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                         - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
                float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

                float ps = sqrt(pa * pa + pb * pb + pc * pc);

                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                //计算权重
                float s = 1;
                if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
                }

                if (s > 0.1 && pd2 != 0) {
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    laserCloudOri->push_back(surfPointsFlat->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    //利用平面点计算
    bool calculateTransformationSurf(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        /*
            Xi = R * Xi-1 + T  ---> Xi-1 = R^-1 * (Xi - T)
            R = R_roll * R_pitch * R_yaw
              = R_rz * R_rx * R_ry
              = |crycrz-srxsrysrz  -crxsrz  srycrz+srxcrysrz|
                |crysrz+srxsrycrz  crxcrz  srysrz-srxcrycrz|
                |-crxsry  srx  crxcry|

            R^-1 = |crycrz-srxsrysrz  crysrz+srxsrycrz  -crxsry|
                   |-crxsrz  crxcrz  srx|
                   |srycrz+srxcrysrz  srysrz-srxcrycrz  crxcry|

            平面点用于计算roll,pitch,tz
            error = R^-1 * (point - T) * coeff

            derror/droll = derror/drz = |-crysrz-srxsrycrz  crycrz-srxsrysrz  0|
                                        |-crxcrz  -crxsrz  0|                    * (point - T) * coeff      
                                        |srxcrycrz-srysrz  srycrz+srxcrysrz  0|

            derror/dpitch = derror/drx = |-crxsrysrz  crxsrycrz  srxsry|
                                         |srxsrz  -srxcrz  crx|           * (point - T) * coeff
                                         |crxcrysrz  -crxcrycrz  -srxcry|

            derror/dtz = derror/dy = |crysrz+srxsrycrz|
                                     |crxcrz|           * -coeff
                                     |srysrz-srxcrycrz|

        */

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = arz;
            matA.at<float>(i, 2) = aty;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 3; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[2] += matX.at<float>(1, 0);
        transformCur[4] += matX.at<float>(2, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    //利用边缘点计算
    bool calculateTransformationCorner(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        /*

            Xi = R * Xi-1 + T  ----> Xi-1 = R^-1 * (Xi - T)

            R = |crycrz-srxsrysrz  -crxsrz  srycrz+srxcrysrz|
                |crysrz+srxsrycrz  crxcrz  srysrz-srxcrycrz|
                |-crxsry  srx  crxcry|

            R^-1 = |crycrz-srxsrysrz  crysrz+srxsrycrz  -crxsry|
                   |-crxsrz  crxcrz  srx|
                   |srycrz+srxcrysrz  srysrz-srxcrycrz  crxcry|

            边缘点用于计算yaw,tx,ty
            error = R^-1 * (point - T) * coeff

            derror/dyaw = derror/dry = |-srycrz-srxcrysrz  srxcrycrz-srysrz  -crxcry|
                                       |0 0 0|                                        * (point - T) * coeff
                                       |crycrz-srxsrysrz  crysrz+srxsrycrz  -crxsry|

            derror/dtx = derror/dz = |-crxsry|
                                     |srx|     * -coeff
                                     |crxcry|

            derror/dty = derror/dx = |crycrz-srxsrysrz|
                                     |-crxsrz|          * -coeff
                                     |srycrz+srxcrysrz|

        */

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c5 = crx*srz;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = ary;
            matA.at<float>(i, 1) = atx;
            matA.at<float>(i, 2) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 3; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[1] += matX.at<float>(0, 0);
        transformCur[3] += matX.at<float>(1, 0);
        transformCur[5] += matX.at<float>(2, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(1, 0) * 100, 2) +
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    bool calculateTransformation(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {10, 10, 10, 10, 10, 10};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[1] += matX.at<float>(1, 0);
        transformCur[2] += matX.at<float>(2, 0);
        transformCur[3] += matX.at<float>(3, 0);
        transformCur[4] += matX.at<float>(4, 0);
        transformCur[5] += matX.at<float>(5, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    //系统初始化
    void checkSystemInitialization(){

        //laserCloudCornerLast保存上一帧的次边缘点
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        //laserCloudSurfLast保存上一帧的次平面点
        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        //构造kdtree
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

        //上一帧次边缘点和次平面点个数
        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        transformSum[0] += imuPitchStart;
        transformSum[2] += imuRollStart;

        systemInitedLM = true;
    }

    //初始位姿估计
    void updateInitialGuess(){

        imuPitchLast = imuPitchCur;
        imuYawLast = imuYawCur;
        imuRollLast = imuRollCur;

        imuShiftFromStartX = imuShiftFromStartXCur;
        imuShiftFromStartY = imuShiftFromStartYCur;
        imuShiftFromStartZ = imuShiftFromStartZCur;

        imuVeloFromStartX = imuVeloFromStartXCur;
        imuVeloFromStartY = imuVeloFromStartYCur;
        imuVeloFromStartZ = imuVeloFromStartZCur;

        if (imuAngularFromStartX != 0 || imuAngularFromStartY != 0 || imuAngularFromStartZ != 0){
            transformCur[0] = - imuAngularFromStartY;
            transformCur[1] = - imuAngularFromStartZ;
            transformCur[2] = - imuAngularFromStartX;
        }
        
        if (imuVeloFromStartX != 0 || imuVeloFromStartY != 0 || imuVeloFromStartZ != 0){
            transformCur[3] -= imuVeloFromStartX * scanPeriod;
            transformCur[4] -= imuVeloFromStartY * scanPeriod;
            transformCur[5] -= imuVeloFromStartZ * scanPeriod;
        }
    }

    //优化
    void updateTransformation(){

        //点太少
        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)
            return;

        //迭代25次
        for (int iterCount1 = 0; iterCount1 < 25; iterCount1++) {
            laserCloudOri->clear();
            coeffSel->clear();

            findCorrespondingSurfFeatures(iterCount1);

            if (laserCloudOri->points.size() < 10)
                continue;
            if (calculateTransformationSurf(iterCount1) == false)
                break;
        }

        //迭代25次
        for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {

            laserCloudOri->clear();
            coeffSel->clear();

            findCorrespondingCornerFeatures(iterCount2);

            if (laserCloudOri->points.size() < 10)
                continue;
            if (calculateTransformationCorner(iterCount2) == false)
                break;
        }
    }

    //计算odometry
    void integrateTransformation(){
        //角度计算
        float rx, ry, rz, tx, ty, tz;
        AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
                           -transformCur[0], -transformCur[1], -transformCur[2], rx, ry, rz);
        
        // R_sum = R_sum * R
        // T_sum = T_sum - (R_ry * R_rx * R_rz * T)

        //roll旋转
        float x1 = cos(rz) * (transformCur[3] - imuShiftFromStartX) 
                 - sin(rz) * (transformCur[4] - imuShiftFromStartY);
        float y1 = sin(rz) * (transformCur[3] - imuShiftFromStartX) 
                 + cos(rz) * (transformCur[4] - imuShiftFromStartY);
        float z1 = transformCur[5] - imuShiftFromStartZ;

        //pitch旋转
        float x2 = x1;
        float y2 = cos(rx) * y1 - sin(rx) * z1;
        float z2 = sin(rx) * y1 + cos(rx) * z1;

        //yaw旋转
        tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
        ty = transformSum[4] - y2;
        tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

        PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart, 
                          imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

        transformSum[0] = rx;
        transformSum[1] = ry;
        transformSum[2] = rz;
        transformSum[3] = tx;
        transformSum[4] = ty;
        transformSum[5] = tz;
    }

    void publishOdometry(){
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[2], -transformSum[0], -transformSum[1]);

        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = -geoQuat.y;
        laserOdometry.pose.pose.orientation.y = -geoQuat.z;
        laserOdometry.pose.pose.orientation.z = geoQuat.x;
        laserOdometry.pose.pose.orientation.w = geoQuat.w;
        laserOdometry.pose.pose.position.x = transformSum[3];
        laserOdometry.pose.pose.position.y = transformSum[4];
        laserOdometry.pose.pose.position.z = transformSum[5];
        pubLaserOdometry.publish(laserOdometry);

        laserOdometryTrans.stamp_ = cloudHeader.stamp;
        laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        laserOdometryTrans.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
        tfBroadcaster.sendTransform(laserOdometryTrans);
    }

    void adjustOutlierCloud(){
        PointType point;
        int cloudSize = outlierCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            point.x = outlierCloud->points[i].y;
            point.y = outlierCloud->points[i].z;
            point.z = outlierCloud->points[i].x;
            point.intensity = outlierCloud->points[i].intensity;
            outlierCloud->points[i] = point;
        }
    }

    void publishCloudsLast(){

        updateImuRollPitchYawStartSinCos();

        int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        for (int i = 0; i < cornerPointsLessSharpNum; i++) {
            TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
        }


        int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        for (int i = 0; i < surfPointsLessFlatNum; i++) {
            TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
        }

        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        }

        frameCount++;

        //隔一帧发一次
        if (frameCount >= skipFrameNum + 1) {

            frameCount = 0;

            adjustOutlierCloud();
            sensor_msgs::PointCloud2 outlierCloudLast2;
            pcl::toROSMsg(*outlierCloud, outlierCloudLast2);
            outlierCloudLast2.header.stamp = cloudHeader.stamp;
            outlierCloudLast2.header.frame_id = "/camera";
            pubOutlierCloudLast.publish(outlierCloudLast2);

            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
            laserCloudCornerLast2.header.frame_id = "/camera";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
            laserCloudSurfLast2.header.frame_id = "/camera";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
        }
    }

    void runFeatureAssociation()
    {

        if (newSegmentedCloud && newSegmentedCloudInfo && newOutlierCloud &&
            std::abs(timeNewSegmentedCloudInfo - timeNewSegmentedCloud) < 0.05 &&
            std::abs(timeNewOutlierCloud - timeNewSegmentedCloud) < 0.05){

            newSegmentedCloud = false;
            newSegmentedCloudInfo = false;
            newOutlierCloud = false;
        }else{
            return;
        }

        adjustDistortion();

        calculateSmoothness();

        markOccludedPoints();

        extractFeatures();

        publishCloud();

        //初始化
        if (!systemInitedLM) {
            checkSystemInitialization();
            return;
        }

        updateInitialGuess();   //IMU位姿估计

        updateTransformation();

        integrateTransformation();

        publishOdometry();

        publishCloudsLast();   
    }
};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Feature Association Started.");

    FeatureAssociation FA;

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        FA.runFeatureAssociation();

        rate.sleep();
    }
    
    ros::spin();
    return 0;
}
