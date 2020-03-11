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

#include "utility.h"


// image projection 类
class ImageProjection{
private:

    ros::NodeHandle nh;                     //句柄

    ros::Subscriber subLaserCloud;          //原始点云订阅句柄
    
    ros::Publisher pubFullCloud;            //
    ros::Publisher pubFullInfoCloud;        //

    ros::Publisher pubGroundCloud;          //
    ros::Publisher pubSegmentedCloud;       //
    ros::Publisher pubSegmentedCloudPure;   //
    ros::Publisher pubSegmentedCloudInfo;   //
    ros::Publisher pubOutlierCloud;         //

  
    pcl::PointCloud<PointType>::Ptr laserCloudIn;   //原始点云

    //
    pcl::PointCloud<PointType>::Ptr fullCloud;      //完整点云  16*1800
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;  //intensity保存距离  16*1800

    //
    pcl::PointCloud<PointType>::Ptr groundCloud;        //地面点云
    pcl::PointCloud<PointType>::Ptr segmentedCloud;     //用于特征提取的点云(分簇的点+地面点(每5个保存一个))
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure; //不包含地面点和外点的剩余点
    pcl::PointCloud<PointType>::Ptr outlierCloud;       //外点

    //NAN点
    PointType nanPoint;

    cv::Mat rangeMat;       //距离矩阵  16*1800
    cv::Mat labelMat;       //分簇标志矩阵  16*1800
    cv::Mat groundMat;      //地面标志矩阵  16*1800
    int labelCount;         //分簇个数，初始值为1
    
    float startOrientation;     //点云开始角度
    float endOrientation;       //点云结束角度

    cloud_msgs::cloud_info segMsg;      //
    std_msgs::Header cloudHeader;       //点云消息头

    //上下左右邻居点
    std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

    //
    uint16_t *allPushedIndX;
    uint16_t *allPushedIndY;

    //
    uint16_t *queueIndX;
    uint16_t *queueIndY;

public:

    //构造函数
    ImageProjection():
        nh("~"){

        //订阅点云消息
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1, &ImageProjection::cloudHandler, this);

        // 16*1800点云
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        
        // 16*1800点云，其中intensity保存距离range
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        //发布地面点云
        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        
        //用于做特征提取的点
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        
        //除去地面点和外点后剩余的点
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        
        //点云信息解释
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        
        //外点
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        //初始化为NAN, intensity = -1
        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();       //分配内存
        resetParameters();      //
    }

    //初始化
    void allocateMemory(){

        //重置
        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        // 16*1800
        fullCloud.reset(new pcl::PointCloud<PointType>());
        
        // 16*1800
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        // 地面点
        groundCloud.reset(new pcl::PointCloud<PointType>());
        
        // 用于特征提取的点
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        
        // 除去地面点和外点后的剩余点
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        
        // 外点
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        //按照图片形式保存点云 raw image
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);      //16线(垂直16线)*1800个点(水平360度，角度分辨率为0.2)

        //按照图片形式保存点的距离在intensity中
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        //每根线的开始下标和结束下标
        segMsg.startRingIndex.assign(N_SCAN, 0);    
        segMsg.endRingIndex.assign(N_SCAN, 0);

        //地面标志矩阵
        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);     //地面标志位
        
        // segmentedCloud中每一个点的列下标
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        
        //保存每个点到原点的距离
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);      //距离

        //上下左右四个临近点
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        //
        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        //
        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    // 成员变量重置
    void resetParameters(){

        //清空
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        //距离矩阵
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));     //距离矩阵
        //地面标志矩阵
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        //分簇标志矩阵
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        //第一簇标志为1
        labelCount = 1;

        //用NAN填充fullCloud和fullInfoCloud
        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    //析构函数
    ~ImageProjection(){}

    //sensor_msgs --> pcl::PointCloud<pcl::PointXYZI>
    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;        //消息头保存
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);     // sensor_msgs --> pcl::PointCloud
    }
    
    //回调函数
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        copyPointCloud(laserCloudMsg);      //sensor_msgs --> pcl::PointCloud<pcl::PointXYZI>
        
        findStartEndAngle();                //计算起始角度和结束角度,视场角
        
        projectPointCloud();                //点云转成图像 pcl::PointCloud<pcl::PointXYZI> --> cv::Mat
        
        groundRemoval();                    //检测并剔除地面点
        
        cloudSegmentation();                //点云分割
        
        publishCloud();                     // 发布点云
        
        resetParameters();                  //重新初始化
    }

    //点云起始角度，结束角度，水平视场角
    void findStartEndAngle(){
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);         //开始角度
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 2].x) + 2 * M_PI;       //结束角度
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;       // 水平视场角
    }

    // image projection
    void projectPointCloud(){

        float verticalAngle, horizonAngle, range;       //水平角度，垂直角度，距离
        
        size_t rowIdn, columnIdn, index, cloudSize;     //行下标，列下标，下标，点云个数
        
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();        //点云个数

        for (size_t i = 0; i < cloudSize; ++i){
            
            //点
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            //计算该点所在的图像行号(0-16) ang_bottom = 15
            //计算行号
            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;      //ang_bottom = 15 ang_res_y = 2.0 计算线号
            
            //异常点
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            //计算列号
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;   //水平夹角
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;     // ang_res_x = 0.2 Horizon_SCAN = 1800
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            //异常点
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            //距离矩阵赋值
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);   //距离
            rangeMat.at<float>(rowIdn, columnIdn) = range;      //rangeMat保存每个点的距离

            //intensity整数部分保存线号，小数部分保存旋转角度
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;   //intensity整数部分表示行号，小数部分表示列号

            index = columnIdn  + rowIdn * Horizon_SCAN;     //计算点下标，图片中按照先行后列排序
            fullCloud->points[index] = thisPoint;           //按照图片形式保存点云

            fullInfoCloud->points[index].intensity = range;     //保存距离
        }
    }


    //地面剔除
    void groundRemoval(){

        size_t lowerInd, upperInd;          
        float diffX, diffY, diffZ, angle;   

        //按照列号遍历 Horizon_SCAN = 1800
        for (size_t j = 0; j < Horizon_SCAN; ++j)
        {
            //groundScanInd = 7 表示下面8根线可能为地面
            for (size_t i = 0; i < groundScanInd; ++i){

                lowerInd = j + ( i )*Horizon_SCAN;     //第i条线的第j个点 
                upperInd = j + (i+1)*Horizon_SCAN;     //第i+1条线的第j个点

                // groundMat -1表示无法判断，0初始化值，1表示地面点
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){
                    groundMat.at<int8_t>(i,j) = -1;     
                    continue;
                }
                    
                //相邻线的点之间的三轴距离差
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                //计算夹角
                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;    // ???

                //sensorMountAngle = 0.0 地面点
                //angle <= 10 表示为地面点, groundMat标志为1
                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }

        //labelMat = -1 //地面点或者不存在的点
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }

        //地面点保存在groundCloud中
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }


    //点云分割
    void cloudSegmentation(){

        //类似广度优先遍历分割
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);


        int sizeOfSegCloud = 0;     //每一个点的下标
        //按照线号遍历
        for (size_t i = 0; i < N_SCAN; ++i) {

            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;    //每一根线的开始下标

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                //labelMat 地面(-1) 初始值(0) 簇(>0) 表示所有存在的点
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    //999999表示outliers点
                    if (labelMat.at<int>(i,j) == 999999){
                        if (i > groundScanInd && j % 5 == 0){       //保存outliers,下采样，每5个保存1个
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);     //outlier点
                            continue;
                        }else{
                            continue;
                        }
                    }

                    // 对于地面点做采样，每5个保留一个
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }

                    //
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1); //地面点标志
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;    //保存列下标
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);   //保存地面点和分割点云
                    ++sizeOfSegCloud;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }

        // segmentedCloudPure保存分割后的点云，其中intensity保存分割号
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }


    //广度优先遍历分割
    void labelComponents(int row, int col){
        
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        //queueIndX.size = 16*1800   queueIndY = 16*1800
        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;      //queueIndX.size = 1
        int queueStartInd = 0;  //queueIndX start index
        int queueEndInd = 1;    //queueIndX end index

        //allPushIndX.size = 16*1800
        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;   //allPushedIndX.size = 1
        
        while(queueSize > 0){
            //queueIndX queueIndY 第一个点
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;        //queueIndX, queueIndY 第一个点已经被处理，size-1
            ++queueStartInd;    //第一个下标向前指
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;  //labelCount = 1;

            //上下左右
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){

                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                //垂直视角上限
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;

                //水平视角360度成环
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;

                if (labelMat.at<int>(thisIndX, thisIndY) != 0)  //已经成簇
                    continue;

                //距离
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                //距离
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                if ((*iter).first == 0)     //相邻两个点为左右关系
                    alpha = segmentAlphaX;      //弧度 segmentAlphaX = 0.2 * M_PI / 180.0
                else                        //相邻两个点为上下关系
                    alpha = segmentAlphaY;      //弧度 segmentAlphaY = 0.2 * M_PI / 180.0

                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));      //计算夹角

                //segmentTheta = 1.0472
                if (angle > segmentTheta){

                    //将临近点添加进queueIndX, queueIndY
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;  //同一簇的点
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        /*
         * 每簇点超过30个则保存，多于5个少于30个判断来自三个线的话保存
         * */
        //超过30个点
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        //segmentValidPointNum = 5
        else if (allPushedIndSize >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            //segmentValidLineNum = 3
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;            
        }

        //特征点标签+1
        if (feasibleSegment == true){
            ++labelCount;
        }
        //不是簇，扔掉
        else{
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    
    void publishCloud(){

        // ???
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);

        sensor_msgs::PointCloud2 laserCloudTemp;

        // outlier点
        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);

        // ???
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);

        //完整点云
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }

        //地面点
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }

        //分割后的点云,不包括地面点
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }

        //距离矩阵点云
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
