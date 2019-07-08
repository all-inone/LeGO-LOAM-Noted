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
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.
#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class mapOptimization{

private:

    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;

    ros::NodeHandle nh;

    ros::Publisher pubLaserCloudSurround;  //发布局部的点云  可视化(rviz里面基本就是全局地图)
    ros::Publisher pubOdomAftMapped;  //发布map_odometry 2hz左右
    ros::Publisher pubKeyPoses;    //发布轨迹点.

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;  //发布最近关键帧上的点云??...

    ros::Subscriber subLaserCloudCornerLast;  //订阅最新的laser边缘点
    ros::Subscriber subLaserCloudSurfLast;    //订阅最新的laser平面点
    ros::Subscriber subOutlierCloudLast;       //订阅最新的laser外点
    ros::Subscriber subLaserOdometry;          //订阅laser发布的odometry
    ros::Subscriber subImu;                     //订阅imu

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;  //存储的是keyframe对应的边沿点点云构成的向量
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;     //存储的是keyframe对应的平面点云构成的向量
    vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;  //存储的是keyframe对应的外点点云构成的向量

    deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;  //保存关键帧时用到的
    deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;    //
    deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames; //
    int latestFrameID;  //保存最新帧的ID

    vector<int> surroundingExistingKeyPosesID;   //保存周围(50米内)的keyposeID
    deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;

    PointType previousRobotPosPoint;
    PointType currentRobotPosPoint;   //当前机器人位置点

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;    //机器人建图过程中的所有轨迹点
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;


    // 结尾有DS代表是downsize,进行过下采样
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;   //周围(50米内)机器人轨迹点
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;  //降采样

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;   //最新接收到的边沿点corner feature set from odomOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;     //最新接收到的平面点 surf feature set from odomOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // 降采样downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;   // 降采样downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;  //最新接收到的外点
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS; //降采样

    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast; //最新接收到的总的平面点  保存平面点和外点  ???????
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS;  //降采样downsampled corner featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;   //L-M优化原始点
    pcl::PointCloud<PointType>::Ptr coeffSel;        //挑选的对应系数..

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;  //地图中选择最新的50帧点云的边沿点,用于匹配
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;   //地图中选择最新的50帧的平面点和外点,用于匹配
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS; //降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;   //降采样

    //kd-tree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;


    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;  //邻近的历史关键帧的边沿点组成的点云  ?
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;  //降采样
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;  //回环检测的到的历史关键帧距离最近的帧前后25帧组成的平面点+边沿点点云
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS; //降采样

    pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;  //关键帧的边沿点云???
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;    //关键帧的平面点点云???
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;  //关键帧的平面点点云降采样???

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;  //全局地图的kd-tree
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;  //500米内的轨迹点
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS; //全局轨迹点降采样后
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;     //全局关键帧
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;  //降采样的全局关键帧的点云

    std::vector<int> pointSearchInd;   //搜索到的50米范围内的轨迹点索引
    std::vector<float> pointSearchSqDis;  //搜索到的50米范围内的轨迹点的欧式距离

    pcl::VoxelGrid<PointType> downSizeFilterCorner;   //边沿点降采样滤波器
    pcl::VoxelGrid<PointType> downSizeFilterSurf;     //平面点降采样滤波器
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;  //外点降采样滤波器
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;  // for histor key frames of loop closure
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;  // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;  // for global map visualization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;  // for global map visualization

    double timeLaserCloudCornerLast;  //接收到最新边缘点的时间戳
    double timeLaserCloudSurfLast;  //接收到最新平面点的时间戳
    double timeLaserOdometry;    //接收到的lase_odometry的时间戳
    double timeLaserCloudOutlierLast; //接收到最新外点的时间戳
    double timeLastGloalMapPublish;

    bool newLaserCloudCornerLast; //接收到最新边沿点标志
    bool newLaserCloudSurfLast;  //接收到最新平面点标志
    bool newLaserOdometry;   //接收到新的laser_odometry标志
    bool newLaserCloudOutlierLast; //接收到最新外点标志


    float transformLast[6];
    float transformSum[6];         //odometry计算得到的到世界坐标系下的转移矩阵
    float transformIncre[6];       //转移增量，只使用了后三个平移增量
    float transformTobeMapped[6];  //以起始位置为原点的世界坐标系下的转换矩阵（猜测与调整的对象）
    float transformBefMapped[6];   //存放mapping之前的Odometry计算的世界坐标系的转换矩阵（注：低频量，不一定与transformSum一样）
    float transformAftMapped[6];   //存放mapping之后的经过mapping微调之后的转换矩阵



    int imuPointerFront;
    int imuPointerLast; //imu最新数据位置

    double imuTime[imuQueLength];//imu数据时间戳队列
    float imuRoll[imuQueLength]; //接收到imu roll保存队列
    float imuPitch[imuQueLength]; //接收到imu pitch保存队列

    std::mutex mtx;

    double timeLastProcessing;   //进行mapping线程的最新时间

    PointType pointOri, pointSel, pointProj, coeff;

    cv::Mat matA0;
    cv::Mat matB0;
    cv::Mat matX0;

    cv::Mat matA1;  //正交矩阵
    cv::Mat matD1;  //特征值
    cv::Mat matV1;  //特征向量

    bool isDegenerate;   //是否出现退化标志
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum;     //从map中选取的最新50帧边沿点点云降采样后的数量
    int laserCloudSurfFromMapDSNum;       //从map中选取的最新50帧平面点点云降采样后的数量
    int laserCloudCornerLastDSNum;  //最新接收到的laser边沿点降采样的点数量
    int laserCloudSurfLastDSNum;    //最新接收到的laser平面点降采样的点数量
    int laserCloudOutlierLastDSNum; //最新接收到的laser外点降采样的点数量
    int laserCloudSurfTotalLastDSNum; //接收到的laser平面点和外点总降采样后的数量

    bool potentialLoopFlag;    //检测到闭环标志
    double timeSaveFirstCurrentScanForLoopClosure;  //记录当前帧检测到回环的时间戳
    int closestHistoryFrameID;
    int latestFrameIDLoopCloure;

    bool aLoopIsClosed;  //回环结束标志

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;                //将局部坐标系下的点转到世界坐标系下
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;    //将点云转换到全局坐标系下?

public:



    mapOptimization():
            nh("~")
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_to_init", 5);

        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu> (imuTopic, 50, &mapOptimization::imuHandler, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);

        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

        downSizeFilterHistoryKeyFrames.setLeafSize(0.4, 0.4, 0.4);
        downSizeFilterSurroundingKeyPoses.setLeafSize(1.0, 1.0, 1.0);

        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);

        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";

        aftMappedTrans.frame_id_ = "/camera_init";
        aftMappedTrans.child_frame_id_ = "/aft_mapped";

        allocateMemory();
    }

    void allocateMemory(){

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
        surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());


        nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
        globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i){
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        imuPointerFront = 0;
        imuPointerLast = -1;

        for (int i = 0; i < imuQueLength; ++i){
            imuTime[i] = 0;
            imuRoll[i] = 0;
            imuPitch[i] = 0;
        }

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat (5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat (3, 1, CV_32F, cv::Scalar::all(0));

        matA1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));
        matD1 = cv::Mat (1, 3, CV_32F, cv::Scalar::all(0));
        matV1 = cv::Mat (3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = cv::Mat (6, 6, CV_32F, cv::Scalar::all(0));

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }


    //基于匀速模型，根据上次微调的结果和odometry这次与上次计算的结果，猜测一个新的世界坐标系的转换矩阵transformTobeMapped
    //T_(tobe)=T_(wAft)*T_(wBef).inverse*T_(sum)
    //R=R(y)R(x)R(z)
    void transformAssociateToMap()
    {
        //通过Sum和bef进行运动的预测.叠加到aft上.猜测下一个T(tobemap)
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
                   - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
                   + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;
        //平移增量
        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
        transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = sin(transformSum[0]);
        float cbcx = cos(transformSum[0]);
        float sbcy = sin(transformSum[1]);
        float cbcy = cos(transformSum[1]);
        float sbcz = sin(transformSum[2]);
        float cbcz = cos(transformSum[2]);

        float sblx = sin(transformBefMapped[0]);
        float cblx = cos(transformBefMapped[0]);
        float sbly = sin(transformBefMapped[1]);
        float cbly = cos(transformBefMapped[1]);
        float sblz = sin(transformBefMapped[2]);
        float cblz = cos(transformBefMapped[2]);

        float salx = sin(transformAftMapped[0]);
        float calx = cos(transformAftMapped[0]);
        float saly = sin(transformAftMapped[1]);
        float caly = cos(transformAftMapped[1]);
        float salz = sin(transformAftMapped[2]);
        float calz = cos(transformAftMapped[2]);

        float srx = -sbcx*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz)
                    - cbcx*sbcy*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                                 - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                    - cbcx*cbcy*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                                 - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx*(cblx*cblz*(caly*salz - calz*salx*saly)
                             - cblx*sblz*(caly*calz + salx*saly*salz) + calx*saly*sblx)
                       - cbcx*cbcy*((caly*calz + salx*saly*salz)*(cblz*sbly - cbly*sblx*sblz)
                                    + (caly*salz - calz*salx*saly)*(sbly*sblz + cbly*cblz*sblx) - calx*cblx*cbly*saly)
                       + cbcx*sbcy*((caly*calz + salx*saly*salz)*(cbly*cblz + sblx*sbly*sblz)
                                    + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) + calx*cblx*saly*sbly);
        float crycrx = sbcx*(cblx*sblz*(calz*saly - caly*salx*salz)
                             - cblx*cblz*(saly*salz + caly*calz*salx) + calx*caly*sblx)
                       + cbcx*cbcy*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx)
                                    + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) + calx*caly*cblx*cbly)
                       - cbcx*sbcy*((saly*salz + caly*calz*salx)*(cbly*sblz - cblz*sblx*sbly)
                                    + (calz*saly - caly*salx*salz)*(cbly*cblz + sblx*sbly*sblz) - calx*caly*cblx*sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                crycrx / cos(transformTobeMapped[0]));

        float srzcrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                                                     - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                       - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                                                       - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                       + cbcx*sbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        float crzcrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*calz*(cbly*sblz - cblz*sblx*sbly)
                                                     - calx*salz*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sbly)
                       - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*salz*(cblz*sbly - cbly*sblx*sblz)
                                                       - calx*calz*(sbly*sblz + cbly*cblz*sblx) + cblx*cbly*salx)
                       + cbcx*cbcz*(salx*sblx + calx*cblx*salz*sblz + calx*calz*cblx*cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
                crzcrx / cos(transformTobeMapped[0]));

        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        //由于刚刚预测的位移增加为bef减去sum(sum当前的lidar_odometry估计的位移量,bef为上一次保存的lidar_odometry位移量,上一次减当前),所以叠加到aft上时应该是减去该增加
        transformTobeMapped[3] = transformAftMapped[3]
                                 - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5]
                                 - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
    }
    //记录odometry发送的转换矩阵与mapping之后的转换矩阵，下一帧点云会使用(有IMU的话会使用IMU进行补偿)
    void transformUpdate()
    {
        //此时transformTobeMapped已经经过LM优化过.
        if (imuPointerLast >= 0) {
            float imuRollLast = 0, imuPitchLast = 0;
            //寻找是否有点云的时间戳小于IMU的时间戳的IMU位置:imuPointerFront
            while (imuPointerFront != imuPointerLast) {
                if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront]) {
                    break;
                }
                imuPointerFront = (imuPointerFront + 1) % imuQueLength;
            }
            //没找到,此时imuPointerFront==imtPointerLast,只能以当前收到的最新的IMU的欧拉角使用
            if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront]) {
                imuRollLast = imuRoll[imuPointerFront];
                imuPitchLast = imuPitch[imuPointerFront];
            } else {
                // 在imu数据充足的情况下可以进行插补
                int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack])
                                   / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod)
                                  / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

                imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
            }
            //imu稍微补偿俯仰角和翻滚角
            transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;  //??更新比例这么小.是认为在这两个方向上没有太大的变化.
            transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
        }

        for (int i = 0; i < 6; i++) {
            transformBefMapped[i] = transformSum[i]; //将当前保存的Lidar_odometry(存在sum中),赋值给BefMapped,
            transformAftMapped[i] = transformTobeMapped[i];  //将优化后并且经过IMU微调后的transform保存到Aft中.用于下次进行预测.
        }
    }

    void updatePointAssociateToMapSinCos(){
        cRoll = cos(transformTobeMapped[0]);
        sRoll = sin(transformTobeMapped[0]);

        cPitch = cos(transformTobeMapped[1]);
        sPitch = sin(transformTobeMapped[1]);

        cYaw = cos(transformTobeMapped[2]);
        sYaw = sin(transformTobeMapped[2]);

        tX = transformTobeMapped[3];
        tY = transformTobeMapped[4];
        tZ = transformTobeMapped[5];
    }
    //根据调整计算后的转移矩阵，将点注册到全局世界坐标系下
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        // 主要进行坐标变换，将局部坐标转换到全局坐标中去
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    //将点云转换到世界坐标系下的Transform
    void updateTransformPointCloudSinCos(PointTypePose *tIn){

        ctRoll = cos(tIn->roll);
        stRoll = sin(tIn->roll);

        ctPitch = cos(tIn->pitch);
        stPitch = sin(tIn->pitch);

        ctYaw = cos(tIn->yaw);
        stYaw = sin(tIn->yaw);

        tInX = tIn->x;
        tInY = tIn->y;
        tInZ = tIn->z;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn){

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);
        //将cloudIn的所有点变换到全局坐标系  ???
        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll* z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){

        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        // 坐标系变换，旋转rpy角
        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw)* pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll)* z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }
        return cloudOut;
    }

    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudOutlierLast = msg->header.stamp.toSec();   //记录接收外点点云时间戳
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
        newLaserCloudOutlierLast = true;  //设置接收标志位
    }

    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
    }

    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr& msg){
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){
        timeLaserOdometry = laserOdometry->header.stamp.toSec();   //记录时间戳
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;  //接收到的数据z朝前,x朝左,y朝上坐标系
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);//不对调.算的roll就是朝左(x)
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;    //接收到标志位
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn){
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength; //移动到下一个位置
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
    }

    /* rz,rx,ry分别对应着标准右手坐标系中的roll,pitch,yaw角,通过查看createQuaternionMsgFromRollPitchYaw()的函数定义可以发现.
        * 当pitch和yaw角给负值后,四元数中的y和z会变成负值,x和w不受影响.由四元数定义可以知道,x,y,z是指旋转轴在三个轴上的投影,w影响
        * 旋转角度,所以由createQuaternionMsgFromRollPitchYaw()计算得到四元数后,其在一般右手坐标系中的x,y,z分量对应到该应用场景下
        * 的坐标系中,geoQuat.x对应实际坐标系下的z轴分量,geoQuat.y对应x轴分量,geoQuat.z对应实际的y轴分量,而由于rx和ry在计算四元数
        * 时给的是负值,所以geoQuat.y和geoQuat.z取负值,这样就等于没变
        */
    void publishTF(){


        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                (transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }
    }

    void visualizeGlobalMapThread(){
        ros::Rate rate(0.2);  //0.2HZ
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }
    }

    void publishGlobalMap(){

        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)  //轨迹点
            return;

        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;

        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 通过KDTree进行最近邻搜索
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);//可视化范围500米
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        // 对globalMapKeyPoses进行下采样
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i){
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],   &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // 对globalMapKeyFrames进行下采样
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(cloudMsgTemp);

        globalMapKeyPoses->clear();
        globalMapKeyPosesDS->clear();
        globalMapKeyFrames->clear();
        globalMapKeyFramesDS->clear();
    }

    void loopClosureThread(){

        if (loopClosureEnableFlag == false)  //检测是否开启回环
            return;

        ros::Rate rate(1);  //1HZ
        while (ros::ok()){
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure(){

        latestSurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloudDS->clear();

        // 资源分配时初始化
        // 在互斥量被析构前不解锁
        std::lock_guard<std::mutex> lock(mtx);

        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
        // 进行半径historyKeyframeSearchRadius内的邻域搜索，
        // currentRobotPosPoint：需要查询的点，
        // pointSearchIndLoop：搜索完的邻域点对应的索引
        // pointSearchSqDisLoop：搜索完的每个领域点点与传讯点之间的欧式距离
        // 0：返回的领域个数，为0表示返回全部的邻域点
        kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        closestHistoryFrameID = -1;
        for (int i = 0; i < pointSearchIndLoop.size(); ++i){
            int id = pointSearchIndLoop[i];
            // 两个时间差值大于30秒
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0){
                closestHistoryFrameID = id;
                break;
            }
        }
        if (closestHistoryFrameID == -1){
            // 找到的点和当前时间上没有超过30秒的
            return false;
        }

        latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        // 点云的xyz坐标进行坐标系变换(分别绕xyz轴旋转)
        *latestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        *latestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

        // 滤掉latestSurfKeyFrameCloud中距离小于0的点???距离值会小于0?
        pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
        int cloudSize = latestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
            // 距离大于0的点放进hahaCloud队列
            if ((int)latestSurfKeyFrameCloud->points[i].intensity >= 0){
                hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
            }
        }
        latestSurfKeyFrameCloud->clear();
        *latestSurfKeyFrameCloud   = *hahaCloud;

        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j){
            if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            // 要求closestHistoryFrameID + j在0到cloudKeyPoses3D->points.size()-1之间,不能超过索引
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID+j],   &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
        }

        // 下采样滤波减少数据量
        downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);  //保存历史关键帧中距离当前帧距离最近帧并且时间大于30秒的.周围前后25帧的点云数据,用于进行回环匹配
        downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);

        if (pubHistoryKeyFrames.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubHistoryKeyFrames.publish(cloudMsgTemp);
        }

        return true;
    }


    void performLoopClosure(){

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        if (potentialLoopFlag == false){

            if (detectLoopClosure() == true){   //如果检测到闭环
                potentialLoopFlag = true;   //设置标志位
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;  //将当前laser_odometry时间戳赋值给回环检测到的时间
            }
            if (potentialLoopFlag == false)   //如果没检测到回环,退出
                return;
        }

        potentialLoopFlag = false;  //重置标志

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);    //最大迭代次数
        icp.setTransformationEpsilon(1e-6); //设置收敛判断条件，越小精度越大，收敛也越慢 (前一个变换矩阵和当前变换矩阵的差异小于阈值时，就认为已经收敛)
        icp.setEuclideanFitnessEpsilon(1e-6); //还有一条收敛条件是均方误差和小于阈值， 停止迭代。
        // 设置RANSAC运行次数
        icp.setRANSACIterations(0);

        icp.setInputSource(latestSurfKeyFrameCloud);
        // 使用detectLoopClosure()函数中下采样刚刚更新nearHistorySurfKeyFrameCloudDS
        icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // 进行icp点云对齐
        icp.align(*unused_result);

        // 为什么匹配分数高直接返回???分数高代表噪声太多  或者如果没收敛
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // 以下在点云icp收敛并且噪声量在一定范围内上进行
        if (pubIcpKeyFrames.getNumSubscribers() != 0){  //如果被订阅
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            // icp.getFinalTransformation()的返回值是Eigen::Matrix<Scalar, 4, 4>
            pcl::transformPointCloud (*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubIcpKeyFrames.publish(cloudMsgTemp);
        }

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        correctionCameraFrame = icp.getFinalTransformation();  //获取icp结果
        // 得到平移和旋转的角度
        pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
        Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch);
        Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        std::lock_guard<std::mutex> lock(mtx);
        gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
        isam->update(gtSAMgraph);
        isam->update();
        gtSAMgraph.resize(0);

        aLoopIsClosed = true; //回环结束
    }

    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint){
        return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                Point3(double(thisPoint.z),   double(thisPoint.x),    double(thisPoint.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint){
        return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

    void extractSurroundingKeyFrames(){

        if (cloudKeyPoses3D->points.empty() == true)  //则还没开始运行
            return;

        // loopClosureEnableFlag 这个变量另外只在loopthread这部分中有用到
        if (loopClosureEnableFlag == true){

            // recentCornerCloudKeyFrames保存的点云数量太少，则清空后重新塞入新的点云直至数量够
            if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum){   //数量小于50帧点云
                recentCornerCloudKeyFrames. clear();  //清空
                recentSurfCloudKeyFrames.   clear();
                recentOutlierCloudKeyFrames.clear();
                int numPoses = cloudKeyPoses3D->points.size(); //Poses3d为历史的lidar位置,只有位置,没有姿态.
                for (int i = numPoses-1; i >= 0; --i){
                    // cloudKeyPoses3D的intensity中存的是索引值
                    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];  //Poses6D保存的历史的lidar位姿.这个包含姿态信息.
                    updateTransformPointCloudSinCos(&thisTransformation);
                    // 依据上面得到的变换thisTransformation，对cornerCloudKeyFrames，surfCloudKeyFrames，surfCloudKeyFrames
                    // 进行坐标变换,后加入到用于回环检测的点云中
                    recentCornerCloudKeyFrames. push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd])); //从
                    recentSurfCloudKeyFrames.   push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                    if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum)  //保存最近的50帧关键帧
                        break;
                }


            }else{
                // recentCornerCloudKeyFrames中点云保存的数量较多
                // pop队列最前端的一个，再push后面一个
                if (latestFrameID != cloudKeyPoses3D->points.size() - 1){

                    recentCornerCloudKeyFrames. pop_front(); //弹出最早的那一帧点云.
                    recentSurfCloudKeyFrames.   pop_front();
                    recentOutlierCloudKeyFrames.pop_front();
                    // 为什么要把recentCornerCloudKeyFrames最前面第一个元素弹出?

                    latestFrameID = cloudKeyPoses3D->points.size() - 1;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    recentCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[latestFrameID])); //将最新的keypose对应的点云加入进去
                    recentSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[latestFrameID]));
                    recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
                }
            }

            for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i){
                // 两个pcl::PointXYZI相加?
                // 注意这里把recentOutlierCloudKeyFrames也加入到了laserCloudSurfFromMap
                *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *recentOutlierCloudKeyFrames[i];
            }
        }else{

            // 下面这部分是不进行闭环的代码
            //
            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();

            //cloudKeyPoses3D虽说是点云，但是是为了保存机器人在建图过程中的轨迹，其中的点就是定周期采样的轨迹点，这一点是在saveKeyFramesAndFactor中计算出的，在第一帧时必然是空的
            //surroundingKeyframeSearchRadius是50米，也就是说是在当前位置进行半径查找，得到附近的轨迹点
            //距离数据保存在pointSearchSqDis中

            kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D);

            // 进行半径surroundingKeyframeSearchRadius内的邻域搜索，
            // currentRobotPosPoint：需要查询的点，
            // pointSearchInd：搜索完的邻域点对应的索引
            // pointSearchSqDis：搜索完的每个领域点点与传讯点之间的欧式距离
            // 0：返回的邻域个数，为0表示返回全部的邻域点
            kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);
            for (int i = 0; i < pointSearchInd.size(); ++i)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);
            //对附近轨迹点的点云进行降采样，轨迹具有一定间隔
            downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
            downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i){
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j){
                    // 双重循环，不断对比surroundingExistingKeyPosesID和surroundingKeyPosesDS中点的index
                    // 如果能够找到一样，说明存在相同的关键帧
                    //例如surroundingExistingKeyPosesID保存的事1-100位姿的ID   ,通过刚才kd-tres搜索,搜索半径委50.如果只搜索到70-120的ID.(101-120还未添加ExistId中去)
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity){
                        existingFlag = true;
                        break;
                    }
                }

                if (existingFlag == false){
                    // 如果surroundingExistingKeyPosesID[i]对比了一轮的已经存在的关键位姿后
                    // 没有找到关键点，那么把这个点从当前队列中删除
                    // 否则的话，existingFlag为true，该关键点就将它留在队列中
                    // 例如刚才的例子.则会提出1-69位姿ID.
                    surroundingExistingKeyPosesID.   erase(surroundingExistingKeyPosesID.   begin() + i);
                    surroundingCornerCloudKeyFrames. erase(surroundingCornerCloudKeyFrames. begin() + i);
                    surroundingSurfCloudKeyFrames.   erase(surroundingSurfCloudKeyFrames.   begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }

            for (int i = 0; i < numSurroundingPosesDS; ++i) {
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter){
                    // *iter是int数值，是intensity的整数部分
                    // 这部分比较有技巧，这里把surroundingExistingKeyPosesID内没有对应的点放进一个队列里
                    // 这个队列专门存放周围存在的关键帧，但是和surroundingExistingKeyPosesID的点不在同一行
                    // 关于行，需要参考intensity数据的存放格式，整数部分和小数部分代表不同意义
                    //例如刚才的例子.会添加101-120位姿的ID.现在ExistingKeyPosesID保存的则是70-120ID
                    if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity){
                        existingFlag = true;
                        break;
                    }
                }
                if (existingFlag == true){
                    continue;
                }else{
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.   push_back(thisKeyInd);
                    surroundingCornerCloudKeyFrames. push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));  //保存周围(50米内)的关键帧的边沿点,平面点.外点点云.
                    surroundingSurfCloudKeyFrames.   push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }
            //累加点云
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i) {
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap   += *surroundingOutlierCloudKeyFrames[i];
            }
        }

        // 进行两次下采样
        // 最后的输出结果是laserCloudCornerFromMapDS和laserCloudSurfFromMapDS
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();

        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }

    void downsampleCurrentScan(){

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();   //降采样后的点数量

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size(); //降采样后的点数量

        laserCloudOutlierLastDS->clear();
        downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
        downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
        laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size(); //降采样后的点数量

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        *laserCloudSurfTotalLast += *laserCloudSurfLastDS;       //保存降采样的平面点和外点
        *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
        downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
    }

    void cornerOptimization(int iterCount){

        updatePointAssociateToMapSinCos();  //先更新预测的T的sin值和cos值.方便使用.
        for (int i = 0; i < laserCloudCornerLastDSNum; i++) {
            pointOri = laserCloudCornerLastDS->points[i];
            // 进行坐标变换,转换到全局坐标中去
            // pointSel:表示选中的点，point select
            pointAssociateToMap(&pointOri, &pointSel);

            // 进行5邻域搜索，
            // pointSel为需要搜索的点，
            // pointSearchInd搜索完的邻域对应的索引
            // pointSearchSqDis 邻域点与查询点之间的距离
            //利用kd树查找最近的5个点，接下来需要计算这五个点的协方差
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0) { //查到的最近的5个点距离超过1m,则不进行
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                //计算算术平均值
                cx /= 5; cy /= 5;  cz /= 5;

                // 下面在求矩阵matA1=[ax,ay,az]t*[ax,ay,az]
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                //计算协方差矩阵
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 求正交阵的特征值和特征向量
                // 特征值：matD1，特征向量：matV1中
                cv::eigen(matA1, matD1, matV1);
                //与里程计的计算类似，计算到直线的距离
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);  //在 5个的中心值处沿着线的方向前后选择0.1米远的点.作为直线上的两个点.
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 因为[(x0-x1),(y0-y1),(z0-z1)]x[(x0-x2),(y0-y2),(z0-z2)]=[XXX,YYY,ZZZ],
                    // [XXX,YYY,ZZZ]=[(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                      * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                      + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                        * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                      + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                                        * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // l12表示的是12点的长度
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                 - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                 + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    float s = 1 - 0.9 * fabs(ld2);

                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    void surfOptimization(int iterCount){
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {//跟Corneroptimization一样.将所有点转到全局地图中去
            pointOri = laserCloudSurfTotalLastDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                //将五个点的坐标值构成矩阵.通过求解Ax=0? 计算平面的法向量
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;
                //验证计算出的平面的有效性
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid) {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                                                              + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    bool LMOptimization(int iterCount){
        //TobeMapped为之前预测的Transform.作为LM优化的初始值
        float srx = sin(transformTobeMapped[0]);
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();
        if (laserCloudSelNum < 50) {    //如果进行配准的点的个数小于50个.则不优化
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        for (int i = 0; i < laserCloudSelNum; i++) {
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];
            //距离求导公式.可以看之前paperReading推到的公式.一模一样.
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                        + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                        + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                         + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                        + ((-cry*crz - srx*sry*srz)*pointOri.x
                           + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                        + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                        + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;
            matB.at<float>(i, 0) = -coeff.intensity;
        }
        //避免A不正定,左乘A转置.
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        //求解matAtA * matX = matAtB
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            //特征值1*6矩阵
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            //特征向量6*6矩阵
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            //特征值取值门槛
            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) { //从小到大查找
                if (matE.at<float>(0, i) < eignThre[i]) { //特征值太小，则认为处在兼并环境中，发生了退化
                    for (int j = 0; j < 6; j++) { //对应的特征向量置为0
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            //计算P矩阵
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {//如果发生退化，只使用预测矩阵P计算
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }
        //积累每次的调整量
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                pow(matX.at<float>(3, 0) * 100, 2) +
                pow(matX.at<float>(4, 0) * 100, 2) +
                pow(matX.at<float>(5, 0) * 100, 2));
        //旋转平移量足够小就停止迭代
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true;
        }
        return false;
    }
    //根据现有地图与最新点云数据进行配准从而更新机器人精确位姿与融合建图，它分为角点优化、平面点优化、配准与更新等部分。
    void scan2MapOptimization(){

        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100) {

            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 10; iterCount++) { //最有的最大次数为10次.

                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount);    //边缘优化
                surfOptimization(iterCount);       //面优化

                if (LMOptimization(iterCount) == true)  //L-M优化
                    break;
            }
            //优化完后.进行更新.
            transformUpdate();
        }
    }


    void saveKeyFramesAndFactor(){
        //计算完当前帧在地图位置后.保存当前帧的pose和提取关键帧.及其点云.
        //此时的AftMapped中保存的是刚刚进过LM优化后的TransformtobeMapped.刚赋值给AftMapped.
        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)
                 +(previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)
                 +(previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z)) < 0.3){ //间隔0.3米保存keyframe
            saveThisKeyFrame = false;
        }



        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
            return;

        previousRobotPosPoint = currentRobotPosPoint;

        if (cloudKeyPoses3D->points.empty()){
            // static Rot3 	RzRyRx (double y, double x, double z),Rotations around Z, x, then y axes
            // RzRyRx依次按照z(transformTobeMapped[2])，x(transformTobeMapped[0])，y(transformTobeMapped[1])坐标轴旋转
            // Point3 (double x, double y, double z)  Construct from z(transformTobeMapped[5]), x(transformTobeMapped[3]), and y(transformTobeMapped[4]) coordinates.
            // Pose3 (const Rot3 &R, const Point3 &t) Construct from R,t. 从旋转和平移构造姿态
            // NonlinearFactorGraph增加一个PriorFactor因子
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                    Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                    Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
                transformLast[i] = transformTobeMapped[i];//保存第一针位姿
        }
        else{
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                    Point3(transformLast[5], transformLast[3], transformLast[4]));
            gtsam::Pose3 poseTo   = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                    Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));

            // 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size()-1, cloudKeyPoses3D->points.size()ses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                    Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
        }

        // gtsam::ISAM2::update函数原型:
        // ISAM2Result gtsam::ISAM2::update	(	const NonlinearFactorGraph & 	newFactors = NonlinearFactorGraph(),
        // const Values & 	newTheta = Values(),
        // const std::vector< size_t > & 	removeFactorIndices = std::vector<size_t>(),
        // const boost::optional< FastMap< Key, int > > & 	constrainedKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	noRelinKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	extraReelimKeys = boost::none,
        // bool 	force_relinearize = false )
        // gtSAMgraph是新加到系统中的因子
        // initialEstimate是加到系统中的新变量的初始点
        isam->update(gtSAMgraph, initialEstimate);
        // update 函数为什么需要调用两次？
        isam->update();

        // 删除内容?
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // Compute an estimate from the incomplete linear delta computed during the last update.
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.roll  = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw   = latestEstimate.rotation().roll();
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);

        if (cloudKeyPoses3D->points.size() > 1){
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i){ //此时transformAftMapped经过gtsam优化后,为最终的transform.保存到transformLast,用于下次的gtsam因子图边构建.
                transformLast[i] = transformAftMapped[i];
                transformTobeMapped[i] = transformAftMapped[i];//将gtsam调整后的transformAftMaped赋值给tobeMapped,在调用该函数前,AftMapped刚由tobeMap赋值.所以此时将调整后的值在赋值回去.
            }
        }

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);



        void correctPoses(){
            if (aLoopIsClosed == true){
                recentCornerCloudKeyFrames. clear();
                recentSurfCloudKeyFrames.   clear();
                recentOutlierCloudKeyFrames.clear();

                int numPoses = isamCurrentEstimate.size();
                for (int i = 0; i < numPoses; ++i)
                {
                    cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
                    cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
                    cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

                    cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                    cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                    cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                    //
                    cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                    cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                    cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                }

                aLoopIsClosed = false;
            }
        }

        void clearCloud(){
            laserCloudCornerFromMap->clear();
            laserCloudSurfFromMap->clear();
            laserCloudCornerFromMapDS->clear();
            laserCloudSurfFromMapDS->clear();
        }

        void run(){

            if (newLaserCloudCornerLast  && std::abs(timeLaserCloudCornerLast  - timeLaserOdometry) < 0.005 &&
                newLaserCloudSurfLast    && std::abs(timeLaserCloudSurfLast    - timeLaserOdometry) < 0.005 &&
                newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
                newLaserOdometry)  //判断接收到数据与间隔时间
            {

                newLaserCloudCornerLast = false; newLaserCloudSurfLast = false; newLaserCloudOutlierLast = false; newLaserOdometry = false;  //重置接收数据标志

                std::lock_guard<std::mutex> lock(mtx);

                if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval) {  //mappingProcessInterval=0.3 大于建图处理时间,则开始处理

                    timeLastProcessing = timeLaserOdometry;
                    //根据匀速模型,进行预测当前的transformtobemapped
                    transformAssociateToMap();
                    //由于帧数的频率大于建图的频率，因此需要提取关键帧进行匹配
                    extractSurroundingKeyFrames();
                    //降采样匹配以及增加地图点云，回环检测
                    downsampleCurrentScan();

                    scan2MapOptimization();

                    saveKeyFramesAndFactor();
                    //发送TF变换
                    correctPoses();

                    publishTF();

                    publishKeyPosesAndFrames();

                    clearCloud();
                }
            }
        }
    };


    int main(int argc, char** argv)
    {
        ros::init(argc, argv, "lego_loam");

        ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

        mapOptimization MO;

        // std::thread 构造函数，将MO作为参数传入构造的线程中使用
        // 进行闭环检测与闭环的功能
        std::thread loopthread(&mapOptimization::loopClosureThread, &MO);

        // 该线程中进行的工作是publishGlobalMap(),将数据发布到ros中，可视化
        std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

        ros::Rate rate(200);
        while (ros::ok())
        {
            ros::spinOnce();

            MO.run();

            rate.sleep();
        }

        loopthread.join();
        visualizeMapThread.join();

        return 0;
    }