# mapOptimization

订阅的话题:
/laser_cloud_corner_last ----> laserCloudCornerLast ----> laserCloudCornerLastDS
/laser_cloud_surf_last ----> laserCloudSurfLast ----> laserCloudSurfLastDS
/outlier_cloud_last ----> laserCloudOutlierLast ----> laserCloudOutlierLastDS 
/laser_odom_to_init ----> transformSum(回调中将6DOF转成z朝前,x朝左,y朝上)存储最新的odom数据

### void transformAssociateToMap()
主要功能: 通过transformBefMapped(存储上一次的transformSum)-transformSum进行匀速运动模型预测,将增量叠加到transformAftMapped,最后存储在transformTobeMapped中.
1.首先将通过transformBefMapped的平移量-transformSum平移量.计算帧间位移.并且通过transformSum的旋转,将增量投影到transformSum所在的坐标系下.
2.对transformAftMapped旋转进行预测: R_tobe=R_w_aft*R_w_bef.inverse*R_w_sum
3.最后将平移增量通过transformTobeMapped旋转至世界坐标系下.并且减去该增量.得到最新的预测量(前面预测的增量为反向预测.)

###  void extractSurroundingKeyFrames()

主要为提取地图中一定范围内的特征点.用于后续的匹配.

如果有开启回环:recentCornerCloudKeyFrames,recentSurfCloudKeyFrames,recentOutlierCloudKeyFrames点云向量保存最新的50帧点云.并且最后将点云叠加到laserCloudCornerFromMap,laserCloudSurfFromMap,中(Surf保存平面点和外点.)

如果没开启回环:先在keypose3D中找到50米范围内的所有keypose3d点.然后进行降采样.得到关键keypose3d.最后将点云叠加到laserCloudCornerFromMap,laserCloudSurfFromMap,中(Surf保存平面点和外点.)

最后降采样得到后续用于匹配的地图点云.laserCloudSurfFromMapDS.laserCloudCornerFromMapDS

### void downsampleCurrentScan()

主要功能: 对订阅的话题点云进行降采样,将降采样后的平面点和外点叠加在一起后,再次降采样.



### void scan2MapOptimization()

主要功能:设置kdtreeCornerFromMap,kdtreeSurfFromMap地图边沿点和平面点kt-trees用于快速搜索.调用cornerOptimization().surfOptimization()进行相对应的特征点的提取.LMOptimization最后进行迭优化.完成更新后transformUpdate()进行数据的更新.

### void cornerOptimization(int iterCount)

主要功能:先通过transformTobeMapped将该帧点云投影到世界坐标系下.然后kd-trees中搜索,在地图中找到最近的5个点,求协方差.然后计算直线的方向.然后在直线上取2个点.计算该点到直线的距离.最后加上对应的权重.将原始点(该帧下的点)存入laserCloudOri.将带权重的距离存入到coeffSel.

### void surfOptimization(int iterCount)

主要功能:先通过transformTobeMapped将该帧点云投影到世界坐标系下.然后kd-trees中搜索,在地图中找到最近的5个点,求协方差.然后计算平面的法向量.验证有效性后.计算出点到平面的距离吼,最后加上对应的权重.将原始点(该帧下的点)存入laserCloudOri.将带权重的距离和法向量存入到coeffSel.

### bool LMOptimization(int iterCount)

主要功能:通过刚才的边沿点和平面的匹配.取出transformTobeMapped.然后进行LM方法求解.得到LM优化后的transformTobeMapped

### void transformUpdate()

主要功能:查询IMU的数据.进行pitch和roll进行插值.稍微更新transformTobeMapped.然后将transformSum赋值给transformBefMapped.用于下一次的运动预测.transformTobeMapped为优化后的位姿.赋值给transformAftMapped(可发布了)

### void saveKeyFramesAndFactor()

主要功能:transformAftMapped取出平移量作为当前机器人的位置.每3米保存keyframe到因子图中.通过因子图优化后.将最新估计的位姿放到cloudKeyPoses3D和cloudKeyPoses6D中.并更新transformAftMapped.最后将transformAftMapped赋值给transformLast和transformTobeMapped.并将关键帧的点云存到cornerCloudKeyFrames.surfCloudKeyFrames.outlierCloudKeyFrames

### void correctPoses()

主要功能: 通过刚刚gtsam的因子图优化后,对所有的cloudKeyPoses3D和cloudKeyPoses6D进行矫正,重新赋值.

### void publishTF()

主要功能: 将最终的transformAftMapped最终发布出去.这里面会将transformAftMapped[2],-[0],-[1]代表为正常坐标系下roll,-pitch,-yaw.然后将对应的赋值本文通用坐标系下.然后进行发布.(z朝前,对应roll,x朝左,对应pitch,y朝上,对应yaw).并将twist用于记录transformBefMapped.用于最后一个节点的线性插值.

### void performLoopClosure()

主要功能: 先检测是否有回环.如果检测到回环.通过ICP匹配或者该帧和提取的局部点云的的相对变换.来矫正原始的估计.最后加入到gtsam中.

### bool detectLoopClosure()

主要功能:进行回环检测,搜索的源是历史保留的所有cloudKeyPoses3D,然后搜索5米范围内的所有keypose,并且间隔大于30秒.并且确定历史距离当前最近的那一阵点云.并将前后25帧点云组合成一个局部地图(世界坐标系下.)