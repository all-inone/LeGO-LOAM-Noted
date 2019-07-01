# LeGO-LOAM-Noted



/* rz,rx,ry分别对应着标准右手坐标系中的roll,pitch,yaw角,通过查看createQuaternionMsgFromRollPitchYaw()的函数定义可以发现.
    * 当pitch和yaw角给负值后,四元数中的y和z会变成负值,x和w不受影响.由四元数定义可以知道,x,y,z是指旋转轴在三个轴上的投影,w影响
    * 旋转角度,所以由createQuaternionMsgFromRollPitchYaw()计算得到四元数后,其在一般右手坐标系中的x,y,z分量对应到该应用场景下
    * 的坐标系中,geoQuat.x对应实际坐标系下的z轴分量,geoQuat.y对应x轴分量,geoQuat.z对应实际的y轴分量,而由于rx和ry在计算四元数
    * 时给的是负值,所以geoQuat.y和geoQuat.z取负值,这样就等于没变
   */