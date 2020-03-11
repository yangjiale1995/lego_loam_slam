Header header           #消息头

#保存每根线开始下标和结束下标,大小为16
int32[] startRingIndex      #从每一线的第5个点开始计算
int32[] endRingIndex		#到每一线的倒数第5个点结束

float32 startOrientation        #起始角
float32 endOrientation          #结束角
float32 orientationDiff         #水平视场角


#segmentedCloud保存地面点和分割点云，其中地面点云做了下采样，每5个保存1个，segmentedCloudGroundFlag保存当前点是否为地面点
#segmentedCloudColInd保存点的列下标，intensity中整数部分保存了行下标，segmentedCloudRange保存了点的距离值
bool[]    segmentedCloudGroundFlag      #地面标志,大小为16*1800
uint32[]  segmentedCloudColInd          #大小为16*1800
float32[] segmentedCloudRange           #距离，大小为16*1800



