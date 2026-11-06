"""启动 SAR 检测节点。

真机 (RealSense D435i)：
    ros2 launch victim_detector sar_detector.launch.py

仿真 (Gazebo D455)：
    ros2 launch victim_detector sar_detector.launch.py \\
        image_topic:=/d455/image

换成微调后的三类模型：
    ros2 launch victim_detector sar_detector.launch.py \\
        weights:=/opt/models/sar_3class.pt keep_classes:=""
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# 名称 -> (默认值, 类型, 说明)
# 类型必须显式写出：launch 传进来的都是字符串，
# 不声明 value_type 的话 int/float/bool 参数会类型不匹配而报错。
ARGS = {
    "image_topic":       ("/camera/camera/color/image_raw", str,   "输入图像话题"),
    "best_effort_qos":   ("true",       bool,  "realsense 必须 true；Gazebo 桥接用 false"),
    "queue_depth":       ("1",          int,   "1 = 只处理最新帧"),
    "weights":           ("yolov8s.pt", str,   "权重路径；离线部署须为容器内绝对路径"),
    "device":            ("auto",       str,   "auto | cpu | cuda:0"),
    "imgsz":             ("640",        int,   "推理分辨率"),
    "conf":              ("0.25",       float, "置信度阈值"),
    "iou":               ("0.45",       float, "NMS IoU"),
    "half":              ("false",      bool,  "FP16；Jetson 上建议 true"),
    "keep_classes":      ("0",          str,   "保留的类 id，逗号分隔；COCO 只要人 = 0，三类模型留空"),
    "class_names":       ("",           str,   "覆盖类名，逗号分隔；留空用模型自带"),
    "publish_annotated": ("true",       bool,  "是否发布标注图（上线建议关掉省带宽）"),
    "max_rate_hz":       ("0.0",        float, "推理限速；Orin 50W 建议 5~10"),
    "log_every_n":       ("30",         int,   "每 N 帧打一行日志"),
}


def generate_launch_description():
    decls = [DeclareLaunchArgument(name, default_value=default, description=doc)
             for name, (default, _t, doc) in ARGS.items()]

    params = {
        name: ParameterValue(LaunchConfiguration(name), value_type=typ)
        for name, (_d, typ, _doc) in ARGS.items()
    }

    node = Node(
        package="victim_detector",
        executable="detector_node",
        name="sar_detector",
        output="screen",
        emulate_tty=True,
        parameters=[params],
    )
    return LaunchDescription(decls + [node])
