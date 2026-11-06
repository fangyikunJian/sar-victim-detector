#!/usr/bin/env python3
"""SAR 目标检测 ROS 2 节点 (Team Lumen Fleet)。

订阅相机图像 -> YOLO 推理 -> 发布检测结果。

所有行为通过 ROS 参数控制，换相机 / 换权重 / 换类别都不用改代码。
完整接口说明见 README_victim_detector.md。
"""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import (BoundingBox2D, Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)


def sensor_qos(best_effort: bool, depth: int) -> QoSProfile:
    """相机话题的 QoS 配置。

    realsense-ros 用 BEST_EFFORT 发布图像。订阅端若用默认的 RELIABLE，
    QoS 不兼容，会一帧都收不到、而且不报任何错 —— 真机上最常见的静默失败。
    """
    return QoSProfile(
        reliability=(ReliabilityPolicy.BEST_EFFORT if best_effort
                     else ReliabilityPolicy.RELIABLE),
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )


def _int_set(spec):
    """"0,2" -> {0, 2}；空串 -> 空集合（表示不过滤）。"""
    return {int(t) for t in str(spec).replace(" ", "").split(",") if t}


def _str_list(spec):
    """"a,b" -> ["a", "b"]；空串 -> []。"""
    return [t for t in (x.strip() for x in str(spec).split(",")) if t]


class SarDetector(Node):

    COLORS = {
        "mannequin": (0, 0, 255),
        "person": (0, 0, 255),
        "cone": (0, 165, 255),
        "first_aid": (0, 200, 0),
    }

    def __init__(self):
        super().__init__("sar_detector")

        p = self.declare_parameter
        # --- 输入 ---
        p("image_topic", "/camera/camera/color/image_raw")
        p("best_effort_qos", True)      # 真机 realsense 必须为 True
        p("queue_depth", 1)             # 1 = 永远处理最新帧，不积压
        # --- 模型 ---
        p("weights", "yolov8s.pt")
        p("device", "auto")             # auto | cpu | cuda:0
        p("imgsz", 640)
        p("conf", 0.25)
        p("iou", 0.45)
        p("half", False)                # GPU 上用 FP16，Jetson 建议开
        # 这两个用逗号分隔的字符串，不用数组：
        #   ROS 2 无法从空列表 [] 推断参数类型，节点会起不来；
        #   而且从 launch 传进来的值本来就是字符串。
        p("keep_classes", "")           # "" = 全部；COCO 只要人就填 "0"
        p("class_names", "")            # 覆盖模型自带类名，"" = 用模型的
        # --- 输出 ---
        p("publish_annotated", True)
        p("detections_topic", "~/detections")
        p("annotated_topic", "~/image_annotated")
        p("status_topic", "~/status")
        p("max_rate_hz", 0.0)           # 0 = 不限速；Orin 50W 建议 5~10
        p("log_every_n", 30)

        def g(key):
            return self.get_parameter(key).value

        self.imgsz = int(g("imgsz"))
        self.conf = float(g("conf"))
        self.iou = float(g("iou"))
        self.half = bool(g("half"))
        self.keep = _int_set(g("keep_classes"))
        self.name_override = _str_list(g("class_names"))
        self.publish_annotated = bool(g("publish_annotated"))
        self.log_every_n = max(1, int(g("log_every_n")))
        rate = float(g("max_rate_hz"))
        self.min_period = (1.0 / rate) if rate > 0 else 0.0

        self.bridge = CvBridge()
        self.n_frames = 0
        self.n_skipped = 0
        self.last_infer_t = 0.0
        self.infer_ms = []

        self.model, self.names, self.device = self._load_model(
            str(g("weights")), str(g("device")))

        depth = max(1, int(g("queue_depth")))
        qos = sensor_qos(bool(g("best_effort_qos")), depth)
        topic = str(g("image_topic"))
        self.sub = self.create_subscription(Image, topic, self.on_image, qos)

        self.pub_det = self.create_publisher(
            Detection2DArray, str(g("detections_topic")), 10)
        self.pub_img = (self.create_publisher(Image, str(g("annotated_topic")), 1)
                        if self.publish_annotated else None)
        self.pub_status = self.create_publisher(String, str(g("status_topic")), 1)
        self.create_timer(5.0, self._publish_status)

        self.get_logger().info(
            "订阅 %s (best_effort=%s, depth=%d)"
            % (topic, g("best_effort_qos"), depth))
        self.get_logger().info(
            "模型 %s on %s, imgsz=%d, conf=%.2f, keep_classes=%s"
            % (g("weights"), self.device, self.imgsz, self.conf,
               sorted(self.keep) or "ALL"))
        self.get_logger().info("类别: %s" % (self.names,))

    # ---------- 模型 ----------

    def _load_model(self, weights, device):
        from ultralytics import YOLO      # 延迟导入，启动失败时报错更清楚
        import torch

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and self.half:
            self.get_logger().warn("CPU 上不支持 half，已自动关闭")
            self.half = False

        model = YOLO(weights)
        model.to(device)
        names = self.name_override or [model.names[i] for i in sorted(model.names)]

        # 预热一次。第一帧推理会慢很多（CUDA 上下文 + cudnn benchmark），
        # 不预热的话上线头几帧会被丢掉。
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=self.imgsz, verbose=False, **self._extra())
        self.get_logger().info(
            "预热完成 %.0f ms" % (1000 * (time.perf_counter() - t0)))
        return model, names, device

    def _extra(self):
        """只有开了 FP16 才传 half。

        ultralytics 8.4 起 half 已废弃（提示改用 quantize），每帧都会刷一条
        警告。默认关闭时干脆不传这个参数，日志就干净了；真要开 FP16
        （Jetson 上建议开）再传，那时的警告是有意义的提示。
        """
        return {"half": True} if self.half else {}

    def _name_of(self, cid):
        return self.names[cid] if 0 <= cid < len(self.names) else str(cid)

    # ---------- 主回调 ----------

    def on_image(self, msg):
        now = time.perf_counter()
        if self.min_period and (now - self.last_infer_t) < self.min_period:
            self.n_skipped += 1
            return
        self.last_infer_t = now

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn("图像转换失败: %s" % exc,
                                   throttle_duration_sec=5.0)
            return

        t0 = time.perf_counter()
        try:
            res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                     iou=self.iou, verbose=False,
                                     **self._extra())[0]
        except Exception as exc:
            self.get_logger().error("推理失败: %s" % exc,
                                    throttle_duration_sec=5.0)
            return
        self.infer_ms.append((time.perf_counter() - t0) * 1000.0)
        if len(self.infer_ms) > 100:
            self.infer_ms.pop(0)

        dets = []
        for b in res.boxes:
            cid = int(b.cls)
            if self.keep and cid not in self.keep:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            dets.append((cid, float(b.conf), x1, y1, x2, y2))

        # 空结果也要发 —— 下游据此知道节点活着、只是这一帧没有目标
        self.pub_det.publish(self._to_msg(dets, msg.header))

        if self.pub_img is not None:
            out = self.bridge.cv2_to_imgmsg(self._draw(frame, dets),
                                            encoding="bgr8")
            out.header = msg.header
            self.pub_img.publish(out)

        self.n_frames += 1
        if self.n_frames % self.log_every_n == 0:
            avg = sum(self.infer_ms) / len(self.infer_ms)
            self.get_logger().info(
                "%d 帧 | 本帧 %d 个目标 | 推理 %.0f ms (上限 %.1f fps) | 跳过 %d"
                % (self.n_frames, len(dets), avg, 1000.0 / avg, self.n_skipped))

    # ---------- 输出 ----------

    def _to_msg(self, dets, header):
        arr = Detection2DArray()
        arr.header = header
        for cid, conf, x1, y1, x2, y2 in dets:
            d = Detection2D()
            d.header = header
            bb = BoundingBox2D()
            bb.center.position.x = (x1 + x2) / 2.0
            bb.center.position.y = (y1 + y2) / 2.0
            bb.size_x = x2 - x1
            bb.size_y = y2 - y1
            d.bbox = bb
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = self._name_of(cid)   # 字符串类名，不是数字
            h.hypothesis.score = conf
            d.results.append(h)
            arr.detections.append(d)
        return arr

    def _draw(self, frame, dets):
        vis = frame.copy()
        for cid, conf, x1, y1, x2, y2 in dets:
            name = self._name_of(cid)
            col = self.COLORS.get(name, (255, 0, 255))
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(vis, "%s %.2f" % (name, conf),
                        (int(x1), max(14, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        return vis

    def _publish_status(self):
        avg = (sum(self.infer_ms) / len(self.infer_ms)) if self.infer_ms else 0.0
        self.pub_status.publish(String(
            data="frames=%d skipped=%d infer_ms=%.1f device=%s"
                 % (self.n_frames, self.n_skipped, avg, self.device)))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SarDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
