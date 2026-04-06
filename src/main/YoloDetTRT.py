import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import ctypes
import pycuda.driver as cuda
import time

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []

class YoloTRT():
    """
    Initialize a YoloTRT object for TensorRT Engine Model.
    :param config: Dictionary that contains the 'model' configuration
    :param library: Path to the TensorRT library
    :param yolo_ver: Version of YOLO being used
    """
    def __init__(self, config: dict, library: str = "../../lib/libmyplugins.so", yolo_ver: str = "v5"):
        # Clear global buffers
        global host_inputs, cuda_inputs, host_outputs, cuda_outputs, bindings
        host_inputs.clear()
        cuda_inputs.clear()
        host_outputs.clear()
        cuda_outputs.clear()
        bindings.clear()

        engine       = config['path']
        classes_file = config['classes']
        conf         = config['confidence']

        self.CONF_THRESH    = conf
        self.IOU_THRESHOLD  = 0.4
        self.yolo_version   = yolo_ver

        # Load classes first
        classes = []
        with open(classes_file, 'r') as f:
            for line in f:
                _, class_name = line.split(':')
                classes.append(class_name.strip())
        self.categories = classes

        # Calculate buffer sizes based on number of classes
        num_classes = len(self.categories)
        self.LEN_ONE_RESULT = num_classes + 5
        self.LEN_ALL_RESULT = 25200 * self.LEN_ONE_RESULT

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        ctypes.CDLL(library)

        with open(engine, 'rb') as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size
        self.context = self.engine.create_execution_context()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

    def PreProcessImg(self, img):
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def Inference(self, img):
        input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
        np.copyto(host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        t1 = time.time()
        self.context.execute_async(self.batch_size, bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        t2 = time.time()
        output = host_outputs[0]
        result_boxes, result_scores, result_classid = self.PostProcess(output, origin_h, origin_w)
        det_res = []
        for j in range(len(result_boxes)):
            box = result_boxes[j]
            class_id = int(result_classid[j])
            if class_id >= len(self.categories):
                continue
            det = dict()
            det["class"] = self.categories[class_id]
            det["conf"] = result_scores[j]
            det["box"] = box
            det_res.append(det)
            self.PlotBbox(box, img, label="{}:{:.2f}".format(self.categories[class_id], result_scores[j]))
        return det_res, t2 - t1

    def PostProcess(self, output, origin_h, origin_w):
        pred = np.reshape(output, (25200, self.LEN_ONE_RESULT))
        objectness = pred[:, 4:5]
        class_probs = pred[:, 5:]
        scores = objectness * class_probs
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        mask = confidences >= self.CONF_THRESH
        pred = pred[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        if len(pred) == 0:
            return np.array([]), np.array([]), np.array([])
        boxes = self.xywh2xyxy(origin_h, origin_w, pred[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        keep = []
        order = np.argsort(-confidences)
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            ious = self.bbox_iou(np.expand_dims(boxes[i], 0), boxes[order[1:]])
            same_class = class_ids[i] == class_ids[order[1:]]
            remove = (ious > self.IOU_THRESHOLD) & same_class
            order = order[1:][~remove]
        keep = np.array(keep)
        return boxes[keep], confidences[keep], class_ids[keep]

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                     np.maximum(inter_rect_y2 - inter_rect_y1 + 1, 0)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    def PlotBbox(self, x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = color or [np.random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)