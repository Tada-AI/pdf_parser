from enum import IntEnum
import yaml
import os
from paddle.base.core import AnalysisConfig as Config
from paddle.inference import create_predictor

from dataclasses import dataclass
from typing import Protocol
import numpy as np
import cv2
from PIL.Image import Image
from io import BytesIO


def load_predictor(
    model_dir, run_mode="paddle", device="CPU", cpu_threads=1, delete_shuffle_pass=False
):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != "GPU" and run_mode != "paddle":
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(
                run_mode, device
            )
        )
    infer_model = os.path.join(model_dir, "model.pdmodel")
    infer_params = os.path.join(model_dir, "model.pdiparams")
    if not os.path.exists(infer_model):
        infer_model = os.path.join(model_dir, "inference.pdmodel")
        infer_params = os.path.join(model_dir, "inference.pdiparams")
        if not os.path.exists(infer_model):
            raise ValueError(
                "Cannot find any inference model in dir: {},".format(model_dir)
            )
    config = Config(infer_model, infer_params)
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(cpu_threads)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config


model_dir = (
    "../PaddleDetection/pretrained_models/picodet_lcnet_x1_0_fgd_layout_table_infer/"
)

SUPPORT_MODELS = {
    "YOLO",
    "PPYOLOE",
    "RCNN",
    "SSD",
    "Face",
    "FCOS",
    "SOLOv2",
    "TTFNet",
    "S2ANet",
    "JDE",
    "FairMOT",
    "DeepSORT",
    "GFL",
    "PicoDet",
    "CenterNet",
    "TOOD",
    "RetinaNet",
    "StrongBaseline",
    "STGCN",
    "YOLOX",
    "YOLOF",
    "PPHGNet",
    "PPLCNet",
    "DETR",
    "CenterTrack",
    "CLRNet",
}


class PredictConfig:
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir: str, use_fd_format=False):
        # parsing Yaml config for Preprocess
        fd_deploy_file = os.path.join(model_dir, "inference.yml")
        ppdet_deploy_file = os.path.join(model_dir, "infer_cfg.yml")
        if use_fd_format:
            if not os.path.exists(fd_deploy_file) and os.path.exists(ppdet_deploy_file):
                raise RuntimeError(
                    "Non-FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = fd_deploy_file
        else:
            if not os.path.exists(ppdet_deploy_file) and os.path.exists(fd_deploy_file):
                raise RuntimeError(
                    "FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = ppdet_deploy_file
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf["arch"]
        self.preprocess_infos = yml_conf["Preprocess"]
        self.min_subgraph_size = yml_conf["min_subgraph_size"]
        self.labels = yml_conf["label_list"]
        self.mask = False
        self.use_dynamic_shape = yml_conf["use_dynamic_shape"]
        if "mask" in yml_conf:
            self.mask = yml_conf["mask"]
        self.tracker = None
        if "tracker" in yml_conf:
            self.tracker = yml_conf["tracker"]
        if "NMS" in yml_conf:
            self.nms = yml_conf["NMS"]
        if "fpn_stride" in yml_conf:
            self.fpn_stride = yml_conf["fpn_stride"]
        if self.arch == "RCNN" and yml_conf.get("export_onnx", False):
            print(
                "The RCNN export model is used for ONNX and it only supports batch_size = 1"
            )
        # self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf["arch"]:
                return True
        raise ValueError(
            "Unsupported arch: {}, expect {}".format(yml_conf["arch"], SUPPORT_MODELS)
        )

    def print_config(self):
        print("-----------  Model Configuration -----------")
        print("%s: %s" % ("Model Arch", self.arch))
        print("%s: " % ("Transform Order"))
        for op_info in self.preprocess_infos:
            print("--%s: %s" % ("transform op", op_info["type"]))
        print("--------------------------------------------")


from .paddle_utils.picodet_postprocess import PicoDetPostProcess


class BytesReadable(Protocol):
    def read(self) -> bytes: ...


# class ImageInfo(TypedDict):
#     im_shape: np.ndarray[np.float32, Any]
#     scale_factor: np.ndarray[np.float32, Any]


def decode_image(image: Image, im_info):
    file = BytesIO()
    image.save(file, "PNG")
    file.seek(0)
    return decode_image_file(file, im_info)


def decode_image_file(im_file: BytesReadable, im_info):
    data = np.frombuffer(im_file.read(), dtype="uint8")
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_info["im_shape"] = np.array(im.shape[:2], dtype=np.float32)
    im_info["scale_factor"] = np.array([1.0, 1.0], dtype=np.float32)
    return im, im_info


def preprocess_image(input_image: Image, preprocess_ops):
    # process image by preprocess_ops
    im_info = {
        "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
        "im_shape": None,
    }
    im, im_info = decode_image(input_image, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    return im, im_info


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs["image"] = np.array((imgs[0],)).astype("float32")
        inputs["im_shape"] = np.array((im_info[0]["im_shape"],)).astype("float32")
        inputs["scale_factor"] = np.array((im_info[0]["scale_factor"],)).astype(
            "float32"
        )
        return inputs

    for e in im_info:
        im_shape.append(np.array((e["im_shape"],)).astype("float32"))
        scale_factor.append(np.array((e["scale_factor"],)).astype("float32"))

    inputs["im_shape"] = np.concatenate(im_shape, axis=0)
    inputs["scale_factor"] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros((im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs["image"] = np.stack(padding_imgs, axis=0)
    return inputs


from PIL.Image import Image


from .paddle_utils.preprocess import (
    Resize,
    NormalizeImage,
    Permute,
    PadStride,
    LetterBoxResize,
    WarpAffine,
    Pad,
    CULaneResize,
)
from utils.normalization import normalize_bbox, Rect

PREPROCESS_OPS = {
    "Resize": Resize,
    "NormalizeImage": NormalizeImage,
    "Permute": Permute,
    "PadStride": PadStride,
    "LetterBoxResize": LetterBoxResize,
    "WarpAffine": WarpAffine,
    "Pad": Pad,
    "CULaneResize": CULaneResize,
}


class LayoutClassification(IntEnum):
    Text = 0
    Title = 1
    List = 2
    Table = 3
    Figure = 4


class ModelReference:
    def __init__(self, model_dir: str, cls_map: dict[int, str]) -> None:
        self.model_dir = model_dir
        self._cls_map = self._validate_map(cls_map)

    def _validate_map(
        self, model_cls_map: dict[int, str]
    ) -> dict[int, LayoutClassification]:
        cls_map: dict[int, LayoutClassification] = dict()

        for model_cls, model_cls_name in model_cls_map.items():
            try:
                cls_map[model_cls] = LayoutClassification[model_cls_name]  # type: ignore
            except KeyError:
                raise Exception(
                    f'"{model_cls_name}" is not a valid LayoutClassification'
                )

        return cls_map

    def get_classification(self, model_cls: int) -> LayoutClassification:
        if model_cls in self._cls_map:
            return self._cls_map[model_cls]
        raise Exception(f'"{model_cls}" not present in class map: {self._cls_map}')


@dataclass
class LayoutDetectionResult:
    score: float
    classification: LayoutClassification
    # Normalized bbox
    bbox: Rect


class LayoutDetector:
    def __init__(self, model_reference: ModelReference) -> None:
        self.model_reference = model_reference
        predictor, predictor_config = load_predictor(model_reference.model_dir)
        self.predictor = predictor
        # TODO: Fix the above to correctly return the config
        self.pred_config = PredictConfig(model_reference.model_dir)

    def detect(self, image: Image):
        inputs = self._preprocess(image)
        result = self._predict()
        # ndarray: N x 6. [class_code, score, x0, y0, x1, y1]
        #   Bounding box is in the image's coordinate system
        result = self._postprocess(inputs, result)
        boxes = result.get("boxes", [])
        return self._normalize_result(image, boxes)

    def _normalize_result(self, image: Image, result):
        image_bbox = (0, 0, image.width, image.height)
        return [
            LayoutDetectionResult(
                classification=self.model_reference.get_classification(r[0]),
                score=r[1],
                bbox=normalize_bbox(image_bbox, r[2:]),
            )
            for r in result
        ]

    def _preprocess(self, image: Image):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop("type")
            preprocessor = PREPROCESS_OPS.get(op_type, None)
            if preprocessor is None:
                raise Exception(f"Invalid preprocessor type: '{op_type}'")
            preprocess_ops.append(preprocessor(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        im, im_info = preprocess_image(image, preprocess_ops)
        input_im_lst.append(im)
        input_im_info_lst.append(im_info)

        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()

        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == "x":
                input_tensor.copy_from_cpu(inputs["image"])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def _postprocess(self, inputs, result):
        # postprocess output of predictor
        np_score_list = result["boxes"]
        np_boxes_list = result["boxes_num"]
        postprocessor = PicoDetPostProcess(
            inputs["image"].shape[2:],
            inputs["im_shape"],
            inputs["scale_factor"],
            strides=self.pred_config.fpn_stride,
            nms_threshold=self.pred_config.nms["nms_threshold"],
        )
        np_boxes, np_boxes_num = postprocessor(np_score_list, np_boxes_list)
        result = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        return result

    def _predict(self, repeats=1):
        """
        Args:
            repeats (int): repeat number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
        """
        np_score_list, np_boxes_list = [], []

        for i in range(repeats):
            self.predictor.run()
            np_score_list.clear()
            np_boxes_list.clear()
            output_names = self.predictor.get_output_names()
            num_outs = int(len(output_names) / 2)
            for out_idx in range(num_outs):
                np_score_list.append(
                    self.predictor.get_output_handle(
                        output_names[out_idx]
                    ).copy_to_cpu()
                )
                np_boxes_list.append(
                    self.predictor.get_output_handle(
                        output_names[out_idx + num_outs]
                    ).copy_to_cpu()
                )
        result = dict(boxes=np_score_list, boxes_num=np_boxes_list)
        return result
