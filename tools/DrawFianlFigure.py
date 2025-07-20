import sys
import os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging

from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    build_davis_test_loader
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model

from evaluation.inference import inference


logger = logging.getLogger("detectron2")

def do_test(cfg, model):

    # data_loader = build_davis_test_loader(cfg)
    result = inference(cfg, model,draw=True)
    return result


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    # model_root_dir = "./Rank_Saliency/Models/RVSOR(26)"
    # model_names = ['model_0039999.pth']

    #best model
    #model_root_dir = "./Rank_Saliency/Models/RVSOR(35)"
    #model_names = ['model_0009999.pth']

    #no tmie 
    # model_root_dir = "./Rank_Saliency/Models/RVSOR(35)"
    # model_names = ['model_0009999.pth']

    # model_root_dir = "./Rank_Saliency/Models/RVSOR(44)"
    # model_names = ['model_0019999.pth']
    
    model_root_dir = "./Rank_Saliency/Models/RVSOR(52)"
    model_names = ['model_0019999.pth']
    # model_root_dir = "./Rank_Saliency/Models/RVSOR(10)"
    # model_names = ['model_0069999.pth']

    ##ST
    # model_root_dir = "./Rank_Saliency/Models/RVSOR(9)"
    # model_names = ['model_0099999.pth']
    ##TS
    # model_root_dir = "./Rank_Saliency/Models/RVSOR(10)"
    # model_names = ['model_0099999.pth']

    # model_root_dir = "./Rank_Saliency/Models/RVSOR(28)"
    # model_names = ['model_0009999.pth','model_0019999.pth','model_0029999.pth','model_0039999.pth']

    result_corre = []
    result_f = []
    result_map=[]

    # Maps = []

    for model_name in model_names:
        model_dir = os.path.join(model_root_dir, model_name)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            model_dir, resume=args.resume
        )
        r_corre, r_f ,r_map= do_test(cfg, model)
        result_corre.append(r_corre)
        result_f.append(r_f)
        result_map.append(r_map)
        #Maps.append(map)

    print("\nSpearman")
    for c in result_corre:
        print("%.4f" % c)

    for f in result_f:
        print('MAE: {}'.format(f['mae']))
        print('F-measure: {}'.format(f['f_measure']))

    for f in result_map:
        print('MAP: {}'.format(f))

if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
