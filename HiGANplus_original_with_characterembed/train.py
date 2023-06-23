import os
from datetime import datetime
import argparse

from lib.utils import yaml2config
from networks import get_model
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="/content/drive/MyDrive/HiGAN_concat/HiGANplus_original_with_characterembed/configs/gan_iam.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)
    run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
    logdir = os.path.join("/content/drive/MyDrive/HiGAN_concat/HiGANplus_original_with_characterembed/runs", os.path.basename(args.config)[:-4] + '-' + str(run_id))

    model = get_model(cfg.model)(cfg, logdir)
    model.train()

