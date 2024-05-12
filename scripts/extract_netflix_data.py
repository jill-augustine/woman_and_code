import sys
import tarfile
from loguru import logger

import waac.config as config

import sys
# ROOT_DIR = Path("/content/drive/MyDrive/W&&C")
ROOT_DIR = config.ROOT_DIR
logger.info(ROOT_DIR)
DATA_DIR = config.DATA_DIR
logger.info(DATA_DIR)
logger.info(str(ROOT_DIR) in sys.path)
# logger.info(sys.path[-2:])
# if str(ROOT_DIR) not in sys.path:
    # sys.path.append(str(ROOT_DIR))

DATA_DIR.mkdir(exist_ok=True, parents=True)
with tarfile.open(DATA_DIR / "nf_prize_dataset.tar.gz", "r:gz") as t:
    names = t.getnames()
    logger.info(names)
    t.extractall(path=DATA_DIR, members = names)

logger.info("Done")
  