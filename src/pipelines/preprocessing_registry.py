"""설정 파일에서 전처리 파이프라인을 선택하는 레지스트리입니다."""

from src.pipelines.pipeline_baseline import BaselinePreprocessingPipeline
from src.pipelines.pipeline_em_v1 import EMV1PreprocessingPipeline
from src.pipelines.pipeline_em_v2 import EMV2PreprocessingPipeline
from src.pipelines.pipeline_em_v3 import EMV3PreprocessingPipeline
from src.pipelines.pipeline_em_v4 import EMV4PreprocessingPipeline
from src.pipelines.pipeline_em_v5 import EMV5PreprocessingPipeline
from src.pipelines.pipeline_em_v6 import EMV6PreprocessingPipeline
from src.pipelines.pipeline_em_v7 import EMV7PreprocessingPipeline
from src.pipelines.pipeline_em_v8 import EMV8PreprocessingPipeline
from src.pipelines.pipeline_em_v9 import EMV9PreprocessingPipeline
from src.pipelines.pipeline_em_v10 import EMV10PreprocessingPipeline
from src.pipelines.pipeline_em_v11 import EMV11PreprocessingPipeline
from src.pipelines.pipeline_em_v12 import EMV12PreprocessingPipeline
from src.pipelines.pipeline_em_v13 import EMV13PreprocessingPipeline
from src.pipelines.pipeline_em_v14 import EMV14PreprocessingPipeline
from src.pipelines.pipeline_em_v15 import EMV15PreprocessingPipeline
from src.pipelines.pipeline_em_v16 import EMV16PreprocessingPipeline
from src.pipelines.pipeline_em_v17 import EMV17PreprocessingPipeline
from src.pipelines.pipeline_em_v18 import EMV18PreprocessingPipeline
from src.pipelines.pipeline_em_v19 import EMV19PreprocessingPipeline
from src.pipelines.pipeline_em_v20 import EMV20PreprocessingPipeline
from src.pipelines.pipeline_em_v21 import EMV21PreprocessingPipeline
from src.pipelines.pipeline_em_v22 import EMV22PreprocessingPipeline
from src.pipelines.pipeline_em_v23 import EMV23PreprocessingPipeline
from src.pipelines.pipeline_em_v24 import EMV24PreprocessingPipeline
from src.pipelines.pipeline_em_v25 import EMV25PreprocessingPipeline
from src.pipelines.pipeline_em_v26 import EMV26PreprocessingPipeline
from src.pipelines.pipeline_em_v27 import EMV27PreprocessingPipeline
from src.pipelines.pipeline_jsj_v1 import JSJV1PreprocessingPipeline
from src.pipelines.pipeline_jsj_v2 import JSJV2PreprocessingPipeline
from src.pipelines.pipeline_jsj_v3 import JSJV3PreprocessingPipeline
from src.pipelines.pipeline_jsj_v4 import JSJV4PreprocessingPipeline
from src.pipelines.pipeline_jsj_v5 import JSJV5PreprocessingPipeline
from src.pipelines.pipeline_jsj_v6 import JSJV6PreprocessingPipeline
from src.pipelines.pipeline_jsj_v7 import JSJV7PreprocessingPipeline
from src.pipelines.pipeline_jsj_v8 import JSJV8PreprocessingPipeline
from src.pipelines.pipeline_jsj_v9 import JSJV9PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f0 import F0PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f0_no_raw import (
    F0NoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f1 import F1PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f2 import F2PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f3 import F3PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f3_no_raw import (
    F3NoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f3_position import (
    F3PositionPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f4 import F4PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f4_no_raw import (
    F4NoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f5 import F5PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f5_no_raw import (
    F5NoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f5_no_raw_missmask import (
    F5NoRawMissmaskPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f5_selective_no_raw import (
    F5SelectiveNoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f8 import F8PreprocessingPipeline
from src.pipelines.jyp_preprocessing.pipeline_jyp_f6 import (
    F6NoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f7 import (
    F7PairContrastNoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f9 import (
    F9GlobalAAPairNoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f10 import (
    F10GlobalAACompositionNoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_f11 import (
    F11AutoConfusionPairNoRawPreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_pipe_comb_v3 import (
    PipeCombV3PreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_pipe_comb_v4 import (
    PipeCombV4PreprocessingPipeline,
)
from src.pipelines.jyp_preprocessing.pipeline_jyp_raw import RawPreprocessingPipeline
from src.pipelines.pipeline_jh_v01 import JHV01PreprocessingPipeline
from src.pipelines.pipeline_jh_v02 import JHV02PreprocessingPipeline
from src.pipelines.pipeline_jh_v03 import JHV03PreprocessingPipeline
from src.pipelines.pipeline_jh_v04 import JHV04PreprocessingPipeline
from src.pipelines.pipeline_jh_v05 import JHV05PreprocessingPipeline
from src.pipelines.pipeline_jh_v06 import JHV06PreprocessingPipeline
from src.pipelines.pipeline_jh_v07 import JHV07PreprocessingPipeline
from src.pipelines.pipeline_jh_v08 import JHV08PreprocessingPipeline
from src.pipelines.pipeline_jh_v09 import JHV09PreprocessingPipeline
from src.pipelines.pipeline_jh_v10 import JHV10PreprocessingPipeline

from src.pipelines.pipeline_em_v28 import EMV28PreprocessingPipeline
from src.pipelines.pipeline_em_v29 import EMV29PreprocessingPipeline
from src.pipelines.pipeline_em_v30 import EMV30PreprocessingPipeline
from src.pipelines.pipeline_em_v31 import EMV31PreprocessingPipeline
from src.pipelines.pipeline_em_v32 import EMV32PreprocessingPipeline
from src.pipelines.pipeline_em_v33 import EMV33PreprocessingPipeline
from src.pipelines.pipeline_em_v34 import EMV34PreprocessingPipeline
from src.pipelines.pipeline_em_v35 import EMV35PreprocessingPipeline
from src.pipelines.pipeline_em_v36 import EMV36PreprocessingPipeline
from src.pipelines.pipeline_em_v37 import EMV37PreprocessingPipeline
from src.pipelines.pipeline_em_v38 import EMV38PreprocessingPipeline
from src.pipelines.pipeline_em_v39 import EMV39PreprocessingPipeline
from src.pipelines.pipeline_em_v40 import EMV40PreprocessingPipeline
from src.pipelines.pipeline_em_v41 import EMV41PreprocessingPipeline
from src.pipelines.pipeline_em_v42 import EMV42PreprocessingPipeline
from src.pipelines.pipeline_em_v43 import EMV43PreprocessingPipeline
from src.pipelines.pipeline_em_v44 import EMV44PreprocessingPipeline
from src.pipelines.pipeline_em_v45 import EMV45PreprocessingPipeline
from src.pipelines.pipeline_em_v46 import EMV46PreprocessingPipeline
from src.pipelines.pipeline_pipeComb_em_v1 import PipeCombEMV1PreprocessingPipeline
from src.pipelines.pipeline_pipeComb_em_v2 import PipeCombEMV2PreprocessingPipeline
from src.pipelines.pipeline_pipeComb_em_v1_001 import (
    PipeCombEMV1001PreprocessingPipeline,
)
from src.pipelines.pipeline_pipeComb_em_v1_006 import (
    PipeCombEMV1006PreprocessingPipeline,
)
from src.pipelines.pipeline_pipeComb_em_v1_007 import (
    PipeCombEMV1007PreprocessingPipeline,
)
from src.pipelines.em_preprocessing.pipeline_em_E1 import EME1PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E2 import EME2PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E3 import EME3PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E4 import EME4PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E5 import EME5PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E6 import EME6PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E7 import EME7PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E8 import EME8PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E9 import EME9PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E10 import EME10PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E11 import EME11PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E12 import EME12PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E13 import EME13PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_E14 import EME14PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F01 import EMF01PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F02 import EMF02PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F03 import EMF03PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F04 import EMF04PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F05 import EMF05PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F06 import EMF06PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F07 import EMF07PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F08 import EMF08PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F09 import EMF09PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F10 import EMF10PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F11 import EMF11PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F12 import EMF12PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F13 import EMF13PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F14 import EMF14PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F15 import EMF15PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F16 import EMF16PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F17 import EMF17PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F18 import EMF18PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F19 import EMF19PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_F20 import EMF20PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_G01 import EMG01PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_G02 import EMG02PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_G03 import EMG03PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_G04 import EMG04PreprocessingPipeline
from src.pipelines.em_preprocessing.pipeline_em_G05 import EMG05PreprocessingPipeline

PIPELINES = {
    "baseline": BaselinePreprocessingPipeline,
    "em_v1": EMV1PreprocessingPipeline,
    "em_v2": EMV2PreprocessingPipeline,
    "em_v3": EMV3PreprocessingPipeline,
    "em_v4": EMV4PreprocessingPipeline,
    "em_v5": EMV5PreprocessingPipeline,
    "em_v6": EMV6PreprocessingPipeline,
    "em_v7": EMV7PreprocessingPipeline,
    "em_v8": EMV8PreprocessingPipeline,
    "em_v9": EMV9PreprocessingPipeline,
    "em_v10": EMV10PreprocessingPipeline,
    "em_v11": EMV11PreprocessingPipeline,
    "em_v12": EMV12PreprocessingPipeline,
    "em_v13": EMV13PreprocessingPipeline,
    "em_v14": EMV14PreprocessingPipeline,
    "em_v15": EMV15PreprocessingPipeline,
    "em_v16": EMV16PreprocessingPipeline,
    "em_v17": EMV17PreprocessingPipeline,
    "em_v18": EMV18PreprocessingPipeline,
    "em_v19": EMV19PreprocessingPipeline,
    "em_v20": EMV20PreprocessingPipeline,
    "em_v21": EMV21PreprocessingPipeline,
    "em_v22": EMV22PreprocessingPipeline,
    "em_v23": EMV23PreprocessingPipeline,
    "em_v24": EMV24PreprocessingPipeline,
    "em_v25": EMV25PreprocessingPipeline,
    "em_v26": EMV26PreprocessingPipeline,
    "em_v27": EMV27PreprocessingPipeline,
    "jsj_v1": JSJV1PreprocessingPipeline,
    "jsj_v2": JSJV2PreprocessingPipeline,
    "jsj_v3": JSJV3PreprocessingPipeline,
    "jsj_v4": JSJV4PreprocessingPipeline,
    "jsj_v5": JSJV5PreprocessingPipeline,
    "jsj_v6": JSJV6PreprocessingPipeline,
    "jsj_v7": JSJV7PreprocessingPipeline,
    "jsj_v8": JSJV8PreprocessingPipeline,
    "jsj_v9": JSJV9PreprocessingPipeline,
    "jyp_raw": RawPreprocessingPipeline,
    "jyp_f0": F0PreprocessingPipeline,
    "jyp_f0_no_raw": F0NoRawPreprocessingPipeline,
    "jyp_f1": F1PreprocessingPipeline,
    "jyp_f2": F2PreprocessingPipeline,
    "jyp_f3_position": F3PositionPreprocessingPipeline,
    "jyp_f3": F3PreprocessingPipeline,
    "jyp_f3_no_raw": F3NoRawPreprocessingPipeline,
    "jyp_f4": F4PreprocessingPipeline,
    "jyp_f4_no_raw": F4NoRawPreprocessingPipeline,
    "jyp_f5": F5PreprocessingPipeline,
    "jyp_f5_no_raw": F5NoRawPreprocessingPipeline,
    "jyp_f5_no_raw_missmask": F5NoRawMissmaskPreprocessingPipeline,
    "jyp_f5_selective_no_raw": F5SelectiveNoRawPreprocessingPipeline,
    "jyp_f8": F8PreprocessingPipeline,
    "jyp_f6": F6NoRawPreprocessingPipeline,
    "jyp_f7": F7PairContrastNoRawPreprocessingPipeline,
    "jyp_f9": F9GlobalAAPairNoRawPreprocessingPipeline,
    "jyp_f10": F10GlobalAACompositionNoRawPreprocessingPipeline,
    "jyp_f11": F11AutoConfusionPairNoRawPreprocessingPipeline,
    "pipeComb_v3": PipeCombV3PreprocessingPipeline,
    "pipeComb_v4": PipeCombV4PreprocessingPipeline,
    "jh_v01": JHV01PreprocessingPipeline,
    "jh_v02": JHV02PreprocessingPipeline,
    "jh_v03": JHV03PreprocessingPipeline,
    "jh_v04": JHV04PreprocessingPipeline,
    "jh_v05": JHV05PreprocessingPipeline,
    "jh_v06": JHV06PreprocessingPipeline,
    "jh_v07": JHV07PreprocessingPipeline,
    "jh_v08": JHV08PreprocessingPipeline,
    "jh_v09": JHV09PreprocessingPipeline,
    "jh_v10": JHV10PreprocessingPipeline,
    "em_v28": EMV28PreprocessingPipeline,
    "em_v29": EMV29PreprocessingPipeline,
    "em_v30": EMV30PreprocessingPipeline,
    "em_v31": EMV31PreprocessingPipeline,
    "em_v32": EMV32PreprocessingPipeline,
    "em_v33": EMV33PreprocessingPipeline,
    "em_v34": EMV34PreprocessingPipeline,
    "em_v35": EMV35PreprocessingPipeline,
    "em_v36": EMV36PreprocessingPipeline,
    "em_v37": EMV37PreprocessingPipeline,
    "em_v38": EMV38PreprocessingPipeline,
    "em_v39": EMV39PreprocessingPipeline,
    "em_v40": EMV40PreprocessingPipeline,
    "em_v41": EMV41PreprocessingPipeline,
    "em_v42": EMV42PreprocessingPipeline,
    "em_v43": EMV43PreprocessingPipeline,
    "em_v44": EMV44PreprocessingPipeline,
    "em_v45": EMV45PreprocessingPipeline,
    "em_v46": EMV46PreprocessingPipeline,
    "pipeComb_em_v1": PipeCombEMV1PreprocessingPipeline,
    "pipeComb_em_v2": PipeCombEMV2PreprocessingPipeline,
    "pipeComb_em_v1_001": PipeCombEMV1001PreprocessingPipeline,
    "pipeComb_em_v1_006": PipeCombEMV1006PreprocessingPipeline,
    "pipeComb_em_v1_007": PipeCombEMV1007PreprocessingPipeline,
    "em_E1": EME1PreprocessingPipeline,
    "em_E2": EME2PreprocessingPipeline,
    "em_E3": EME3PreprocessingPipeline,
    "em_E4": EME4PreprocessingPipeline,
    "em_E5": EME5PreprocessingPipeline,
    "em_E6": EME6PreprocessingPipeline,
    "em_E7": EME7PreprocessingPipeline,
    "em_E8": EME8PreprocessingPipeline,
    "em_E9": EME9PreprocessingPipeline,
    "em_E10": EME10PreprocessingPipeline,
    "em_E11": EME11PreprocessingPipeline,
    "em_E12": EME12PreprocessingPipeline,
    "em_E13": EME13PreprocessingPipeline,
    "em_E14": EME14PreprocessingPipeline,
    "em_F01": EMF01PreprocessingPipeline,
    "em_F02": EMF02PreprocessingPipeline,
    "em_F03": EMF03PreprocessingPipeline,
    "em_F04": EMF04PreprocessingPipeline,
    "em_F05": EMF05PreprocessingPipeline,
    "em_F06": EMF06PreprocessingPipeline,
    "em_F07": EMF07PreprocessingPipeline,
    "em_F08": EMF08PreprocessingPipeline,
    "em_F09": EMF09PreprocessingPipeline,
    "em_F10": EMF10PreprocessingPipeline,
    "em_F11": EMF11PreprocessingPipeline,
    "em_F12": EMF12PreprocessingPipeline,
    "em_F13": EMF13PreprocessingPipeline,
    "em_F14": EMF14PreprocessingPipeline,
    "em_F15": EMF15PreprocessingPipeline,
    "em_F16": EMF16PreprocessingPipeline,
    "em_F17": EMF17PreprocessingPipeline,
    "em_F18": EMF18PreprocessingPipeline,
    "em_F19": EMF19PreprocessingPipeline,
    "em_F20": EMF20PreprocessingPipeline,
    "em_G01": EMG01PreprocessingPipeline,
    "em_G02": EMG02PreprocessingPipeline,
    "em_G03": EMG03PreprocessingPipeline,
    "em_G04": EMG04PreprocessingPipeline,
    "em_G05": EMG05PreprocessingPipeline,
}


def create_preprocessing_pipeline(config: dict):
    """설정의 이름과 파라미터에 맞는 전처리 파이프라인을 생성합니다."""
    pipeline_name = config.get("name", "baseline")
    try:
        pipeline_class = PIPELINES[pipeline_name]
    except KeyError as error:
        available = ", ".join(sorted(PIPELINES))
        raise ValueError(
            f"지원하지 않는 전처리 파이프라인입니다: {pipeline_name}. 사용 가능: {available}"
        ) from error
    parameters = {key: value for key, value in config.items() if key != "name"}
    return pipeline_class(**parameters)
