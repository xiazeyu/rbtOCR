from func import *
from paddleocr import PaddleOCR
import pprint

ocr = PaddleOCR(lang="ch",
                det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/",
                rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/",
                cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/",
                use_gpu=True,
                gpu_mem=4000,
                use_angle_cls=True,
                use_space_char=False
                )

TT = "TEST"
imgs_path = gen_imgs_path(TT)
output = []
for img_path in imgs_path:
    result = ocr.ocr(img_path, cls=True)
    #print(result)
    output.append((img_path, determine_real_address(result)))
    print(output[-1])

pprint.pprint(output)

print(get_acc(output, TT))
