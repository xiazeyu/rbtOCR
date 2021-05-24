from constants import *
import numpy as np
import os


def get_prov(det_str: str) -> [bool, str]:
    det_str = det_str[:6]
    found = False
    fnd_prov = None
    for ext_prov in COV_LIST:
        if not found and ext_prov in det_str:
            found = True
            fnd_prov = ext_prov
    if not found:
        for keyword in QUICK_COV_LIST.keys():
            if not found and keyword in det_str:
                found = True
                fnd_prov = QUICK_COV_LIST[keyword]
    if found and fnd_prov == "贵阳":
        fnd_prov = "贵州"
    return [found, fnd_prov]


def get_real_test_label(file_name: str, test_set: str):
    if test_set == "BIG":
        return TEST_LABEL_BIG[file_name]
    if test_set == "SMALL":
        return TEST_LABEL_SMALL[file_name]
    if test_set == "TEST":
        return TEST_LABEL_TEST[file_name]


def get_pos_std(position) -> float:
    np_pos = np.array(position)
    return np_pos.std(axis=0).sum()


def determine_real_address(result) -> str:
    light_result = [(get_pos_std(line[0]), line[1][0]) for line in result]
    light_result.sort(key=lambda s: s[0], reverse=True)
    for _, det_str in light_result:
        found, fnd_prov = get_prov(det_str)
        # print(det_str, found, fnd_prov)
        if found:
            return fnd_prov


def gen_imgs_path(test_set: str):
    if test_set == "BIG":
        return ['./datasets/big/' + i for i in TEST_LABEL_BIG.keys()]
    if test_set == "SMALL":
        return ['./datasets/small/' + i for i in TEST_LABEL_SMALL.keys()]
    if test_set == "TEST":
        return ['./datasets/test/' + i for i in TEST_LABEL_TEST.keys()]


def get_acc(result, test_set: str) -> float:
    tot_cnt = 0
    ok_cnt = 0
    failed = []
    if test_set == "BIG":
        tot_cnt = len(TEST_LABEL_BIG)
    if test_set == "SMALL":
        tot_cnt = len(TEST_LABEL_SMALL)
    if test_set == "TEST":
        tot_cnt = len(TEST_LABEL_TEST)
    for line in result:
        img_path, prov = line
        img_name = os.path.basename(img_path)
        if test_set == "BIG" :
            if TEST_LABEL_BIG[img_name] != prov:
                failed.append(img_name)
            else:
                ok_cnt += 1
        if test_set == "SMALL":
            if TEST_LABEL_SMALL[img_name] != prov:
                failed.append(img_name)
            else:
                ok_cnt += 1
        if test_set == "TEST":
            if TEST_LABEL_TEST[img_name] != prov:
                failed.append(img_name)
            else:
                ok_cnt += 1
    return ok_cnt / tot_cnt, failed
