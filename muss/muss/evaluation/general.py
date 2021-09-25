# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from easse.cli import report, get_orig_and_refs_sents, evaluate_system_output

from muss.utils.helpers import write_lines, get_temp_filepath

'''A simplifier is a function with signature: simplifier(complex_filepath, output_pred_filepath)'''


# def evaluate_simplifier(simplifier, test_set, orig_sents_path=None, refs_sents_paths=None, quality_estimation=False):
#     orig_sents, _ = get_orig_and_refs_sents(
#         test_set, orig_sents_path=orig_sents_path, refs_sents_paths=refs_sents_paths
#     )
#     orig_sents_path = get_temp_filepath()
#     write_lines(orig_sents, orig_sents_path)
#     sys_sents_path = simplifier(orig_sents_path)
#     return evaluate_system_output(
#         test_set,
#         sys_sents_path=sys_sents_path,
#         orig_sents_path=orig_sents_path,
#         refs_sents_paths=refs_sents_paths,
#         metrics=['sari', 'bleu', 'fkgl'],
#         quality_estimation=quality_estimation,
#     )


# def get_easse_report(simplifier, test_set, orig_sents_path=None, refs_sents_paths=None):
#     orig_sents, _ = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths)
#     orig_sents_path = get_temp_filepath()
#     write_lines(orig_sents, orig_sents_path)
#     sys_sents_path = simplifier(orig_sents_path)
#     report_path = get_temp_filepath()
#     report(
#         test_set,
#         sys_sents_path=sys_sents_path,
#         orig_sents_path=orig_sents_path,
#         refs_sents_paths=refs_sents_paths,
#         report_path=report_path,
#     )
#     return report_path



def evaluate_simplifier(simplifier, test_set, orig_sents_path=None, refs_sents_paths=None, quality_estimation=False):
    # when do generate & find best para: change test_set to custom    
    # when do evaludation, change original sentence to asset, while output is generated from custom data
    
    print('get ref from ( evaluate_simplifier)',test_set)
    # change this path using test_set
    if test_set == 'asset_valid':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/token_asset_0803/valid.complex'
    elif test_set == 'asset_test':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/token_asset_0803/test.complex'
        
    elif test_set == 'asset_valid_simple':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/token_asset_simple_NER_0810/valid.complex'
    elif test_set == 'asset_test_simple':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/token_asset_simple_NER_0810/test.complex'
    
    elif test_set == 'asset_valid_ABCD':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/asset_ABCD_C1C2B2/valid.complex'
    elif test_set == 'asset_test_ABCD':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/asset_ABCD_C1C2B2/test.complex'
    
    elif test_set =='asset_valid_both':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/asset_both_0912/valid.complex'
    
    elif test_set == 'asset_valid_NER_ABCD':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/asset_ABCD_NER/valid.complex'
    else:
        print('test_set is',test_set)
        assert False, "unexpectec test set"
    
    ################# 
    test_set = test_set.replace('_simple','')
    test_set = test_set.replace('_ABCD','')
    test_set = test_set.replace('_both','')
    test_set = test_set.replace('_NER','')
    print('test_set in evaluate_simplifier when get ori',test_set)
    
    orig_sents, _ = get_orig_and_refs_sents(
        test_set, orig_sents_path=orig_sents_path, refs_sents_paths=refs_sents_paths
    )
    
    
    orig_sents_path = get_temp_filepath()
    write_lines(orig_sents, orig_sents_path)

    

    sys_sents_path = simplifier(token_ori_sents_path)

    return evaluate_system_output(
        test_set,
        sys_sents_path=sys_sents_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        metrics=['sari', 'bleu', 'fkgl'],
        quality_estimation=quality_estimation,
    )



def get_easse_report(simplifier, test_set, orig_sents_path=None, refs_sents_paths=None):
    orig_sents, _ = get_orig_and_refs_sents(test_set, orig_sents_path, refs_sents_paths)
    orig_sents_path = get_temp_filepath()
    write_lines(orig_sents, orig_sents_path)

    # change this path using test_set
    if test_set == 'asset_valid':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/token_asset_0803/valid.complex'
    elif test_set == 'asset_test':
        token_ori_sents_path = '/content/drive/MyDrive/muss/resources/datasets/token_asset_0803/test.complex'
    else:
        assert False, "unexpectec test set"

    sys_sents_path = simplifier(token_ori_sents_path)
    report_path = get_temp_filepath()

    # print('test_set',test_set)
    # print('sys_sents_path',sys_sents_path)
    # print('orig_sents_path',orig_sents_path)
    # print('refs_sents_paths',refs_sents_paths)
    # print('report_path',report_path)

    
    report(
        test_set,
        sys_sents_path=sys_sents_path,
        orig_sents_path=orig_sents_path,
        refs_sents_paths=refs_sents_paths,
        report_path=report_path,
    )
    return report_path

