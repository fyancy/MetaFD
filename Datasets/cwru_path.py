"""
CW 数据集
0 1 2 3分别代表对轴承施加的由轻到重的载荷程度(load)，
轴承转速分别对应1797, 1772, 1750, 1730 (rpm).
数据集均为.csv格式，样本数目不足，允许overlapping.
"""

import os
root_dir = r"G:\dataset\CWdata_12k"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]


# NC
NC_0 = get_file(os.path.join(root_dir, r'NC\0'))
NC_1 = get_file(os.path.join(root_dir, r'NC\1'))
NC_2 = get_file(os.path.join(root_dir, r'NC\2'))
NC_3 = get_file(os.path.join(root_dir, r'NC\3'))

# IF
IF_7_file = [r'007\IF\0', r'007\IF\1', r'007\IF\2', r'007\IF\3']
IF_14_file = [r'014\IF\0', r'014\IF\1', r'014\IF\2', r'014\IF\3']
IF_21_file = [r'021\IF\0', r'021\IF\1', r'021\IF\2', r'021\IF\3']
IF_7 = [get_file(os.path.join(root_dir, f)) for f in IF_7_file]
IF_14 = [get_file(os.path.join(root_dir, f)) for f in IF_14_file]
IF_21 = [get_file(os.path.join(root_dir, f)) for f in IF_21_file]

# OF
OF_7_file = [r'007\OF\0', r'007\OF\1', r'007\OF\2', r'007\OF\3']
OF_14_file = [r'014\OF\0', r'014\OF\1', r'014\OF\2', r'014\OF\3']
OF_21_file = [r'021\OF\0', r'021\OF\1', r'021\OF\2', r'021\OF\3']
OF_7 = [get_file(os.path.join(root_dir, f)) for f in OF_7_file]
OF_14 = [get_file(os.path.join(root_dir, f)) for f in OF_14_file]
OF_21 = [get_file(os.path.join(root_dir, f)) for f in OF_21_file]

# RoF
RoF_7_file = [r'007\RoF\0', r'007\RoF\1', r'007\RoF\2', r'007\RoF\3']
RoF_14_file = [r'014\RoF\0', r'014\RoF\1', r'014\RoF\2', r'014\RoF\3']
RoF_21_file = [r'021\RoF\0', r'021\RoF\1', r'021\RoF\2', r'021\RoF\3']
RoF_7 = [get_file(os.path.join(root_dir, f)) for f in RoF_7_file]
RoF_14 = [get_file(os.path.join(root_dir, f)) for f in RoF_14_file]
RoF_21 = [get_file(os.path.join(root_dir, f)) for f in RoF_21_file]

# Tasks with 10-way
T0 = [NC_0, IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0], RoF_7[0], RoF_14[0], RoF_21[0]]
T1 = [NC_1, IF_7[1], IF_14[1], IF_21[1], OF_7[1], OF_14[1], OF_21[1], RoF_7[1], RoF_14[1], RoF_21[1]]
T2 = [NC_2, IF_7[2], IF_14[2], IF_21[2], OF_7[2], OF_14[2], OF_21[2], RoF_7[2], RoF_14[2], RoF_21[2]]
T3 = [NC_3, IF_7[3], IF_14[3], IF_21[3], OF_7[3], OF_14[3], OF_21[3], RoF_7[3], RoF_14[3], RoF_21[3]]

# Tasks with 7-way
# T7w = [NC_0, IF_7[0], IF_14[0], OF_7[0], OF_14[0], RoF_7[0], RoF_14[0]]
# T4w = [NC_1, IF_21[0], OF_21[0], RoF_21[0]]

# Tasks with 6-way
T6w = [IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0]]
T4w = [NC_0, RoF_7[0], RoF_14[0], RoF_21[0]]  # brand new categories

# T_sq = [NC_3, IF_7[3], OF_7[3]]
# T_sa = [NC_3, OF_7[3], RoF_7[3]]


if __name__ == "__main__":
    print(T0)
    print(T1)
    print(T2)
    print(T3)
    pass
