import pandas as pd
import numpy as np
import MYDI
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
import seaborn as sns
import os
import torch.nn.functional as F
import torch
from scipy import signal
import copy

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nUsing cuda')
else:
    device = torch.device('cpu')
    print('\nUsing CPU')

# 全局参数
NOTE_ON_WEIGHT = 1  # 音头的赋值
DUR_WEIGHT = 0.01  # 音头的赋值
# 获取调色板颜色列表
palette = sns.color_palette("Set1", 50)  # 选择 'husl' 调色板，并指定颜色数量


# 函数：显示PR
def show_pr(ar, disp_xrange=None, title='Pianoroll',
            set_figure=1, aspect=1, tight_layout=1,
            subplot_position=None, close_axis=1, fontsize=10):
    if set_figure:
        prfig = plt.figure()

    if disp_xrange:
        if type(disp_xrange) == tuple:
            ar = ar[:, disp_xrange[0]:disp_xrange[1]]
        elif type(disp_xrange) == int:
            ar = ar[:, 0:disp_xrange]
    # else:
    #     if ar.shape[1] > 500:  # 太长则截断显示
    #         ar = ar[:, 0:200]

    if subplot_position:
        subplot_position.imshow(ar, cmap='gray')
        subplot_position.set_title(title, fontsize=fontsize)
        subplot_position.invert_yaxis()
        subplot_position.axis('off')
    else:
        plt.imshow(ar, cmap='gray', aspect=f'{aspect}')
        plt.title(title, fontsize=fontsize)
        plt.gca().invert_yaxis()
        plt.grid(color='lightgrey', alpha=0.2)
        if close_axis:
            plt.axis('off')

    if tight_layout:
        plt.tight_layout()
    if set_figure:
        plt.show()
        return prfig


# def get_label_bbox(df):
#     bbox_ls = []
#     last_note = None
#     for event_i, note in df.iterrows():
#         if note != last_note:
#             bbox_ls.append()
#             patch.append()
#     return pr
class template:
    def __init__(self, uid, tid, notes, template_df):
        self.uid = uid
        self.tid = tid
        self.notes = notes
        self.template_df = template_df


def get_csv_pr(csv_path, disp=0, TPB=24, orig_TPB=480):
    def DF2PR(df, on_weight=1, dur_weight=0.2):
        pitch_range = df.pitch.max() - df.pitch.min() + 1
        tick_range = df.note_off_tick.max() - df.note_on_tick.min() + 1 + list(df.dur)[-1]

        pr = np.zeros((pitch_range, tick_range))
        for event_i, event in df.iterrows():
            pr_note_on_tick = event.note_on_tick - df.note_on_tick.min()
            pr_pitch = event.pitch - df.pitch.min()
            pr[pr_pitch, pr_note_on_tick] = on_weight
            for on_tick in range(1, event.dur):
                pr[pr_pitch, pr_note_on_tick + on_tick] = dur_weight
        return pr

    df = pd.read_csv(csv_path)
    df = MYDI.change_TPB_df(df, new_ticks_per_beat=TPB, orig_TPB=orig_TPB)
    pr = DF2PR(df, on_weight=NOTE_ON_WEIGHT, dur_weight=DUR_WEIGHT)
    if disp:
        show_pr(pr[:, :500])
    return df, pr


# 矩形绘制函数
def plot_rectangle(bbox, edgecolor='red', fill=False, Linestyle='--', ax_position=None):
    rectangle = Rectangle((bbox[0] - 0.5, bbox[1] - 0.5), bbox[2], bbox[3] + 0.5, fill=fill, edgecolor=edgecolor,
                          linestyle=Linestyle)
    if ax_position:
        ax_position.add_patch(rectangle)
    else:
        plt.gca().add_patch(rectangle)


def box_xyxy2yxhw(box_xyxy):
    return (box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1])


def view_pr_bbox(pr_list, bbox_ls, max_time_range=None):
    print('===================================')
    print('Drawing bbox...')
    suptitle = 'Training Dataset'
    n_col = 1
    title_list = [f'Score-{i}' for i in range(len(pr_list))]

    n_pr = len(pr_list)
    nrows = int(np.ceil(n_pr / n_col))
    if max_time_range is not None:
        disp_range = min(pr_list[0].shape[1], max_time_range)
    else:
        disp_range = pr_list[0].shape[1]
    w_pr = disp_range / 20

    plt.figure(figsize=(w_pr * n_col, 1.8 * nrows))

    for i, pr in enumerate(pr_list):
        plt.subplot(nrows, n_col, i + 1)
        show_pr(pr, title=title_list[i], set_figure=0, tight_layout=0, disp_xrange=disp_range)

        for bbox in bbox_ls[i]:
            box_variant = box_xyxy2yxhw(bbox.box)
            plot_rectangle(box_variant, edgecolor=palette[bbox.tid], fill=False, Linestyle='--')
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(suptitle)

    plt.show()


def id2path(n):
    pop_id = str(n).zfill(3)
    return 'POP909/' + pop_id + '/' + pop_id + '.mid'  # midi路径


class BBox:
    def __init__(self, box, tid):
        self.box = box
        self.tid = tid


song_id = 8
track_number = 0
max_time_range = 1000

# 确保以二进制模式打开文件
pkl_path = 'labeled data/T_dict/8-0.pkl'
with open(pkl_path, 'rb') as file:
    dict = pickle.load(file)


def get_labeled_dict(song_id, track_number):
    pkl_path = f'labeled data/T_dict/{song_id}-{track_number}.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            dict = pickle.load(file)
        return dict
    else:
        print(f'missing file {song_id}-{track_number}')
        return 0


def convert_pacth(patch, crop_info, orig_TPB=480):
    patch_df = MYDI.change_TPB_df(patch.template_df, orig_TPB=orig_TPB)
    patch_df.note_on_tick -= crop_info[1]
    patch_df.note_off_tick -= crop_info[1]
    patch_df.pitch -= crop_info[0]
    return patch_df


def DictLabel_2_Bboxes(song_id, track_number, orig_TPB, crop_info):
    dict = get_labeled_dict(song_id, track_number)
    if dict:
        bboxes = []
        for t_id, t in dict.items():
            for patch in t:
                patch_df = convert_pacth(patch, crop_info, orig_TPB=orig_TPB)

                y1 = patch_df.pitch.min()
                x1 = patch_df.note_on_tick.min()
                y2 = patch_df.pitch.max() + 1
                x2 = patch_df.note_off_tick.max()

                if x2 > max_time_range:
                    break

                bboxes.append(BBox((x1, y1, x2, y2), t_id))
        return bboxes


def pad_small_pr(pr, pitch_range=30, time_pad=None):
    if pr.shape[0] < pitch_range:
        pad_n = int(np.ceil((pitch_range - pr.shape[0]) / 2))
        pr_padded = np.pad(pr, pad_width=((pad_n, pad_n), (time_pad, time_pad)), mode='constant')
        return pr_padded
    else:
        return pr


def save_fig(name, folder=''):
    if name is None:
        return
    path = '' + folder
    file_name = path + '/' + name + '.png'
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    plt.savefig(file_name, dpi=500)


# 多子图绘图
def show_multi_pr(pr_list, n_col=5, title_list=None, suptitle='', max_time_range=None,
                  save_folder='', save_name=None, palette_ls=None, show=1):
    n_pr = len(pr_list)
    nrows = int(np.ceil(n_pr / n_col))
    if max_time_range is not None:
        disp_range = min(pr_list[0].shape[1], max_time_range)
    else:
        disp_range = pr_list[0].shape[1]
    w_pr = disp_range / 10

    h_pr = pr_list[0].shape[0] / 5
    if h_pr * nrows < 1.5:
        h_pr = 1.5 / nrows
    if show:
        fig = plt.figure(figsize=(3 * n_col, 3 * nrows))

    for i, pr in enumerate(pr_list):
        plt.subplot(nrows, n_col, i + 1)
        if title_list is not None:
            show_pr(pr, title=title_list[i], set_figure=0, tight_layout=0)
        else:
            show_pr(pr, title='', set_figure=0, tight_layout=0)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
    if suptitle:
        plt.suptitle(suptitle)

    if save_name is not None:
        save_fig(name=save_name, folder=save_folder)
    if show:
        plt.show()
        return fig


def zncc(patch1, patch2):
    # Convert patches to float64 for precision
    patch1 = patch1.astype(np.float64)
    patch2 = patch2.astype(np.float64)

    # Compute the mean of each patch
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)

    # Subtract the mean from each patch
    norm1 = patch1 - mean1
    norm2 = patch2 - mean2

    # Compute the numerator
    numerator = np.sum(norm1 * norm2)

    # Compute the denominator
    denominator = np.sqrt(np.sum(norm1 ** 2) * np.sum(norm2 ** 2))

    # Return the ZNCC value
    return numerator / denominator if denominator != 0 else 0


def compute_shift_zncc(matrix1, kernel):
    rows2, cols2 = kernel.shape

    # 计算填充量
    pad_height = rows2 - 1

    # 对较大的矩阵进行填充
    padded_matrix1 = np.pad(matrix1, ((pad_height, pad_height), (0, 0)),
                            mode='constant', constant_values=0)

    # 提取填充后的矩阵的尺寸
    padded_rows1, padded_cols1 = padded_matrix1.shape

    # 用于存储 ZNCC 结果的矩阵
    zncc_results = np.zeros((matrix1.shape[0], 1))

    # 遍历所有可能的位置
    for i in range(matrix1.shape[0]):
        # 提取当前区域
        patch = padded_matrix1[i:i + rows2, :]
        # 计算 ZNCC 值
        current_zncc = zncc(patch, kernel)
        zncc_results[i] = current_zncc

    return zncc_results


def full_padding(x, k):
    pad_y = k.shape[0] - 1
    pad_x = k.shape[1] - 1
    return np.pad(x, mode='constant', pad_width=((pad_y, pad_y), (pad_x, pad_x)))


def cross_correlate(kernel, feature_map, pad_mode='full'):
    def np2tensor(x):
        return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)

    # padding模式：full-全pad
    if pad_mode == 'full':
        padding = (kernel.shape[0] - 1, kernel.shape[1] - 1)
    elif pad_mode == 'same':
        padding = 0
        feature_map = np.pad(feature_map, mode='constant',
                             pad_width=((0, kernel.shape[0] - 1),
                                        (0, kernel.shape[1] - 1)))
    else:
        padding = 0

    # 将 NumPy 数组转换为 PyTorch 张量，并移动到指定设备
    if type(kernel) is np.ndarray:
        kernel = np2tensor(kernel)
    if type(feature_map) is np.ndarray:
        feature_map = np2tensor(feature_map)

    # 使用卷积计算互相关
    output = F.conv2d(feature_map, kernel, padding=padding)

    output_np = output.squeeze().cpu().numpy()
    return output_np


# def shift_rmse(X, Y):
def calculate_rmse(array1, array2):
    # 检查数组形状是否相同
    if array1.shape != array2.shape:
        max_x = max(array1.shape[0], array2.shape[0])
        max_y = max(array1.shape[1], array2.shape[1])
        array1 = np.pad(array1, pad_width=((0, max_x - array1.shape[0]), (0, max_y - array1.shape[1])), mode='constant')
        array2 = np.pad(array2, pad_width=((0, max_x - array2.shape[0]), (0, max_y - array2.shape[1])), mode='constant')
        # raise ValueError("数组形状不一致")

    # 计算差异
    difference = array1.astype(float) - array2.astype(float)

    # 计算平方差
    squared_difference = np.square(difference)

    # 计算均方根误差
    mean_squared_error = np.mean(squared_difference)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return root_mean_squared_error


# 基元类
# 属性：x-音头；p-音高；d-时值
class Basis:
    def __init__(self, x, p, d, i=None):
        self.i = i
        self.x = x
        self.p = p
        self.d = d

    def __str__(self):
        return f'Basis-{self.i}: x={self.x}, p={self.p}, d={self.d}'

    def __repr__(self):
        return f'Basis-{self.i}: x={self.x}, p={self.p}, d={self.d}'


# 根据二维矩阵识别出音符集
def pr2basis_set(croped_pr):
    basis_set = []
    notes_in_pr_indices = np.where(croped_pr == NOTE_ON_WEIGHT)
    i_b = 0
    for p, x in np.array(notes_in_pr_indices).T:
        d = 0
        # 计数每个音符的时值
        while x + d + 1 < croped_pr.shape[1]:
            if croped_pr[p, x + d + 1] == DUR_WEIGHT:
                d += 1
            else:
                break
        basis_set.append(Basis(i=i_b, p=p, x=x, d=d))
    return basis_set


def spawn_basis_on_mat(T, basis):
    T[basis.p, basis.x] = NOTE_ON_WEIGHT  # 添加音头
    if basis.x + 1 < T.shape[1] and basis.d > 0:  # 添加延音(延音超出音形边界则截断)
        T[basis.p, basis.x + 1: min(basis.x + basis.d, T.shape[1])] = DUR_WEIGHT


# 由音符集创造音形矩阵
def get_matrix(basis_set, T_shape=None, template_id=None, disp=0):
    # 如果未输入形状，根据音符集自动计算大小
    if T_shape is None:
        T_shape = (np.max([basis.p for basis in basis_set]) + 1,
                   np.max([(basis.x + basis.d) for basis in basis_set]) + 1)

    T = np.zeros(T_shape)
    for basis_i, basis in enumerate(basis_set):
        spawn_basis_on_mat(T, basis)

    if disp:
        basis_set_text = "\n".join(map(str, basis_set))
        show_pr(T, title=f'Template - {template_id}\n{basis_set_text}')
    return T


# 获取音形变体矩阵
# variation_type：'flip'-翻转；'resize'-缩放
# variation_parameter：'flip':'ud'-上下翻转；'lr'-左右翻转；'diag'-上下左右翻转
# variation_parameter：'resize':(音高缩放比，时间缩放比) 只改变音头不改变时值
def get_variant(basis_set, orig_shape, prescale, xrescale, flip_type='orig', disp=0):
    variant_basis_set = copy.deepcopy(basis_set)

    # 如果输入要进行翻转
    if flip_type == 'ud':  # 上下翻转，将音形内音符的音高对称
        for basis in variant_basis_set:
            basis.p = orig_shape[0] - basis.p
    elif flip_type == 'lr':  # 左右翻转，将音形内音符的音头时间对称
        for basis in variant_basis_set:
            basis.x = orig_shape[1] - basis.x
    elif flip_type == 'diag':
        for basis in variant_basis_set:
            basis.p = orig_shape[0] - basis.p
            basis.x = orig_shape[1] - basis.x

    # 如果输入要进行横纵缩放
    if prescale != 1:
        for basis in variant_basis_set:
            basis.p = int(round(basis.p * prescale))

    if xrescale != 1:
        for basis in variant_basis_set:
            basis.x = int(round(basis.x * xrescale))

    variant_matrix = get_matrix(variant_basis_set)

    basis_set_text = "\n".join(map(str, variant_basis_set))
    if disp: show_pr(variant_matrix, title=f'Variant:flip-{flip_type}, '
                                           f'p rescale-{prescale}, '
                                           f'x rescale-{xrescale}\n'
                                           f'{basis_set_text}')
    return variant_matrix


# 高斯模糊函数，输入矩阵，输出模糊结果
# size：核函数尺寸
# sigma：标准差
def blur(matrix, size=10, sigma=2, pad='full', normalize=1):
    def gaussian_filter(size, sigma, plot=0):
        kernel_1d = signal.windows.gaussian(size, std=sigma).reshape(size, 1)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        if normalize:
            kernel_2d /= kernel_2d.sum()  # 正则化确保和为1

        if plot:
            # 显示滤波器
            plt.imshow(kernel_2d, cmap='gray', interpolation='nearest')
            plt.title('2D Gaussian Filter')
            plt.colorbar()
            plt.show()

        return kernel_2d

    g = gaussian_filter(size, sigma)
    # result = convolve2d(matrix, g, mode='same')
    result = cross_correlate(g, matrix, pad_mode=pad)
    result = result / np.sum(result) * np.sum(matrix)
    return result


def similarity(X, Y):
    response_map = cross_correlate(X, Y, pad_mode='full')
    self_correlation = np.sum(X * X)
    response_ratio_map = response_map / self_correlation
    cc = np.max(response_ratio_map)
    return cc


def plot_confusion_matrix(cm, labels=None, title='Confusion Matrix', cmap='Blues'):
    """
    绘制混淆矩阵的函数。

    参数:
    - cm: 2D 数组或类似矩阵，表示混淆矩阵。
    - labels: 标签列表，用于显示 x 和 y 轴上的类别名称（默认值为 None）。
    - title: 图形标题（默认值为 'Confusion Matrix'）。
    - cmap: 颜色映射，默认为 'Blues'。

    返回:
    - 无（显示混淆矩阵）。
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()


orig_TPB = 480
track_number = 0
id_range = range(1, 30)
flip_type_choices = ['orig', 'ud', 'lr', 'diag']
x_rescale_choices = [1, 0.8, 1.2]
p_rescale_choices = [1, 0.8, 1.2]
# x_rescale_choices = [1, 0.6, 0.8, 1.2, 1.4]
# p_rescale_choices = [1, 0.6, 0.8, 1.2, 1.4]

bbox_ls = []
pr_ls = []
crop_info_ls = []
# 把音形标注转换为bbox
for id in id_range:
    song_path = id2path(id)
    orig_TPB = MYDI.get_tpb(song_path)

    csv_path = f'labeled data/Labeled Songs/{song_id}-{track_number}.csv'
    pr, crop_info = MYDI.get_file_pr(song_path, track_number=track_number + 1, disp=0, TPB=24, return_crop_info=1)
    crop_info_ls.append(crop_info)

    bboxes = DictLabel_2_Bboxes(id, track_number=0, orig_TPB=orig_TPB, crop_info=crop_info)
    if bboxes:
        bbox_ls.append(bboxes)
        pr_ls.append(pr)

view_pr_bbox(pr_ls, bbox_ls, max_time_range=max_time_range)

# 浏览所有音形，给相似类加上相同音形号
orig_TPB = MYDI.get_tpb(id2path(1))
dict_pr_ls = []
for song_i in id_range:
    dict = get_labeled_dict(song_i, track_number)
    if dict:
        template_pr_ls = []
        for t_id, t in dict.items():
            if not t:
                continue

            print('convert_pacth..')
            # template_df = convert_pacth(t[0], crop_info_ls[song_i-1], orig_TPB=orig_TPB)

            patch_df = MYDI.change_TPB_df(t[0].template_df, orig_TPB=orig_TPB)

            print('DF2PR..')
            template_pr = MYDI.DF2PR(patch_df, on_weight=NOTE_ON_WEIGHT, dur_weight=DUR_WEIGHT)

            template_pr = pad_small_pr(template_pr, pitch_range=10, time_pad=0)

            template_pr_ls.append(template_pr)
        dict_pr_ls.append(template_pr_ls)

# n_col = max([len(d) for d in dict_pr_ls])
# n_row = len(dict_pr_ls)
# _, axs = plt.subplots(n_row, n_col, figsize=(2 * n_col, 2 * n_row))
# for ax in axs.flatten():
#     ax.axis('off')  # 关闭坐标轴
#     ax.set_frame_on(False)  # 关闭边框
#
# for song_i, dict in enumerate(dict_pr_ls):
#     for t_id, t in enumerate(dict):
#         show_pr(t, set_figure=0, tight_layout=0, subplot_position=axs[song_i][t_id],
#                 title=f'T-{t_id} in Song-{song_i}', fontsize=10)
#
# save_fig('dict', folder='./')
# plt.show()

# 获取跨乐曲统一字典
all_template_dict = []
for dict in dict_pr_ls:
    for t in dict:
        all_template_dict.append(t)

# plt.ion()
# fig = plt.figure()
# now_dict_fig = plt.figure()
#
# for tid, t in enumerate(uni_dict):
#     plt.close(now_dict_fig)
#     now_dict_fig = show_pr(t)
#
#     ttype = input("音型号:")
#     ttype = int(ttype)
#     sim_template_n_ls[ttype].append(tid)
#
#     if ttype not in range(len(uni_dict)):
#         uni_dict.append(t)
#         plt.close(fig)
#         show_multi_pr(uni_dict, title_list=[f'T-{i}' for i in range(len(uni_dict))], n_col=10, show=1)
# plt.draw()
# plt.show()


# uni_dict = []
# tid = 0
# sim_template_n_ls = [[] for _ in range(len(all_template_dict))]
# # 持续更新直到手动中断
# for tid, t in enumerate(all_template_dict):
#     show_multi_pr(uni_dict+[np.ones((3, 3)), t], title_list=[f'T-{i}' for i in range(len(uni_dict)+2)],
#                   n_col=5, show=1)
#
#     # 接收用户输入，更新图形
#     ttype = input("音型号:")
#     if ttype.lower() == 'exit':
#         break
#     ttype = int(ttype)
#     sim_template_n_ls[ttype].append(tid)
#
#     if ttype not in range(len(uni_dict)):
#         uni_dict.append(t)
#
# with open(f'uni_dict.pkl', 'wb') as file:
#     pickle.dump((uni_dict, sim_template_n_ls), file)
show_multi_pr(all_template_dict, title_list=[f'T-{i}' for i in range(len(all_template_dict))], n_col=10)

uni_dict = []
sim_template_n_ls = [[] for _ in range(len(all_template_dict))]
sim_thr = 0.9
err_thr = 0.001
sim_matrix = np.zeros(shape=(len(all_template_dict), len(all_template_dict)))
for i, t in enumerate(all_template_dict):
    variant_ls = []
    for x_rescale in x_rescale_choices:
        for p_rescale in p_rescale_choices:
            for flip_type in flip_type_choices:
                variant = get_variant(pr2basis_set(t), t.shape,
                                      p_rescale, x_rescale, flip_type=flip_type, disp=0)
                variant = blur(variant, size=3, sigma=1, pad='same')
                variant_ls.append(variant)

    find_sim = 0
    for j, exist_t in enumerate(all_template_dict[:i]):
        sim = 0
        err = np.inf
        match_v = None
        if abs(np.sum(t) - np.sum(exist_t)) > NOTE_ON_WEIGHT:
            continue
        for v in variant_ls:
            # cc = similarity(v, exist_t)
            # if cc > sim:
            #     sim = cc
            #     match_v = v

            rmse = calculate_rmse(v, exist_t)
            if rmse < err:
                err = rmse
                match_v = v

        # print((i, j), sim)
        # if sim > sim_thr:
        print((i, j), err)
        if err < err_thr:
            show_multi_pr([match_v, exist_t], title_list=[f'T-{i}', f'T-{j}'], n_col=2, suptitle='same')
            sim_template_n_ls[j].append(i)
            find_sim = 1
            break
        else:
            show_multi_pr([t, exist_t], title_list=[f'T-{i}', f'T-{j}'], n_col=2, suptitle='diff')

    if find_sim == 0:
        uni_dict.append(t)
        sim_template_n_ls[len(uni_dict)].append(i)

sns.heatmap(sim_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=np.arange(0, len(all_template_dict), 1),
            yticklabels=np.arange(0, len(all_template_dict), 1))

print('end')
