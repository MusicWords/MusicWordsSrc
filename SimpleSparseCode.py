import matplotlib.pyplot as plt
import numpy as np
import MYDI
from scipy import signal
import copy
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
import pandas as pd
import os
import pickle
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nUsing cuda')
else:
    device = torch.device('cpu')
    print('\nUsing CPU')
# 获取调色板颜色列表
palette = sns.color_palette("Set1", 50)  # 选择 'husl' 调色板，并指定颜色数量

# 全局参数
NOTE_ON_WEIGHT = 1  # 音头的赋值
DUR_WEIGHT = 0.01  # 音头的赋值
DPI = 200
TPB = 24
from scipy.optimize import linear_sum_assignment


def calculate_iou(box1, box2):
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    if x1_intersection >= x2_intersection or y1_intersection >= y2_intersection:
        return 0.0

    intersection_area = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def calculate_average_iou(gt_boxes, pred_boxes):
    # 计算 IoU 矩阵
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt, pred)

    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # 注意负号使得 IoU 最大化

    # 计算匹配到的 IoU 的平均值
    matched_ious = [iou_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    average_iou = np.mean(matched_ious)
    return average_iou


# 函数：显示PR
def show_pr(ar, disp_xrange=None, title='Pianoroll',
            set_figure=1, aspect=1, tight_layout=1,
            subplot_position=None, close_axis=1):
    if set_figure:
        plt.figure()

    if disp_xrange:
        if type(disp_xrange) == tuple:
            ar = ar[:, disp_xrange[0]:disp_xrange[1]]
        elif type(disp_xrange) == int:
            ar = ar[:, 0:disp_xrange]
    # else:
    #     if ar.shape[1] > 500:  # 太长则截断显示
    #         ar = ar[:, 0:200]

    if subplot_position:
        subplot_position.imshow(ar, cmap='gray', aspect=f'{aspect}')
        subplot_position.set_title(title)
        subplot_position.invert_yaxis()
        subplot_position.axis('off')
    else:
        plt.imshow(ar, cmap='gray', aspect=f'{aspect}')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(color='lightgrey', alpha=0.2)
        if close_axis:
            plt.axis('off')

    if tight_layout:
        plt.tight_layout()
    if set_figure:
        plt.show()


def get_track_df(path, track_number=0):
    df_ls = MYDI.get_event_df_ls(path)
    return df_ls[track_number].iloc[:, :5]


def change_TPB_df(df, new_ticks_per_beat=TPB, old_tpb=480):
    out_df = df.copy()
    tick_rescale_rate = new_ticks_per_beat / old_tpb
    out_df.note_off_tick = np.round(out_df.note_off_tick * tick_rescale_rate).astype(int)
    out_df.note_on_tick = np.round(out_df.note_on_tick * tick_rescale_rate).astype(int)
    out_df.dur = np.round(out_df.dur * tick_rescale_rate).astype(int)
    return out_df


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


def id2path(n):
    pop_id = str(n).zfill(3)
    return 'POP909/' + pop_id + '/' + pop_id + '.mid'  # midi路径


# 函数：读取PR
# track_number: 0-旋律；1-桥；2-伴奏
def get_file_pr(path, track_number=0, disp=0, TPB=TPB):
    df = get_track_df(path, track_number=track_number)
    df = MYDI.change_TPB_df(df, new_ticks_per_beat=TPB, orig_TPB=MYDI.get_tpb(path))
    pr = DF2PR(df, on_weight=NOTE_ON_WEIGHT, dur_weight=DUR_WEIGHT)
    if disp:
        show_pr(pr[:, :500])
    return pr


# （padding是为了允许音形尺寸大于一部分midi的尺寸，并且让音形能更快找到更多的匹配对象）
# pad_width=((顶部行数, 底部行数), (左侧行数, 右侧行数))
def pad_small_pr(pr, pitch_range=30, time_pad=None):
    if pr.shape[0] < pitch_range:
        pad_n = int(np.ceil((pitch_range - pr.shape[0]) / 2))
        pr_padded = np.pad(pr, pad_width=((0, 2 * pad_n), (time_pad, time_pad)), mode='constant')
        return pr_padded
    else:
        return pr


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


# 音形类
# 属性：basis_set-音符集；template_shape-尺寸；disp-是否绘制显示音形；matrix-二维矩阵，初始化时计算
class Template:
    def __init__(self, basis_set, template_shape=None,
                 template_id=None, source=None, disp=0):
        self.basis_set = basis_set
        self.template_shape = template_shape
        self.template_id = template_id
        self.disp = disp
        self.n_basis = len(self.basis_set)
        self.source = source

        if template_shape is None:
            ps = [b.p for b in basis_set]
            xs = [b.x for b in basis_set]
            offs = [b.x + b.d for b in basis_set]
            min_p = min(ps)
            max_p = max(ps)
            min_x = min(xs)
            max_off = max(offs)
            self.template_shape = (max_p - min_p + 1, max_off - min_x + 1)
            for b in basis_set:
                b.x = b.x - min_x
                b.p = b.p - min_p

        # 记录音符信息用于标题
        self.basis_set_text = '\nBasis: '
        for basis in basis_set:
            self.basis_set_text = self.basis_set_text + f'\n{basis.i}, x={basis.x}, p={basis.p}, d={basis.d}'

        # 获得音形矩阵
        self.matrix = get_matrix(self.basis_set, self.template_shape, template_id=self.template_id, disp=self.disp)
        self.dataset = []
        self.variant_ls = []
        self.spawn_map = None


def np2tensor(x):
    return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)


def tensor2np(x):
    return x.cpu().detach().numpy().squeeze()


def full_padding(x, k):
    pad_y = k.shape[0] - 1
    pad_x = k.shape[1] - 1
    return np.pad(x, mode='constant', pad_width=((pad_y, pad_y), (pad_x, pad_x)))


def cross_correlate(kernel, feature_map, pad_mode='full'):
    def np2tensor(x):
        return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)

    # padding模式：full-全pad
    if pad_mode == 'full':
        feature_map = full_padding(feature_map, kernel)
    elif pad_mode == 'same':
        pad_top = (kernel.shape[0] - 1) // 2
        pad_bottom = (kernel.shape[0] - 1) - pad_top
        pad_left = (kernel.shape[1] - 1) // 2
        pad_right = (kernel.shape[1] - 1) - pad_left
        feature_map = np.pad(feature_map, mode='constant',
                             pad_width=((pad_top, pad_bottom), (pad_left, pad_right)))
    elif pad_mode == 'same-corner':
        feature_map = np.pad(feature_map, mode='constant',
                             pad_width=((0, kernel.shape[0] - 1),
                                        (0, kernel.shape[1] - 1)))

    # 将 NumPy 数组转换为 PyTorch 张量，并移动到指定设备
    if type(kernel) is np.ndarray:
        kernel = np2tensor(kernel)
    if type(feature_map) is np.ndarray:
        feature_map = np2tensor(feature_map)

    # 使用卷积计算互相关
    output = F.conv2d(feature_map, kernel)

    output_np = output.squeeze().cpu().numpy()
    return output_np


# 高斯模糊函数，输入矩阵，输出模糊结果
# size：核函数尺寸
# sigma：标准差
def blur(matrix, size=10, sigma=2, pad='full', normalize=1, divide_max=1):
    if size == 1:
        return matrix

    def gaussian_filter(size, sigma, plot=0):
        if size // 2 == 0:
            warnings.warn("Even blur kernel size is not recommended", UserWarning)
        kernel_1d = signal.windows.gaussian(size, std=sigma).reshape(size, 1)
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        if normalize:
            kernel_2d /= kernel_2d.sum()  # 正则化确保和为1
        if divide_max:
            kernel_2d /= kernel_2d.max()  # 高斯核最大值为1

        if plot:
            # 显示滤波器
            plt.imshow(kernel_2d, cmap='gray', interpolation='nearest')
            plt.title('2D Gaussian Filter')
            plt.colorbar()
            plt.show()

        return kernel_2d

    g = gaussian_filter(size, sigma)
    result = cross_correlate(g, matrix, pad_mode=pad)
    return result


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


# 变体类
class Variant:
    def __init__(self, matrix, flip, x_rescale, p_rescale, n_basis,
                 blur_size=3, blur_sigma=2, template_i=None, variant_id=None):
        self.flip = flip
        self.x_rescale = x_rescale
        self.p_rescale = p_rescale
        self.n_basis = n_basis
        self.blur_size = blur_size
        self.blur_sigma = blur_sigma

        self.matrix = matrix
        self.blur_matrix = blur(self.matrix, size=blur_size, sigma=blur_sigma)
        self.self_correlation = None

        self.template_i = template_i
        self.variant_id = variant_id

        self.rmap_ls = []
        self.capture_ls = []
        self.update_patch_ls = None


def get_all_variants(_template,
                     _flip_type_choices=['orig'],
                     _x_rescale_choices=[1],
                     _p_rescale_choices=[1],
                     disp=0, ncols=4,
                     blur_size=10, blur_sigma=2, percise_sigma=1):
    if disp:
        n_variants = len(_flip_type_choices) * len(_x_rescale_choices) * len(_p_rescale_choices)
        nrows = int(np.ceil(n_variants / ncols))
        plt.figure(figsize=(3 * nrows, 3 * ncols))
        variant_i = 0

    variant_ls = []
    for x_rescale in _x_rescale_choices:
        for p_rescale in _p_rescale_choices:
            for flip_type in flip_type_choices:
                variant_mat = get_variant(basis_set=_template.basis_set, orig_shape=_template.matrix.shape,
                                          flip_type=flip_type, xrescale=x_rescale, prescale=p_rescale,
                                          disp=0)

                if _template.source == 'learned':
                    sigma = percise_sigma
                else:
                    sigma = blur_sigma
                variant_obj = Variant(matrix=variant_mat, template_i=_template.template_id,
                                      n_basis=_template.n_basis, flip=flip_type,
                                      x_rescale=x_rescale, p_rescale=p_rescale,
                                      blur_size=blur_size, blur_sigma=sigma)
                variant_ls.append(variant_obj)

                # 绘制该音形所有变体
                if disp:
                    plt.subplot(nrows, ncols, variant_i + 1)
                    show_pr(variant_obj.blur_matrix, title=f'Variant of {_template.template_id} '
                                                           f'({flip_type}, '
                                                           f'{p_rescale}, '
                                                           f'{x_rescale})',
                            set_figure=0, tight_layout=0)
                    plt.axis('off')
                    variant_i = variant_i + 1

    plt.tight_layout()
    plt.show()
    return variant_ls


def gen_random_template(template_shape, n_basis_range, duration_range, template_i, disp):
    basis_set = []
    n_basis = np.random.choice(np.arange(n_basis_range[0], n_basis_range[1]))

    # 生成n_basis个basis的随机参数
    random_x_arr = np.random.randint(0, template_shape[1], (n_basis,))
    random_p_arr = np.random.randint(0, template_shape[0], (n_basis,))
    random_d_arr = np.random.randint(duration_range[0], duration_range[1], (n_basis,))
    # 裁去空白部分
    random_x_arr = random_x_arr - np.min(random_x_arr)
    random_p_arr = random_p_arr - np.min(random_p_arr)
    random_d_arr = random_d_arr - np.min(random_d_arr)
    # 随机生成基元集
    for basis_i in range(n_basis):
        basis_set.append(Basis(random_x_arr[basis_i],
                               random_p_arr[basis_i],
                               random_d_arr[basis_i],
                               i=basis_i))
    # 生成音形
    template = Template(basis_set, template_shape, template_id=template_i, disp=disp, source='random')
    return template


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
    return abs(numerator / denominator) if denominator != 0 else 0


def compute_max_zncc(matrix1, matrix2):
    matrix1 = full_padding(matrix1, matrix2)

    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape

    max_zncc_value = -1  # ZNCC range is between -1 and 1, start with the lowest possible value

    # Loop through all possible positions of the top-left corner of the smaller matrix
    for i in range(rows1 - rows2 + 1):
        for j in range(cols1 - cols2 + 1):
            # Extract the current patch from the larger matrix
            patch = matrix1[i:i + rows2, j:j + cols2]
            # Compute the ZNCC value between the current patch and the smaller matrix
            current_zncc = zncc(patch, matrix2)
            # Update the maximum ZNCC value if the current one is higher
            if current_zncc > max_zncc_value:
                max_zncc_value = current_zncc

    return max_zncc_value


def regenerate_different_template(t_ls, regenerate_thr,
                                  template_shape, n_basis_range, duration_range,
                                  blur_sigma, blur_size, template_i=None):
    regenerate = 1
    while regenerate:
        regenerate = 0
        template = gen_random_template(template_shape=template_shape,
                                       n_basis_range=n_basis_range,
                                       duration_range=duration_range, template_i=template_i, disp=0)

        # 每生成一个音形将其与已有音形做互相关，若表明二者相似，则重新生成
        template_mat = blur(template.matrix, size=blur_size, sigma=blur_sigma)
        h_pad, w_pad = template_mat.shape[0] - 1, template_mat.shape[0] - 1
        for template_exist in t_ls:
            if template_exist is not None:
                template_exist_mat_padded = np.pad(template_exist.matrix, pad_width=((h_pad, h_pad), (w_pad, w_pad)),
                                                   mode='constant')
                template_exist_mat = blur(template_exist_mat_padded, size=5, sigma=2)
                max_zncc = compute_max_zncc(template_exist_mat, template_mat)
                # 如果与某已有模板高度相似，重新生成
                if max_zncc > regenerate_thr:
                    print(f'Regenerating template {template_i} (cause: not unique)')
                    print(f'CC:{round(max_zncc, 2)}')
                    # 如果是因为相似而跳出for, 回到while开头（regenerate = 1）
                    regenerate = 1
                    break
        # 如果运行完了整个for，添加模板，regenerate = 0，从而跳出while
    return template


# 计算一个音形的所有变体在所有数据上的响应图
def get_rmaps_of_one_template(template, pr, gamma=0.06, disp=0, gen_surpress=0.2):
    if disp:
        n_rmap = len(template.variant_ls)
        plt.figure(figsize=(10, 3 * n_rmap))

    for i_rmap, variant in enumerate(template.variant_ls):
        response_map = cross_correlate(variant.blur_matrix, pr, pad_mode='none')

        # 将响应图除以自相关，以统一量纲，转化为相对最佳匹配的匹配度比例
        # 期待响应值在0~1，而实际由于延音和模糊化，响应值可能大于1，这表明目标附近有其他模板外音符
        variant.self_correlation = np.sum(blur(variant.matrix, sigma=variant.blur_sigma,
                                               size=variant.blur_size, pad='same') * variant.matrix)
        response_map = response_map / variant.self_correlation

        if template.source == 'learned':
            response_map = response_map - gen_surpress
        # # 音符个数惩罚
        response_map = response_map - variant.n_basis * gamma

        # 将响应图存入对应的变体的响应图列表中
        variant.rmap_ls.append(response_map)
        if disp:
            plt.subplot(n_rmap, 1, i_rmap + 1)
            show_pr(response_map, disp_xrange=(0, 500),
                    title=f'Response map of variant\n'
                          f'{variant.flip},{variant.x_rescale},{variant.p_rescale}',
                    set_figure=0, tight_layout=0)
            plt.axis('off')
    if disp:
        plt.show()

    return


# 矩形绘制函数
def plot_rectangle(bbox, edgecolor='red', fill=False, Linestyle='--', ax_position=None):
    rectangle = Rectangle((bbox[0] - 0.5, bbox[1] - 0.5), bbox[2], bbox[3], fill=fill, edgecolor=edgecolor,
                          linestyle=Linestyle)
    if ax_position:
        ax_position.add_patch(rectangle)
    else:
        plt.gca().add_patch(rectangle)


# 找出响应最大值：（响应值，二维坐标，乐谱编号，变体）
def find_max_response(dict):
    max_response_ls = []
    for template in dict:  # 遍历所有的音形
        for variant in template.variant_ls:  # 所有的变体
            for pr_i, rmap in enumerate(variant.rmap_ls):  # 所有的乐谱
                max_cord = np.unravel_index(np.argmax(rmap), rmap.shape)
                max_response_candidate = (rmap[max_cord], max_cord, pr_i, variant)
                max_response_ls.append(max_response_candidate)
    # 所有响应里最大的那个，记录下相应（响应值，坐标，乐谱编号，变体）
    max_of_all = max(max_response_ls, key=lambda x: x[0])
    return max_of_all


def expand_box(Bbox, expand_w, expand_h):
    return (Bbox[0] - expand_w, Bbox[1] - expand_h,
            Bbox[2] + expand_w, Bbox[3] + expand_h)


# 确定需要更新的响应图的区域（用最大响应位置和变体形状计算，同时防止出界）
def get_update_bbox_xyxy(max_cord, variant_added, variant_update, pr_i):
    rmap_2b_update = variant_update.rmap_ls[pr_i]
    # 需要用到的尺寸信息（添加的变体，当前要更新响应图的变体，两变体的互相关patch）
    variant_h_added = variant_added.blur_matrix.shape[0]
    variant_w_added = variant_added.blur_matrix.shape[1]
    variant_h_update = variant_update.blur_matrix.shape[0]
    variant_w_update = variant_update.blur_matrix.shape[1]
    # 计算不考虑响应图边界的需要更新的坐标范围
    patch_direct_ap = max_cord[0] - variant_h_update + 1
    patch_direct_ax = max_cord[1] - variant_w_update + 1
    patch_direct_bp = max_cord[0] + variant_h_added - 1
    patch_direct_bx = max_cord[1] + variant_w_added - 1
    # 计算响应图需要更新的坐标范围
    rmap_update_corner_ap = max(patch_direct_ap, 0)  # 左下角p坐标
    rmap_update_corner_ax = max(patch_direct_ax, 0)  # 左下角x坐标
    rmap_update_corner_bp = min(patch_direct_bp + 1, rmap_2b_update.shape[0])  # 右上角p坐标
    rmap_update_corner_bx = min(patch_direct_bx + 1, rmap_2b_update.shape[1])  # 右上角x坐标

    update_bbox_xyxy = (rmap_update_corner_ax, rmap_update_corner_ap,
                        rmap_update_corner_bx, rmap_update_corner_bp)
    return update_bbox_xyxy


# # 确定需要更新的响应图的区域（用最大响应位置和变体形状计算，同时防止出界）
# def get_update_bbox_xyxy(max_cord, variant_added, variant_update, pr_i, cross_update_patch):
#     rmap_2b_update = variant_update.rmap_ls[pr_i]
#     # 需要用到的尺寸信息（添加的变体，当前要更新响应图的变体，两变体的互相关patch）
#     variant_h_added = variant_added.blur_matrix.shape[0]
#     variant_w_added = variant_added.blur_matrix.shape[1]
#     variant_h_update = variant_update.blur_matrix.shape[0]
#     variant_w_update = variant_update.blur_matrix.shape[1]
#     patch_h = cross_update_patch.shape[0]
#     patch_w = cross_update_patch.shape[1]
#     # 计算不考虑响应图边界的需要更新的坐标范围
#     patch_direct_ap = max_cord[0] - variant_h_update + 1
#     patch_direct_ax = max_cord[1] - variant_w_update + 1
#     patch_direct_bp = max_cord[0] + variant_h_added - 1
#     patch_direct_bx = max_cord[1] + variant_w_added - 1
#     # 计算响应图需要更新的坐标范围
#     rmap_update_corner_ap = max(patch_direct_ap, 0)  # 左下角p坐标
#     rmap_update_corner_ax = max(patch_direct_ax, 0)  # 左下角x坐标
#     rmap_update_corner_bp = min(patch_direct_bp + 1, rmap_2b_update.shape[0])  # 右上角p坐标
#     rmap_update_corner_bx = min(patch_direct_bx + 1, rmap_2b_update.shape[1])  # 右上角x坐标
#     # 计算更新补丁的裁切坐标
#     patch_corner_ap = max(0, 0 - patch_direct_ap)  # 左下角p坐标
#     patch_corner_ax = max(0, 0 - patch_direct_ax)  # 左下角x坐标
#     patch_corner_bp = min(patch_h, patch_h - (patch_direct_bp - rmap_2b_update.shape[0] + 1))  # 右上角p坐标
#     patch_corner_bx = min(patch_w, patch_w - (patch_direct_bx - rmap_2b_update.shape[1] + 1))  # 右上角x坐标
#
#     update_bbox_xyxy = (rmap_update_corner_ax, rmap_update_corner_ap,
#                         rmap_update_corner_bx, rmap_update_corner_bp)
#     patch_region_xyxy = (patch_corner_ax, patch_corner_ap,
#                          patch_corner_bx, patch_corner_bp)
#     return update_bbox_xyxy, patch_region_xyxy


def box_xyxy2xywh(box_xyxy):
    return (box_xyxy[0], box_xyxy[1], box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1])


def bbox_crop(matrix, box):
    return matrix[box[1]: box[3], box[0]:box[2]]


def add_patch(pr, patch, cord):
    pr[cord[0]:cord[0] + patch.shape[0], cord[1]:cord[1] + patch.shape[1]] = \
        pr[cord[0]:cord[0] + patch.shape[0], cord[1]:cord[1] + patch.shape[1]] + patch


def minus_patch(pianoroll, patch, cord):
    pianoroll[cord[0]:cord[0] + patch.shape[0],
    cord[1]:cord[1] + patch.shape[1]] = pianoroll[cord[0]:cord[0] + patch.shape[0],
                                        cord[1]:cord[1] + patch.shape[1]] - patch
    return pianoroll


# 更新需要被更新的响应图
def update_rmap(max_cord, variant_added, pr_i, dict, residual_pr, repulsion=0.1):
    # print('updating rmap')
    for v_2b_updated in seek_all_variants(dict):
        # 确定需要更新的响应图的区域（用最大响应位置和变体形状计算，同时防止出界）
        update_bbox = get_update_bbox_xyxy(max_cord, variant_added, v_2b_updated, pr_i)

        # 裁切更新补丁
        pr_update_xleft = update_bbox[0]
        pr_update_xright = update_bbox[2] + v_2b_updated.blur_matrix.shape[1] - 1
        rmap_update_xleft = pr_update_xleft
        rmap_update_xright = update_bbox[2]

        limited_residual_pr = residual_pr[:, pr_update_xleft:pr_update_xright]
        update_patch = cross_correlate(v_2b_updated.blur_matrix, limited_residual_pr, pad_mode='none') \
                       / v_2b_updated.self_correlation

        v_2b_updated.rmap_ls[pr_i][:, rmap_update_xleft: rmap_update_xright] = update_patch - repulsion

    # print('updating done')
    return update_bbox


# # 更新需要被更新的响应图
# def update_rmap_2(max_cord, variant_added, pr_i, dict, cross_update_patch_matrix):
#     all_variants = seek_all_variants(dict)
#
#     for v_2b_updated in all_variants:
#         cross_update_patch = cross_update_patch_matrix[v_2b_updated.variant_id][variant_added.variant_id]
#         # 确定需要更新的响应图的区域（用最大响应位置和变体形状计算，同时防止出界）
#         update_bbox, patch_region = get_update_bbox_xyxy(max_cord, variant_added, v_2b_updated,
#                                                          pr_i, cross_update_patch)
#         # 裁切更新补丁
#         r_patch = bbox_crop(cross_update_patch, patch_region)
#         # 覆盖更新补丁(要用当前选择的变体的自相关)
#         update_region = bbox_crop(v_2b_updated.rmap_ls[pr_i], update_bbox)
#
#         v_2b_updated.rmap_ls[pr_i][update_bbox[1]: update_bbox[3],
#         update_bbox[0]: update_bbox[2]] = update_region - r_patch
#
#     return update_bbox


def show_search_step(disp, save, rmap_box_xyxy, patch_box_xyxy, expanded_bbox_xyxy,
                     old_variant_ls, variant, find_in_pr, max_response=None,
                     left_space=100, right_space=200, disp_all_x=0,
                     save_folder='', save_name=None):
    if disp or save:
        rmap_box = list(box_xyxy2xywh(rmap_box_xyxy))
        patch_box = list(box_xyxy2xywh(patch_box_xyxy))
        expanded_bbox = list(box_xyxy2xywh(expanded_bbox_xyxy))

        updated_all_variants = seek_all_variants(dict)

        if not disp_all_x:
            if rmap_box[0] > left_space:
                rmap_disp_range = (rmap_box[0] - left_space, rmap_box[0] + right_space)
                rmap_box[0] = left_space
            else:
                rmap_disp_range = left_space + right_space
            if patch_box[0] > left_space:
                pr_disp_range = (patch_box[0] - left_space, patch_box[0] + right_space)
                expanded_bbox[0] = expanded_bbox[0] - patch_box[0] + left_space
                patch_box[0] = left_space
            else:
                pr_disp_range = left_space + right_space
        else:
            rmap_disp_range = None
            pr_disp_range = None

        print('Drawing searching step...')
        _, axs = plt.subplots(4, 1, dpi=DPI)

        show_pr(variant.matrix, set_figure=0, tight_layout=0, subplot_position=axs[0],
                title=f'Variant of template:{variant.template_i}')

        show_pr(find_in_pr, set_figure=0, tight_layout=0, disp_xrange=pr_disp_range, subplot_position=axs[1],
                title=f'Original piano roll')
        plot_rectangle(patch_box, edgecolor='red', fill=False, Linestyle='--', ax_position=axs[1])
        plot_rectangle(expanded_bbox, edgecolor='yellow', fill=False, Linestyle='-', ax_position=axs[1])

        show_pr(encoding_pr_this, set_figure=0, tight_layout=0, disp_xrange=pr_disp_range, subplot_position=axs[2],
                title=f'Encoding progress')
        plot_rectangle(patch_box, edgecolor='red', fill=False, Linestyle='--', ax_position=axs[2])

        show_pr(residual_pr, set_figure=0, tight_layout=0, disp_xrange=pr_disp_range, subplot_position=axs[3],
                title=f'Risidual piano roll')
        plot_rectangle(patch_box, edgecolor='red', fill=False, Linestyle='--', ax_position=axs[3])
        save_fig(name=save_name, folder=save_folder)
        if disp:
            plt.show()

        # n_rows = len(old_variant_ls) + 1
        # n_cols = 2
        # _, axs = plt.subplots(n_rows, n_cols, dpi=400)
        #
        # # 音形原型图
        # show_pr(variant.matrix, title=f'Variant of template:{variant.template_i}',
        #         set_figure=0, tight_layout=0, subplot_position=axs[0][0])
        #
        # position = axs[0][1]
        # show_pr(find_in_pr, set_figure=0, disp_xrange=pr_disp_range,
        #         title=f'Cropping on piano-roll: {pr_i}', subplot_position=position)
        # plot_rectangle(patch_box, edgecolor='red', fill=False, Linestyle='--', ax_position=position)
        # plot_rectangle(expanded_bbox, edgecolor='yellow', fill=False, Linestyle='-', ax_position=position)
        #
        # for vi, v in enumerate(updated_all_variants):
        #     position = axs[vi + 1][0]
        #     show_pr(old_variant_ls[vi].rmap_ls[pr_i], set_figure=0, disp_xrange=rmap_disp_range,
        #             subplot_position=position,
        #             title=f'RMap of Template-{v.template_i}\n(R={str(np.round(max_response, 2))})'
        #                   f'(variant: {v.flip},{v.x_rescale},{v.p_rescale}) in score No.{pr_i}')
        #     plot_rectangle(rmap_box, edgecolor='red', fill=False, Linestyle='--', ax_position=position)
        #
        #     position = axs[vi + 1][1]
        #     show_pr(v.rmap_ls[pr_i], set_figure=0, disp_xrange=rmap_disp_range, subplot_position=position,
        #             title=f'Updated response map\n bbox:{box_xyxy2xywh(rmap_box_xyxy)}')
        #     plot_rectangle(rmap_box, edgecolor='red', fill=False, Linestyle='--', ax_position=position)
        #
        # plt.show()


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


# 多子图绘图
def show_multi_pr(pr_list, n_col=5, title_list=None, suptitle='', max_time_range=None,
                  save_folder='', save_name=None, palette_ls=None):
    n_pr = len(pr_list)
    nrows = int(np.ceil(n_pr / n_col))
    if max_time_range is not None:
        disp_range = min(pr_list[0].shape[1], max_time_range)
    else:
        disp_range = pr_list[0].shape[1]
    w_pr = disp_range / 20

    h_pr = pr_list[0].shape[0] / 10
    if h_pr * nrows < 1.5:
        h_pr = 1.5 / nrows
    plt.figure(figsize=(w_pr * n_col, h_pr * nrows))

    for i, pr in enumerate(pr_list):
        plt.subplot(nrows, n_col, i + 1)
        if title_list is not None:
            show_pr(pr, title=title_list[i], set_figure=0, tight_layout=0, disp_xrange=disp_range)
        else:
            show_pr(pr, title='', set_figure=0, tight_layout=0, disp_xrange=max_time_range)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
    if suptitle:
        plt.suptitle(suptitle)

    if save_name is not None:
        save_fig(name=save_name, folder=save_folder)
    plt.show()


def view_captured_dataset(disp, dict, img_name):
    if disp:
        print('===================================')
        print('Drawing encoding result...')
        n_rows = max(len(T.dataset) for T in dict) + 2
        n_cols = len(dict)
        _, axs = plt.subplots(n_rows, n_cols, dpi=DPI)
        for ax in axs.flatten():
            ax.axis('off')  # 关闭坐标轴
            ax.set_frame_on(False)  # 关闭边框
        for template in dict:
            # 音形原型图
            disp_row = 1
            show_pr(template.matrix, title='', set_figure=0, tight_layout=0,
                    subplot_position=axs[n_rows - disp_row][template.template_id])
            # axs[n_rows-1][template.template_id].set_title(f'Template-{template.template_id}', fontsize=10)
            # 音形号文本
            disp_row = 2
            axs[n_rows - disp_row][template.template_id].text(x=0.5, y=0.2, s=f'T-{template.template_id}',
                                                              ha='center', fontsize=5)
            # 捕获patch集合
            disp_row = 3
            capture_i = 0
            for pr in template.dataset:
                # plt.subplot(n_rows, n_cols, (capture_i - 1) * n_cols + template.template_id +1)
                show_pr(pr, title='', set_figure=0, tight_layout=0,
                        subplot_position=axs[n_rows - capture_i - disp_row][template.template_id])
                capture_i = capture_i + 1
        save_fig(name=img_name, folder='Captured dataset')
        plt.show()


def view_dataset_and_spawn_map(disp, dict, img_name):
    if disp:
        print('===================================')
        print('Drawing spawn maps')
        n_rows = max(len(T.dataset) for T in dict if T is not None) + 4
        n_cols = len(dict)
        _, axs = plt.subplots(n_rows, n_cols, dpi=DPI)
        for ax in axs.flatten():
            ax.axis('off')  # 关闭坐标轴
            ax.set_frame_on(False)  # 关闭边框
        for template in dict:
            if template is None:
                continue
            # 音形原型图
            disp_row = 1
            show_pr(template.spawn_map, title='', set_figure=0, tight_layout=0,
                    subplot_position=axs[n_rows - disp_row][template.template_id])
            # 音形号文本
            disp_row = 2
            axs[n_rows - disp_row][template.template_id].text(x=0.5, y=0.2, s=f'Spawn Map',
                                                              ha='center', fontsize=5)
            # 音形原型图
            disp_row = 3
            show_pr(template.matrix, title='', set_figure=0, tight_layout=0,
                    subplot_position=axs[n_rows - disp_row][template.template_id])
            # 音形号文本
            disp_row = 4
            axs[n_rows - disp_row][template.template_id].text(x=0.5, y=0.2, s=f'T-{template.template_id}',
                                                              ha='center', fontsize=5,
                                                              color=palette[template.template_id])
            # 捕获patch集合
            disp_row = 5
            capture_i = 0
            for pr in template.dataset:
                # plt.subplot(n_rows, n_cols, (capture_i - 1) * n_cols + template.template_id +1)
                show_pr(pr, title='', set_figure=0, tight_layout=0,
                        subplot_position=axs[n_rows - capture_i - disp_row][template.template_id])
                capture_i = capture_i + 1

        save_fig(name=img_name, folder='Captured dataset and spawn map')
        plt.show()


def view_relearing_result(disp, old_dict, dict, img_name, palette):
    if disp:
        print('===================================')
        print('Drawing dictionary re-learning result')
        n_rows = max(len(T.dataset) for T in old_dict if T is not None) + 6
        _, axs = plt.subplots(n_rows, len(old_dict), dpi=DPI)
        for ax in axs.flatten():
            ax.axis('off')  # 关闭坐标轴
            ax.set_frame_on(False)  # 关闭边框
        for template in old_dict:
            if template is None:
                continue
            new_template = dict[template.template_id]
            if new_template is not None:
                # 新学的音形
                disp_row = 1
                show_pr(new_template.matrix, title='', set_figure=0, tight_layout=0,
                        subplot_position=axs[n_rows - disp_row][template.template_id])
                # 音形号文本
                disp_row = 2
                axs[n_rows - disp_row][template.template_id].text(x=0.5, y=0.2, ha='center', fontsize=5,
                                                                  s=f'Re-learned T-{template.template_id}',
                                                                  color=palette[template.template_id])
            # 音形原型图
            disp_row = 3
            show_pr(template.spawn_map, title='', set_figure=0, tight_layout=0,
                    subplot_position=axs[n_rows - disp_row][template.template_id])
            # 音形号文本
            disp_row = 4
            axs[n_rows - disp_row][template.template_id].text(x=0.5, y=0.2, s=f'Spawn Map',
                                                              ha='center', fontsize=5)
            # 音形原型图
            disp_row = 5
            show_pr(template.matrix, title='', set_figure=0, tight_layout=0,
                    subplot_position=axs[n_rows - disp_row][template.template_id])
            # 音形号文本
            disp_row = 6
            axs[n_rows - disp_row][template.template_id].text(x=0.5, y=0.2, s=f'T-{template.template_id}',
                                                              ha='center', fontsize=5,
                                                              color=palette[template.template_id])
            # 捕获patch集合
            disp_row = 7
            capture_i = 0
            for pr in template.dataset:
                show_pr(pr, title='', set_figure=0, tight_layout=0,
                        subplot_position=axs[n_rows - capture_i - disp_row][template.template_id])
                capture_i = capture_i + 1
        save_fig(name=img_name, folder='Relearing result')
        plt.show()


def pad_patch_set(matrices):
    # 找到最大矩阵的尺寸
    max_rows = max(matrix.shape[0] for matrix in matrices)
    max_cols = max(matrix.shape[1] for matrix in matrices)

    result = []
    # 遍历所有矩阵，将它们填充到最大尺寸
    for matrix in matrices:
        padded_matrix = np.pad(matrix,
                               ((0, max_rows - matrix.shape[0]), (0, max_cols - matrix.shape[1])),
                               mode='constant')
        result.append(padded_matrix)

    return result


# 获取一个音形全部变体在全部数据集上的最大响应
def get_template_max_response(template):
    max_r_arr = np.array([[np.max(rmap) for rmap in variant.rmap_ls] for variant in template.variant_ls])
    return np.max(max_r_arr)


def PatchSet_2_SpawnMap(template, size, sigma):
    # 将该音形收集到的所有patch pad到统一尺寸然后求平均
    avg_patch = np.sum(template.dataset, axis=0) / len(template.dataset)
    # 对平均响应图模糊化，以容许patch对不齐的情况
    return blur(avg_patch, size=size, sigma=sigma, pad='same', normalize=0, divide_max=1)


def cord_2_pxd(spawn_map, cord, dur_ratio_thr):
    (p, x) = np.array(cord)[:, 0]
    # 计数每个音符的时值
    d = 0
    while x + d + 1 < spawn_map.shape[1]:

        if max_dense * DUR_WEIGHT / NOTE_ON_WEIGHT * dur_ratio_thr < \
                spawn_map[p, x + d + 1] < max_dense * dur_ratio_thr:
            d += 1
        else:
            break
    return p, x, d


def update_avg_patch(T, basis):
    T[basis.p, basis.x] = 0  # 添加音头
    if basis.x + 1 < T.shape[1] and basis.d > 0:  # 添加延音(延音超出音形边界则截断)
        T[basis.p, basis.x + 1: min(basis.x + basis.d, T.shape[1])] = 0


def view_pr_bbox(disp, pr_list, pr_bbox_list, img_name, max_time_range=None):
    if disp:
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

            for info in pr_bbox_list[i]:
                box_variant = box_xyxy2xywh(info.bbox)
                plot_rectangle(box_variant, edgecolor=palette[info.variant.template_i], fill=False, Linestyle='--')
                box_patch_crop = box_xyxy2xywh(info.box_expanded)
                plot_rectangle(box_patch_crop, edgecolor=palette[info.variant.template_i], fill=False, Linestyle='-')
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(suptitle)

        save_fig(name=img_name, folder='Encoding Result')
        plt.show()


def non_max_suppression(boxes, scores, iou_threshold):
    if type(boxes) == list:
        boxes = np.array(boxes)
    if type(scores) == list:
        scores = np.array(scores)
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def non_max_suppression_old(boxes, scores, iou_threshold=0.5):
    """
    执行非极大值抑制 (NMS)。

    参数:
    boxes -- 形状为 (N, 4) 的 numpy 数组，包含 N 个边界框，每个边界框的格式为 [x1, y1, x2, y2]。
    scores -- 形状为 (N,) 的 numpy 数组，包含每个边界框的置信度分数。
    iou_threshold -- IoU 阈值，用于决定是否抑制重叠的框。

    返回:
    保留的边界框的索引列表。
    """
    if type(boxes) == list:
        boxes = np.array(boxes)

    if len(boxes) == 0:
        return []

    # 计算边界框的面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按分数从高到低排序
    indices = np.argsort(scores)[::-1]

    keep = []

    while len(indices) > 0:
        # 选择分数最高的边界框
        i = indices[0]
        keep.append(i)

        # 计算与当前框的重叠区域
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        # 计算重叠区域的宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # 计算重叠区域的面积
        inter_area = w * h
        union_area = areas[i] + areas[indices[1:]] - inter_area
        iou = inter_area / union_area

        # 保留 IoU 小于阈值的边界框
        indices = indices[np.where(iou <= iou_threshold)[0] + 1]

    return keep


class Patch_info:
    def __init__(self, bbox, score, pr_i, variant, captured_patch, box_expanded):
        self.bbox = bbox
        self.score = score
        self.pr_i = pr_i
        self.variant = variant
        self.captured_patch = captured_patch
        self.box_expanded = box_expanded


def draw_curve(disp, data, title=None, threshold=None):
    if disp == 1:
        plt.figure()
        plt.plot(data)
        plt.title(title)
        if threshold:
            plt.axhline(threshold, color='red')
        plt.show()


def prepare_update_patches(dict):
    all_variants = []
    variant_id = 0
    for t in dict:
        for v in t.variant_ls:
            all_variants.append(v)
            v.variant_id = variant_id
            variant_id = variant_id + 1

    n_variants = len(all_variants)

    cross_update_patch_matrix = [[None for _ in range(n_variants)] for _ in range(n_variants)]

    for v1 in all_variants:
        for v2 in all_variants[v1.variant_id:]:
            cross_patch = cross_correlate(v1.blur_matrix, v2.blur_matrix, pad_mode='full')
            cross_update_patch_matrix[v1.variant_id][v2.variant_id] = cross_patch
            cross_update_patch_matrix[v2.variant_id][v1.variant_id] = cross_patch
    return cross_update_patch_matrix


def save_fig(name, folder=''):
    if name is None:
        return
    path = OUT_PATH + folder
    file_name = path + '/' + name + '.png'
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    plt.savefig(file_name, dpi=DPI)


def get_next_folder_name(root_directory):
    try:
        os.makedirs(root_directory + '/1')
    except FileExistsError:
        pass
        # 获取根目录下的所有文件夹
        existing_directories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

    # 找到现有文件夹中的最大序号
    if existing_directories:
        max_existing_number = max([int(d) for d in existing_directories])
    else:
        max_existing_number = 0

    # 计算下一个文件夹的序号
    new_folder_number = max_existing_number + 1
    new_folder_name = root_directory + '/' + str(new_folder_number) + '/'

    try:
        os.makedirs(new_folder_name)
    except FileExistsError:
        pass
    return new_folder_name


def seek_all_variants(dict):
    v_ls = []
    for t in dict:
        for v in t.variant_ls:
            v_ls.append(v)
    return v_ls


# 保存参数
def save_vars_to_csv(filename, **kwargs):
    """
    将变量名及其值保存为 CSV 文件。
    参数:
    filename (str): 保存文件的名称。
    **kwargs: 要保存的变量名及其值。
    """
    # 创建一个字典来存储变量名和对应的值
    data = {'Variable': [], 'Value': []}

    # 将传入的变量名和变量值填充到字典中
    for var_name, var_value in kwargs.items():
        data['Variable'].append(var_name)
        data['Value'].append(var_value)

    # 将字典转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(filename, index=False)
    print(f"关键参数已保存到 {filename}")


class BBox:
    def __init__(self, box, tid):
        self.box = box
        self.tid = tid


class template:
    def __init__(self, uid, tid, notes, template_df):
        self.uid = uid
        self.tid = tid
        self.notes = notes
        self.template_df = template_df


def read_label(track_number=0, id_range=(1, 30)):
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

    def get_df(id, track_number, crop_info):
        track_df = get_track_df(id2path(id), track_number=track_number)
        gt_df_temp = MYDI.change_TPB_df(track_df, new_ticks_per_beat=TPB, orig_TPB=MYDI.get_tpb(id2path(id)))
        gt_df_temp.note_on_tick -= crop_info[1]
        gt_df_temp.note_off_tick -= crop_info[1]
        gt_df_temp.pitch -= crop_info[0]
        gt_df_temp['gt_class'] = -1
        gt_df_temp['pred_class'] = -1
        return gt_df_temp

    def DictLabel_2_Bboxes(song_id, track_number, orig_TPB, crop_info, gt_df):
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

                    for i, note in patch_df.iterrows():
                        gt_df.loc[(gt_df['pitch'] == note.pitch) & (gt_df['note_on_tick'] == note.note_on_tick),
                                  'gt_class'] = t_id
                    bboxes.append((x1, y1, x2, y2))
            return bboxes, gt_df

    # def get_csv_pr(csv_path, disp=0, TPB=TPB, orig_TPB=480):
    #     df = pd.read_csv(csv_path)
    #     df = MYDI.change_TPB_df(df, new_ticks_per_beat=TPB, orig_TPB=orig_TPB)
    #     pr = DF2PR(df, on_weight=NOTE_ON_WEIGHT, dur_weight=DUR_WEIGHT)
    #     if disp:
    #         show_pr(pr[:, :500])
    #     return df, pr

    # 读取标注
    bbox_ls = []
    pr_ls = []
    crop_info_ls = []
    gt_df_ls = []
    # 把音形标注转换为bbox
    for id in id_range:

        song_path = id2path(id)
        orig_TPB = MYDI.get_tpb(song_path)

        # csv_path = f'labeled data/Labeled Songs/{id}-{track_number}.csv'
        # df, pr_csv = get_csv_pr(csv_path, disp=1, TPB=TPB, orig_TPB=480)

        pr, crop_info = MYDI.get_file_pr(song_path, track_number=track_number + 1, disp=0, TPB=TPB, return_crop_info=1)
        crop_info_ls.append(crop_info)

        gt_df = get_df(id, track_number + 1, crop_info=crop_info)
        gt_df_ls.append(gt_df)

        bboxes, gt_df = DictLabel_2_Bboxes(id, track_number=0, orig_TPB=orig_TPB, crop_info=crop_info, gt_df=gt_df)
        if bboxes:
            bbox_ls.append(bboxes)
            pr_ls.append(pr)

    # view_pr_bbox(pr_ls, bbox_ls, max_time_range=max_time_range)
    return bbox_ls, gt_df_ls


def bbox_label_df(bbox_ls, t_id_ls, gt_df):
    for ind, bbox in enumerate(bbox_ls):
        (x1, y1, x2, y2) = bbox
        gt_df.loc[
            ((y1 <= gt_df['pitch']) & (gt_df['pitch'] < y2)) &
            ((x1 <= gt_df['note_on_tick']) & (gt_df['note_on_tick'] < x2)),
            'pred_class'
        ] = t_id_ls[ind]

    return gt_df


def calc_classify_acc(true_labels, pred_labels):
    # 计算 Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(true_labels, pred_labels)
    print("Adjusted Rand Index (ARI):", ari)

    # 计算同质性分数（Homogeneity）
    homogeneity = homogeneity_score(true_labels, pred_labels)
    print("Homogeneity:", homogeneity)

    # 计算完整性分数（Completeness）
    completeness = completeness_score(true_labels, pred_labels)
    print("Completeness:", completeness)

    # 计算 V-Measure
    v_measure = v_measure_score(true_labels, pred_labels)
    print("V-Measure:", v_measure)
    return ari, homogeneity, completeness, v_measure


def calc_avg_c(df_ls):
    acc_metrics_avg = np.zeros(4)
    for df in df_ls:
        true_labels = np.array(df['gt_class'])
        pred_labels = np.array(df['pred_class'])
        ari, homogeneity, completeness, v_measure = calc_classify_acc(true_labels, pred_labels)
        acc_metrics_avg += np.array([ari, homogeneity, completeness, v_measure])
    acc_metrics_avg = acc_metrics_avg / len(df_ls)
    return acc_metrics_avg


def rmse(matrix_a, matrix_b):
    # 确保两个矩阵的形状一致
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("两矩阵的形状必须一致")

    # 计算RMSE
    mse = np.mean((matrix_a - matrix_b) ** 2)
    rmse_value = np.sqrt(mse)

    return rmse_value


import matplotlib.pyplot as plt


def plot_metrics_curves(metrics_curves):
    """
    绘制多个评估指标的曲线图，其中 RMSE 使用独立的纵坐标轴。

    Parameters:
    - metrics_curves: list of lists, 每个子列表包含每轮的6个评估指标。
                      假设顺序为：Average IOU, ARI, Homogeneity, Completeness, V-Measure, RMSE。
    """
    # 将指标分别拆分为不同的列表，方便绘制
    metrics_curves = list(zip(*metrics_curves))  # 转置，使每个元素为各个指标的时间序列

    # 定义指标名称和配色
    metric_names = ["Average IOU", "ARI", "Homogeneity", "Completeness", "V-Measure", "RMSE"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    line_styles = ['-', '--', '-.', ':', '-', '--']  # 不同的线条样式

    # 创建一个图形
    fig, ax1 = plt.subplots(figsize=(20, 8))

    # 绘制除 RMSE 外的所有指标
    for i, (metric, color, line_style) in enumerate(zip(metrics_curves[:-1], colors[:-1], line_styles[:-1])):
        ax1.plot(metric, color=color, linestyle=line_style, marker='o', label=metric_names[i])

    # 设置左侧纵坐标轴（默认）
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metrics")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    # 设置横轴刻度为 1 至 10
    ax1.set_xticks(range(1, 11))  # 假设你希望横轴的刻度是 1 到 10
    ax1.set_xticklabels(range(1, 11), fontsize=10)  # 可以调整字体大小
    ax1.legend(loc="upper left", fontsize=9)

    # 创建一个共享横坐标的第二纵坐标轴
    ax2 = ax1.twinx()

    # 绘制 RMSE 指标，使用右侧纵坐标
    ax2.plot(metrics_curves[-1], color=colors[-1], linestyle=line_styles[-1], marker='x', label=metric_names[-1])

    # 设置右侧纵坐标轴
    ax2.set_ylabel("RMSE", color=colors[-1], fontsize=10)
    ax2.tick_params(axis='y', labelsize=10, labelcolor=colors[-1])

    # 为 RMSE 添加图例
    ax2.legend(loc="upper right", fontsize=9)

    # 美化图形
    plt.title("Training Metrics Curves")
    plt.show()


# 示例用法
# metrics_curves = [ ... ]  # 填入您训练中逐轮记录的metrics列表
# plot_metrics_curves(metrics_curves)


# —————————————————————————————————————————————————————主程序——————————————————————————————————————————————————————
if __name__ == '__main__':

    print('Start')
    # =============================乐谱数据读取参数=================================
    midi_path = 'classical/001.mid'
    read_pop909 = 1
    id_range = range(2, 3)  # 乐谱范围
    time_range = None  # 乐谱取用的时间范围
    disp_time_range = None
    long_disp_range = (0, 1000)  # 最大乐谱显示长度
    min_pr_pitch_range = 40  # 应pad到的音高范围
    track_number = 1
    # 加载标注
    gt_bbox_ls, gt_df_ls = read_label(track_number=track_number - 1, id_range=id_range)

    # =============================音形随机生成参数=================================
    # 随机种子
    seed = 37
    # seed = np.random.randint(1, 100)
    np.random.seed(seed)
    # =======随机基本参数=======
    init_template_shape = (10, 20)  # 音形的初始化形状（高，长）
    init_n_basis_range = (5, 10)  # 音形的初始化基元音符个数
    init_n_templates = 4  # 音形的初始化个数
    init_duration_range = (1, 20)  # 音形的初始化基元时值的取值范围

    # =======重新生成阈值=======
    regenerate_thr = 0.8  # 独特性阈值：为避免生成音形与已有音形相似，重新生成音形的相似阈值（音形与已有音形）
    Rthr = 0.3  # 有用性阈值：为避免生成音形在数据集中缺少相似对象，重新生成音形的相似阈值（音形与数据集）

    # =======变体参数=======
    flip_type_choices = ['orig']  # ['orig', 'ud', 'lr', 'diag']
    x_rescale_choices = [1]
    p_rescale_choices = [1, 0.8, 1.2]

    # =============================编码参数=================================
    epsilon_response = 0.2  # 终止编码的响应阈值
    blur_size = 5  # size：核函数尺寸（最好为奇数）
    blur_sigma = blur_size / 4  # sigma：标准差 越小越集中，影响范围越小
    percise_sigma = 1  # 在拼图阶段，对于学习到的音形，要比随机生成的音形更加严苛
    time_pad_n = blur_size // 2  # 左右pad的时间宽度
    expand_h, expand_w = (10, 10)  # 拓宽音形视野的单侧大小
    gamma = 0.018  # 音形匹配计算响应图时对音符数量做惩罚
    gen_surpress = 0.25
    repulsion = 0.6

    # =======NMS后处理=======
    nms = 0
    iou_thr = 0.2

    # =============================字典学习参数=================================
    # =======patch=======
    spawn_map_blur_size = 3  # patch平均图的模糊范围参数（最好为奇数）
    spawn_map_blur_sigma = spawn_map_blur_size // 2  # patch平均图的模糊sigma参数

    # =======预处理=======
    merge_similarity_thr = 0.8  # 为避免学习音形时与已有音形相似，合并音形的相似阈值（patch图与已有patch图）
    min_patch_n = 3

    # =======***基元添加***=======
    basis_spawn_ratio_thr = 0.5  # 音形添加基元的响应阈值（≈相对于数据中最大相应的比例）
    dur_ratio_thr = basis_spawn_ratio_thr  # 音形添加基元延音的响应阈值
    min_relearned_basis_n = 3  # 音形学习的最少basis个数
    max_relearned_basis_n = 8  # 音形学习的最大basis个数

    # =============================大循环参数=================================
    dict_expand_n = 3  # 每轮字典满载时需要扩充的大小
    stop_learning = 0  # 音形提取算法的停止条件
    n_learn_round = 10  # 循环学习次数

    # =============================绘图参数=================================
    disp_dataset = 0
    disp_dict = 0
    disp_variants = 0
    disp_rmaps = 0
    disp_search_process = 0
    save_search_process = 0
    disp_patch_reverse = 0
    disp_patch_set = 0
    disp_bbox = 0
    disp_spawn_map = 0
    disp_relearing_result = 0

    # 绘图保存路径
    OUT_PATH = get_next_folder_name('Result')
    # 保存关键参数
    save_vars_to_csv(OUT_PATH + 'Parameters.csv',
                     seed=seed, id_range=id_range, time_range=time_range, track_number=track_number,
                     read_pop909=read_pop909, midi_path=midi_path,
                     n_learn_round=n_learn_round, dict_expand_n=dict_expand_n,
                     regenerate_thr=regenerate_thr, Rthr=Rthr, gen_surpress=gen_surpress, gamma=gamma,
                     merge_similarity_thr=merge_similarity_thr, min_patch_n=min_patch_n,
                     basis_spawn_ratio_thr=basis_spawn_ratio_thr, epsilon_response=epsilon_response,
                     min_relearned_basis_n=min_relearned_basis_n, max_relearned_basis_n=max_relearned_basis_n,
                     blur_size=blur_size, blur_sigma=blur_sigma, percise_sigma=percise_sigma,
                     expand_h=expand_h, expand_w=expand_w, repulsion=repulsion,
                     spawn_map_blur_size=spawn_map_blur_size, spawn_map_blur_sigma=spawn_map_blur_sigma,
                     nms=nms, iou_thr=iou_thr,
                     flip_type_choices=flip_type_choices, x_rescale_choices=x_rescale_choices,
                     p_rescale_choices=p_rescale_choices,
                     init_n_templates=init_n_templates, init_duration_range=init_duration_range,
                     init_template_shape=init_template_shape, init_n_basis_range=init_n_basis_range,
                     conmment='')
    # ———————————————————————————————————————————————加载数据——————————————————————————————————————————————————————
    # 第一步：加载midi, 输出量化PianoRoll
    if read_pop909:
        pr_list = []
        for i_pr in id_range:
            pr = get_file_pr(id2path(i_pr), track_number)
            pr_list.append(pr)
    else:
        pr_list = []
        pr = get_file_pr(midi_path, track_number)
        pr_list.append(pr)

    for priii, pr in enumerate(pr_list):
        if time_range:
            pr_list[priii] = pr[:, time_range[0]:time_range[1]]

    if disp_dataset:
        show_multi_pr(pr_list, max_time_range=disp_time_range,
                      title_list=[f'Score-{i}' for i in range(len(pr_list))], n_col=1)

    for priii, pr in enumerate(pr_list):
        # 将音高范围较小的乐谱进行上下padding，pad到统一默认尺寸。然后用音形与乐谱做相关运算。
        pr_list[priii] = pad_small_pr(pr, pitch_range=min_pr_pitch_range, time_pad=time_pad_n)

    # ———————————————————————————————————————————————学习开始——————————————————————————————————————————————————————
    # 在最开始，初始化一个长度为init_n_templates的字典
    dict = [None] * init_n_templates
    # 【随机生成-拼图-音形学习】大循环
    metrics_curves = []
    for learning_round in range(n_learn_round):
        # 如果字典没有空位，则扩充大小
        if None not in dict:
            dict += [None] * dict_expand_n
        print(f'第{learning_round}轮学习开始，扩充后字典长度：{len(dict)}')

        # ———————————————————————————————————————————————音形随机生成——————————————————————————————————————————————————————

        print('===================================')
        print('Generating random dictionary')
        print('===================================')
        # 第二步：初始化音形字典, 对字典每个空位随机生成不重复且有用的音形
        for template_i, template in enumerate(dict):
            Rmax = 0  # 最大响应值初始化为0
            # 若该音形在数据集上没有足够大的响应值，则持续重新生成音形
            while True:
                if template is None:
                    print('Getting template..')
                    # 随机生成指定个数、尺寸、基元数范围的音形字典
                    # 每生成一个音形将其与已有音形做互相关，若表明二者相似，则重新生成
                    template = regenerate_different_template(dict, regenerate_thr, init_template_shape,
                                                             init_n_basis_range, init_duration_range,
                                                             blur_sigma, blur_size,
                                                             template_i=template_i)
                    print('Got template')

                # 第三步：产生音形变体
                # variant_bank: 列表，每个元素是一个变体列表，也就是某个音形的所有变体
                # 模糊化处理：将音符值高斯扩散到附近，以允许一定范围的音符变动
                # disp=1: 绘制所有变体
                print('Getting variants..')
                template.variant_ls = get_all_variants(template,
                                                       _flip_type_choices=flip_type_choices,
                                                       _x_rescale_choices=x_rescale_choices,
                                                       _p_rescale_choices=p_rescale_choices,
                                                       blur_size=blur_size, blur_sigma=blur_sigma,
                                                       percise_sigma=percise_sigma,
                                                       disp=disp_variants, ncols=4)
                print('Got variants')

                # 第四步：将音形变体逐个与乐谱做相关运算
                # disp=1: 绘制所有响应图
                print('Computing response maps')
                for pr in tqdm(pr_list):
                    get_rmaps_of_one_template(template, pr, gamma=gamma, disp=disp_rmaps, gen_surpress=gen_surpress)
                print('Computation done')

                # 检查随机音形的最大响应值是否超过阈值(该音形所有变体在全部数据集上的最大值)，否则回炉重造
                Rmax = get_template_max_response(template)
                print(f'Template {template_i}: R={str(np.round(Rmax, 2))}')
                if Rmax < Rthr:
                    template = None
                    print(f'Regenerating template {template_i} (cause: not useful)')
                else:
                    dict[template_i] = template
                    break

        # cross_update_patch_matrix = prepare_update_patches(dict)

        # 绘制字典
        if disp_dict:
            show_multi_pr([template.matrix for template in dict], suptitle='Dictionary',
                          title_list=[f'T-{t.template_id}\n({t.source})' for t in dict], n_col=5,
                          save_folder='Dictionary', save_name=f'Round-{learning_round}',
                          palette_ls=[palette[t.template_id] for t in dict])

        # ———————————————————————————————————————————————拼图阶段——————————————————————————————————————————————————————

        # 第五步：
        # 两种策略：A-在全部数据集上学习，有新数据再增量学习（√）；B-在第一条数据上学习，然后不断增量学习
        # 1-找出响应最大值
        # 2-更新响应图
        # → 循环直到响应值过小
        # 找出响应最大值：（响应值，二维坐标，乐谱编号，变体）
        print('===================================')
        print('Encoding phase begin')

        print('===================================')
        print('Searching begin')
        encoding_response_curve = []
        encoding_pr_ls = [np.zeros_like(pr) for pr in pr_list]
        reconstruction_pr_ls = [np.zeros_like(pr) for pr in pr_list]
        # 用于收集各条乐谱bbox的列表
        pr_bboxes_list = [[] for _ in range(len(pr_list))]
        while 1:
            # 查找响应最大位置
            (max_response, max_cord, pr_i, variant) = find_max_response(dict)
            find_in_pr = pr_list[pr_i]
            # 响应低于阈值：停止
            if max_response < epsilon_response:
                break

            old_variant_ls = copy.deepcopy(seek_all_variants(dict))
            # 2-更新响应图
            variant_bbox = (max_cord[1], max_cord[0], max_cord[1] + variant.blur_matrix.shape[1],
                            max_cord[0] + variant.blur_matrix.shape[0])
            # rmap_update_bbox = update_rmap(max_cord, variant, pr_i, dict, cross_update_patch_matrix)
            encoding_pr_this = encoding_pr_ls[pr_i]
            add_patch(encoding_pr_this, variant.blur_matrix, max_cord)
            add_patch(reconstruction_pr_ls[pr_i], variant.matrix, max_cord)

            residual_pr = find_in_pr - encoding_pr_ls[pr_i]
            residual_pr[residual_pr < 0] = 0

            rmap_update_bbox = update_rmap(max_cord, variant, pr_i, dict, residual_pr, repulsion=repulsion)
            print(f'{round(max_response, 2), len(pr_bboxes_list[pr_i])}')

            # 根据扩展后的bbox裁切原乐谱
            box_expanded = expand_box(variant_bbox, expand_w, expand_h)
            box_expanded = np.array(box_expanded)
            if box_expanded[0] < 0:
                box_expanded[0] = 0
            if box_expanded[1] < 0:
                box_expanded[1] = 0
            cropped_patch = bbox_crop(find_in_pr, box_expanded)

            # 记录patch相关信息
            patch_info = Patch_info(bbox=variant_bbox, score=max_response, pr_i=pr_i, variant=variant,
                                    captured_patch=cropped_patch, box_expanded=box_expanded)
            pr_bboxes_list[pr_i].append(patch_info)
            encoding_response_curve.append(max_response)

            # 显示搜索结果
            show_search_step(disp_search_process, save_search_process, rmap_update_bbox, variant_bbox, box_expanded,
                             old_variant_ls, variant, find_in_pr, max_response=max_response,
                             left_space=100, right_space=200, disp_all_x=1,
                             save_folder=f'Encoding process/Round-{learning_round}',
                             save_name=f'{len(pr_bboxes_list[pr_i])} template, R={round(max_response, 2)}')
        print('Search done.')
        print('===================================')
        # draw_curve(1, encoding_response_curve, title='Encoding response curve', threshold=epsilon_response)

        # 对检测出的相似块做非极大值抑制筛选
        if nms:
            print('===================================')
            print('Doing NMS')
            print('===================================')
            for pr_i, pr_bboxes in enumerate(pr_bboxes_list):
                if pr_bboxes:
                    keep_indices = non_max_suppression([info.bbox for info in pr_bboxes],
                                                       [info.score for info in pr_bboxes], iou_threshold=iou_thr)
                    pr_bboxes_list[pr_i] = [pr_bboxes[i] for i in keep_indices]

        # 查看拼图情况
        view_pr_bbox(disp_bbox, pr_list, pr_bboxes_list, img_name=f'Round-{learning_round}')

        avg_avg_iou = 0
        for pr_i in range(len(pr_list)):
            pred_bbox_ls = [patch_info.bbox for patch_info in pr_bboxes_list[pr_i]]
            pred_tid_ls = [patch_info.variant.template_i for patch_info in pr_bboxes_list[pr_i]]
            gt_df = gt_df_ls[pr_i]

            # 分类
            gt_df = bbox_label_df(pred_bbox_ls, pred_tid_ls, gt_df)

            # IOU
            average_iou = calculate_average_iou(gt_bbox_ls[pr_i], pred_bbox_ls)
            avg_avg_iou += average_iou

        avg_avg_iou = avg_avg_iou / len(pr_list)
        print(avg_avg_iou)

        # 计算准确率
        avg_acc_metrics = calc_avg_c(gt_df_ls)
        print(avg_acc_metrics)

        # 计算重构误差
        for pr_i, rec in enumerate(reconstruction_pr_ls):
            err = rmse(rec, pr_list[pr_i])
        print(err)

        metrics_curves.append(metrics)

        plot_metrics_curves(np.array(metrics_curves))
        # ———————————————————————————————————————————————整理patch阶段——————————————————————————————————————————————————————
        # 截取乐谱块并做逆变换
        print('===================================')
        print('Cropping patches and doing reverse transformation')
        print('===================================')
        bbox_ls = []
        for patch_info_ls in pr_bboxes_list:
            if patch_info_ls:
                for patch_info in patch_info_ls:
                    # 截取乐谱块，识别音符集
                    basis_set = pr2basis_set(patch_info.captured_patch)
                    if not basis_set:
                        continue

                    # 并做逆变换
                    variant = patch_info.variant
                    patch_reversed = get_variant(basis_set, patch_info.captured_patch.shape,
                                                 1 / variant.p_rescale, 1 / variant.x_rescale,
                                                 flip_type=variant.flip, disp=0)
                    if disp_patch_reverse:
                        show_multi_pr([patch_info.captured_patch, patch_reversed], n_col=2,
                                      title_list=['Cropped patch', 'Reversed version'])

                    # 存储逆变换后的patch
                    dict[variant.template_i].dataset.append(patch_reversed)

        # 查看裁剪还原出的图片块组成的数据集
        view_captured_dataset(disp_patch_set, dict, img_name=f'Round-{learning_round}')

        # 0. 先检查有无高度相似的数据
        print('===================================')
        print('Preparing spawn maps...')
        print('===================================')
        for template_i, template in enumerate(dict):
            # 如捕获的patch过少，清零该音形，等待再次随机生成
            if len(template.dataset) <= min_patch_n:
                dict[template_i] = None
                print(f'Deleted useless template {template_i}')
            else:
                # 将该音形所有patch填充到统一最大尺寸
                template.dataset = pad_patch_set(template.dataset)
                spawn_map = PatchSet_2_SpawnMap(template, spawn_map_blur_size, spawn_map_blur_sigma)
                similar_arr = []
                # 对每个音形的patch平均图，与前面已计算的平均图做互相关，如果高度相似，则将patch数据集转移合并
                for compare_i, template_compare in enumerate(dict[:template_i]):
                    if template_compare is not None:
                        max_zncc = compute_max_zncc(template_compare.spawn_map, spawn_map)
                        similar_arr.append(max_zncc)
                        if max_zncc > merge_similarity_thr:  # 如果patch图与前面的音形patch图近似
                            template.dataset = template.dataset + template_compare.dataset  # 夺取其数据集
                            dict[compare_i] = None  # 消灭该音形，等待重新随机生成
                            print(f'Merging template {compare_i} and {template_i}')
                print(f'{template_i} similarity to others: {similar_arr}')
                # 合并后更新平均图
                template.dataset = pad_patch_set(template.dataset)
                template.spawn_map = PatchSet_2_SpawnMap(template, spawn_map_blur_size, spawn_map_blur_sigma)

        # 如果全部音形查找结果不足，直接重新生成
        if all(t is None for t in dict):
            continue
        # 显示所有生成图
        view_dataset_and_spawn_map(disp_spawn_map, dict, img_name=f'Round-{learning_round}')

        # ———————————————————————————————————————————————字典学习阶段——————————————————————————————————————————————————————
        # 第六步：从捕获并反变换的图片集中学习新的原型字典
        print('===================================')
        print('Re-learning phase begin')
        print('===================================')
        old_dict = copy.deepcopy(dict)

        # 1. 将每个音形下的图片块模求平均并模糊化，然后不断添加基元，更新平均patch
        for template_i, template in enumerate(dict):
            # 如捕获的音形过少，则不启动音形学习，直接再次随机生成
            if template is not None:
                # 开始根据平均patch图生成新音形
                basis_set = []
                print(f'Learning template{template_i}')
                spawn_map = copy.deepcopy(template.spawn_map)
                max_dense = np.max(spawn_map)
                while True:
                    # 根据最大密度位置生成basis构成新音形（每次更新平均patch之后重新计算模糊化音形生成图）
                    cur_max_dense = np.max(spawn_map)
                    # 如果密度过小，停止该音形学习，向字典添加该音形
                    if cur_max_dense < max_dense * basis_spawn_ratio_thr:
                        # 如果停止时没有生成足够多的基元，则重新生成该音形
                        if len(basis_set) < min_relearned_basis_n:
                            dict[template_i] = None
                            break
                        else:
                            # 生成音形
                            dict[template_i] = Template(basis_set, template_id=template_i, source='learned')
                            break

                    max_dense_cord = np.where(spawn_map == cur_max_dense)  # 密度最大位置
                    (p, x, d) = cord_2_pxd(spawn_map, max_dense_cord, dur_ratio_thr)  # 计算出最大位置的基元信息（估计时值）

                    # 生成基元
                    new_basis = Basis(p=p, x=x, d=d)
                    basis_set.append(new_basis)
                    print(f'added basis {new_basis}')
                    # 如果基元数量过多，停止该音形学习，向字典添加该音形
                    if len(basis_set) == max_relearned_basis_n:
                        # 生成音形
                        dict[template_i] = Template(basis_set, template_id=template_i, source='learned')
                        break

                    #  在平均patch上减去新添加的basis以更新
                    new_template_mat = get_matrix(basis_set=basis_set, T_shape=spawn_map.shape)
                    blured_new_template_mat = blur(new_template_mat, size=spawn_map_blur_size,
                                                   sigma=spawn_map_blur_sigma,
                                                   pad='same', normalize=0, divide_max=1)
                    spawn_map = template.spawn_map - blured_new_template_mat

        # 显示所有生成图
        view_relearing_result(disp_relearing_result, old_dict, dict, img_name=f'Round-{learning_round}',
                              palette=palette)

        print('Re-learning phase end')
        print('===================================')

    print(metrics_curves)
    print(metrics_curves[-1,:])
    print('end')
