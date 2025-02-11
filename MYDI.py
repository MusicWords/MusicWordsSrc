import mido
from mido import Message, MidiFile, MidiTrack
import numpy as np
import pypianoroll
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
PR_height = 37
Note_on_weight = 1
Dur_weight = 1

# 读取单个midi文件为消息列表
def read_midi_msg(path = '001resave.mid',track_i = 1):
    msg_ls = []
    mid = MidiFile(path)
    for i, track in enumerate(mid.tracks):
        # print('Track {}: {}'.format(i, track.name))
        if i == track_i:
            for j,msg, in enumerate(track):
                if not msg.is_meta:
                    if msg.type == 'note_on' or msg.type == 'note_off':
                        msg_ls.append(msg)
    return msg_ls


def readin(path, n_mid = 909):
    mids = []

    for i in range(n_mid):
        mid = pypianoroll.read(path)
        mid = mid.tracks[0].pianoroll.transpose()

        mids.append(mid)
    return mids

# 消息列表 → 三维表示
def msg2event(msg_ls):
    pitch_ls = []
    vel_ls = []
    head_ls = []
    dur_ls = []
    tail_ls = []
    head_rlt_ls = []
    tail_rlt_ls = []
    head_msg_index_ls = []
    tail_msg_index_ls = []
    orig_other_msg_ls = []
    orig_other_msg_index_ls = []

    unclosed_pitch = []
    abs_time = 0

    for i,msg in enumerate(msg_ls):
        abs_time = abs_time+msg.time

        if (msg.type!='note_on') and (msg.type!='note_off'):
            orig_other_msg_index_ls.append(i)
            orig_other_msg_ls.append(msg)
            continue

        if msg.velocity!=0 and msg.type=='note_on':
            pitch_on = msg.note

            pitch_ls.append(pitch_on)
            vel_ls.append(msg.velocity)
            head_ls.append(abs_time)
            head_rlt_ls.append(msg.time)
            head_msg_index_ls.append(i)

            unclosed_pitch.append((pitch_on,abs_time))

        if msg.velocity==0 or msg.type=='note_off':
            pitch_off = msg.note

            for pitch_head in unclosed_pitch:
                if pitch_off == pitch_head[0]:
                    dur_ls.append(abs_time-pitch_head[1])
                    tail_ls.append(abs_time)
                    tail_rlt_ls.append(msg.time)
                    tail_msg_index_ls.append(i)

                    unclosed_pitch.remove(pitch_head)
                    break

    return np.array([pitch_ls, head_ls, tail_ls, dur_ls, vel_ls, head_rlt_ls, tail_rlt_ls, head_msg_index_ls, tail_msg_index_ls]).T, np.array([orig_other_msg_index_ls, orig_other_msg_ls]).T

def get_event_ls_ls(path,track_i_range):

    event_ls_ls = []
    other_msg_ls_ls = []
    for track_i in track_i_range:
        # msg_ls = read_midi_msg(path, track_i=track_i)
        msg_ls = list(MidiFile(path).tracks[track_i])
        event_ls,other_msg = msg2event(msg_ls)
        event_ls_ls.append(event_ls)
        other_msg_ls_ls.append(other_msg)

    event_ls_ls
    return event_ls_ls, other_msg_ls_ls

def arr_2_df_ls(arr_ls):
    df_ls = []
    for arr in arr_ls:
        df = pd.DataFrame(arr)
        df.columns = ['pitch', 'note_on_tick', 'note_off_tick', 'dur', 'vel', 'head_rlt', 'tail_rlt', 'head_msg_i', 'tail_msg_i']
        df_ls.append(df)
    return df_ls

def df_2_arr_ls(df_ls):
    arr_ls = []
    for df in df_ls:
        arr = np.array(df)
        arr_ls.append(arr)
    return arr_ls

def get_event_df_ls(path,track_i_range = range(1,4)):
    event_ls_ls, _ = get_event_ls_ls(path,track_i_range)
    return arr_2_df_ls(event_ls_ls)

def event2mid_track(event_ls):
    on_event_ls = np.array([event_ls[:,0], event_ls[:,1], event_ls[:,5], event_ls[:,4], event_ls[:,7]]).T # [pitch, head, relative_head, vel, head_msg_ind]
    off_event_ls = np.array([event_ls[:,0], event_ls[:,2], event_ls[:,6], np.zeros_like(event_ls[:,4]), event_ls[:,8]]).T# [pitch, tail, relative_tail, vel, tail_msg_ind]
    on_off_ls = np.vstack((on_event_ls, off_event_ls))
    # 按照原msg索引排序
    sorted_on_off_ls = on_off_ls[on_off_ls[:, 4].argsort()]
    return sorted_on_off_ls # [pitch, head/tail, relative_head/relative_tail, vel]

def create_new_midi(out_path, event_ls_ls, bpm = 69, ticks_per_beat = 480, numerator = 1, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time_signature_time=0):
    # 若单轨输入则转为列表
    if type(event_ls_ls) != list:
        event_ls_ls = list([event_ls_ls])

    # 创建midi
    new_mid = mido.MidiFile()

    print('No ref')
    # 设置元信息
    new_mid.ticks_per_beat = ticks_per_beat

    meta_track = MidiTrack()
    new_mid.tracks.append(meta_track)

    # 创建时间签名消息
    time_signature_msg = mido.MetaMessage('time_signature',
                                     numerator=numerator,  # 分子
                                     denominator=denominator,  # 分母
                                     clocks_per_click=clocks_per_click,  # 每拍的时钟数
                                     notated_32nd_notes_per_beat=notated_32nd_notes_per_beat,  # 每拍的 32 分音符数
                                     time=time_signature_time)  # 事件发生的时间
    meta_track.append(time_signature_msg)

    tempo = mido.bpm2tempo(bpm)  # 将 BPM 转换为每分钟微秒数
    meta_track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    meta_track.append(mido.MetaMessage('end_of_track'))

    # 分别添加轨道
    for track_i, event_ls in enumerate(event_ls_ls):
        # 时间序列排序
        sorted_on_off_ls = event2mid_track(event_ls)
        # 创建轨道
        track = MidiTrack()
        new_mid.tracks.append(track)


        for i, event in enumerate(sorted_on_off_ls):
                msg = Message('note_on', channel = track_i, note=event[0], velocity=event[3], time=event[2])
                track.append(msg)
                msg_i = msg_i + 1
        # 结束
        # track.append(mido.MetaMessage('end_of_track'))

    # 保存新的 MIDI 文件
    new_mid.save(out_path)
    return new_mid

def create_new_midi_with_ref(out_path, event_ls_ls, other_msg_ls_ls, ref_path = None, bpm = 69, ticks_per_beat = 480, numerator = 1, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time_signature_time=0):
    # 若单轨输入则转为列表
    if type(event_ls_ls) != list:
        event_ls_ls = list([event_ls_ls])

    # 创建midi
    new_mid = mido.MidiFile()
    ref_mid = mido.MidiFile(ref_path)

    # 有参考midi元轨道
    new_mid.tracks.append(ref_mid.tracks[0])

    # 分别添加轨道
    for track_i, event_ls in enumerate(event_ls_ls):
        # 时间序列排序
        sorted_on_off_ls = event2mid_track(event_ls)
        # 创建轨道
        track = MidiTrack()
        new_mid.tracks.append(track)

        # 读取其他类信息列表
        other_msg_ls = other_msg_ls_ls[track_i].copy()
        other_msg_index = other_msg_ls[:,0].copy()

        event_i = 0
        # 遍历输入 MIDI 文件的每个轨道和消息
        for msg_i in range(len(list(ref_mid.tracks[track_i+1]))):
            if np.any(other_msg_index == msg_i):
                other_msg = other_msg_ls[np.where(other_msg_index==msg_i)[0][0], 1]
                track.append(other_msg)

            else:
                event = sorted_on_off_ls[event_i]
                msg = Message('note_on', channel = track_i, note=event[0], velocity=event[3], time=event[2])
                track.append(msg)
                event_i = event_i + 1
        # 结束
        # track.append(mido.MetaMessage('end_of_track'))

    # 保存新的 MIDI 文件
    new_mid.save(out_path)
    return new_mid

def get_meta_info(midi_path):
    # msg_time = 0
    bpm_ls = []
    for meta_msg in list(MidiFile(midi_path).tracks[0]):
        # print(meta_msg)
        if meta_msg.type == 'set_tempo':
            bpm_ls.append((np.round(mido.tempo2bpm(meta_msg.tempo),2), meta_msg.time))
        # elif meta_msg.type == 'time_signature':
            # notated_32nd_notes_per_beat = meta_msg.notated_32nd_notes_per_beat
            # print((meta_msg.numerator, meta_msg.denominator), meta_msg.time)
            # print(meta_msg.clocks_per_click)
            # print(notated_32nd_notes_per_beat)
    return bpm_ls

#获取名称信息
def get_pop_name(n):
    df = pd.read_excel('POP909/index.xlsx')
    df.set_index('song_id', inplace=True)
    return df.loc[int(n)]


def id2path(n):
    pop_id = str(n).zfill(3)
    return 'POP909/'+ pop_id +'/'+ pop_id +'.mid' # midi路径

def get_pop_melody_df(n):
    df_ls = get_event_df_ls(id2path(n))
    return df_ls[0].iloc[:,:5]

def get_pop_df(n, track_i=0):
    df_ls = get_event_df_ls(id2path(n))
    return df_ls[track_i].iloc[:,:5]

def get_pop_info(n):
    _, meta_msg_ls = get_event_ls_ls(id2path(n), range(1,4))
    bpm_ls = get_meta_info(id2path(n))
    pop_name = get_pop_name(n)
    return pop_name, bpm_ls, meta_msg_ls

def print_pop_info(i):
    pop_name, bpm_ls, meta_msg_ls = get_pop_info(i)
    pop_info = pop_name.copy()
    pop_info['ticks_per_beat'] = MidiFile(id2path(i)).ticks_per_beat
    pop_info['first_bpm'] = bpm_ls[0][0]
    print('<<<<<<<<<<<<<<<<<<<<<<<<',i,': ',pop_name['name'],'>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(pop_info,'\n')
    print(bpm_ls,'\n')

def change_TPB_df(df, new_ticks_per_beat = 24):
    out_df = df.copy()
    tick_rescale_rate = new_ticks_per_beat/480
    out_df.note_off_tick = np.round(out_df.note_off_tick*tick_rescale_rate).astype(int)
    out_df.note_on_tick = np.round(out_df.note_on_tick*tick_rescale_rate).astype(int)
    out_df.dur = np.round(out_df.dur*tick_rescale_rate).astype(int)
    return out_df


def adjust_res(x, scale_rate):
    return int(np.round(x*scale_rate))

def DF2PR(df, on_weight = 0.8, dur_weight = 0.2):
    pitch_range = df.pitch.max() - df.pitch.min() + 1
    tick_range = df.note_off_tick.max() - df.note_on_tick.min() + 1 + list(df.dur)[-1]
    # print(pitch_range,tick_range)

    pr = np.zeros((pitch_range, tick_range))
    for event_i, event in df.iterrows():
        pr_note_on_tick = event.note_on_tick - df.note_on_tick.min()
        pr_pitch = event.pitch - df.pitch.min()
        pr[pr_pitch, pr_note_on_tick] = on_weight
        for on_tick in range(1,event.dur):
            pr[pr_pitch, pr_note_on_tick+on_tick] = dur_weight
    return pr

def show_pr(ar, tick_on=0, x_start = 0, title = 'Pianoroll', size = 'xl', set_figure=1):
    if set_figure:
        if size=='xl':
            plt.figure(figsize = (50,50))
        elif size=='s':
            plt.figure(figsize = (5,5))
        else:
            plt.figure()

    plt.imshow(ar, cmap='gray')
    plt.gca().invert_yaxis()
    x_interval = int(ar.shape[1]*0.01)
    if x_interval==0: x_interval=1
    plt.title(title)

    if tick_on:
        x_tick_values = range(0, ar.shape[1], x_interval)
        x_tick_labels = range(x_start, x_start+ar.shape[1], x_interval)
        plt.gca().xaxis.set_major_locator(FixedLocator(x_tick_values))
        plt.gca().xaxis.set_major_formatter(FixedFormatter(x_tick_labels))
        for tick_label in plt.gca().xaxis.get_ticklabels():
            tick_label.set_rotation(-45)  # 设置旋转角度
            # tick_label.set_ha('right')    # 靠右对齐
        plt.yticks(range(0, ar.shape[0], 1))
    plt.grid(color='lightgrey', alpha = 0.2)

def get_pr(id, Note_on_weight=1, Dur_weight=0.1, disp=0):
    df = get_pop_melody_df(id)
    df = change_TPB_df(df)
    pr = DF2PR(df, on_weight=Note_on_weight, dur_weight=Dur_weight)
    pr = np.pad(pr, ((0, PR_height - pr.shape[0]), (0, 0)), mode='constant')
    if disp: show_pr(pr[:, :500])
    return pr

def get_max_prange(max_id=100):
    max_prange = 0
    for id in range(1,max_id+1):
        # print(id)
        prange = get_pr(id, Note_on_weight, Dur_weight).shape[0]
        if (prange>max_prange):
            max_prange = prange
    return max_prange


def freq2pitch(freq):
    pitch = 12 * np.log2(freq/440)+69
    return int(pitch)

def pitch2freq(pitch):
    freq = 440 * 2 ** ((pitch-69)/12)
    return round(freq,2)

def freq2pitch_float(freq):
    pitch = 12 * np.log2(freq/440)+69
    return pitch

def pitch2freq_float(pitch):
    freq = 440 * 2 ** ((pitch-69)/12)
    return np.round(freq,2)

def generate_pitch_map():
    pitch_map = {}
    base_pitch = 0  # MIDI pitch of the first A0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i in range(21,88+21):  # There are 88 keys on a piano
        note_name = note_names[i % 12]
        octave = i // 12 - 1
        pitch = base_pitch + i
        pitch_map[f"{note_name}{octave}"] = pitch
    return pitch_map

name2pitch_dict = generate_pitch_map()
notes_all = list(name2pitch_dict.keys())

def name_lat2sharp(input_string):
    # 创建映射字典
    mapping = {
        'Db': 'C#',
        'Eb': 'D#',
        'Gb': 'F#',
        'Bb': 'A#'
    }

    # 提取前缀和数字部分
    prefix = input_string[:-1]
    suffix = input_string[-1]

    # 如果有映射，进行映射
    if prefix in mapping:
        output_string = mapping[prefix] + suffix
        return output_string
    else:
        # 无映射，返回原字符串
        return input_string


def name2pitch(name):
    name = name_lat2sharp(name)
    return name2pitch_dict[name]


def pitch2name(pitch):
    return notes_all[pitch-21]

def name2freq(note):
    return pitch2freq(name2pitch(note))





def plot_track(event_df_ls, track_range=range(1), tick_range=()):
    plt.figure(figsize=(35, 40))
    for track_i in track_range:
        # 选择旋律轨道
        event_df = event_df_ls[track_i]

        # 音高子图
        ax = plt.subplot(6, 1, 2 * track_i + 1)
        plt.xlim(tick_range)
        plt.xticks(np.arange(tick_range[0], tick_range[1], int((tick_range[1] - tick_range[0]) / 15)))

        plt.ylim(event_df.pitch.min() - 3, event_df.pitch.max() + 3)
        pitch_ticks = np.arange(event_df.pitch.min() - 3, event_df.pitch.max() + 3, 1)
        note_ticks = [pitch2name(p) for p in pitch_ticks]
        plt.yticks(pitch_ticks, note_ticks)

        for i, note in event_df.iterrows():
            # 绿色绘制音高
            rect = plt.Rectangle((note.note_on_tick, note.pitch - 0.5), note.dur, 1, facecolor='g',
                                 edgecolor=(0, 0, 0, 0.5), alpha=0.7)
            ax.add_patch(rect)