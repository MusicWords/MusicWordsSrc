import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import pandas as pd
from music21 import converter, chord, tempo, meter
import mido
from mido import Message, MidiFile, MidiTrack
import MYDI

# 全局参数
NOTE_ON_WEIGHT = 1  # 音头的赋值
DUR_WEIGHT = 0.1  # 音头的赋值
DPI = 200
TPB = 12


def get_tpb(path):
    return mido.MidiFile(path).ticks_per_beat


def get_event_ls_ls(path):
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

        for i, msg in enumerate(msg_ls):
            abs_time = abs_time + msg.time

            if (msg.type != 'note_on') and (msg.type != 'note_off'):
                orig_other_msg_index_ls.append(i)
                orig_other_msg_ls.append(msg)
                continue

            if msg.velocity != 0 and msg.type == 'note_on':
                pitch_on = msg.note

                pitch_ls.append(pitch_on)
                vel_ls.append(msg.velocity)
                head_ls.append(abs_time)
                head_rlt_ls.append(msg.time)
                head_msg_index_ls.append(i)

                unclosed_pitch.append((pitch_on, abs_time))

            if msg.velocity == 0 or msg.type == 'note_off':
                pitch_off = msg.note

                for pitch_head in unclosed_pitch:
                    if pitch_off == pitch_head[0]:
                        dur_ls.append(abs_time - pitch_head[1])
                        tail_ls.append(abs_time)
                        tail_rlt_ls.append(msg.time)
                        tail_msg_index_ls.append(i)

                        unclosed_pitch.remove(pitch_head)
                        break

        return np.array([pitch_ls, head_ls, tail_ls, dur_ls, vel_ls, head_rlt_ls, tail_rlt_ls, head_msg_index_ls,
                         tail_msg_index_ls]).T, np.array([orig_other_msg_index_ls, orig_other_msg_ls]).T

    event_ls_ls = []
    other_msg_ls_ls = []
    track_i_range = len(mido.MidiFile(path).tracks)
    for track_i in range(track_i_range):
        msg_ls = list(MidiFile(path).tracks[track_i])
        event_ls, other_msg = msg2event(msg_ls)
        event_ls_ls.append(event_ls)
        other_msg_ls_ls.append(other_msg)

    event_ls_ls
    return event_ls_ls, other_msg_ls_ls


def arr_2_df_ls(arr_ls):
    df_ls = []
    for arr in arr_ls:
        df = pd.DataFrame(arr)
        df.columns = ['pitch', 'note_on_tick', 'note_off_tick', 'dur', 'vel',
                      'head_rlt', 'tail_rlt', 'head_msg_i', 'tail_msg_i']
        df_ls.append(df)
    return df_ls


def get_event_df_ls(path):
    event_ls_ls, _ = get_event_ls_ls(path)
    return arr_2_df_ls(event_ls_ls)


def get_track_df(path, track_number=0):
    df_ls = get_event_df_ls(path)
    return df_ls[track_number].iloc[:, :5]


def change_TPB_df(df, new_ticks_per_beat=24, orig_TPB=480):
    out_df = df.copy()
    tick_rescale_rate = new_ticks_per_beat / orig_TPB
    out_df.note_off_tick = np.round(out_df.note_off_tick * tick_rescale_rate).astype(int)
    out_df.note_on_tick = np.round(out_df.note_on_tick * tick_rescale_rate).astype(int)
    out_df.dur = np.round(out_df.dur * tick_rescale_rate).astype(int)
    return out_df


def DF2PR(df, on_weight=1, dur_weight=0.2):
    pitch_range = df.pitch.max() - df.pitch.min() + 1
    tick_range = df.note_off_tick.max() + 1 + list(df.dur)[-1]

    pr = np.zeros((pitch_range, tick_range))
    for event_i, event in df.iterrows():
        pr_note_on_tick = event.note_on_tick
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
def get_file_pr(path, track_number=0, disp=0, tpb=TPB):
    df = get_track_df(path, track_number=track_number)
    df = change_TPB_df(df, new_ticks_per_beat=tpb, orig_TPB=get_tpb(path))
    pr = DF2PR(df, on_weight=NOTE_ON_WEIGHT, dur_weight=DUR_WEIGHT)
    return pr

#
# class Midi:
#     def __init__(self, path, tpb=TPB):
#         self.path = path
#         self.tpb = tpb
#         self.prettyMidi = pretty_midi.PrettyMIDI(path)
#         self.score = converter.parse(path)
#
#         # 获取所有的速度变更点
#         self.tempo_times, tempi = self.prettyMidi.get_tempo_changes()
#         # 获取每拍的时间位置（即每拍的绝对时间）
#         self.beats = self.prettyMidi.get_beats()
#         # 获取小节线的时间位置
#         self.downbeats = self.prettyMidi.get_downbeats()
#         # 获取拍号变更信息
#         self.time_signatures = self.prettyMidi.time_signature_changes
#         # 使用 music21 加载 MIDI 文件并进行调性分析
#         self.key = self.score.analyze('key')
#         # 获取音符DF
#         self.dfs = self.get_dfs()
#         self.melody_df = self.dfs[0]
#         self.bridge_df = self.dfs[1]
#         self.piano_df = self.dfs[2]
#         # 获取量化PR
#         self.prs = [DF2PR(df) for df in self.dfs]
#         self.melody_pr = self.prs[0]
#         self.bridge_pr = self.prs[1]
#         self.piano_pr = self.prs[2]
#
#     def get_dfs(self):
#         # 初始化空列表来存储音符信息
#         dfs = []
#         # 遍历每个乐器轨道，提取音符信息
#         for instrument in self.prettyMidi.instruments:
#             notes = []
#             for note in instrument.notes:
#                 notes.append({
#                     'instrument': instrument.name,  # 乐器名称
#                     'start_time': note.start,  # 音符开始时间
#                     'end_time': note.end,  # 音符结束时间
#                     'pitch': note.pitch,  # 音高
#                     'velocity': note.velocity,  # 力度
#                     'duration': note.end - note.start  # 持续时间
#                 })
#             df = pd.DataFrame(notes)
#             dfs.append(df)
#         return dfs
#
#     # def quantize_df(self, df, quantize_beat=QUANTIZE_BEAT):
#     #     def quantize_array(arr, interval):
#     #         # 将数组中的每个元素量化到最近的间隔
#     #         quantized_arr = np.round(arr / interval) * interval
#     #         return quantized_arr
#     #
#     #     out_df = df.copy()
#     #     out_df.end_time = quantize_array(out_df.end_time, interval=quantize_beat)
#     #     out_df.start_time = quantize_array(out_df.start_time, interval=quantize_beat)
#     #     out_df.duration = quantize_array(out_df.duration, interval=quantize_beat)
#     #     return out_df
#
#     def get_pr(self, df):
#         quantized_df = self.quantize_df(df)
#
#         pitch_range = df.pitch.max() - df.pitch.min() + 1
#         tick_range = df.end_time.max() - df.start_time.min() + 1
#
#         pr = np.zeros((pitch_range, tick_range))
#
#     def __str__(self):
#         return f'Midi-{self.path}'
#
#     def __repr__(self):
#         return f'Midi-{self.path}'
