import numpy as np
import MYDI
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, RectangleSelector, MultiCursor
from matplotlib.path import Path
import seaborn as sns
import pandas as pd
import uuid
import pickle
import os
plt.rc("font",family='Microsoft YaHei') # 显示中文

class template:
    def __init__(self, uid, tid, notes, template_df):
        self.uid = uid
        self.tid = tid
        self.notes = notes
        self.template_df = template_df

class SelectFromCollection:
    def __init__(self, midi_id, track_i, alpha_other=0.3, tick_range = 30000, step_size = 5000, T_dict = {0:[]}, figsize = (20, 3)):
        self.step_size = step_size
        self.alpha_other = alpha_other
        self.midi_id = midi_id
        self.midi_df = MYDI.get_pop_df(midi_id, track_i=track_i)
        self.midi_df['TemplateType'] = None
        self.midi_df['TemplateID'] = None
        self.T_dict = T_dict
        self.load_labeled_data()
        self.track_i = track_i
        self.T_disp_w = 800
        self.T_disp_h = 10

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1, autoscale_on=False, xlim = (0,tick_range), ylim=(self.midi_df.pitch.min() - 3, self.midi_df.pitch.max() + 3))  # 添加多光标
        MultiCursor(fig.canvas, (ax,), color='r', lw=0.1, horizOn=True, vertOn=True)
        ax.set_title("单击拖拽框选音形")
        pts = ax.scatter(self.midi_df.note_on_tick, self.midi_df.pitch, s=80, marker='s')
        plt.tight_layout()

        self.canvas = ax.figure.canvas
        self.collection = pts
        self.xys = pts.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = pts.get_facecolors()
        self.colors = sns.color_palette("Set1", n_colors=50)[2:]

        # self.template_ls = []
        self.num_template = len(T_dict)
        self.ax = ax

        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        # self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.lasso = RectangleSelector(ax, onselect=self.onselect_rect)
        self.ind = []

        fig.canvas.mpl_connect("key_press_event", self.key_press)
        # 将滚动事件连接到图形
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def load_labeled_data(self):
        try:
            self.midi_df = pd.read_csv(f'Labeled Songs/{self.midi_id}.csv')
            # 如果文件存在，将文件内容读取到 df 中
            print(self.midi_df)  # 这里你可以对数据框进行操作或者打印出来
            print("已加载本首音乐标注数据")
        except FileNotFoundError:
            # 如果文件不存在，会触发 FileNotFoundError 异常
            print("无已标注数据")

    # 创建一个滚动事件处理函数
    def on_scroll(self, event):
        if event.button == 'up':
            self.ax.set_xlim(self.ax.get_xlim()[0] - self.step_size, self.ax.get_xlim()[1] - self.step_size)  # 向右滚动
        elif event.button == 'down':
            self.ax.set_xlim(self.ax.get_xlim()[0] + self.step_size, self.ax.get_xlim()[1] + self.step_size)  # 向左滚动
        plt.draw()  # 重新绘制图形

    def onselect_rect(self, eclick, erelease):
        verts = [(eclick.xdata, eclick.ydata),(eclick.xdata, erelease.ydata),(erelease.xdata, erelease.ydata),(erelease.xdata, eclick.ydata)]
        path = Path(verts)
        # print('verts:',verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    # def onselect(self, verts):
    #     path = Path(verts)
    #     self.ind = np.nonzero(path.contains_points(self.xys))[0]
    #     self.fc[:, -1] = self.alpha_other
    #     self.fc[self.ind, -1] = 1
    #     self.collection.set_facecolors(self.fc)
    #     self.canvas.draw_idle()

    def draw_dict(self):
        global axs, fig1
        fig1.clf()
        # axs = fig1.subplots(self.num_template)
        axs = fig1.subplots(ncols=min(self.num_template,max_rows), nrows=self.num_template//max_rows+1).reshape(-1)
        fig1.suptitle('选择音形类别')

        for tid in range(len(self.T_dict)):
            ax = axs[tid]
            if len(self.T_dict[tid])!=0:
                t_df = self.T_dict[tid][0].template_df
                min_t = t_df.note_on_tick.min()
                min_p = t_df.pitch.min()
                range_t = t_df.note_on_tick.max() - t_df.note_on_tick.min()
                range_p = t_df.pitch.max() - t_df.pitch.min()
                if range_t < self.T_disp_w: ax.set_xlim(-self.T_disp_w*0.2,self.T_disp_w*1.2)
                if range_p < self.T_disp_h: ax.set_ylim(-self.T_disp_h*0.2,self.T_disp_h*1.2)

                ax.scatter(t_df.note_on_tick-min_t, t_df.pitch-min_p, s=80, marker='s')
                ax.text(int(self.T_disp_w*0.3), int(self.T_disp_h*0.5), f'T-{tid}', fontsize=20, color='red')
                ax.text(int(self.T_disp_w*0.3), int(self.T_disp_h*0.5+5), f'已有{len(self.T_dict[tid])}个', fontsize=12, color='red')

    def key_press(self, event):
        if event.key == "enter":
            print("Selected Notes:")
            selected_notes_t_p = self.xys[self.ind]
            selected_note_ls = []
            uid = uuid.uuid4()
            for (t,p) in selected_notes_t_p:
                condition = (self.midi_df['note_on_tick'] == t) & (self.midi_df['pitch'] == p)
                selected_note = self.midi_df.loc[condition]
                self.midi_df.loc[condition, 'TemplateType'] = selected_tid

                self.midi_df.loc[condition, 'TemplateID'] = uid

                selected_note_ls.append(selected_note.copy())
                print(selected_note)
            template_df = pd.concat(selected_note_ls, ignore_index=True)
            # print(self.midi_df)

            self.fc[:, -1] = self.alpha_other
            self.fc[self.ind, :] = [self.colors[selected_tid][0], self.colors[selected_tid][1], self.colors[selected_tid][2],1]

            # print('tid','lenT',selected_tid+1 ,len(self.T_dict))
            if selected_tid+1 == len(self.T_dict):
                self.T_dict[self.num_template] = []
                self.num_template = self.num_template+1

                new_template = template(uid, selected_tid, selected_note_ls, template_df)
                self.T_dict[selected_tid].append(new_template)

            else:
                new_template = template(uid, selected_tid, selected_note_ls, template_df)
                self.T_dict[selected_tid].append(new_template)

            self.draw_dict()
            self.ax.set_title(f"已添加音形 类型: {selected_tid} 音符数: {len(selected_note_ls)}")
            print("已添加音形 ")
            self.collection.set_facecolors(self.fc)
            self.canvas.draw_idle()
            event.canvas.draw()
            fig1.canvas.draw()
            plt.show()

        elif event.key == "t":
            try:
                os.makedirs('Labeled Songs')
            except FileExistsError:
                pass
            self.midi_df.to_csv(f'Labeled Songs/{self.midi_id}-{self.track_i}.csv', index=False)  # index=False 表示不保存行索引

            # 保存字典到文件
            # if len(list(self.T_dict)[-1])==0: self.T_dict.popitem()
            with open(f'T_dict-{self.track_i}.pkl', 'wb') as file:
                pickle.dump(self.T_dict, file)

            self.ax.set_title("已保存！")
            print("已保存")

        elif event.key == "L":
            self.load_labeled_data()
            self.ax.set_title("已加载之前保存的标注进度")
            print("已加载之前保存的标注进度")
            print(self.midi_df)

        elif event.key.isdigit():
            select_template(int(event.key))

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
        note_ticks = [MYDI.pitch2name(p) for p in pitch_ticks]
        plt.yticks(pitch_ticks, note_ticks)

        for i, note in event_df.iterrows():
            # 绿色绘制音高
            rect = plt.Rectangle((note.note_on_tick, note.pitch - 0.5), note.dur, 1, facecolor='g',
                                 edgecolor=(0, 0, 0, 0.5), alpha=0.7)
            ax.add_patch(rect)

    # ----------------------------------------------------------------
def enter_axes(event):
    event.inaxes.patch.set_facecolor('yellow')
    event.canvas.draw()

def leave_axes(event):
    if type(fig1.get_axes())==list:
        if fig1.get_axes().index(event.inaxes) != selected_tid:
            event.inaxes.patch.set_facecolor('white')
            event.canvas.draw()

def select_template(type_id):
    global axs
    axs[type_id].patch.set_facecolor('blue')
    global selected_tid
    selected_tid = type_id
    fig1.suptitle('已选择模板类型：' + str(selected_tid))

    for ax in axs:
        if ax != axs[type_id]:
            ax.set_facecolor('white')

    fig1.canvas.draw()

def press_sub(event):
    if event.inaxes:
        print(event.inaxes)
        event.inaxes.patch.set_facecolor('blue')
        global selected_tid
        selected_tid = fig1.get_axes().index(event.inaxes)
        fig1.suptitle('已选择模板类型：'+str(selected_tid))

        for ax in axs:
            if ax != event.inaxes:
                ax.set_facecolor('white')

        event.canvas.draw()


if __name__ == '__main__':
    label_song_id = 2 # 歌曲编号 1~909
    label_track_id = 0 # 轨道编号 0-Melody; 1-Bridge; 2-Piano
    tick_range = 17000 # 显示宽度
    pianoroll_figsize = (15,3) # 窗口大小
    max_rows = 5
    # 初始化
    selected_tid = None
    loaded_T_dict = {0: []}
    # 加载已标注的字典
    try:
        with open(f'T_dict-{label_track_id}.pkl', 'rb') as file:
            loaded_T_dict = pickle.load(file)
    except FileNotFoundError:
        print("文件不存在，无法加载字典。")
    except Exception as e:
        print(f"发生了其他错误: {str(e)}")

    # 标注器
    selector = SelectFromCollection(midi_id=label_song_id, track_i=label_track_id, T_dict=loaded_T_dict, tick_range=tick_range, figsize = pianoroll_figsize)

    # ----------------------------------------------------------------
    # 选择器
    fig1, axs = plt.subplots(ncols=min(len(loaded_T_dict),max_rows), nrows=len(loaded_T_dict)//max_rows+1, figsize = (19,5))

    fig1.canvas.mpl_connect('button_press_event', press_sub)
    fig1.canvas.mpl_connect('axes_enter_event', enter_axes)
    fig1.canvas.mpl_connect('axes_leave_event', leave_axes)
    fig1.canvas.mpl_connect("key_press_event", selector.key_press)
    if type(axs) == np.ndarray: selector.draw_dict()
    plt.show()