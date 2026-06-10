import multiprocessing
import queue

import numpy as np
from enum import StrEnum

from typing import List

class IncomingMessageType(StrEnum):
    SET_SAMPLE = 'set_sample'
    LOAD_PROJECT_DATA = 'load_project_data'

class OutgoingMessageType(StrEnum):
    SET_PROJECT_FOLDER = 'set_project_folder'
    SET_EDIT_WINDOW = 'set_edit_window'
    SET_ACTIVE_SNAPSHOT = 'set_active_snapshot'

def _visualization_worker(ui_process_send_queue: multiprocessing.Queue, ui_process_recv_queue: multiprocessing.Queue):
    import tkinter
    import tkinter.filedialog

    from matplotlib.figure import Figure
    from matplotlib.animation import FuncAnimation


    # Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                NavigationToolbar2Tk)
    
    import pathlib

    root = tkinter.Tk()
    root.wm_title("Sample Editor")

    wave_graph = Figure(figsize=(5, 4), dpi=100)

    canvas = FigureCanvasTkAgg(wave_graph, master=root)  # A tk.DrawingArea.
    canvas.draw()

        # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()

    canvas.mpl_connect("key_press_event", key_press_handler)

    window_menu = tkinter.Menu(root)

    sample_snapshots = []
    active_snapshot_index = None

    wave_axes = None
    wave_artist = None
    existing_x_bounds = None
    def _plot_sample(sample: np.ndarray):
        nonlocal wave_axes, wave_artist, existing_x_bounds, canvas
        if wave_axes is None:
            wave_axes = wave_graph.gca()
            wave_artist = wave_axes.plot(np.arange(sample.shape[0]), sample)[0]
            new_x_bound_min, new_x_bound_max = wave_axes.get_xbound()
            existing_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))

        try:
            wave_artist.set_ydata(sample)
            canvas.draw()
        except BaseException as ex:
            pass

    def _set_project_folder():
        dialog_result = tkinter.filedialog.askdirectory()

        if dialog_result is not None and dialog_result != ():
            current_project_folder = pathlib.Path(dialog_result)
            ui_process_send_queue.put_nowait((OutgoingMessageType.SET_PROJECT_FOLDER, current_project_folder))

            print('Current project folder set to ' + str(current_project_folder))

    def _undo_snapshot():
        nonlocal active_snapshot_index
        if active_snapshot_index is not None and active_snapshot_index >= 1:
            active_snapshot_index -= 1
            ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, active_snapshot_index))
            _plot_sample(sample_snapshots[active_snapshot_index])

    def _redo_snapshot():
        nonlocal active_snapshot_index
        if active_snapshot_index is not None and active_snapshot_index < len(sample_snapshots) - 1:
            active_snapshot_index += 1
            ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, active_snapshot_index))
            _plot_sample(sample_snapshots[active_snapshot_index])

    window_project_menu = tkinter.Menu(root)
    window_project_menu.add_command(label='Set Project Folder...', command=_set_project_folder)
    window_menu.add_cascade(label='Project', menu=window_project_menu)
    window_edit_menu = tkinter.Menu(root)
    window_edit_menu.add_command(label='Undo (Use Previous Snapshot)', command=_undo_snapshot, accelerator='Ctrl+Z')
    window_edit_menu.add_command(label='Redo (Use Next Snapshot)', command=_redo_snapshot, accelerator='Ctrl+Y')
    window_menu.add_cascade(label='Edit', menu=window_edit_menu)
    root.configure(menu=window_menu)

    def draw_wave(_):
        nonlocal sample_snapshots, active_snapshot_index, existing_x_bounds
        new_msg = None
        try:
            new_msg = ui_process_recv_queue.get_nowait()

            if new_msg[0] == IncomingMessageType.SET_SAMPLE:
                _, new_sample, is_checkpoint = new_msg

                _plot_sample(new_sample)

                if is_checkpoint:
                    sample_snapshots.append(new_sample)
                    active_snapshot_index = len(sample_snapshots) - 1
            elif new_msg[0] == IncomingMessageType.LOAD_PROJECT_DATA:
                _, project_samples = new_msg
                sample_snapshots = project_samples
                active_snapshot_index = len(project_samples) - 1
                _plot_sample(sample_snapshots[active_snapshot_index])
        except queue.Empty:
            pass

        new_x_bound_min, new_x_bound_max = wave_axes.get_xbound()
        new_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))
        if new_x_bounds != existing_x_bounds:
            existing_x_bounds = new_x_bounds
            ui_process_send_queue.put_nowait((OutgoingMessageType.SET_EDIT_WINDOW, existing_x_bounds))

        return wave_artist

    wave_anim = FuncAnimation(wave_graph, draw_wave)
    wave_anim.resume()

    toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
    tkinter.mainloop()

class SampleEditorNativeUI:
    def __init__(self):
        self.ui_process_send_queue = multiprocessing.Queue()
        self.ui_process_recv_queue = multiprocessing.Queue()
        self.vis_process = multiprocessing.Process(target=_visualization_worker, args=(self.ui_process_send_queue, self.ui_process_recv_queue))
        self.vis_process.start()

    def load_project_data(self, project_samples: List[np.ndarray]):
        self.ui_process_recv_queue.put_nowait((IncomingMessageType.LOAD_PROJECT_DATA, project_samples))

    def update_current_sample(self, sample: np.ndarray, is_checkpoint: bool, checkpoint_desc: str):
        self.ui_process_recv_queue.put_nowait((IncomingMessageType.SET_SAMPLE, sample, is_checkpoint))

    def get_events(self):
        try:
            while True:
                yield self.ui_process_send_queue.get_nowait()
        except queue.Empty:
            pass