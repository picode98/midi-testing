import multiprocessing
import threading
import queue

import numpy as np
from enum import StrEnum

from typing import List, Tuple

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

    class SnapshotHistoryWidget(tkinter.Frame):
        def __init__(self, master, height: int):
            super().__init__(master, height=height)
            self.history_widgets: List[Tuple[Figure, FigureCanvasTkAgg]] = []
            self.height = height

            self.scrolling_canvas = tkinter.Canvas(self, height=height)
            self.scrolling_canvas.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)

            self.scrollbar = tkinter.Scrollbar(self, orient=tkinter.HORIZONTAL, command=self.scrolling_canvas.xview)
            self.scrollbar.pack(side=tkinter.BOTTOM, fill=tkinter.X, expand=False)

            # def _on_canvas_scroll(start, end):
            #     self.scrolling_canvas.xview(tkinter.SCROLL, 3, 'units')
            #     self.scrollbar.set(start, end)
            #     print((start, end))

            self.scrolling_canvas.configure(yscrollcommand=self.scrollbar.set)

            self.scrolling_frame = tkinter.Frame(self.scrolling_canvas)
            self.scrolling_canvas.create_window((0, 0), window=self.scrolling_frame, anchor='nw')
            self.scrolling_frame.bind("<Configure>", lambda e: self.scrolling_canvas.configure(scrollregion=self.scrolling_canvas.bbox("all")))

            self.active_index = None

        def set_active(self, index: int):
            if self.active_index is not None:
                self.history_widgets[self.active_index][1].get_tk_widget().configure(highlightthickness=0)

            self.history_widgets[index][1].get_tk_widget().configure(highlightthickness=2, highlightcolor='blue')
            self.active_index = index

        def add_entry(self, entry: np.ndarray):
            new_figure = Figure(figsize=(1.5 * self.height / 100, self.height / 100), dpi=100)
            new_figure.gca().plot(np.arange(entry.shape[0]), entry)

            new_canvas = FigureCanvasTkAgg(new_figure, master=self.scrolling_frame)
            new_canvas.get_tk_widget().pack(side=tkinter.LEFT, expand=False, padx=5)
            new_canvas.draw()

            def _on_click(event, index=len(self.history_widgets)):
                print('Setting ' + str(index) + ' to active.')
                self.set_active(index)
                self.event_generate('<<ActiveChanged>>')

            new_canvas.mpl_connect('button_release_event', _on_click)

            self.history_widgets.append((new_figure, new_canvas))

            self.scrolling_canvas.xview_moveto(1.0 - self.scrolling_canvas.xview()[1])
            self.scrollbar.set(*self.scrolling_canvas.xview())

        def clear_entries(self):
            for figure, widget in self.history_widgets:
                widget.get_tk_widget().pack_forget()

            self.history_widgets.clear()
            self.active_index = None

    class SampleEditorNativeUIApplication(tkinter.Tk):
        def __init__(self):
            super().__init__()
            self.wm_title("Sample Editor")

            wave_graph = Figure(figsize=(5, 4), dpi=100)

            self.active_wave_graph = FigureCanvasTkAgg(wave_graph, master=self)  # A tk.DrawingArea.
            self.active_wave_graph.draw()

                # pack_toolbar=False will make it easier to use a layout manager later on.
            self.active_wave_toolbar = NavigationToolbar2Tk(self.active_wave_graph, self, pack_toolbar=False)
            self.active_wave_toolbar.update()

            self.active_wave_graph.mpl_connect("key_press_event", key_press_handler)
            self.active_wave_graph.mpl_connect("button_release_event", self._on_active_plot_moved)

            self.history_view = SnapshotHistoryWidget(self, 100)
            self.history_view.bind('<<ActiveChanged>>', self._on_sample_history_selection)

            self.window_menu = tkinter.Menu(self)

            self.window_project_menu = tkinter.Menu(self)
            self.window_project_menu.add_command(label='Set Project Folder...', command=self._set_project_folder)
            self.window_menu.add_cascade(label='Project', menu=self.window_project_menu)
            self.window_edit_menu = tkinter.Menu(self)
            self.window_edit_menu.add_command(label='Undo (Use Previous Snapshot)', command=self._undo_snapshot, accelerator='Ctrl+Z')
            self.window_edit_menu.add_command(label='Redo (Use Next Snapshot)', command=self._redo_snapshot, accelerator='Ctrl+Y')
            self.window_menu.add_cascade(label='Edit', menu=self.window_edit_menu)
            self.configure(menu=self.window_menu)

            self.history_view.pack(side=tkinter.BOTTOM, fill=tkinter.X)
            self.active_wave_toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
            self.active_wave_graph.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

            self.sample_snapshots = []
            self.active_snapshot_index = None

            self.wave_axes = None
            self.wave_artist = None
            self.existing_x_bounds = None

            self.synth_handlers = {IncomingMessageType.LOAD_PROJECT_DATA: self._on_project_load, IncomingMessageType.SET_SAMPLE: self._on_sample_set}
            self.queue_pump_thread = threading.Thread(target=self._invoke_synth_handlers, name='Queue Pump Thread')
            self.queue_pump_thread.start()


        def _invoke_synth_handlers(self):
            while True:
                event_data = ui_process_recv_queue.get()
                self.after('idle', self.synth_handlers[event_data[0]], *event_data[1:])

        def _plot_sample(self, sample: np.ndarray):
            if self.wave_axes is None:
                self.wave_axes = self.active_wave_graph.figure.gca()
                self.wave_artist = self.wave_axes.plot(np.arange(sample.shape[0]), sample)[0]
                new_x_bound_min, new_x_bound_max = self.wave_axes.get_xbound()
                self.existing_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))

            self.wave_artist.set_ydata(sample)
            self.active_wave_graph.draw()

        def _on_project_load(self, project_samples: List[np.ndarray]):
            self.sample_snapshots = project_samples
            self.active_snapshot_index = len(project_samples) - 1
            self._plot_sample(self.sample_snapshots[self.active_snapshot_index])

            self.history_view.clear_entries()
            for sample in self.sample_snapshots:
                self.history_view.add_entry(sample)

            self.history_view.set_active(self.active_snapshot_index)

        def _on_sample_history_selection(self, event):
            self.active_snapshot_index = self.history_view.active_index
            ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, self.active_snapshot_index))
            self._plot_sample(self.sample_snapshots[self.active_snapshot_index])

        def _on_active_plot_moved(self, event):
            new_x_bound_min, new_x_bound_max = self.wave_axes.get_xbound()
            new_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))
            if new_x_bounds != self.existing_x_bounds:
                self.existing_x_bounds = new_x_bounds
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_EDIT_WINDOW, self.existing_x_bounds))

        def _on_sample_set(self, new_sample: np.ndarray, is_checkpoint: bool):
            self._plot_sample(new_sample)

            if is_checkpoint:
                self.sample_snapshots.append(new_sample)
                self.active_snapshot_index = len(self.sample_snapshots) - 1
                self.history_view.add_entry(new_sample)
                self.history_view.set_active(self.active_snapshot_index)

        def _set_project_folder(self):
            dialog_result = tkinter.filedialog.askdirectory()

            if dialog_result is not None and dialog_result != () and dialog_result != '':
                current_project_folder = pathlib.Path(dialog_result)
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_PROJECT_FOLDER, current_project_folder))

                print('Current project folder set to ' + str(current_project_folder))

        def _undo_snapshot(self):
            if self.active_snapshot_index is not None and self.active_snapshot_index >= 1:
                self.active_snapshot_index -= 1
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, self.active_snapshot_index))
                self._plot_sample(self.sample_snapshots[self.active_snapshot_index])
                self.history_view.set_active(self.active_snapshot_index)

        def _redo_snapshot(self):
            if self.active_snapshot_index is not None and self.active_snapshot_index < len(self.sample_snapshots) - 1:
                self.active_snapshot_index += 1
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, self.active_snapshot_index))
                self._plot_sample(self.sample_snapshots[self.active_snapshot_index])
                self.history_view.set_active(self.active_snapshot_index)

    # root = tkinter.Tk()
    # root.wm_title("Sample Editor")

    # def draw_wave(_):
    #     nonlocal sample_snapshots, active_snapshot_index, existing_x_bounds
    #     new_msg = None
    #     try:
    #         new_msg = ui_process_recv_queue.get_nowait()

    #         if new_msg[0] == IncomingMessageType.SET_SAMPLE:
    #             _, new_sample, is_checkpoint = new_msg

    #             _plot_sample(new_sample)

    #             if is_checkpoint:
    #                 sample_snapshots.append(new_sample)
    #                 active_snapshot_index = len(sample_snapshots) - 1
    #                 history_view.add_entry(new_sample)
    #                 history_view.set_active(active_snapshot_index)
    #         elif new_msg[0] == IncomingMessageType.LOAD_PROJECT_DATA:
    #             _, project_samples = new_msg
    #             sample_snapshots = project_samples
    #             active_snapshot_index = len(project_samples) - 1
    #             _plot_sample(sample_snapshots[active_snapshot_index])

    #             history_view.clear_entries()
    #             for sample in sample_snapshots:
    #                 history_view.add_entry(sample)

    #             history_view.set_active(active_snapshot_index)
    #     except queue.Empty:
    #         pass

    #     new_x_bound_min, new_x_bound_max = wave_axes.get_xbound()
    #     new_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))
    #     if new_x_bounds != existing_x_bounds:
    #         existing_x_bounds = new_x_bounds
    #         ui_process_send_queue.put_nowait((OutgoingMessageType.SET_EDIT_WINDOW, existing_x_bounds))

    #     return wave_artist

    # wave_anim = FuncAnimation(wave_graph, draw_wave)
    # wave_anim.resume()

    application = SampleEditorNativeUIApplication()
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