import multiprocessing
import threading
import queue
import pathlib

import numpy as np
from enum import StrEnum

from sample_editor_effect import EffectSettings, SampleEffect

from typing import List, Tuple, Dict, Optional


class IncomingMessageType(StrEnum):
    SET_SAMPLE = 'set_sample'
    LOAD_PROJECT_DATA = 'load_project_data'
    UPDATE_RECORDING_STATE = 'update_recording_state'

class OutgoingMessageType(StrEnum):
    SET_PROJECT_FOLDER = 'set_project_folder'
    SET_EDIT_WINDOW = 'set_edit_window'
    SET_ACTIVE_SNAPSHOT = 'set_active_snapshot'
    SET_EFFECT_SETTING_VALUE = 'set_effect_setting_value'
    SET_SAMPLE_SIZE = 'set_sample_size'
    RECORD_START = 'record_start'
    RECORD_PAUSE = 'record_pause'
    RECORD_CONTINUE = 'record_continue'
    RECORD_STOP = 'record_stop'
    APPLICATION_EXIT = 'application_exit'

class WavelengthAddMode(StrEnum):
    ADD_SINE_WAVE = 'add_sine_wave'
    ADD_SILENCE = 'add_silence'
    ADD_LOOP = 'add_loop'

def _visualization_worker(ui_process_send_queue: multiprocessing.Queue, ui_process_recv_queue: multiprocessing.Queue, effect_settings_info: Dict[str, Dict[str, EffectSettings]], initial_setting_values: Dict[str, Dict[str, int | float | str]]):
    import wx
    import wx.svg
    import wx.lib.newevent
    import wx.lib.scrolledpanel

    from matplotlib.figure import Figure


    # Implement the default Matplotlib key bindings.
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.backends.backend_wxagg import (FigureCanvasWxAgg, NavigationToolbar2WxAgg)

    ActiveItemChangedEvent, EVT_ACTIVE_ITEM_CHANGED = wx.lib.newevent.NewEvent()

    class SVGButton(wx.Button):
        def __init__(self, parent: wx.Window, id: int, label: str, svg_filepath: pathlib.Path) -> None:
            super().__init__(parent, id, label)

            self.set_image(svg_filepath)
            # self.Bind(wx.EVT_SIZE, lambda event: self.render_image())

        def set_image(self, svg_filepath: pathlib.Path):
            # dc = wx.PaintDC(self)
            # dc.SetBackground(wx.Brush(wx.Colour('white')))
            # dc.Clear()

            self.img: wx.svg.SVGimage = wx.svg.SVGimage.CreateFromFile(str(svg_filepath))
            self.render_image()

        def render_image(self):
            img_size = self.GetSize()
            img_size.IncBy(-10, -10)
            rendered_bitmap = self.img.ConvertToScaledBitmap(img_size)
            self.SetBitmap(wx.BitmapBundle(rendered_bitmap))

            # ctx = wx.GraphicsContext.Create(dc)
            # self.img.RenderToGC(ctx, scale)

    class SnapshotHistoryWidget(wx.lib.scrolledpanel.ScrolledPanel):
        def __init__(self, parent, height: int):
            super().__init__(parent)
            self.history_widgets: List[Tuple[Figure, FigureCanvasWxAgg]] = []
            self.height = height

            self.item_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
            self.SetSizer(self.item_sizer)

            self.SetupScrolling(scroll_x=True, scroll_y=False)

            self.active_index = None

        def set_active(self, index: int):
            # if self.active_index is not None:
            #     self.history_widgets[self.active_index][1].SetWindowStyle(wx.BORDER_SIMPLE)

            # self.history_widgets[index][1].get_tk_widget().configure(highlightthickness=2, highlightcolor='blue')
            self.active_index = index

        def add_entry(self, sample: np.ndarray, sample_desc: str):
            new_figure = Figure(figsize=(1.5 * self.height / 100, self.height / 100), dpi=100)
            new_axes = new_figure.gca()
            new_axes.set_axis_off()
            new_axes.plot(np.arange(sample.shape[0]), sample)

            desc_text = wx.StaticText(self, wx.ID_ANY, label=sample_desc, style=wx.ALIGN_CENTRE_HORIZONTAL)
            desc_text.Wrap(int(1.5 * self.height))

            item_label_sizer = wx.BoxSizer(orient=wx.VERTICAL)
            item_label_sizer.Add(desc_text, wx.SizerFlags().Align(wx.CENTER))

            new_canvas = FigureCanvasWxAgg(self, wx.ID_ANY, figure=new_figure)
            new_canvas.draw()

            item_label_sizer.Add(new_canvas)
            self.item_sizer.Add(item_label_sizer)

            def _on_click(event, index=len(self.history_widgets)):
                print('Setting ' + str(index) + ' to active.')
                self.set_active(index)
                wx.PostEvent(self.GetEventHandler(), ActiveItemChangedEvent(index=index))

            new_canvas.mpl_connect('button_release_event', _on_click)

            wx.CallLater(1000, lambda: new_canvas._on_size(None))

            self.history_widgets.append((new_figure, new_canvas))

        def clear_entries(self):
            self.item_sizer.Clear()
            self.history_widgets.clear()
            self.active_index = None

    class ModifyNumWavelengthsDialog(wx.Dialog):
        def __init__(self, parent, current_wavelengths: int):
            super().__init__(parent, wx.ID_ANY, title='Modify Wavelengths')

            self.result_wavelengths: Optional[int] = None
            self.result_add_mode: Optional[WavelengthAddMode] = None

            self.form_sizer = wx.GridSizer(cols=2, gap=wx.Size(5, 5))
            self.form_sizer.Add(wx.StaticText(self, wx.ID_ANY, 'Number of wavelengths:'))
            self.num_wavelengths_entry = wx.SpinCtrl(self, wx.ID_ANY, initial=current_wavelengths, min=1, max=(2 ** 31 - 1))
            self.form_sizer.Add(self.num_wavelengths_entry)
            self.form_sizer.Add(wx.StaticText(self, wx.ID_ANY, 'Fill additional wavelengths with:'))
            self.fill_option_sine_button = wx.RadioButton(self, wx.ID_ANY, label='Sine wave', style=wx.RB_GROUP)
            self.fill_option_silence_button = wx.RadioButton(self, wx.ID_ANY, label='Silence')
            self.fill_option_silence_button.SetValue(True)
            self.fill_option_loop_button = wx.RadioButton(self, wx.ID_ANY, label='Loop of the current sample')
            self.form_sizer.Add(self.fill_option_sine_button)
            self.form_sizer.Add(wx.Size(0, 0))
            self.form_sizer.Add(self.fill_option_silence_button)
            self.form_sizer.Add(wx.Size(0, 0))
            self.form_sizer.Add(self.fill_option_loop_button)

            self.main_sizer = wx.BoxSizer(orient=wx.VERTICAL)
            self.btn_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
            self.ok_btn = wx.Button(self, wx.ID_OK, label='OK')
            self.ok_btn.Bind(wx.EVT_BUTTON, self._on_OK_click)
            self.cancel_btn = wx.Button(self, wx.ID_CANCEL, label='Cancel')
            self.cancel_btn.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL))
            self.btn_sizer.AddStretchSpacer(prop=1)
            self.btn_sizer.Add(self.ok_btn)
            self.btn_sizer.AddSpacer(5)
            self.btn_sizer.Add(self.cancel_btn)

            self.main_sizer.Add(self.form_sizer, wx.SizerFlags().Expand().Border(wx.ALL, 5))
            self.main_sizer.Add(self.btn_sizer, wx.SizerFlags().Expand().Border(wx.ALL, 5))
            self.SetSizerAndFit(self.main_sizer)

        def _on_OK_click(self, event):
            self.result_wavelengths = self.num_wavelengths_entry.GetValue()
            self.result_add_mode = (WavelengthAddMode.ADD_SINE_WAVE if self.fill_option_sine_button.GetValue() else (WavelengthAddMode.ADD_SILENCE if self.fill_option_silence_button.GetValue() else WavelengthAddMode.ADD_LOOP))
            self.EndModal(wx.ID_OK)

    class SampleEditorNativeUIApplication(wx.App):
        def OnInit(self):
            self.main_window = wx.Frame(None, wx.ID_ANY, title='Sample Editor')

            wave_graph = Figure(figsize=(5, 4), dpi=100)

            self.active_wave_graph = FigureCanvasWxAgg(self.main_window, wx.ID_ANY, figure=wave_graph)
            self.active_wave_graph.draw()

                # pack_toolbar=False will make it easier to use a layout manager later on.
            self.active_wave_toolbar = NavigationToolbar2WxAgg(self.active_wave_graph)
            self.active_wave_toolbar.update()

            self.active_wave_last_drawn_num_wavelengths = None
            self.active_wave_last_drawn_sample_length = None

            self.bottom_toolbar_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
            self.active_wave_num_wavelengths_text = wx.StaticText(self.main_window, wx.ID_ANY, label='[unknown] wavelengths')
            self.active_wave_modify_wavelengths_button = wx.Button(self.main_window, wx.ID_ANY, label='Modify...')
            self.active_wave_modify_wavelengths_button.Bind(wx.EVT_BUTTON, self._on_num_wavelengths_button_click)

            self.bottom_toolbar_sizer.Add(self.active_wave_toolbar, proportion=1)
            self.bottom_toolbar_sizer.Add(self.active_wave_num_wavelengths_text, wx.SizerFlags().CenterVertical())
            self.bottom_toolbar_sizer.AddSpacer(5)
            self.bottom_toolbar_sizer.Add(self.active_wave_modify_wavelengths_button, wx.SizerFlags().CenterVertical())

            self.active_wave_graph.mpl_connect("key_press_event", key_press_handler)
            self.active_wave_graph.mpl_connect("button_release_event", self._on_active_plot_moved)

            self.history_view = SnapshotHistoryWidget(self.main_window, 100)
            self.history_view.Bind(EVT_ACTIVE_ITEM_CHANGED, self._on_sample_history_selection)

            self.window_menu = wx.MenuBar()

            self.window_project_menu = wx.Menu()
            self.main_window.Bind(wx.EVT_MENU, lambda _: self._set_project_folder(), self.window_project_menu.Append(-1, item='Set Project Folder...'))
            self.window_menu.Append(self.window_project_menu, 'Project')
            self.window_edit_menu = wx.Menu()
            self.main_window.Bind(wx.EVT_MENU, lambda _: self._undo_snapshot(), self.window_edit_menu.Append(-1, item='Undo (Use Previous Snapshot)\tCtrl+Z'))
            self.main_window.Bind(wx.EVT_MENU, lambda _: self._redo_snapshot(), self.window_edit_menu.Append(-1, item='Redo (Use Next Snapshot)\tCtrl+Y'))
            self.window_menu.Append(self.window_edit_menu, 'Edit')
            self.main_window.SetMenuBar(self.window_menu)

            self.sample_view_sizer = wx.BoxSizer(orient=wx.VERTICAL)
            self.sample_view_sizer.Add(self.active_wave_graph, wx.SizerFlags(proportion=1).Expand())
            self.sample_view_sizer.Add(self.bottom_toolbar_sizer, wx.SizerFlags().Expand())
            self.sample_view_sizer.Add(self.history_view, wx.SizerFlags().Expand())

            self.effect_setting_controls: Dict[str, Dict[str, wx.TextCtrl | wx.SpinCtrl | wx.SpinCtrlDouble]] = dict()
            self.effect_settings_container = wx.Notebook(self.main_window, wx.ID_ANY)
            self.effect_page_indices: Dict[str, int] = dict()
            for idx, (effect_name, effect_settings) in enumerate(sorted(effect_settings_info.items(), key=lambda x: x[0])):
                new_page = wx.NotebookPage(self.effect_settings_container, wx.ID_ANY)
                new_page_sizer = wx.BoxSizer(orient=wx.VERTICAL)
                new_page_form_sizer = wx.GridSizer(2, wx.Size(5, 5))

                self.effect_setting_controls[effect_name] = dict()
                for setting_key, setting in sorted(effect_settings.items(), key=lambda x: x[1].setting_name):
                    new_page_form_sizer.Add(wx.StaticText(new_page, wx.ID_ANY, setting.setting_name))
                    if setting.data_type == int:
                        text_input = wx.SpinCtrl(new_page, wx.ID_ANY, initial=initial_setting_values[effect_name][setting_key], min=(1 - 2 ** 31 if setting.range_min is None else setting.range_min),
                                                 max=(2 ** 31 if setting.range_max is None else setting.range_max))
                    elif setting.data_type == float:
                        text_input = wx.SpinCtrlDouble(new_page, wx.ID_ANY, initial=initial_setting_values[effect_name][setting_key], min=(1 - 2 ** 31 if setting.range_min is None else setting.range_min),
                                                       max=(2 ** 31 if setting.range_max is None else setting.range_max), inc=0.01)
                    else:
                        text_input = wx.TextCtrl(new_page, wx.ID_ANY, initial_setting_values[effect_name][setting_key])

                    text_input.Bind(wx.EVT_TEXT, lambda event, input_ctl=text_input, effect_name=effect_name, setting_key=setting_key: self._on_effect_setting_changed(input_ctl, effect_name, setting_key))
                    new_page_form_sizer.Add(text_input)
                    self.effect_setting_controls[effect_name][setting_key] = text_input

                new_page_sizer.Add(new_page_form_sizer, wx.SizerFlags().Border(wx.ALL, 5))
                new_page.SetSizerAndFit(new_page_sizer)
                self.effect_settings_container.AddPage(new_page, effect_name)
                self.effect_page_indices[effect_name] = idx

            self.record_button = SVGButton(self.main_window, wx.ID_ANY, 'Record', pathlib.Path('./record_button.svg'))
            self.record_button.Bind(wx.EVT_BUTTON, self._on_record_button_click)
            self.pause_button = SVGButton(self.main_window, wx.ID_ANY, 'Record', pathlib.Path('./record_button.svg'))
            self.pause_button.Bind(wx.EVT_BUTTON, self._on_pause_button_click)
            self.pause_button.Show(False)
            self.restart_button = SVGButton(self.main_window, wx.ID_ANY, 'Restart', pathlib.Path('./restart_button.svg'))
            self.restart_button.Bind(wx.EVT_BUTTON, self._on_restart_button_click)
            self.restart_button.Show(False)
            self.elapsed_time_text = wx.StaticText(self.main_window, wx.ID_ANY, '')
            self.elapsed_time_text.Show(False)

            self.main_controls_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
            self.main_controls_sizer.Add(self.record_button)
            self.main_controls_sizer.AddSpacer(5)
            self.main_controls_sizer.Add(self.pause_button)
            self.main_controls_sizer.AddSpacer(5)
            self.main_controls_sizer.Add(self.restart_button)
            self.main_controls_sizer.AddSpacer(5)
            self.main_controls_sizer.Add(self.elapsed_time_text)

            self.right_panel_sizer = wx.BoxSizer(orient=wx.VERTICAL)
            self.right_panel_sizer.Add(self.main_controls_sizer, wx.SizerFlags(proportion=0).Expand())
            self.right_panel_sizer.Add(self.effect_settings_container, wx.SizerFlags(proportion=1).Expand())

            self.main_window_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
            self.main_window_sizer.Add(self.sample_view_sizer, wx.SizerFlags(proportion=1).Expand())
            self.main_window_sizer.Add(self.right_panel_sizer, wx.SizerFlags(proportion=1).Expand())
            self.main_window.SetSizerAndFit(self.main_window_sizer)

            self.sample_snapshots: List[Tuple[np.ndarray, int, str]] = []
            self.active_snapshot_index: Optional[int] = None
            self.project_path: Optional[pathlib.Path] = None
            self.recording_active: bool = False
            self.recording_paused: bool = False
            self.recording_path: Optional[str] = None

            self.wave_axes = None
            self.wave_artist = None
            self.existing_x_bounds = None
            
            self.application_exiting = False

            self.synth_handlers = {IncomingMessageType.LOAD_PROJECT_DATA: self._on_project_load, IncomingMessageType.SET_SAMPLE: self._on_sample_set, IncomingMessageType.UPDATE_RECORDING_STATE: self._on_recording_state_update}
            self.queue_pump_thread = threading.Thread(target=self._invoke_synth_handlers, name='Queue Pump Thread')
            self.queue_pump_thread.start()

            self.SetTopWindow(self.main_window)
            self.main_window.Show()

            return True
        
        def OnExit(self) -> int:
            self.application_exiting = True
            ui_process_recv_queue.put_nowait(None)
            ui_process_send_queue.put_nowait((OutgoingMessageType.APPLICATION_EXIT,))
            self.queue_pump_thread.join()

            return 0           

        def _invoke_synth_handlers(self):
            while True:
                event_data = ui_process_recv_queue.get()

                if self.application_exiting:
                    return
                
                wx.CallAfter(self.synth_handlers[event_data[0]], *event_data[1:])

        def _plot_sample(self, sample: np.ndarray, num_wavelengths: int):
            if self.wave_axes is None or num_wavelengths != self.active_wave_last_drawn_num_wavelengths or len(sample) != self.active_wave_last_drawn_sample_length:
                self.wave_axes = self.active_wave_graph.figure.gca()
                self.wave_axes.clear()
                self.wave_artist = self.wave_axes.plot(np.arange(sample.shape[0]), sample)[0]
                self.wave_axes.set_xticks(np.linspace(0, len(sample), num_wavelengths + 1), [str(x) for x in range(num_wavelengths + 1)])
                self.wave_axes.grid(visible=True, which='major', axis='x')

                new_x_bound_min, new_x_bound_max = self.wave_axes.get_xbound()
                self.existing_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))

                self.active_wave_last_drawn_num_wavelengths = num_wavelengths
                self.active_wave_last_drawn_sample_length = len(sample)

            self.wave_artist.set_ydata(sample)
            self.active_wave_graph.draw()

            self.active_wave_num_wavelengths_text.SetLabelText(str(num_wavelengths) + (' wavelengths' if num_wavelengths != 1 else ' wavelength'))

        def _on_project_load(self, project_path: pathlib.Path, project_samples: List[Tuple[np.ndarray, int, str]], project_effect_settings: Dict[str, Dict[str, int | float | str]]):
            self.project_path = project_path
            self.sample_snapshots = project_samples
            self.active_snapshot_index = len(project_samples) - 1
            self._plot_sample(self.sample_snapshots[self.active_snapshot_index][0], self.sample_snapshots[self.active_snapshot_index][1])

            self.history_view.clear_entries()
            for sample, _, desc in self.sample_snapshots:
                self.history_view.add_entry(sample, desc)

            self.history_view.set_active(self.active_snapshot_index)

            for effect_name, effect_settings in project_effect_settings.items():
                if effect_name in self.effect_setting_controls:
                    for setting_key, setting_value in project_effect_settings[effect_name].items():
                        if setting_key in self.effect_setting_controls[effect_name]:
                            self.effect_setting_controls[effect_name][setting_key].SetValue(setting_value)

        def _on_num_wavelengths_button_click(self, event):
            dialog = ModifyNumWavelengthsDialog(self.main_window, self.sample_snapshots[self.active_snapshot_index][1])
            dialog.ShowModal()
            if dialog.result_wavelengths is not None:
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_SAMPLE_SIZE, dialog.result_wavelengths, dialog.result_add_mode))

        def _on_record_button_click(self, event):
            if self.recording_active:
                ui_process_send_queue.put_nowait((OutgoingMessageType.RECORD_STOP,))
            else:
                file_path = wx.SaveFileSelector(parent=self.main_window, what='Recording destination file', extension='wav')
                if len(file_path) > 0:
                    self.recording_path = file_path
                    ui_process_send_queue.put_nowait((OutgoingMessageType.RECORD_START, file_path))

        def _on_pause_button_click(self, event):
            if self.recording_paused:
                ui_process_send_queue.put_nowait((OutgoingMessageType.RECORD_CONTINUE,))
            else:
                ui_process_send_queue.put_nowait((OutgoingMessageType.RECORD_PAUSE,))

        def _on_restart_button_click(self, event):
            assert self.recording_path is not None
            ui_process_send_queue.put_nowait((OutgoingMessageType.RECORD_START, self.recording_path))

        def _on_sample_history_selection(self, event):
            self.active_snapshot_index = self.history_view.active_index
            assert self.active_snapshot_index is not None
            ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, self.active_snapshot_index))
            self._plot_sample(self.sample_snapshots[self.active_snapshot_index][0], self.sample_snapshots[self.active_snapshot_index][1])

        def _on_active_plot_moved(self, event):
            new_x_bound_min, new_x_bound_max = self.wave_axes.get_xbound()
            new_x_bounds = (int(round(new_x_bound_min)), int(round(new_x_bound_max)))
            if new_x_bounds != self.existing_x_bounds:
                self.existing_x_bounds = new_x_bounds
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_EDIT_WINDOW, self.existing_x_bounds))

        def _on_sample_set(self, new_sample: np.ndarray, num_wavelengths: int, is_checkpoint: bool, checkpoint_desc: str, last_effect_name: str):
            self._plot_sample(new_sample, num_wavelengths)

            if is_checkpoint:
                self.sample_snapshots.append((new_sample, num_wavelengths, checkpoint_desc))
                self.active_snapshot_index = len(self.sample_snapshots) - 1
                self.history_view.add_entry(new_sample, checkpoint_desc)
                self.history_view.set_active(self.active_snapshot_index)

            if len(last_effect_name) > 0:
                self.effect_settings_container.SetSelection(self.effect_page_indices[last_effect_name])

        def _on_recording_state_update(self, recording_active: bool, recording_paused: bool, elapsed_time: Optional[float]):
            if recording_active:
                assert elapsed_time is not None
                self.elapsed_time_text.SetLabelText(f'{int(elapsed_time) // 3600}:{(int(elapsed_time) // 60) % 60:02d}:{int(elapsed_time) % 60:02d}')

            if self.recording_active != recording_active or self.recording_paused != recording_paused:
                if recording_active:
                    self.record_button.set_image(pathlib.Path('./stop_button.svg'))
                    self.record_button.SetLabel('Stop')
                    self.restart_button.Show(True)

                    self.pause_button.Show(True)
                    if recording_paused:
                        self.pause_button.set_image(pathlib.Path('./play_button.svg'))
                        self.pause_button.SetLabel('Play')
                    else:
                        self.pause_button.set_image(pathlib.Path('./pause_button.svg'))
                        self.pause_button.SetLabel('Pause')

                    self.elapsed_time_text.Show(True)
                else:
                    self.recording_path = None
                    self.record_button.set_image(pathlib.Path('./record_button.svg'))
                    self.record_button.SetLabel('Record')
                    self.restart_button.Show(False)
                    self.pause_button.Show(False)
                    self.elapsed_time_text.Show(False)

                self.main_controls_sizer.RepositionChildren(wx.Size(0, 0))
            
            self.recording_active = recording_active
            self.recording_paused = recording_paused

        def _on_effect_setting_changed(self, input_ctl: wx.TextCtrl, effect_name: str, setting_key: str):
            text_value = input_ctl.GetTextValue()
            settings = effect_settings_info[effect_name][setting_key]

            if len(text_value) > 0:
                try:
                    if settings.data_type == int:
                        new_value = int(text_value)
                    elif settings.data_type == float:
                        new_value = float(text_value)
                    else:
                        new_value = text_value

                    ui_process_send_queue.put_nowait((OutgoingMessageType.SET_EFFECT_SETTING_VALUE, effect_name, setting_key, new_value))
                except ValueError:
                    print('Invalid numeric value "' + text_value + '" for setting ' + setting_key + ' of effect ' + effect_name + '.')

        def _set_project_folder(self):
            folder_dialog = wx.DirDialog(parent=self.main_window, message='Select a project folder')
            result = folder_dialog.ShowModal()

            if result == wx.ID_OK:
                current_project_folder = pathlib.Path(folder_dialog.GetPath())
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_PROJECT_FOLDER, current_project_folder))

                print('Current project folder set to ' + str(current_project_folder))

        def _undo_snapshot(self):
            if self.active_snapshot_index is not None and self.active_snapshot_index >= 1:
                self.active_snapshot_index -= 1
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, self.active_snapshot_index))
                self._plot_sample(self.sample_snapshots[self.active_snapshot_index][0], self.sample_snapshots[self.active_snapshot_index][1])
                self.history_view.set_active(self.active_snapshot_index)

        def _redo_snapshot(self):
            if self.active_snapshot_index is not None and self.active_snapshot_index < len(self.sample_snapshots) - 1:
                self.active_snapshot_index += 1
                ui_process_send_queue.put_nowait((OutgoingMessageType.SET_ACTIVE_SNAPSHOT, self.active_snapshot_index))
                self._plot_sample(self.sample_snapshots[self.active_snapshot_index][0], self.sample_snapshots[self.active_snapshot_index][1])
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
    application.MainLoop()

class SampleEditorNativeUI:
    def __init__(self, effect_settings_info: Dict[str, Dict[str, EffectSettings]], initial_setting_values: Dict[str, Dict[str, int | float | str]]):
        self.ui_process_send_queue = multiprocessing.Queue()
        self.ui_process_recv_queue = multiprocessing.Queue()
        self.vis_process = multiprocessing.Process(target=_visualization_worker, args=(self.ui_process_send_queue, self.ui_process_recv_queue, effect_settings_info, initial_setting_values))
        self.vis_process.start()

    def load_project_data(self, project_path: pathlib.Path, project_samples: List[Tuple[np.ndarray, int, str]], project_setting_values: Dict[str, Dict[str, int | float | str]]):
        self.ui_process_recv_queue.put_nowait((IncomingMessageType.LOAD_PROJECT_DATA, project_path, project_samples, project_setting_values))

    def update_current_sample(self, sample: np.ndarray, num_wavelengths: int, is_checkpoint: bool, checkpoint_desc: str, last_effect_name: str):
        self.ui_process_recv_queue.put_nowait((IncomingMessageType.SET_SAMPLE, sample, num_wavelengths, is_checkpoint, checkpoint_desc, last_effect_name))

    def update_recording_status(self, recording_active: bool, recording_paused: bool, elapsed_time: Optional[float]):
        self.ui_process_recv_queue.put_nowait((IncomingMessageType.UPDATE_RECORDING_STATE, recording_active, recording_paused, elapsed_time))

    def get_events(self):
        try:
            while True:
                yield self.ui_process_send_queue.get_nowait()
        except queue.Empty:
            pass