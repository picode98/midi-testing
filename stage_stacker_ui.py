import queue
import multiprocessing
import threading
import dataclasses
import json
from typing import Type

import webview
from stage_stacker_effect import SampleStage

def _ui_runner(incoming_evt_queue: multiprocessing.Queue, outgoing_evt_queue: multiprocessing.Queue):
    class JSAPI:
        def add_effect(self, location: str, type_name: str):
            outgoing_evt_queue.put(('add_effect', location, type_name))

        def reorder_effects(self, location: str, prev_idx: int, new_idx: int):
            outgoing_evt_queue.put(('reorder', location, prev_idx, new_idx))

        def set_parameter(self, location: str, index: int, param_name: str, new_value):
            outgoing_evt_queue.put(('set_param', location, index, param_name, new_value))



    ui_window = webview.create_window(title='Stage Stacker', url='stage_stacker_ui.html',
                                      js_api=JSAPI())

    def _incoming_evt_thread():
        while True:
            msg = incoming_evt_queue.get()
            ui_window.evaluate_js(msg)

    threading.Thread(name='incoming_evt_thread', target=_incoming_evt_thread).start()
    webview.start(debug=True)


class StageStackerWebviewUI:
    def __init__(self):
        self.ui_process_send_queue = multiprocessing.Queue()
        self.ui_process_recv_queue = multiprocessing.Queue()
        self.ui_process = multiprocessing.Process(name='ui_process', target=_ui_runner, args=(self.ui_process_send_queue, self.ui_process_recv_queue))
        self.ui_process.start()

    def register_effect(self, effect_type: Type[SampleStage], location: str):
        self.ui_process_send_queue.put(f"registerEffect('{location}', '{effect_type.__name__}', '{effect_type.display_name or effect_type.__name__}')")

    def add_effect(self, effect: SampleStage, location: str):
        self.ui_process_send_queue.put(f"addEffect('{location}', '{effect.display_name or effect.__class__.__name__}', {json.dumps([dataclasses.asdict(param) for param in effect.get_parameters()])});")

    def update_effect(self, location: str, index: int, effect: SampleStage):
        self.ui_process_send_queue.put(f"updateEffect('{location}', {index}, '{effect.__class__.__name__}', {json.dumps([dataclasses.asdict(param) for param in effect.get_parameters()])});")

    def clear_effects(self, location: str):
        self.ui_process_send_queue.put(f"clearEffects('{location}');")

    def get_events(self):
        try:
            while True:
                yield self.ui_process_recv_queue.get_nowait()
        except queue.Empty:
            pass
