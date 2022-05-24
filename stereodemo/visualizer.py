import copy
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .methods import IntParameter, EnumParameter, StereoOutput, StereoMethod, Calibration, InputPair
   
def show_color_disparity (name: str, disparity_map: np.ndarray):
    min_disp = 0
    max_disp = 64
    norm_disparity_map = 255*((disparity_map-min_disp) / (max_disp-min_disp))
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_MAGMA)
    cv2.namedWindow (name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow (name, disparity_color)

class Settings:
    def __init__(self):
        self.show_axes = False

class Source:
    def __init__(self):
        pass

    @abstractmethod
    def get_next_pair(self) -> InputPair:
        return InputPair(None, None, None, None)


class Visualizer:
    def __init__(self, stereo_methods: Dict[str, StereoMethod], source: Source):
        gui.Application.instance.initialize()

        self.vis = gui.Application.instance
        self.source = source

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor_future = None
        self._progress_dialog = None
        self._last_progress_update_time = None

        self.stereo_methods = stereo_methods
        self.stereo_methods_output = {}
        self.input = InputPair (None, None, None, None)
        self._downsample_factor = 0

        self.window = gui.Application.instance.create_window("Stereo Demo", 1280, 1024)
        w = self.window  # to make the code more concise

        self.settings = Settings()

        # 3D widget
        self._scene = gui.SceneWidget()        
        self._scene.scene = rendering.Open3DScene(w.renderer)
        # self._scene.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XZ)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self._scene.set_on_key(self._on_key_pressed)

        self._clear_outputs ()

        for name, o in self.stereo_methods_output.items():
            if o.point_cloud is not None:
                self._scene.scene.add_geometry(name, o.point_cloud, rendering.MaterialRecord())

        self._reset_camera()

        em = w.theme.font_size
        self.separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._next_image_button = gui.Button("Next Image")
        self._next_image_button.set_on_clicked(self._next_image_clicked)
        self._settings_panel.add_child(self._next_image_button)
        
        horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        label = gui.Label("Input downsampling")
        label.tooltip = "Number of /2 downsampling steps to apply on the input"
        horiz.add_child(label)
        downsampling_slider = gui.Slider(gui.Slider.INT)
        downsampling_slider.set_limits(0, 4)
        downsampling_slider.int_value = self._downsample_factor
        downsampling_slider.set_on_value_changed(self._downsampling_changed)
        horiz.add_child(downsampling_slider)
        self._settings_panel.add_child(horiz)

        self._settings_panel.add_fixed(self.separation_height)

        self.algo_list = gui.ListView()
        self.algo_list.set_items(list(stereo_methods.keys()))
        self.algo_list.selected_index = 0
        self.algo_list.set_max_visible_items(7)
        self.algo_list.set_on_selection_changed(self._on_algo_list_selected)
        self._settings_panel.add_child(self.algo_list)

        self.method_params_proxy = gui.WidgetProxy()
        self._settings_panel.add_child (self.method_params_proxy)

        self.last_runtime = gui.Label("")
        self._settings_panel.add_child (self.last_runtime)

        self.input_status = gui.Label("No input.")
        self._settings_panel.add_child (self.input_status)

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        reset_cam_button = gui.Button("Reset Camera")
        reset_cam_button.set_on_clicked(self._reset_camera)
        view_ctrls.add_child(reset_cam_button)
        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)
        
        horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        label = gui.Label("Max depth (m)")
        label.tooltip = "Max depth to render in meters"
        horiz.add_child(label)
        self.depth_range_slider = gui.Slider(gui.Slider.INT)
        self.depth_range_slider.set_limits(1, 1000)
        self.depth_range_slider.int_value = 100
        self.depth_range_slider.set_on_value_changed(lambda v: self._update_rendering())
        horiz.add_child(self.depth_range_slider)
        view_ctrls.add_child(horiz)
        
        self._settings_panel.add_fixed(self.separation_height)
        self._settings_panel.add_child(view_ctrls)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        
        self._on_algo_list_selected(self.algo_list.selected_value, False)
        self._apply_settings()

        self.read_next_pair ()

    def _on_key_pressed (self, keyEvent):
        if keyEvent.key == gui.KeyName.Q:
            self.vis.quit()
            return gui.SceneWidget.EventCallbackResult.HANDLED
        return gui.SceneWidget.EventCallbackResult.IGNORED
    
    def _downsampling_changed(self, v):
        self._downsample_factor = int(v)
        self._process_input (self.full_res_input)

    def _downsample_input (self, input: InputPair):
        for i in range(0, self._downsample_factor):
            if np.max(input.left_image.shape[:2]) < 250:
                break
            input.left_image = cv2.pyrDown(input.left_image)
            input.right_image = cv2.pyrDown(input.right_image)
            if input.input_disparity is not None:
                input.input_disparity = cv2.pyrDown(input.input_disparity)
            input.calibration.downsample(input.left_image.shape[1], input.left_image.shape[0])

    def read_next_pair (self):
        input = self.source.get_next_pair ()
        self._process_input (input)

    def _process_input (self, input):
        if self._downsample_factor > 0:
            self.full_res_input = input
            input = copy.deepcopy(input)
            self._downsample_input (input)
        else:
            self.full_res_input = input
        cv2.namedWindow ("Input image", cv2.WINDOW_KEEPRATIO)
        cv2.imshow ("Input image", np.hstack([input.left_image, input.right_image]))
        self.input = input
        self.input_status.text = f"Input: {input.left_image.shape[1]}x{input.left_image.shape[0]} " + input.status

        if self.input.has_data():
            assert self.input.left_image.shape[1] == self.input.calibration.width and self.input.left_image.shape[0] == self.input.calibration.height
            self.o3dCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.input.left_image.shape[1],
                                                                        height=self.input.left_image.shape[0],
                                                                        fx=self.input.calibration.fx,
                                                                        fy=self.input.calibration.fy,
                                                                        cx=self.input.calibration.cx0,
                                                                        cy=self.input.calibration.cy)

            self._clear_outputs ()
            self._run_current_method ()

    def update_once (self):
        if self.executor_future is not None:
            self._check_run_complete()
        return gui.Application.instance.run_one_tick()

    def _clear_outputs (self):
        for name in self.stereo_methods.keys():
            self.stereo_methods_output[name] = StereoOutput(
                disparity_pixels=None,
                color_image_bgr=None,
                computation_time=np.nan)
            if self._scene.scene.has_geometry(name):
                self._scene.scene.remove_geometry(name)

    def _reset_camera (self):
        # bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-10, 0,-10]), np.array([0,3,0]))
        bbox = self._scene.scene.bounding_box
        min_bound, max_bound = bbox.min_bound.copy(), bbox.max_bound.copy()
        min_bound[0] = min(min_bound[0], -5)
        min_bound[2] = min(min_bound[2], -5)
        max_bound[0] = max(max_bound[0],  5)
        max_bound[1] = max(max_bound[1],  2)
        max_bound[2] = 0
        bbox.min_bound, bbox.max_bound = min_bound, max_bound

        self._scene.setup_camera(60.0, bbox, np.array([0,0,0]))
        eye = np.array([0, 0.5,  1.0])
        lookat = np.array([0, 0, -1.0])
        up = np.array([0, 1.0, 0])
        self._scene.look_at(lookat, eye, up)

    def _build_stereo_method_widgets(self, name):
        em = self.window.theme.font_size
        method = self.stereo_methods[name]
        container = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        label = gui.Label(method.description)
        label.text_color = gui.Color(1.0, 0.5, 0.0)
        container.add_child(label)
        self._reload_settings_functions = []
        for name, param in method.parameters.items():
            if isinstance(param, IntParameter):
                horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
                label = gui.Label(name)
                label.tooltip = param.description
                horiz.add_child(label)
                slider = gui.Slider(gui.Slider.INT)
                slider.set_limits(param.min, param.max)
                slider.int_value = param.value
                def set_value_from_method(slider=slider, method=method, name=name):
                    slider.int_value = method.parameters[name].value
                self._reload_settings_functions.append(set_value_from_method)
                # workaround late binding
                # https://docs.python-guide.org/writing/gotchas/#:~:text=Python's%20closures%20are%20late%20binding,surrounding%20scope%20at%20call%20time.
                def callback(value, method=method, name=name, slider=slider):
                    p = method.parameters[name]
                    p.set_value(int(value))
                    slider.int_value = p.value
                slider.set_on_value_changed(callback)
                horiz.add_child(slider)
                container.add_child(horiz)
            elif isinstance(param, EnumParameter):
                horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
                label = gui.Label(name)
                label.tooltip = param.description
                horiz.add_child(label)
                combo = gui.Combobox()
                for value in param.values:
                    combo.add_item(value)
                combo.selected_index = param.index
                def callback(combo_idx, combo_val, method=method, name=name, combo=combo):
                    method.parameters[name].set_index(combo.selected_index)
                combo.set_on_selection_changed(callback)
                def set_value_from_method(combo=combo, method=method, name=name):
                    combo.selected_index = method.parameters[name].index
                self._reload_settings_functions.append(set_value_from_method)
                horiz.add_child(combo)
                container.add_child(horiz)
            
        horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))        
        apply_button = gui.Button("Apply")
        apply_button.horizontal_padding_em = 3
        apply_button.set_on_clicked(self._run_current_method)
        horiz.add_child(apply_button)
        horiz.add_fixed(self.separation_height)
        reset_default = gui.Button("Reset defaults")            
        reset_default.set_on_clicked(self._reset_method_defaults)
        horiz.add_child(reset_default)
        container.add_child(horiz)
        return container

    def _on_algo_list_selected(self, name: str, is_dbl_click: bool):
        self.method_params_proxy.set_widget(self._build_stereo_method_widgets(name))
        self._update_runtime ()
        for other_name in self.stereo_methods_output.keys():
            self._scene.scene.show_geometry(other_name, False)
        self._scene.scene.show_geometry(name, True)
        self._apply_settings()
        if self.stereo_methods_output[name].disparity_pixels is None:
            self._run_current_method ()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _next_image_clicked(self):
        self.read_next_pair ()

    def _apply_settings(self):
        self._scene.scene.show_axes(self.settings.show_axes)

    def _reset_method_defaults(self):
        name = self.algo_list.selected_value
        method = self.stereo_methods[name]
        method.reset_defaults()
        for m in self._reload_settings_functions:
            m()

    def _check_run_complete(self):                
        if not self.executor_future.done():
            if self._progress_dialog is None:
                self._progress_dialog = self._show_progress_dialog("Running the current method", f"Computing {self.algo_list.selected_value}...")
            now = time.time()
            if (now - self._last_progress_update_time > 0.1):
                self._last_progress_update_time = now
                self._run_progress.value += (1.0 - self._run_progress.value) / 16.0
            return

        if self._progress_dialog:
            self.window.close_dialog ()
        self._progress_dialog = None

        stereo_output = self.executor_future.result()
        self.executor_future = None

        name = self.algo_list.selected_value
        show_color_disparity (name, stereo_output.disparity_pixels)

        self.stereo_methods_output[name] = stereo_output
        self._update_rendering ([name])
        self._update_runtime ()
    
    def _update_rendering (self, names_to_update=None):
        if names_to_update is None:
            names_to_update = list(self.stereo_methods_output.keys())

        for name in names_to_update:
            stereo_output = self.stereo_methods_output[name]
            if stereo_output.disparity_pixels is None:
                continue

            depth_meters = StereoMethod.depth_meters_from_disparity(stereo_output.disparity_pixels, self.input.calibration)

            if self._scene.scene.has_geometry(name):
                self._scene.scene.remove_geometry(name)

            o3d_color = o3d.geometry.Image(cv2.cvtColor(stereo_output.color_image_bgr, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(depth_meters)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color,
                                                                    o3d_depth,
                                                                    1,
                                                                    depth_trunc=self.depth_range_slider.int_value,
                                                                    convert_rgb_to_intensity=False)
            stereo_output.point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.o3dCameraIntrinsic)
            stereo_output.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            self._scene.scene.add_geometry(name, stereo_output.point_cloud, rendering.MaterialRecord())
    
    def _run_current_method(self):
        if self.executor_future is not None:
            return self._check_run_complete ()

        if not self.input.has_data():
            return

        name = self.algo_list.selected_value

        def do_beefy_work():
            stereo_output = self.stereo_methods[name].compute_disparity (self.input)
            return stereo_output

        self._last_progress_update_time = time.time()
        self.executor_future = self.executor.submit (do_beefy_work)
    
    def _show_progress_dialog(self, title, message):
        # A Dialog is just a widget, so you make its child a layout just like
        # a Window.
        dlg = gui.Dialog(title)

        # Add the message text
        em = self.window.theme.font_size
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        self._run_progress = gui.ProgressBar()
        self._run_progress.value = 0.1  # 10% complete
        prog_layout = gui.Horiz(em)
        prog_layout.add_child(self._run_progress)
        dlg_layout.add_child(prog_layout)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)
        return dlg

    def _update_runtime (self):
        name = self.algo_list.selected_value
        output = self.stereo_methods_output[name]
        if np.isnan(output.computation_time):
            self.last_runtime.text = "No output yet."
        else:
            self.last_runtime.text = f"Computation time: {output.computation_time*1e3:.1f} ms"

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        settings_width = 17 * layout_context.theme.font_size
        r = self.window.content_rect
        self._scene.frame = gui.Rect(0, r.y, r.get_right() - settings_width, r.height)
        # height = min(
        #     r.height,
        #     self._settings_panel.calc_preferred_size(
        #         layout_context, gui.Widget.Constraints()).height)
        height = r.height
        self._settings_panel.frame = gui.Rect(r.get_right() - settings_width, r.y, settings_width, height)
