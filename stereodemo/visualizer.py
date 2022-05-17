import time
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import cv2

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .methods import IntParameter, EnumParameter, StereoMethod

@dataclass
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_meters: float

    def to_json(self):
        return json.dumps(self.__dict__)

    def from_json(json_str):
        d = json.loads(json_str)
        return Calibration(**d)

@dataclass
class MethodOutput:
    disparity_pixels: np.ndarray
    computation_time: float
    point_cloud: o3d.geometry.PointCloud
   

def show_color_disparity (name: str, disparity_map: np.ndarray):
    min_disp = 0
    max_disp = 64
    norm_disparity_map = 255*((disparity_map-min_disp) / (max_disp-min_disp))
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_MAGMA)
    cv2.imshow (name, disparity_color)

class Settings:
    def __init__(self):
        self.show_axes = False

@dataclass
class InputPair:
    left_image: np.ndarray
    right_image: np.ndarray
    calibration: Calibration
    status: str

    def has_data(self):
        return self.left_image is not None

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

        self.stereo_methods = stereo_methods
        self.stereo_methods_output = {}
        self.input = InputPair (None, None, None, None)

        self._clear_outputs ()

        self.window = gui.Application.instance.create_window("Stereo Demo", 1024, 768)
        w = self.window  # to make the code more concise

        self.settings = Settings()

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        # self._scene.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XZ)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        for name, o in self.stereo_methods_output.items():
            self._scene.scene.add_geometry(name, o.point_cloud, rendering.MaterialRecord())

        self._reset_camera()

        em = w.theme.font_size
        self.separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._next_image_button = gui.Button("Next Image")
        self._next_image_button.set_on_clicked(self._next_image_clicked)
        self._settings_panel.add_child(self._next_image_button)
        self._settings_panel.add_fixed(self.separation_height)

        self.algo_list = gui.ListView()
        self.algo_list.set_items(list(stereo_methods.keys()))
        self.algo_list.selected_index = 0
        self.algo_list.set_max_visible_items(4)
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
        self._settings_panel.add_fixed(self.separation_height)
        self._settings_panel.add_child(view_ctrls)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        
        self._on_algo_list_selected(self.algo_list.selected_value, False)
        self._apply_settings()

        self.read_next_pair ()

    def read_next_pair (self):
        input = self.source.get_next_pair ()
        cv2.imshow ("Input image", np.hstack([input.left_image, input.right_image]))
        self.input = input
        self.input_status.text = input.status

        if self.input.has_data():
            assert self.input.left_image.shape[1] == self.input.calibration.width and self.input.left_image.shape[0] == self.input.calibration.height
            self.o3dCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.input.left_image.shape[1],
                                                                        height=self.input.left_image.shape[0],
                                                                        fx=self.input.calibration.fx,
                                                                        fy=self.input.calibration.fy,
                                                                        cx=self.input.calibration.cx,
                                                                        cy=self.input.calibration.cy)

            self._clear_outputs ()
            self._run_current_method ()

    def update_once (self):
        return gui.Application.instance.run_one_tick()

    def _clear_outputs (self):
        for name in self.stereo_methods.keys():
            self.stereo_methods_output[name] = MethodOutput(
                disparity_pixels=None,
                computation_time=np.nan,
                point_cloud = o3d.t.geometry.PointCloud()
            )

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
        apply_button = gui.Button("Run")
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
        self._update_method_output (name)
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

    def _run_current_method(self):
        # self.window.show_message_box ("Please wait.", "Computing...")

        if not self.input.has_data():
            return
        name = self.algo_list.selected_value

        output = self.stereo_methods_output[name]

        disparity, computation_time = self.stereo_methods[name].compute_disparity (self.input.left_image, self.input.right_image)
        show_color_disparity (name, disparity)

        old_seterr = np.seterr(divide='ignore')
        depth_meters = np.float32(self.input.calibration.baseline_meters * self.input.calibration.fx) / disparity
        depth_meters = np.nan_to_num(depth_meters)
        depth_meters = np.clip (depth_meters, -1.0, 10.0)
        np.seterr(**old_seterr)


        o3d_left = o3d.geometry.Image(self.input.left_image)
        o3d_depth = o3d.geometry.Image(depth_meters)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_left,
                                                                  o3d_depth,
                                                                  1,
                                                                  depth_trunc=10.0,
                                                                  convert_rgb_to_intensity=False)
        output.point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.o3dCameraIntrinsic)
        output.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        self._scene.scene.remove_geometry(name)
        self._scene.scene.add_geometry(name, output.point_cloud, rendering.MaterialRecord())

        output.disparity_pixels = disparity
        output.computation_time = computation_time
        self._update_method_output (name)

        # self.window.close_dialog ()
    
    def _update_method_output (self, name):
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
