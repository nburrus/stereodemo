from dis import dis
import time
from typing import Dict, List

from dataclasses import dataclass
from abc import abstractmethod

import numpy as np
import cv2

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .methods import IntParameter, StereoMethod

@dataclass
class Calibration:
   width: int
   height: int
   fx: float
   fy: float
   cx: float
   cy: float
   baseline_meters: float

@dataclass
class MethodOutput:
    disparity_pixels: np.ndarray
    computation_time: float
    point_cloud: o3d.geometry.PointCloud
   

def show_color_disparity (name: str, disparity_map: np.ndarray):
    min_disp = 0
    max_disp = disparity_map.shape[1] // 2
    norm_disparity_map = 255*((disparity_map-min_disp) / (max_disp-min_disp))
    disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map, 1), cv2.COLORMAP_MAGMA)
    cv2.imshow (name, disparity_color)

class Settings:
    def __init__(self):
        self.show_axes = False

class Visualizer:
    def __init__(self, stereo_methods: Dict[str, StereoMethod]):
        gui.Application.instance.initialize()

        self.vis = gui.Application.instance

        self.set_input (None, None, None)
        self.stereo_methods = stereo_methods
        self.stereo_methods_output = {}

        for name in self.stereo_methods.keys():
            self.stereo_methods_output[name] = MethodOutput(
                disparity_pixels=None,
                computation_time=np.nan,
                point_cloud = o3d.t.geometry.PointCloud()
            )        

        self.window = gui.Application.instance.create_window("Stereo Demo", 1024, 768)
        w = self.window  # to make the code more concise

        self.settings = Settings()

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XZ)
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

        eye = np.array([0, 0, 0])
        lookat = np.array([0, 0.28, 1.0])
        up = np.array([0, 1, 0])
        self._scene.look_at(eye, lookat, up)

        for name, o in self.stereo_methods_output.items():
            self._scene.scene.add_geometry(name, o.point_cloud, rendering.MaterialRecord())

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._next_image_button = gui.Button("Next Image")
        self._next_image_button.set_on_clicked(self._next_image_clicked)
        view_ctrls.add_child(self._next_image_button)
        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)
        
        self._settings_panel.add_child(view_ctrls)

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

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        
        self._on_algo_list_selected(self.algo_list.selected_value, False)
        self._apply_settings()

    def set_input (self, left_image: np.ndarray, right_image: np.ndarray, calibration: Calibration) -> None:
        self.left_image = left_image
        self.right_image = right_image
        self.calibration = calibration
        if left_image is not None:
            self.o3dCameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width=left_image.shape[1],
                                                                        height=left_image.shape[0],
                                                                        fx=self.calibration.fx,
                                                                        fy=self.calibration.fy,
                                                                        cx=self.calibration.cx,
                                                                        cy=self.calibration.cy)

    def update_once (self):
        return gui.Application.instance.run_one_tick()

    def _build_stereo_method_widgets(self, name):
        em = self.window.theme.font_size
        method = self.stereo_methods[name]
        container = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        for name, param in method.parameters.items():
            if isinstance(param, IntParameter):
                horiz = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
                horiz.add_child(gui.Label(name))
                slider = gui.Slider(gui.Slider.INT)
                slider.set_limits(param.min, param.max)
                slider.int_value = param.value
                callback = lambda value, param=param: setattr(param, 'value', int(value))
                slider.set_on_value_changed(callback)
                horiz.add_child(slider)
                container.add_child(horiz)
            apply_button = gui.Button("Run")
            apply_button.set_on_clicked(self._apply_method_clicked)
        container.add_child(apply_button)
        return container

    def _on_algo_list_selected(self, name: str, is_dbl_click: bool):
        # self.method_params_proxy
        print ("Selected ", self.stereo_methods[name])
        self.method_params_proxy.set_widget(self._build_stereo_method_widgets(name))
        self._update_method_output (name)
        for other_name in self.stereo_methods_output.keys():
            self._scene.scene.show_geometry(other_name, False)
        self._scene.scene.show_geometry(name, True)
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _next_image_clicked(self):
        print ("next image clicked!")

    def _apply_settings(self):
        self._scene.scene.show_axes(self.settings.show_axes)

    def _apply_method_clicked(self):
        if self.left_image is None:
            return
        name = self.algo_list.selected_value

        output = self.stereo_methods_output[name]

        tstart = time.time()
        disparity = self.stereo_methods[name].compute_disparity (self.left_image, self.right_image)
        show_color_disparity (name, disparity)
        tend = time.time()

        depth_meters = np.float32(16.0 * self.calibration.baseline_meters * self.calibration.fx) / disparity
        depth_meters = np.nan_to_num(depth_meters)
        depth_meters = np.clip (depth_meters, -1.0, 10.0)


        o3d_left = o3d.geometry.Image(self.left_image)
        o3d_depth = o3d.geometry.Image(depth_meters)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_left,
                                                                  o3d_depth,
                                                                  1,
                                                                  depth_trunc=5.0,
                                                                  convert_rgb_to_intensity=False)
        output.point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.o3dCameraIntrinsic)
        output.point_cloud.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        self._scene.scene.remove_geometry(name)
        self._scene.scene.add_geometry(name, output.point_cloud, rendering.MaterialRecord())

        computation_time=(tend - tstart)
        output.disparity_pixels = disparity
        output.computation_time = computation_time
        self._update_method_output (name)
    
    def _update_method_output (self, name):
        output = self.stereo_methods_output[name]
        if np.isnan(output.computation_time):
            self.last_runtime.text = "No output yet."
        else:
            self.last_runtime.text = f"Computation time for {name}: {output.computation_time*1e3:.1f} ms"        

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)
