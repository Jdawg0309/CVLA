"""Images tab - image loading and processing operations."""

import imgui
from typing import Optional, Callable, Any

from ui.ribbon.ribbon_tab import RibbonTab
from ui.ribbon.ribbon_group import RibbonGroup
from ui.ribbon.ribbon_button import RibbonButton
from state.actions import (
    LoadImage, CreateSampleImage, ApplyKernel, ApplyTransform,
    FlipImageHorizontal, UseResultAsInput, ClearImage,
    SetSelectedKernel, SetSamplePattern, SetSampleSize,
    SetTransformRotation, SetTransformScale, SetImageColorMode,
)


# Kernel options
KERNELS = [
    ("Sobel X", "sobel_x"),
    ("Sobel Y", "sobel_y"),
    ("Laplacian", "laplacian"),
    ("Gaussian 3x3", "gaussian_3x3"),
    ("Gaussian 5x5", "gaussian_5x5"),
    ("Sharpen", "sharpen"),
    ("Emboss", "emboss"),
    ("Edge Detect", "edge_detect"),
]


class ImagesTab(RibbonTab):
    """Images tab with loading, processing, and visualization."""

    def __init__(self):
        groups = [
            RibbonGroup("Source", [
                RibbonButton("Load\nFile", "File", tooltip="Load image from file"),
                RibbonButton(
                    "Sample", "Sam",
                    tooltip="Create sample pattern",
                    dropdown_items=[
                        ("Checkerboard", "checkerboard"),
                        ("Gradient", "gradient"),
                        ("Circle", "circle"),
                        ("Noise", "noise"),
                    ]
                ),
                RibbonButton("Clear", "Clr", tooltip="Clear current image"),
            ]),
            RibbonGroup("Kernels", [
                RibbonButton(
                    "Sobel", "Sob",
                    tooltip="Edge detection (Sobel)",
                    dropdown_items=[
                        ("Sobel X", "sobel_x"),
                        ("Sobel Y", "sobel_y"),
                    ]
                ),
                RibbonButton("Gaussian", "Gau", tooltip="Gaussian blur"),
                RibbonButton("Sharpen", "Sha", tooltip="Sharpen"),
                RibbonButton("Edge", "Edg", tooltip="Edge detection"),
            ]),
            RibbonGroup("Transform", [
                RibbonButton("Rotate", "Rot", tooltip="Rotate image"),
                RibbonButton("Scale", "Sc", tooltip="Scale image"),
                RibbonButton("Flip H", "Flp", tooltip="Flip horizontally"),
            ]),
            RibbonGroup("Pipeline", [
                RibbonButton("Use as\nInput", "<<", tooltip="Use result as new input"),
                RibbonButton("Apply", "Go", tooltip="Apply selected kernel"),
            ]),
        ]
        super().__init__(groups)

        self._selected_kernel = "sobel_x"
        self._rotation = 0.0
        self._scale = 1.0

    def render(
        self,
        state: Any,
        dispatch: Optional[Callable] = None,
        camera: Any = None,
        view_config: Any = None,
    ) -> None:
        """Render Images tab with controls."""
        has_image = state and state.current_image is not None
        has_result = state and state.processed_image is not None

        # Source group
        imgui.begin_group()

        if imgui.button("File\nLoad\nImage", 64, 56):
            # TODO: File dialog
            pass

        if imgui.is_item_hovered():
            imgui.set_tooltip("Load an image from file")

        imgui.same_line()

        if imgui.button("Sam\nSample", 64, 56):
            imgui.open_popup("##sample_popup")

        if imgui.is_item_hovered():
            imgui.set_tooltip("Create a sample pattern image")

        if imgui.begin_popup("##sample_popup"):
            patterns = [
                ("Checkerboard", "checkerboard"),
                ("Gradient", "gradient"),
                ("Circle", "circle"),
                ("Noise", "noise"),
            ]
            for name, pattern in patterns:
                if imgui.menu_item(name)[0] and dispatch:
                    dispatch(SetSamplePattern(pattern=pattern))
                    dispatch(CreateSampleImage())
            imgui.end_popup()

        imgui.same_line()

        if imgui.button("Clr\nClear", 64, 56) and has_image and dispatch:
            dispatch(ClearImage())

        if imgui.is_item_hovered():
            imgui.set_tooltip("Clear the current image")

        imgui.spacing()
        imgui.text_disabled("Source")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Kernel selection and preview
        imgui.begin_group()
        imgui.text("Kernel:")
        imgui.same_line()

        current_kernel = state.selected_kernel if state else "sobel_x"
        current_name = next((k[0] for k in KERNELS if k[1] == current_kernel), "Sobel X")

        imgui.push_item_width(100)
        if imgui.begin_combo("##kernel", current_name):
            for name, kernel_id in KERNELS:
                if imgui.selectable(name, kernel_id == current_kernel)[0]:
                    if dispatch:
                        dispatch(SetSelectedKernel(kernel=kernel_id))
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.same_line()

        if imgui.button("Apply", 60, 24) and has_image and dispatch:
            dispatch(ApplyKernel())

        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply selected kernel to image")

        imgui.spacing()
        imgui.text_disabled("Convolution")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Transform controls
        imgui.begin_group()
        imgui.text("Rotation:")
        imgui.same_line()

        rotation = state.input_transform_rotation if state else 0.0
        imgui.push_item_width(60)
        changed, new_rot = imgui.slider_float("##rot", rotation, 0, 360, "%.0f")
        imgui.pop_item_width()
        if changed and dispatch:
            dispatch(SetTransformRotation(rotation=new_rot))

        imgui.same_line()

        if imgui.button("Apply##rot", 50, 20) and has_image and dispatch:
            dispatch(ApplyTransform())

        if imgui.is_item_hovered():
            imgui.set_tooltip("Apply rotation to image")

        imgui.text("Scale:")
        imgui.same_line()

        scale = state.input_transform_scale if state else 1.0
        imgui.push_item_width(60)
        changed, new_scale = imgui.slider_float("##scale", scale, 0.1, 3.0, "%.1f")
        imgui.pop_item_width()
        if changed and dispatch:
            dispatch(SetTransformScale(scale=new_scale))

        imgui.spacing()
        imgui.text_disabled("Transform")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Color mode
        imgui.begin_group()
        imgui.text("Color Mode:")

        color_mode = state.image_color_mode if state else "rgb"
        modes = [("RGB", "rgb"), ("Gray", "grayscale"), ("Heat", "heatmap")]

        for name, mode in modes:
            is_active = mode == color_mode
            if is_active:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.5, 0.7, 1.0)

            if imgui.button(name, 45, 20):
                if dispatch:
                    dispatch(SetImageColorMode(mode=mode))

            if is_active:
                imgui.pop_style_color()

            imgui.same_line()

        imgui.new_line()
        imgui.spacing()
        imgui.text_disabled("Display")
        imgui.end_group()

        imgui.same_line()
        imgui.dummy(16, 1)
        imgui.same_line()

        # Pipeline controls
        imgui.begin_group()

        if imgui.button("<<\nUse as\nInput", 64, 56) and has_result and dispatch:
            dispatch(UseResultAsInput())

        if imgui.is_item_hovered():
            imgui.set_tooltip("Use the processed result as new input")

        imgui.same_line()

        if imgui.button("Flp\nFlip H", 64, 56) and has_image and dispatch:
            dispatch(FlipImageHorizontal())

        if imgui.is_item_hovered():
            imgui.set_tooltip("Flip image horizontally")

        imgui.spacing()
        imgui.text_disabled("Pipeline")
        imgui.end_group()
