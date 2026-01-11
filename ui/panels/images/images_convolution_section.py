"""
Images tab convolution section.
"""

import imgui
from typing import Callable

from state import (
    AppState,
    Action,
    ApplyKernel,
    SetSelectedKernel,
)
from ui.panels.images.images_tab_constants import KERNEL_OPTIONS


def _render_convolution_section(state: AppState, dispatch: Callable[[Action], None]) -> None:
    """Render convolution controls."""
    imgui.push_style_color(imgui.COLOR_HEADER, 0.18, 0.15, 0.15, 0.8)

    if imgui.collapsing_header("Convolution (CNN Core)", imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        imgui.indent(10)

        imgui.text_colored("Kernels detect features", 0.8, 0.8, 0.4, 1.0)
        imgui.text_disabled("Select a kernel and apply:")
        imgui.spacing()

        imgui.push_item_width(260)
        if imgui.begin_combo("##kernel", state.selected_kernel):
            for name, desc in KERNEL_OPTIONS:
                is_selected = (name == state.selected_kernel)
                label = f"{name}: {desc}"
                if imgui.selectable(label, is_selected)[0]:
                    dispatch(SetSelectedKernel(kernel_name=name))
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.spacing()

        try:
            from domain.images import get_kernel_by_name
            kernel = get_kernel_by_name(state.selected_kernel)
            imgui.text("Kernel Matrix:")
            imgui.begin_child("##kernel_view", 0, 70, border=True)
            for row in kernel:
                row_str = "  ".join(f"{v:>6.2f}" for v in row)
                imgui.text(row_str)
            imgui.end_child()
        except Exception:
            pass

        imgui.spacing()

        if imgui.button("Apply Convolution", width=260):
            dispatch(ApplyKernel(kernel_name=state.selected_kernel))

        imgui.spacing()
        imgui.text("Quick Apply:")
        imgui.columns(2, "##quick", border=False)

        quick_kernels = ['sobel_x', 'sobel_y', 'gaussian_blur', 'sharpen']
        for i, k in enumerate(quick_kernels):
            if imgui.button(k.replace('_', ' ').title(), width=125):
                dispatch(SetSelectedKernel(kernel_name=k))
                dispatch(ApplyKernel(kernel_name=k))
            if i % 2 == 0:
                imgui.next_column()

        imgui.columns(1)

        imgui.unindent(10)
        imgui.spacing()

    imgui.pop_style_color()
