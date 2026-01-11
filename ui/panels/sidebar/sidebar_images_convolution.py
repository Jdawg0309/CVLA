"""
Sidebar image convolution section.
"""

import imgui
from ui.panels.sidebar.sidebar_vision import list_kernels, get_kernel_by_name, apply_kernel


def _render_image_convolution_section(self):
    """Render convolution section."""
    if self.current_image is None:
        return

    if self._section("Convolution (CNN Core)", "Conv"):
        imgui.text_colored("Kernels detect features", 0.8, 0.8, 0.4, 1.0)
        imgui.text_disabled("Select a kernel and apply:")

        imgui.spacing()

        kernels = list_kernels()
        kernel_names = [k[0] for k in kernels]

        imgui.push_item_width(-1)
        if imgui.begin_combo("##kernel", self.selected_kernel):
            for name, desc in kernels:
                label = f"{name}: {desc}"
                if imgui.selectable(label[:40], name == self.selected_kernel)[0]:
                    self.selected_kernel = name
            imgui.end_combo()
        imgui.pop_item_width()

        imgui.spacing()

        kernel = get_kernel_by_name(self.selected_kernel)
        imgui.text("Kernel Matrix:")
        imgui.begin_child("##kernel_view", 0, 80, border=True)
        for row in kernel:
            row_str = "  ".join(f"{v:>6.2f}" for v in row)
            imgui.text(row_str)
        imgui.end_child()

        imgui.spacing()

        if imgui.button("Apply Convolution", width=-1):
            self.processed_image = apply_kernel(
                self.current_image,
                self.selected_kernel,
                normalize_output=True
            )

        imgui.spacing()
        imgui.text("Quick Apply:")
        quick_kernels = ['sobel_x', 'sobel_y', 'gaussian_blur', 'sharpen']
        imgui.columns(2, "##quick_kernels", border=False)
        for i, k in enumerate(quick_kernels):
            if imgui.button(k.replace('_', ' ').title(), width=-1):
                self.selected_kernel = k
                self.processed_image = apply_kernel(
                    self.current_image, k, normalize_output=True
                )
            if i % 2 == 0:
                imgui.next_column()
        imgui.columns(1)

        self._end_section()
