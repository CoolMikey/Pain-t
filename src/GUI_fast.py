import datetime
import json
import os
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# My classes
from image_filters import ImageFilters
from pixel_wise_operations import *

PREFERENCES_FILE = "preferences.json"

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    ctk = tk


class ModernImageEditor:

    @property
    def image_size(self):
        return self.preferences.get("image_size", 512)

    def __init__(self, root):
        self.output_thumb_canvas = None
        self.output_thumb_scrollbar = None
        self.output_thumbnails_container = None
        self.root = root
        self.root.title("Pain't")  # Changed title

        # Load and set icon
        try:
            logo_img = Image.open("./images/Logo_big.png")
            logo_img = logo_img.resize((32, 32))
            logo_tk = ImageTk.PhotoImage(logo_img)
            self.root.iconphoto(False, logo_tk)
        except Exception as e:
            print("Failed to load logo:", e)

        self.root.geometry("1700x1000")
        self.root.configure(bg="#1e1e1e")

        # Create thread pool for background tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.active_tasks = []  # Track ongoing tasks
        self.slider_timer = None  # For debouncing slider movements

        self.load_preferences()
        self.setup_ui()
        self.create_menu()

        self.drag_preview = None
        self.drag_data = {
            "widget": None,
            "image_path": None,
            "type": None,
            "custom_image": None
        }
        self.dragged_filter = None
        self.dragged_filter_name = ""
        self.current_filter = None
        self.current_filter_name = ""
        self.current_filter_param = None
        self.input_image = None
        self.output_image = None
        self.operation_type = None
        self.operation_options = None
        self.operation_name = None
        self.configure_styles()

    def setup_ui(self):
        toolbar_frame = tk.Frame(self.root, bg="#1e1e1e", height=60)
        toolbar_frame.pack(side="top", fill="x")

        canvas = tk.Canvas(toolbar_frame, bg="#1e1e1e", height=60, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        self.filters_inner_frame = tk.Frame(canvas, bg="#1e1e1e")
        canvas.create_window((0, 0), window=self.filters_inner_frame, anchor="nw")
        self.filters_inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.setup_toolbar()

        # === LEFT PANEL: INPUT IMAGE THUMBNAILS ===
        self.left_panel = tk.Frame(self.root, bg="#2b2b2b", width=200)
        self.left_panel.pack_propagate(False)
        self.left_panel.pack(side="left", fill="y")
        tk.Label(self.left_panel, text="Input Images", bg="#2b2b2b", fg="white").pack(pady=5)

        # Create the canvas first
        self.input_thumb_canvas = tk.Canvas(self.left_panel, bg="#2b2b2b", highlightthickness=0)

        # Then create the scrollbar and reference the canvas
        self.input_thumb_scrollbar = ttk.Scrollbar(self.left_panel, orient="vertical",
                                                   command=self.input_thumb_canvas.yview,
                                                   style="Dark.Vertical.TScrollbar")

        # Configure the canvas with the scrollbar
        self.input_thumb_canvas.configure(yscrollcommand=self.input_thumb_scrollbar.set)

        # Pack them in the right order
        self.input_thumb_scrollbar.pack(side="right", fill="y")
        self.input_thumb_canvas.pack(side="left", fill="both", expand=True)

        # Create the container inside the canvas
        self.input_thumbnails_container = tk.Frame(self.input_thumb_canvas, bg="#2b2b2b")
        self.input_thumb_canvas.create_window((0, 0), window=self.input_thumbnails_container, anchor="nw")

        # Configure the scrolling
        def on_input_thumb_configure(event):
            self.input_thumb_canvas.configure(scrollregion=self.input_thumb_canvas.bbox("all"))

        self.input_thumbnails_container.bind("<Configure>", on_input_thumb_configure)

        # Load the sample images into the container
        self.load_sample_images(self.input_thumbnails_container, folder="sample_images", image_type="input")

        # === RIGHT PANEL: OUTPUT IMAGE THUMBNAILS ===
        self.right_panel = tk.Frame(self.root, bg="#2b2b2b", width=100)
        self.right_panel.pack_propagate(False)
        self.right_panel.pack(side="right", fill="y")
        refresh_btn = tk.Button(self.right_panel, text="Refresh", command=self.update_output_images_display,
                                bg="#444", fg="white")
        refresh_btn.pack(pady=5)
        tk.Label(self.right_panel, text="Output Images", bg="#2b2b2b", fg="white").pack(pady=5)
        self.output_thumb_canvas = tk.Canvas(self.right_panel, bg="#2b2b2b", highlightthickness=0)
        self.output_thumb_scrollbar = ttk.Scrollbar(self.right_panel, orient="vertical",
                                                    command=self.output_thumb_canvas.yview,
                                                    style="Dark.Vertical.TScrollbar")
        self.output_thumb_canvas.configure(yscrollcommand=self.output_thumb_scrollbar.set)
        self.output_thumb_scrollbar.pack(side="right", fill="y")
        self.output_thumb_canvas.pack(side="left", fill="both", expand=True)
        self.output_thumbnails_container = tk.Frame(self.output_thumb_canvas, bg="#2b2b2b")
        self.output_thumb_canvas.create_window((0, 0), window=self.output_thumbnails_container, anchor="nw")

        def on_thumb_configure(event):  # Enable scrolling when contents overflow
            self.output_thumb_canvas.configure(scrollregion=self.output_thumb_canvas.bbox("all"))

        self.output_thumbnails_container.bind("<Configure>", on_thumb_configure)
        self.update_output_images_display()

        # === CENTER PANEL ===
        self.center_panel = tk.Frame(self.root, bg="#1e1e1e")
        self.center_panel.pack(side="left", fill="both", expand=True)
        self.setup_center_layout()

    def setup_toolbar(self):
        self.setup_toolbar_pixelwise()
        self.setup_toolbar_filters()

    def setup_toolbar_pixelwise(self):
        pixelwise_operations = [
            "grayscale", "adjust_brightness", "adjust_contrast", "negative", "binarize"
        ]

        no_options = {
            'scale_start': None,
            'scale_stop': None,
            'slider_val': None
        }

        options_with_slider = {'scale_start': 0, 'scale_stop': 255, 'slider_val': 128}

        pixelwise_options = {
            "grayscale": no_options,
            "adjust_brightness": {'scale_start': 1, 'scale_stop': 40, 'slider_val': 10},
            "adjust_contrast": {'scale_start': 1, 'scale_stop': 40, 'slider_val': 10},
            "negative": no_options,
            "binarize": options_with_slider
        }

        for pixelwise_operation in pixelwise_operations:
            box = tk.Frame(self.filters_inner_frame, width=70, height=50, bg="#333333", bd=1, relief="raised")
            box.pack(side="left", padx=4, pady=4)
            box.pack_propagate(False)

            label = tk.Label(box, text=pixelwise_operation.replace("_", "\n"), bg="#333333", fg="white",
                             font=("Arial", 7))
            label.pack(expand=True)

            # Bind both box and label to the same events
            for widget in (box, label):
                widget.bind("<ButtonPress-1>",
                            lambda event, name=pixelwise_operation: self.start_pixelwise_drag(event, name))
                widget.bind("<B1-Motion>", lambda event, name=pixelwise_operation: self.do_pixelwise_drag(event, name))
                widget.bind("<ButtonRelease-1>",
                            lambda event, name=pixelwise_operation,
                                   options=pixelwise_options[pixelwise_operation]: self.stop_pixelwise_drag(event, name,
                                                                                                            options))

    def start_pixelwise_drag(self, event, name):
        print("started dragging", name)

        if self.drag_preview:
            self.drag_preview.destroy()

        # Create a floating label (drag preview)
        self.drag_preview = tk.Toplevel(self.root)
        self.drag_preview.overrideredirect(True)  # Remove window borders
        self.drag_preview.attributes("-topmost", True)

        preview_label = tk.Label(self.drag_preview, text=name, bg="#555", fg="white", font=("Arial", 8), padx=5, pady=2)
        preview_label.pack()

        # Position the preview near the mouse
        self.drag_preview.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

    def do_pixelwise_drag(self, event, name):
        # Move drag preview to follow mouse
        if self.drag_preview:
            self.drag_preview.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

    def stop_pixelwise_drag(self, event, name, options):
        if self.drag_preview:
            self.drag_preview.destroy()
            self.drag_preview = None

        x = event.x_root
        y = event.y_root

        if self.operations_canvas.winfo_rootx() <= x <= self.operations_canvas.winfo_rootx() + self.operations_canvas.winfo_width() and \
                self.operations_canvas.winfo_rooty() <= y <= self.operations_canvas.winfo_rooty() + self.operations_canvas.winfo_height():
            print("Stopped inside operations_canvas")
            print("Options:", options)
            self.operation_type = 'bitwise'
            self.operation_options = options
            self.operation_name = name
            print("set self.operation_name to:", self.operation_name)
            if hasattr(self, "operation_label"):
                self.operation_label.config(text=f"Operation: {self.operation_name}")
            self.apply_operation()

        else:
            print("Stopped outside operations_canvas")

        print("Stopped moving", name)

    def apply_operation(self, slider_moved=False):
        if self.operation_type is None:
            return
        elif self.operation_name is None:
            print("operation_name is none")
            return

        # Show loading indicator
        self.show_loading_indicator()

        # Submit operation to thread pool
        future = self.thread_pool.submit(self._process_operation, slider_moved)
        self.active_tasks.append(future)
        future.add_done_callback(self._operation_complete)

    def _process_operation(self, slider_moved):
        """Process the operation in a background thread"""
        try:
            if self.operation_type == 'bitwise':
                # Hide matrix in main thread
                self.root.after(0, self.hide_matrix)

                if self.operation_options['scale_start'] is None:
                    # Hide slider in main thread
                    self.root.after(0, self.hide_slider)

                    if self.operation_name == 'grayscale':
                        return grayscale(image=self.input_image)
                    elif self.operation_name == 'negative':
                        return negative(image=self.input_image)
                    else:
                        print("operation not found:", self.operation_name)
                        return None
                else:
                    if not slider_moved:
                        # Show slider in main thread
                        self.root.after(0, lambda: self.show_slider(
                            min_val=self.operation_options['scale_start'],
                            max_val=self.operation_options['scale_stop'],
                            default_val=self.slider.get()))

                    slider_val = self.operation_options['slider_val']

                    if self.operation_name == 'binarize':
                        return binarize(image=self.input_image, threshold=slider_val)
                    elif self.operation_name == 'adjust_brightness':
                        return adjust_brightness(image=self.input_image, factor=slider_val / 10)
                    elif self.operation_name == 'adjust_contrast':
                        return adjust_contrast(image=self.input_image, factor=slider_val / 10)
                    else:
                        print("Not found bitwise:", self.operation_name)
                        return None

            elif self.operation_type == 'filter':
                options = self.operation_options
                filter_name = self.operation_name
                fast_apply = self.fast_apply_enabled.get()

                if options.get("has_custom_kernel"):
                    try:
                        kernel = [[float(entry.get()) for entry in row] for row in self.matrix_entries]
                        ImageFilters.custom_kernel = kernel
                    except Exception as e:
                        print("Error parsing custom kernel: ", e)
                        return None

                # Special case: filters that use kernel_size from slider
                if filter_name in ["motion_blur_filter", "gaussian_filter"]:
                    if not slider_moved:
                        self.root.after(0, lambda: self.show_slider(
                            min_val=3, max_val=7,
                            default_val=options["kernel_size"],
                            tick_interval=2))

                    kernel_size = self.slider.get()
                    options["kernel_size"] = kernel_size  # update option

                    if filter_name == "motion_blur_filter":
                        # Show matrix in main thread
                        matrix = ImageFilters.get_motion_blur_kernel(kernel_size=kernel_size)
                        self.root.after(0, lambda: self.show_matrix(size=kernel_size, matrix_entries=matrix))

                        return ImageFilters.motion_blur_filter(
                            self.input_image, kernel_size=kernel_size, fast_apply=fast_apply)

                    elif filter_name == "gaussian_filter":
                        sigma = options.get("sigma", None)

                        # Show matrix in main thread
                        matrix = ImageFilters.get_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
                        self.root.after(0, lambda: self.show_matrix(size=kernel_size, matrix_entries=matrix))

                        return ImageFilters.gaussian_filter(
                            self.input_image, kernel_size=kernel_size, sigma=sigma, fast_apply=fast_apply)

                else:
                    # Hide slider in main thread
                    self.root.after(0, self.hide_slider)

                    if filter_name == "averaging_filter":
                        return ImageFilters.averaging_filter(self.input_image, fast_apply)
                    elif filter_name == "sharpening_filter":
                        return ImageFilters.sharpening_filter(self.input_image, fast_apply)
                    elif filter_name == "edge_detection_filter":
                        return ImageFilters.edge_detection_filter(self.input_image, fast_apply)
                    elif filter_name == "emboss_filter":
                        return ImageFilters.emboss_filter(self.input_image, fast_apply)
                    elif filter_name == "horizontal_sobel_filter":
                        return ImageFilters.horizontal_sobel_filter(self.input_image, fast_apply)
                    elif filter_name == "vertical_sobel_filter":
                        return ImageFilters.vertical_sobel_filter(self.input_image, fast_apply)
                    elif filter_name == "custom_filter":
                        kernel = self.get_matrix_from_entries()
                        ImageFilters.custom_kernel = kernel
                        if kernel is None:
                            print("Invalid kernel; cannot apply custom filter.")
                            return None
                        return ImageFilters.custom_filter(self.input_image, fast_apply=fast_apply)
                    else:
                        print("Unknown filter:", filter_name)
                        return None
        except Exception as e:
            print(f"Error processing operation: {e}")
            return None

    def _operation_complete(self, future):
        """Handle operation completion"""
        try:
            output_image = future.result()
            if output_image is not None:
                self.output_image = output_image
                self.output_image.update_image()
                self.show_output_image()
                self.update_stats_on_screen(fast=True)

            # Remove task from active tasks
            if future in self.active_tasks:
                self.active_tasks.remove(future)

            # Hide loading indicator
            self.hide_loading_indicator()
        except Exception as e:
            print(f"Operation completion error: {e}")
            self.hide_loading_indicator()

    def show_loading_indicator(self):
        """Show a loading indicator during processing"""
        # Remove any existing loading indicator
        self.hide_loading_indicator()

        # Create new loading indicator
        self.loading_frame = tk.Frame(self.output_canvas, bg="#3a3a3a")
        self.loading_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.loading_label = tk.Label(self.loading_frame, text="Processing...",
                                      bg="#3a3a3a", fg="white",
                                      font=("Arial", 12, "bold"))
        self.loading_label.pack(pady=10, padx=20)

    def hide_loading_indicator(self):
        """Hide the loading indicator"""
        if hasattr(self, 'loading_frame') and self.loading_frame:
            self.loading_frame.destroy()
            self.loading_frame = None

    def setup_center_layout(self):
        # === TOP AREA: Editor Canvas with Filter Details ===
        top_frame = tk.Frame(self.center_panel, bg="#1e1e1e")
        top_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.input_canvas = self.create_image_box(top_frame, "Input Image", row=0, column=0, bg_color="#3a3a3a")

        # === Filter Controls (now between input and output) ===
        self.filter_details_frame = tk.Frame(top_frame, bg="#1e1e1e", width=250)
        self.filter_details_frame.grid(row=0, column=1, padx=5, sticky="n")
        tk.Label(self.filter_details_frame, text="Filter Details", fg="white", bg="#1e1e1e",
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        self.filter_controls_container = tk.Frame(self.filter_details_frame, bg="#2a2a2a")
        self.filter_controls_container.pack(fill="x", pady=5)
        self.slider = tk.Scale(self.filter_controls_container, from_=0, to=255, orient="horizontal", bg="#2a2a2a",
                               fg="white", command=self.slider_moved)
        self.slider.pack(fill="x", pady=5)
        self.slider.pack_forget()
        self.matrix_frame = tk.Frame(self.filter_controls_container, bg="#2a2a2a")  # Matrix (for kernels etc.)
        self.matrix_frame.pack(pady=5)
        self.matrix_frame.pack_forget()

        self.run_button = tk.Button(self.filter_controls_container, text="Run", command=self.apply_operation, bg="#555",
                                    fg="white")
        self.run_button.pack(pady=10)

        # === Output Canvas and Operations Canvas ===
        self.output_canvas = self.create_image_box(top_frame, "Output Image", row=0, column=2, bg_color="#3a3a3a")
        self.operations_canvas = self.create_operations_box(top_frame, row=0, column=3)

        scroll_container = tk.Frame(self.center_panel, bg="#1e1e1e")
        scroll_container.pack(fill="both", expand=True, padx=10, pady=10)
        scroll_canvas = tk.Canvas(scroll_container, bg="#1e1e1e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=scroll_canvas.yview,
                                  style="Dark.Vertical.TScrollbar")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)
        bottom_frame = tk.Frame(scroll_canvas, bg="#1e1e1e")
        scroll_canvas.create_window((0, 0), window=bottom_frame, anchor="nw")

        def on_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        bottom_frame.bind("<Configure>", on_configure)

        def _on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.hist_box = self.create_analysis_box(bottom_frame, "Histogram", 0)
        self.proj_box = self.create_analysis_box(bottom_frame, "Projection", 1)
        self.diff_box = self.create_analysis_box(bottom_frame, "Difference", 2)

        # Initialize plots
        self.initialize_plots()

        # ======== projekcje ==========#
        self.projection_source_var = tk.StringVar(value="input")
        self.projection_source_var.trace_add("write", lambda *args: self.update_projection_plot())
        self.projection_type_var = tk.StringVar(value="Horizontal")
        self.projection_type_var.trace_add("write", lambda *args: self.update_projection_plot())
        tk.Label(self.proj_controls_frame, text="Source:", bg="#1e1e1e", fg="white").pack(side="left", padx=(0, 5))
        proj_source_menu = tk.OptionMenu(self.proj_controls_frame, self.projection_source_var, "input", "output")
        proj_source_menu.config(bg="#333", fg="white", highlightthickness=0, activebackground="#444",
                                activeforeground="white")
        proj_source_menu["menu"].config(bg="#333", fg="white")
        proj_source_menu.pack(side="left", padx=5)
        tk.Label(self.proj_controls_frame, text="Type:", bg="#1e1e1e", fg="white").pack(side="left", padx=(20, 5))
        proj_type_menu = tk.OptionMenu(self.proj_controls_frame, self.projection_type_var, "Horizontal", "Vertical")
        proj_type_menu.config(bg="#333", fg="white", highlightthickness=0, activebackground="#444",
                              activeforeground="white")
        proj_type_menu["menu"].config(bg="#333", fg="white")
        proj_type_menu.pack(side="left", padx=5)

        # ======== histogramy ========= #
        self.stats_source_var = tk.StringVar(value="input")
        self.histogram_type_var = tk.StringVar(value="RGB")
        tk.Label(self.hist_controls_frame, text="Source:", bg="#1e1e1e", fg="white").pack(side="left", padx=(0, 5))
        hist_source_menu = tk.OptionMenu(self.hist_controls_frame, self.stats_source_var, "input", "output")
        hist_source_menu.config(bg="#333", fg="white", highlightthickness=0, activebackground="#444",
                                activeforeground="white")
        hist_source_menu["menu"].config(bg="#333", fg="white")
        hist_source_menu.pack(side="left", padx=5)
        tk.Label(self.hist_controls_frame, text="Histogram Type:", bg="#1e1e1e", fg="white").pack(side="left",
                                                                                                  padx=(20, 5))
        hist_type_menu = tk.OptionMenu(self.hist_controls_frame, self.histogram_type_var, "RGB", "R", "G", "B",
                                       "Grayscale")
        hist_type_menu.config(bg="#333", fg="white", highlightthickness=0, activebackground="#444",
                              activeforeground="white")
        hist_type_menu["menu"].config(bg="#333", fg="white")
        hist_type_menu.pack(side="left", padx=5)
        calc_btn = tk.Button(self.hist_controls_frame, text="Calculate", command=self.update_stats_on_screen, bg="#555",
                             fg="white")
        calc_btn.pack(side="right", padx=10)

    def initialize_plots(self):
        """Initialize all plots once to avoid recreation"""
        # Histogram plot initialization
        self.setup_histogram_plot()

        # Projection plot initialization
        self.setup_projection_plot()

        # Difference view initialization (using a label)
        self.diff_image_label = tk.Label(self.diff_plot_frame, bg="#2f2f2f")
        self.diff_image_label.pack(expand=True, fill="both")

    def setup_histogram_plot(self):
        """Initialize histogram figure and canvas once"""
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(4, 3))
        self.hist_fig.patch.set_facecolor('#2f2f2f')
        self.hist_ax.set_facecolor('#2f2f2f')

        # Set up appearance
        self.hist_ax.set_xlim(0, 255)
        self.hist_ax.set_xlabel("Pixel Intensity", color="white")
        self.hist_ax.set_ylabel("Frequency", color="white")
        self.hist_ax.tick_params(axis='x', colors='white')
        self.hist_ax.tick_params(axis='y', colors='white')
        self.hist_ax.set_title("Histogram", color="white")

        # Create canvas
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.hist_plot_frame)
        self.hist_canvas.draw()
        self.hist_canvas_widget = self.hist_canvas.get_tk_widget()
        self.hist_canvas_widget.pack(fill="both", expand=True)

    def setup_projection_plot(self):
        """Initialize projection figure and canvas once"""
        self.proj_fig, self.proj_ax = plt.subplots(figsize=(4, 3))
        self.proj_fig.patch.set_facecolor('#2f2f2f')
        self.proj_ax.set_facecolor('#2f2f2f')

        # Set up appearance
        self.proj_ax.set_xlabel("Pixel Index", color="white")
        self.proj_ax.set_ylabel("Intensity Sum", color="white")
        self.proj_ax.tick_params(axis='x', colors='white')
        self.proj_ax.tick_params(axis='y', colors='white')
        self.proj_ax.set_title("Projection", color="white")

        # Create canvas
        self.proj_canvas = FigureCanvasTkAgg(self.proj_fig, master=self.proj_plot_frame)
        self.proj_canvas.draw()
        self.proj_canvas_widget = self.proj_canvas.get_tk_widget()
        self.proj_canvas_widget.pack(fill="both", expand=True)

    def update_projection_plot(self):
        # Use thread pool to calculate projections in background
        self.thread_pool.submit(self._compute_projection_data)

    def _compute_projection_data(self):
        """Calculate projection data in background thread"""
        source = self.projection_source_var.get()
        proj_type = self.projection_type_var.get()

        image_obj = self.input_image if source == "input" else self.output_image
        if not image_obj:
            print("No image for projection.")
            return

        np_img = np.array(image_obj.image)
        height, width = np_img.shape[:2]

        if proj_type == "Horizontal":
            r = np.sum(np_img[:, :, 0], axis=1)
            g = np.sum(np_img[:, :, 1], axis=1)
            b = np.sum(np_img[:, :, 2], axis=1)
            x = np.arange(height)
            title = "Horizontal Projection"
        else:
            r = np.sum(np_img[:, :, 0], axis=0)
            g = np.sum(np_img[:, :, 1], axis=0)
            b = np.sum(np_img[:, :, 2], axis=0)
            x = np.arange(width)
            title = "Vertical Projection"

        # Schedule UI update in main thread
        self.root.after(0, lambda: self._update_projection_plot(x, r, g, b, title, source))

    def _update_projection_plot(self, x, r, g, b, title, source):
        """Update the projection plot with new data"""
        # Clear previous data
        self.proj_ax.clear()

        # Add new data
        self.proj_ax.bar(x, r, color='red', label='Red')
        self.proj_ax.bar(x, g, bottom=r, color='green', label='Green')
        self.proj_ax.bar(x, b, bottom=np.array(r) + np.array(g), color='blue', label='Blue')

        # Update appearance
        self.proj_ax.set_title(f"{source.capitalize()} Image - {title}", color="white")
        self.proj_ax.set_xlabel("Pixel Index", color="white")
        self.proj_ax.set_ylabel("Intensity Sum", color="white")
        self.proj_ax.tick_params(axis='x', colors='white')
        self.proj_ax.tick_params(axis='y', colors='white')

        # Refresh plot
        self.proj_fig.tight_layout()
        self.proj_canvas.draw_idle()  # More efficient than full redraw

    def create_image_box(self, parent, label, row, column, bg_color="#3a3a3a"):
        size = self.image_size
        frame = tk.Frame(parent, bg=bg_color, width=size, height=size, bd=2, relief="groove")
        frame.grid(row=row, column=column, padx=5, pady=5)
        frame.grid_propagate(False)  # Keep frame from resizing to contents

        placeholder = tk.Label(
            frame,
            text=label + "\n(Drop an image here)",
            bg=bg_color,
            fg="white",
            justify="center",
            wraplength=size - 20
        )
        placeholder.pack(expand=True, fill="both", padx=200, pady=200, anchor="center")
        return frame

    def create_analysis_box(self, parent, label, column):
        box = tk.Frame(parent, bg="#2f2f2f", width=self.image_size, height=self.image_size // 2, bd=1, relief="sunken")
        box.grid(row=0, column=column, padx=5, pady=5, sticky="nsew")
        box.grid_propagate(False)
        plot_frame = tk.Frame(box, bg="#2f2f2f")
        plot_frame.pack(expand=True, fill="both")
        controls_frame = tk.Frame(box, bg="#1e1e1e")
        controls_frame.pack(fill="x")

        if label == "Histogram":
            self.hist_plot_frame = plot_frame
            self.hist_controls_frame = controls_frame
        elif label == "Projection":
            self.proj_plot_frame = plot_frame
            self.proj_controls_frame = controls_frame
        elif label == "Difference":
            self.diff_plot_frame = plot_frame
            self.diff_controls_frame = controls_frame

        return box

    def create_operations_box(self, parent, row, column):
        frame = tk.Frame(parent, bg="#252525", width=400, height=400, bd=2, relief="ridge")
        frame.grid(row=row, column=column, padx=10, pady=5)
        self.operation_label = tk.Label(frame, text="No Operation", bg="#252525", fg="white",
                                        font=("Arial", 10, "bold"))
        self.operation_label.pack(pady=5)
        btn_style = {
            "bg": "#444444",
            "fg": "white",
            "activebackground": "#555555",
            "relief": "flat",
            "width": 20
        }

        tk.Button(frame, text="Save Input", command=self.save_input_image, **btn_style).pack(pady=2)
        tk.Button(frame, text="Swap Images", command=self.swap_images, **btn_style).pack(pady=2)
        tk.Button(frame, text="Save Output", command=self.save_output_image, **btn_style).pack(pady=2)
        return frame

    def load_sample_images(self, container, folder="./sample_images", image_type="input"):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for img_file in os.listdir(folder):
            print("image:", img_file)
            if img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                path = os.path.join(folder, img_file)
                img = Image.open(path)
                img.thumbnail((self.preferences.get("thumbnail_size", 80),) * 2)
                img_tk = ImageTk.PhotoImage(img)

                lbl = tk.Label(container, image=img_tk, bg="#2b2b2b", cursor="hand2")
                lbl.image = img_tk  # prevent GC
                lbl.pack(pady=4)

                lbl.bind("<ButtonPress-1>", lambda e, p=path, t=image_type: self.start_drag_image(e, p, t))
                lbl.bind("<B1-Motion>", self.do_drag_image)
                lbl.bind("<ButtonRelease-1>", self.stop_drag_image)

    def create_menu(self):
        menubar = tk.Menu(self.root, bg="#2b2b2b", fg="white", activebackground="#444", activeforeground="white")

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image_file)
        menubar.add_cascade(label="File", menu=file_menu)

        self.fast_apply_enabled = tk.BooleanVar(value=False)
        improvements_menu = tk.Menu(menubar, tearoff=0)
        improvements_menu.add_checkbutton(label="Use Fast Apply", variable=self.fast_apply_enabled)
        menubar.add_cascade(label="Improvements", menu=improvements_menu)

        # options_menu = tk.Menu(menubar, tearoff=0)
        # options_menu.add_command(label="Preferences", command=self.open_preferences_window)
        # menubar.add_cascade(label="Options", menu=options_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def open_preferences_window(self):
        pref_window = tk.Toplevel(self.root)
        pref_window.title("Preferences")
        pref_window.geometry("300x500")
        pref_window.configure(bg="#2b2b2b")

        tk.Label(pref_window, text="Preferences", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)

        image_size_frame = tk.Frame(pref_window, bg="#2b2b2b")
        image_size_frame.pack(pady=5, fill="x", padx=10)

        tk.Label(image_size_frame, text="Default Image Display Size:", bg="#2b2b2b", fg="white").pack(anchor="w")

        sizes = [256, 512, 768]
        self.image_size_var = tk.IntVar(value=self.preferences.get("image_size", 512))

        for size in sizes:
            tk.Radiobutton(
                image_size_frame, text=f"{size}px", variable=self.image_size_var, value=size,
                bg="#2b2b2b", fg="white", selectcolor="#444444", activebackground="#2b2b2b"
            ).pack(anchor="w")

        thumbnail_size_frame = tk.Frame(pref_window, bg="#2b2b2b")
        thumbnail_size_frame.pack(pady=10, fill="x", padx=10)
        tk.Button(pref_window, text="Save & Apply", command=lambda: self.apply_preferences(pref_window)).pack(pady=10)
        thumb_sizes = [64, 80, 96, 128]
        self.thumbnail_size_var = tk.IntVar(value=self.preferences.get("thumbnail_size", 80))

        for size in thumb_sizes:
            tk.Radiobutton(
                thumbnail_size_frame, text=f"{size}px", variable=self.thumbnail_size_var, value=size,
                bg="#2b2b2b", fg="white", selectcolor="#444444", activebackground="#2b2b2b"
            ).pack(anchor="w")
        tk.Button(pref_window, text="Save", command=lambda: self.apply_preferences(pref_window)).pack(pady=10)

    def apply_preferences(self, window):
        selected_image_size = self.image_size_var.get()
        selected_thumb_size = self.thumbnail_size_var.get()
        self.preferences["image_size"] = selected_image_size
        self.preferences["thumbnail_size"] = selected_thumb_size
        self.save_preferences()
        window.destroy()
        for widget in self.root.winfo_children():
            widget.destroy()

        self.setup_ui()
        self.create_menu()  # Ensure menu is recreated too

    def load_preferences(self):
        default_prefs = {
            "image_size": 512,
            "thumbnail_size": 80
        }

        if os.path.exists(PREFERENCES_FILE):
            try:
                with open(PREFERENCES_FILE, "r") as f:
                    prefs = json.load(f)
                    default_prefs.update(prefs)
            except Exception as e:
                print("Error loading preferences:", e)
        self.preferences = default_prefs

    def save_preferences(self):
        try:
            with open(PREFERENCES_FILE, "w") as f:
                json.dump(self.preferences, f, indent=4)
        except Exception as e:
            print("Error saving preferences:", e)

    def setup_toolbar_filters(self):
        filter_names = [
            "averaging_filter", "gaussian_filter", "sharpening_filter", "edge_detection_filter",
            "emboss_filter", "horizontal_sobel_filter", "vertical_sobel_filter",
            "motion_blur_filter", "custom_filter"
        ]

        filter_options = {
            "averaging_filter": {
                "kernel_size": 3,
                "has_custom_kernel": False,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Averaging Filter"
            },
            "gaussian_filter": {
                "kernel_size": 3,
                "sigma": None,  # defaults to kernel_size / 3
                "has_custom_kernel": False,
                "needs_sigma": True,
                "has_matrix": True,
                "label": "Gaussian Filter"
            },
            "sharpening_filter": {
                "kernel_size": 3,
                "has_custom_kernel": False,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Sharpening"
            },
            "edge_detection_filter": {
                "kernel_size": 3,
                "has_custom_kernel": False,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Edge Detection"
            },
            "emboss_filter": {
                "kernel_size": 3,
                "has_custom_kernel": False,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Emboss"
            },
            "horizontal_sobel_filter": {
                "kernel_size": 3,
                "has_custom_kernel": False,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Horizontal Sobel"
            },
            "vertical_sobel_filter": {
                "kernel_size": 3,
                "has_custom_kernel": False,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Vertical Sobel"
            },
            "motion_blur_filter": {
                "kernel_size": 5,
                "has_custom_kernel": True,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Motion Blur"
            },
            "custom_filter": {
                "kernel_size": 3,  # default unless user selects otherwise
                "has_custom_kernel": True,
                "needs_sigma": False,
                "has_matrix": True,
                "label": "Custom Filter"
            }
        }

        for filter_name in filter_names:
            box = tk.Frame(self.filters_inner_frame, width=70, height=50, bg="#444444", bd=1, relief="raised")
            box.pack(side="left", padx=4, pady=4)
            box.pack_propagate(False)

            label = tk.Label(box, text=filter_name.replace("_", "\n"), bg="#444444", fg="white", font=("Arial", 7),
                             justify="center")
            label.pack(expand=True)

            for widget in (box, label):
                widget.bind("<ButtonPress-1>", lambda e, name=filter_name: self.start_filter_drag(e, name))
                widget.bind("<B1-Motion>", self.do_drag_filter)
                widget.bind("<ButtonRelease-1>",
                            lambda e, name=filter_name, options=filter_options[filter_name]: self.stop_filter_drag(e,
                                                                                                                   name,
                                                                                                                   options))

    def start_filter_drag(self, event, name):
        print("Started dragging filter:", name)

        if self.drag_preview:
            self.drag_preview.destroy()

        self.drag_preview = tk.Toplevel(self.root)
        self.drag_preview.overrideredirect(True)
        self.drag_preview.attributes("-topmost", True)

        preview_label = tk.Label(self.drag_preview, text=name, bg="#555", fg="white", font=("Arial", 8), padx=5, pady=2)
        preview_label.pack()

        self.drag_preview.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

    def do_drag_filter(self, event):
        if self.drag_preview:
            self.drag_preview.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

    def stop_filter_drag(self, event, name, options):
        if self.drag_preview:
            self.drag_preview.destroy()
            self.drag_preview = None

        x = event.x_root
        y = event.y_root

        FILTER_KERNEL_GETTERS = {
            "averaging_filter": ImageFilters.get_averaging_kernel,
            "sharpening_filter": ImageFilters.get_sharpening_kernel,
            "edge_detection_filter": ImageFilters.get_edge_detection_kernel,
            "emboss_filter": ImageFilters.get_emboss_kernel,
            "horizontal_sobel_filter": ImageFilters.get_horizontal_sobel_kernel,
            "vertical_sobel_filter": ImageFilters.get_vertical_sobel_kernel,
            "motion_blur_filter": lambda: ImageFilters.get_motion_blur_kernel(
                kernel_size=self.operation_options.get("kernel_size", 3)),
            "gaussian_filter": lambda: ImageFilters.get_gaussian_kernel(
                kernel_size=self.operation_options.get("kernel_size", 3),
                sigma=self.operation_options.get("sigma", None)),
            "custom_filter": lambda: ImageFilters.get_custom_kernel(
                kernel_size=self.operation_options.get("kernel_size", 3))
        }

        if self.operations_canvas.winfo_rootx() <= x <= self.operations_canvas.winfo_rootx() + self.operations_canvas.winfo_width() and \
                self.operations_canvas.winfo_rooty() <= y <= self.operations_canvas.winfo_rooty() + self.operations_canvas.winfo_height():
            print("Dropped filter inside operations_canvas:", name)
            self.operation_type = 'filter'
            self.operation_name = name
            self.operation_options = options
            # self.options
            self.operation_label.config(text=f"Filter: {self.operation_name}")
            if options.get("has_matrix"):
                self.show_matrix(size=options.get("kernel_size", 3),
                                 matrix_entries=FILTER_KERNEL_GETTERS[self.operation_name]())
            else:
                self.hide_matrix()
            self.apply_operation()

            # self.prepare_filter_for_run(name)
        else:
            print("Dropped filter outside canvas.")

    def save_input_image(self):
        os.makedirs("./saved_images", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"input_{timestamp}.png"
        save_path = os.path.join("./saved_images", filename)
        self.input_image.save(save_path)
        self.update_output_images_display()
        print(f"Input image saved as {save_path}")

    def save_output_image(self):
        os.makedirs("./saved_images", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_{timestamp}.png"
        save_path = os.path.join("./saved_images", filename)
        self.output_image.save(save_path)
        self.update_output_images_display()
        print(f"Output image saved as {save_path}")

    def swap_images(self):
        self.input_image, self.output_image = self.output_image, self.input_image
        if self.input_image:
            self.input_image_tk = self.display_image_on_canvas(self.input_canvas, self.input_image)
        if self.output_image:
            self.output_image_tk = self.display_image_on_canvas(self.output_canvas, self.output_image)

    def display_image_on_canvas(self, canvas, image):
        image_tk = ImageTk.PhotoImage(image.image)
        for widget in canvas.winfo_children():
            widget.destroy()
        lbl = tk.Label(canvas, image=image_tk, bg="#3a3a3a")
        lbl.pack(expand=True, fill="both")
        return image_tk

    def start_drag_image(self, event, image_path, image_type):
        self.drag_data["widget"] = event.widget
        self.drag_data["image_path"] = image_path
        self.drag_data["type"] = image_type
        self.drag_data["custom_image"] = CustomImageClass(image_path=image_path)  # <-- new line

        print(f"Started dragging {image_type} image: {image_path}")

        try:  # Thumbnail preview
            thumb_img = self.drag_data["custom_image"].get_thumbnail(size=64)
            self.drag_image_tk = ImageTk.PhotoImage(thumb_img)

            self.drag_preview = tk.Label(self.root, image=self.drag_image_tk, bd=0, highlightthickness=0)
            self.drag_preview.place(x=event.x_root, y=event.y_root)
        except Exception as e:
            print("Failed to load preview image:", e)

    def do_drag_image(self, event):
        if self.drag_preview:
            x = self.root.winfo_pointerx() - self.root.winfo_rootx()
            y = self.root.winfo_pointery() - self.root.winfo_rooty()
            self.drag_preview.place(x=x + 10, y=y + 10)

    def stop_drag_image(self, event):
        # Reset widget visual
        if self.drag_data["widget"]:
            self.drag_data["widget"].config(relief="flat")

        if self.drag_preview:
            self.drag_preview.destroy()
            self.drag_preview = None

        x = self.root.winfo_pointerx()
        y = self.root.winfo_pointery()

        if self.input_canvas.winfo_rootx() <= x <= self.input_canvas.winfo_rootx() + self.input_canvas.winfo_width() and \
                self.input_canvas.winfo_rooty() <= y <= self.input_canvas.winfo_rooty() + self.input_canvas.winfo_height():
            print("Dropped on input image canvas!")
            self.load_image_to_input(self.drag_data["custom_image"])
        else:
            print("Dropped outside input canvas.")

        self.drag_data = {
            "widget": None,
            "image_path": None,
            "type": None,
            "custom_image": None
        }

    def load_image_to_input(self, image_path_or_obj):
        if isinstance(image_path_or_obj, CustomImageClass):
            custom_img = image_path_or_obj
        else:
            custom_img = CustomImageClass(image_path=image_path_or_obj)

        max_dim = 450
        if custom_img.width > max_dim or custom_img.height > max_dim:  # Resize if dimensions are too large
            scale_factor = min(max_dim / custom_img.width, max_dim / custom_img.height)
            new_width = int(custom_img.width * scale_factor)
            new_height = int(custom_img.height * scale_factor)

            custom_img.image = custom_img.image.resize((new_width, new_height))
            custom_img.width, custom_img.height = new_width, new_height
            custom_img.pixel_data = custom_img._convert_to_list()

        self.input_image = custom_img  # Set after resize is done
        self.input_image_tk = ImageTk.PhotoImage(self.input_image.image)

        for widget in self.input_canvas.winfo_children():
            widget.destroy()

        lbl = tk.Label(self.input_canvas, image=self.input_image_tk, bg="#3a3a3a")
        lbl.pack(expand=True, fill="both")

        # Update stats when new image is loaded
        self.update_stats_on_screen(fast=True)
        print("Loaded CustomImageClass object into input canvas.")

    def show_slider(self, min_val=0, max_val=255, default_val=128, tick_interval=1):
        self.slider.config(from_=min_val, to=max_val, resolution=tick_interval)
        self.slider.set(default_val)
        self.slider.pack(fill="x", pady=5)

    def hide_slider(self):
        self.slider.pack_forget()

    def slider_moved(self, value):
        # Debounce slider movements to prevent excessive reprocessing
        if hasattr(self, 'slider_timer') and self.slider_timer:
            self.root.after_cancel(self.slider_timer)

        self.operation_options['slider_val'] = int(value)

        # Wait a brief period before applying operation to reduce UI lag
        self.slider_timer = self.root.after(150, lambda: self.apply_operation(slider_moved=True))

    def show_matrix(self, size, matrix_entries=[]):
        self.hide_matrix()  # Clear existing first

        self.matrix_entries = []

        for row in range(size):
            row_entries = []
            for col in range(size):
                e = tk.Entry(self.matrix_frame, width=4, justify="center")
                if row < len(matrix_entries) and col < len(matrix_entries[row]):
                    value = matrix_entries[row][col]
                else:
                    value = 0
                e.insert(0, str(value))
                e.grid(row=row, column=col, padx=2, pady=2)
                row_entries.append(e)
            self.matrix_entries.append(row_entries)
        self.matrix_frame.pack()

    def hide_matrix(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        self.matrix_frame.pack_forget()

    def get_matrix_from_entries(self):
        kernel = []
        try:
            for row_entries in self.matrix_entries:
                row = []
                for entry in row_entries:
                    val = float(entry.get())
                    row.append(val)
                kernel.append(row)
        except Exception as e:
            print("Error reading kernel from entries:", e)
            return None
        return kernel

    def show_output_image(self):
        if not self.output_image:
            print("No output image to show.")
            return

        self.output_image_tk = ImageTk.PhotoImage(self.output_image.image)

        for widget in self.output_canvas.winfo_children():
            widget.destroy()

        lbl = tk.Label(self.output_canvas, image=self.output_image_tk, bg="#3a3a3a")
        lbl.pack(expand=True, fill="both")

        print("Displayed output image.")

    def update_stats_on_screen(self, fast=False):
        """Update all statistical visualizations"""
        # Submit each update to the thread pool
        self.thread_pool.submit(self._compute_histogram_data)
        self.thread_pool.submit(self._compute_projection_data)

        # Update difference view if both images are available
        if self.input_image and self.output_image:
            self.thread_pool.submit(self._compute_diff_image)

    def _compute_histogram_data(self):
        """Compute histogram data in background thread"""
        source = self.stats_source_var.get()
        hist_type = self.histogram_type_var.get()
        image_obj = self.input_image if source == "input" else self.output_image

        if not image_obj:
            return

        try:
            # Ensure image data is updated
            image_obj.update_image()
            np_img = np.array(image_obj.image)

            # Schedule UI update in main thread with computed data
            if hist_type == "RGB":
                r_data = np_img[:, :, 0].flatten()
                g_data = np_img[:, :, 1].flatten()
                b_data = np_img[:, :, 2].flatten()
                gray_data = None
                title = "RGB Histogram"
            elif hist_type == "R":
                r_data = np_img[:, :, 0].flatten()
                g_data = b_data = gray_data = None
                title = "Red Channel"
            elif hist_type == "G":
                g_data = np_img[:, :, 1].flatten()
                r_data = b_data = gray_data = None
                title = "Green Channel"
            elif hist_type == "B":
                b_data = np_img[:, :, 2].flatten()
                r_data = g_data = gray_data = None
                title = "Blue Channel"
            elif hist_type == "Grayscale":
                gray_data = (0.2989 * np_img[:, :, 0] + 0.5870 * np_img[:, :, 1] + 0.1140 * np_img[:, :, 2]).astype(
                    np.uint8).flatten()
                r_data = g_data = b_data = None
                title = "Grayscale"

            self.root.after(0, lambda: self._update_histogram_plot(
                hist_type, title, r_data, g_data, b_data, gray_data))

        except Exception as e:
            print(f"Error computing histogram: {e}")

    def _update_histogram_plot(self, hist_type, title, r_data, g_data, b_data, gray_data):
        """Update histogram plot with pre-computed data"""
        # Clear previous plot
        self.hist_ax.clear()

        # Update with new data based on histogram type
        if hist_type == "RGB":
            self.hist_ax.hist(r_data, bins=256, color="red", alpha=0.5, label="Red")
            self.hist_ax.hist(g_data, bins=256, color="green", alpha=0.5, label="Green")
            self.hist_ax.hist(b_data, bins=256, color="blue", alpha=0.5, label="Blue")
            self.hist_ax.legend()
        elif hist_type == "R":
            self.hist_ax.hist(r_data, bins=256, color="red")
        elif hist_type == "G":
            self.hist_ax.hist(g_data, bins=256, color="green")
        elif hist_type == "B":
            self.hist_ax.hist(b_data, bins=256, color="blue")
        elif hist_type == "Grayscale":
            self.hist_ax.hist(gray_data, bins=256, color="gray")

        # Set appearance
        self.hist_ax.set_title(title, color="white")
        self.hist_ax.set_xlim(0, 255)
        self.hist_ax.set_xlabel("Pixel Intensity", color="white")
        self.hist_ax.set_ylabel("Frequency", color="white")
        self.hist_ax.tick_params(axis='x', colors='white')
        self.hist_ax.tick_params(axis='y', colors='white')

        # Update the plot efficiently
        self.hist_fig.tight_layout()
        self.hist_canvas.draw_idle()  # More efficient than full redraw

    def _compute_diff_image(self):
        """Compute difference image in background thread"""
        try:
            input_np = np.array(self.input_image.image, dtype=np.int16)
            output_np = np.array(self.output_image.image, dtype=np.int16)

            if input_np.shape != output_np.shape:
                print("Input and output image dimensions do not match")
                return

            # Compute absolute difference
            diff_array = np.abs(input_np - output_np)
            diff_array = np.clip(diff_array, 0, 255).astype(np.uint8)

            # Create PIL image
            pil_diff = Image.fromarray(diff_array)

            # Schedule UI update in main thread
            self.root.after(0, lambda: self._update_diff_display(pil_diff))

        except Exception as e:
            print(f"Error computing difference image: {e}")

    def _update_diff_display(self, pil_diff):
        """Update difference display with pre-computed image"""
        try:
            # Create PhotoImage from PIL image
            self.diff_photo = ImageTk.PhotoImage(pil_diff)

            # Update existing label instead of creating a new one
            self.diff_image_label.config(image=self.diff_photo)

        except Exception as e:
            print(f"Error updating difference display: {e}")

    def update_output_images_display(self):
        folder = "./saved_images"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Remove existing thumbnails
        for widget in self.output_thumbnails_container.winfo_children():
            widget.destroy()

        thumb_size = self.preferences.get("thumbnail_size", 80)

        # Load thumbnails in sorted order (newest first)
        for img_file in sorted(os.listdir(folder), reverse=True):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                path = os.path.join(folder, img_file)
                try:
                    img = Image.open(path)
                    img.thumbnail((thumb_size, thumb_size))
                    img_tk = ImageTk.PhotoImage(img)

                    lbl = tk.Label(self.output_thumbnails_container, image=img_tk, bg="#2b2b2b", cursor="hand2")
                    lbl.image = img_tk  # prevent GC
                    lbl.pack(pady=4, padx=5)

                    # Bind click to load into input
                    lbl.bind("<Button-1>", lambda e, p=path: self.load_image_to_input(p))
                except Exception as e:
                    print(f"Failed to load {img_file}:", e)

    def open_image_file(self):
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Open Image", filetypes=filetypes)

        if filepath:
            print("Selected file:", filepath)
            try:
                self.load_image_to_input(filepath)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")  # Allows customization of scrollbar elements

        style.element_create("custom.Vertical.Scrollbar.trough", "from", "clam")
        style.element_create("custom.Horizontal.Scrollbar.trough", "from", "clam")

        style.configure("Dark.Vertical.TScrollbar",
                        troughcolor="#1e1e1e",
                        background="#444444",
                        darkcolor="#333333",
                        lightcolor="#333333",
                        arrowcolor="white",
                        bordercolor="#1e1e1e",
                        gripcount=0,
                        relief="flat",
                        width=10)

        style.configure("Dark.Horizontal.TScrollbar",
                        troughcolor="#1e1e1e",
                        background="#444444",
                        darkcolor="#333333",
                        lightcolor="#333333",
                        arrowcolor="white",
                        bordercolor="#1e1e1e",
                        gripcount=0,
                        relief="flat",
                        width=10)

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Modern Image Editor\nVersion 2.0\nCreated by Mike Matuszyk for biometria@2025"
        )

    def __del__(self):
        """Clean up resources when the application closes"""
        # Shut down thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)

        # Close matplotlib figures to prevent memory leaks
        if hasattr(self, 'hist_fig'):
            plt.close(self.hist_fig)
        if hasattr(self, 'proj_fig'):
            plt.close(self.proj_fig)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Pain't")
    app = ModernImageEditor(root)
    root.mainloop()