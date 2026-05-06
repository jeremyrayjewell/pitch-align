import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from processing import process_chain
from utils import INPUT_DIR, OUTPUT_DIR, ensure_directories, load_audio, make_output_path, save_audio


class PitchAlignApp:
    def __init__(self):
        ensure_directories()
        self.root = tk.Tk()
        self.root.title("pitch-align")
        self.root.geometry("920x860")
        self.root.minsize(820, 720)

        self.is_processing = False
        self.worker_thread = None
        self.process_button = None
        self.cancel_button = None
        self.status_var = tk.StringVar(value="Idle")
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        
        # Timing
        self.process_start_time = None
        self.stage_start_time = None
        self.update_timer_id = None
        self.processing_timeout_seconds = 300.0

        self.pitch_enabled_var = tk.BooleanVar(value=True)
        self.skip_long_files_var = tk.BooleanVar(value=False)
        self.key_var = tk.StringVar(value="D")
        self.scale_var = tk.StringVar(value="major")
        self.pitch_strength_var = tk.DoubleVar(value=0.9)
        self.pitch_mix_var = tk.DoubleVar(value=1.0)
        self.hard_tune_var = tk.BooleanVar(value=False)

        self.dc_remove_enabled_var = tk.BooleanVar(value=True)
        self.highpass_enabled_var = tk.BooleanVar(value=True)
        self.highpass_cutoff_var = tk.DoubleVar(value=80.0)
        self.low_shelf_enabled_var = tk.BooleanVar(value=False)
        self.low_shelf_gain_var = tk.DoubleVar(value=0.0)
        self.noise_gate_enabled_var = tk.BooleanVar(value=False)
        self.noise_gate_threshold_var = tk.DoubleVar(value=-45.0)
        self.noise_gate_release_var = tk.DoubleVar(value=120.0)
        self.mid_boost_enabled_var = tk.BooleanVar(value=True)
        self.mid_boost_gain_var = tk.DoubleVar(value=3.0)
        self.presence_boost_enabled_var = tk.BooleanVar(value=False)
        self.presence_boost_gain_var = tk.DoubleVar(value=0.0)
        self.de_esser_enabled_var = tk.BooleanVar(value=False)
        self.de_esser_intensity_var = tk.DoubleVar(value=0.25)
        self.notch_enabled_var = tk.BooleanVar(value=False)
        self.notch_frequency_var = tk.DoubleVar(value=60.0)
        self.notch_q_var = tk.DoubleVar(value=20.0)
        self.high_cut_enabled_var = tk.BooleanVar(value=False)
        self.high_cut_freq_var = tk.DoubleVar(value=9000.0)
        self.high_cut_mix_var = tk.DoubleVar(value=0.35)
        self.high_shelf_enabled_var = tk.BooleanVar(value=False)
        self.high_shelf_gain_var = tk.DoubleVar(value=0.0)
        self.compression_enabled_var = tk.BooleanVar(value=False)
        self.compression_intensity_var = tk.DoubleVar(value=0.35)
        self.compression_threshold_var = tk.DoubleVar(value=-18.0)
        self.compression_attack_var = tk.DoubleVar(value=12.0)
        self.compression_release_var = tk.DoubleVar(value=90.0)
        self.compression_makeup_var = tk.DoubleVar(value=0.0)
        self.saturation_enabled_var = tk.BooleanVar(value=False)
        self.saturation_amount_var = tk.DoubleVar(value=0.2)
        self.stereo_width_enabled_var = tk.BooleanVar(value=False)
        self.stereo_width_amount_var = tk.DoubleVar(value=0.3)
        self.stereo_balance_enabled_var = tk.BooleanVar(value=False)
        self.stereo_balance_var = tk.DoubleVar(value=0.0)
        self.reverb_enabled_var = tk.BooleanVar(value=False)
        self.reverb_mix_var = tk.DoubleVar(value=0.12)
        self.reverb_decay_var = tk.DoubleVar(value=0.22)
        self.reverb_predelay_var = tk.DoubleVar(value=20.0)
        self.limiter_enabled_var = tk.BooleanVar(value=True)
        self.limiter_ceiling_var = tk.DoubleVar(value=0.98)

        self._build_ui()

    def _format_time(self, seconds):
        """Format seconds into readable time string."""
        if seconds < 1:
            return f"{seconds:.1f}s"
        elif seconds < 60:
            return f"{int(seconds)}s"
        else:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"

    def _update_elapsed_time(self):
        """Periodically update status with elapsed time."""
        if self.is_processing and self.process_start_time:
            elapsed = time.time() - self.process_start_time
            current_status = self.status_var.get()
            
            # Extract the base status (before any time info)
            if current_status.endswith(")") and " (" in current_status:
                # Status strings are built as: "<base stage> (<elapsed>)".
                # Use rsplit to preserve any earlier parentheses present in the base stage text.
                base_status = current_status.rsplit(" (", 1)[0]
            else:
                base_status = current_status
            
            # Update with elapsed time
            formatted_elapsed = self._format_time(elapsed)
            new_status = f"{base_status} ({formatted_elapsed})"
            self.status_var.set(new_status)
            
            # Schedule next update
            self.update_timer_id = self.root.after(500, self._update_elapsed_time)

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        self._build_file_controls(container)
        self._build_scrollable_controls(container)
        self._build_actions(container)

    def _build_file_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="File Controls", padding=12)
        frame.pack(fill="x", pady=(0, 12))

        ttk.Label(frame, text="Input file").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.input_path_var, width=88).grid(row=1, column=0, padx=(0, 8), sticky="ew")
        ttk.Button(frame, text="Browse...", command=self.select_input_file).grid(row=1, column=1, sticky="ew")

        ttk.Label(frame, text="Output file").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.output_path_var, width=88).grid(row=3, column=0, padx=(0, 8), sticky="ew")
        ttk.Button(frame, text="Browse...", command=self.select_output_file).grid(row=3, column=1, sticky="ew")

        frame.columnconfigure(0, weight=1)

    def _build_scrollable_controls(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True, pady=(0, 12))

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.controls_frame = ttk.Frame(canvas)

        self.controls_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas_window = canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))

        self._build_pitch_controls(self.controls_frame)
        self._build_dsp_controls(self.controls_frame)

    def _build_pitch_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Pitch Controls", padding=12)
        frame.pack(fill="x", pady=(0, 12))

        ttk.Checkbutton(frame, text="Enable pitch align", variable=self.pitch_enabled_var).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        ttk.Checkbutton(frame, text="Skip pitch alignment for files >60s", variable=self.skip_long_files_var).grid(
            row=1, column=0, sticky="w", pady=(0, 8)
        )

        ttk.Label(frame, text="Key").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self.key_var,
            values=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
            state="readonly",
            width=8,
        ).grid(row=2, column=1, sticky="w", padx=(8, 18))

        ttk.Label(frame, text="Scale").grid(row=2, column=2, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=self.scale_var,
            values=[
                "major",
                "minor",
                "harmonic minor",
                "melodic minor",
                "major pentatonic",
                "minor pentatonic",
                "blues",
                "dorian",
                "phrygian",
                "lydian",
                "mixolydian",
                "locrian",
                "whole tone",
                "chromatic",
            ],
            state="readonly",
            width=18,
        ).grid(row=2, column=3, sticky="w", padx=(8, 0))

        self._slider_row(frame, 3, "Pitch strength", self.pitch_strength_var, 0.0, 1.0, "Blend to scale")
        self._slider_row(frame, 4, "Pitch mix", self.pitch_mix_var, 0.0, 1.0, "Dry/Wet")
        ttk.Checkbutton(frame, text="Hard tune mode", variable=self.hard_tune_var).grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )

        frame.columnconfigure(4, weight=1)

    def _build_dsp_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="DSP Controls", padding=12)
        frame.pack(fill="both", expand=True)

        controls = [
            ("DC remove", self.dc_remove_enabled_var, None, "Utility"),
            ("Highpass", self.highpass_enabled_var, [(self.highpass_cutoff_var, 40.0, 400.0, "Cutoff Hz")], "Filters"),
            ("Low shelf", self.low_shelf_enabled_var, [(self.low_shelf_gain_var, -12.0, 12.0, "Gain dB")], "Tone"),
            (
                "Noise gate",
                self.noise_gate_enabled_var,
                [
                    (self.noise_gate_threshold_var, -70.0, -10.0, "Threshold dB"),
                    (self.noise_gate_release_var, 20.0, 400.0, "Release ms"),
                ],
                "Dynamics",
            ),
            ("Mid boost", self.mid_boost_enabled_var, [(self.mid_boost_gain_var, 0.0, 12.0, "Gain dB")], "Tone"),
            ("Presence boost", self.presence_boost_enabled_var, [(self.presence_boost_gain_var, 0.0, 12.0, "Gain dB")], "Tone"),
            ("De-esser", self.de_esser_enabled_var, [(self.de_esser_intensity_var, 0.0, 1.0, "Intensity")], "Dynamics"),
            (
                "Notch",
                self.notch_enabled_var,
                [
                    (self.notch_frequency_var, 40.0, 12000.0, "Frequency Hz"),
                    (self.notch_q_var, 1.0, 40.0, "Q"),
                ],
                "Filters",
            ),
            (
                "High cut",
                self.high_cut_enabled_var,
                [
                    (self.high_cut_freq_var, 4000.0, 16000.0, "Freq Hz"),
                    (self.high_cut_mix_var, 0.0, 1.0, "Mix"),
                ],
                "Tone",
            ),
            ("High shelf", self.high_shelf_enabled_var, [(self.high_shelf_gain_var, -12.0, 12.0, "Gain dB")], "Tone"),
            (
                "Compression",
                self.compression_enabled_var,
                [
                    (self.compression_intensity_var, 0.0, 1.0, "Intensity"),
                    (self.compression_threshold_var, -36.0, -6.0, "Threshold dB"),
                    (self.compression_attack_var, 1.0, 60.0, "Attack ms"),
                    (self.compression_release_var, 20.0, 300.0, "Release ms"),
                    (self.compression_makeup_var, 0.0, 12.0, "Makeup dB"),
                ],
                "Dynamics",
            ),
            ("Saturation", self.saturation_enabled_var, [(self.saturation_amount_var, 0.0, 1.0, "Amount")], "Tone"),
            ("Stereo width", self.stereo_width_enabled_var, [(self.stereo_width_amount_var, 0.0, 1.0, "Amount")], "Stereo"),
            ("Stereo balance", self.stereo_balance_enabled_var, [(self.stereo_balance_var, -1.0, 1.0, "Left/Right")], "Stereo"),
            (
                "Reverb",
                self.reverb_enabled_var,
                [
                    (self.reverb_mix_var, 0.0, 0.5, "Wet/Dry"),
                    (self.reverb_decay_var, 0.05, 1.2, "Decay s"),
                    (self.reverb_predelay_var, 0.0, 120.0, "Predelay ms"),
                ],
                "Space",
            ),
            ("Limiter", self.limiter_enabled_var, [(self.limiter_ceiling_var, 0.8, 1.0, "Ceiling")], "Safety"),
        ]

        row = 0
        current_group = None
        for label, enabled_var, sliders, group in controls:
            if group != current_group:
                ttk.Label(frame, text=group, font=("Segoe UI", 10, "bold")).grid(
                    row=row, column=0, sticky="w", pady=(10 if row else 0, 6)
                )
                row += 1
                current_group = group

            ttk.Checkbutton(frame, text=label, variable=enabled_var).grid(row=row, column=0, sticky="w", padx=(0, 12))
            if sliders:
                slider_parent = ttk.Frame(frame)
                slider_parent.grid(row=row, column=1, sticky="ew", pady=4)
                for idx, (variable, min_value, max_value, suffix) in enumerate(sliders):
                    self._slider_row(slider_parent, idx, suffix, variable, min_value, max_value, suffix)
                slider_parent.columnconfigure(1, weight=1)
            else:
                ttk.Label(frame, text="On/Off").grid(row=row, column=1, sticky="w")
            row += 1

        frame.columnconfigure(1, weight=1)

    def _slider_row(self, parent, row, label, variable, min_value, max_value, suffix):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
        ttk.Scale(parent, variable=variable, from_=min_value, to=max_value, orient="horizontal").grid(
            row=row, column=1, sticky="ew", padx=(0, 8)
        )
        ttk.Label(parent, textvariable=self._formatted_var(variable, suffix)).grid(row=row, column=2, sticky="e")

    def _build_actions(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="x")

        self.process_button = ttk.Button(frame, text="Process Audio", command=self.start_processing)
        self.process_button.pack(side="left")
        
        self.cancel_button = ttk.Button(frame, text="Cancel", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(side="left", padx=(10, 0))
        
        ttk.Label(frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(side="right")

    def _formatted_var(self, variable, suffix):
        formatted = tk.StringVar()

        def update_label(*_):
            value = variable.get()
            if abs(value) >= 100:
                formatted.set(f"{value:.0f} {suffix}")
            else:
                formatted.set(f"{value:.2f} {suffix}")

        variable.trace_add("write", update_label)
        update_label()
        return formatted

    def select_input_file(self):
        selected = filedialog.askopenfilename(
            title="Select audio file",
            initialdir=INPUT_DIR,
            filetypes=[("Audio Files", "*.wav *.mp3"), ("WAV Files", "*.wav"), ("MP3 Files", "*.mp3")],
        )
        if selected:
            self.input_path_var.set(selected)
            self.output_path_var.set(str(make_output_path(selected)))

    def select_output_file(self):
        selected = filedialog.asksaveasfilename(
            title="Select output file",
            initialdir=OUTPUT_DIR,
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav")],
        )
        if selected:
            source_path = self.input_path_var.get().strip() or (INPUT_DIR / "untitled.wav")
            self.output_path_var.set(str(make_output_path(source_path, selected)))

    def _collect_settings(self):
        return {
            "pitch_align_enabled": self.pitch_enabled_var.get(),
            "skip_long_files": self.skip_long_files_var.get(),
            "key": self.key_var.get(),
            "scale": self.scale_var.get(),
            "pitch_strength": float(self.pitch_strength_var.get()),
            "pitch_mix": float(self.pitch_mix_var.get()),
            "hard_tune": self.hard_tune_var.get(),
            "dc_remove_enabled": self.dc_remove_enabled_var.get(),
            "highpass_enabled": self.highpass_enabled_var.get(),
            "highpass_cutoff": float(self.highpass_cutoff_var.get()),
            "low_shelf_enabled": self.low_shelf_enabled_var.get(),
            "low_shelf_gain": float(self.low_shelf_gain_var.get()),
            "noise_gate_enabled": self.noise_gate_enabled_var.get(),
            "noise_gate_threshold": float(self.noise_gate_threshold_var.get()),
            "noise_gate_release": float(self.noise_gate_release_var.get()),
            "mid_boost_enabled": self.mid_boost_enabled_var.get(),
            "mid_boost_gain": float(self.mid_boost_gain_var.get()),
            "presence_boost_enabled": self.presence_boost_enabled_var.get(),
            "presence_boost_gain": float(self.presence_boost_gain_var.get()),
            "de_esser_enabled": self.de_esser_enabled_var.get(),
            "de_esser_intensity": float(self.de_esser_intensity_var.get()),
            "notch_enabled": self.notch_enabled_var.get(),
            "notch_frequency": float(self.notch_frequency_var.get()),
            "notch_q": float(self.notch_q_var.get()),
            "high_cut_enabled": self.high_cut_enabled_var.get(),
            "high_cut_freq": float(self.high_cut_freq_var.get()),
            "high_cut_mix": float(self.high_cut_mix_var.get()),
            "high_shelf_enabled": self.high_shelf_enabled_var.get(),
            "high_shelf_gain": float(self.high_shelf_gain_var.get()),
            "compression_enabled": self.compression_enabled_var.get(),
            "compression_intensity": float(self.compression_intensity_var.get()),
            "compression_threshold": float(self.compression_threshold_var.get()),
            "compression_attack": float(self.compression_attack_var.get()),
            "compression_release": float(self.compression_release_var.get()),
            "compression_makeup": float(self.compression_makeup_var.get()),
            "saturation_enabled": self.saturation_enabled_var.get(),
            "saturation_amount": float(self.saturation_amount_var.get()),
            "stereo_width_enabled": self.stereo_width_enabled_var.get(),
            "stereo_width_amount": float(self.stereo_width_amount_var.get()),
            "stereo_balance_enabled": self.stereo_balance_enabled_var.get(),
            "stereo_balance": float(self.stereo_balance_var.get()),
            "reverb_enabled": self.reverb_enabled_var.get(),
            "reverb_mix": float(self.reverb_mix_var.get()),
            "reverb_decay": float(self.reverb_decay_var.get()),
            "reverb_predelay": float(self.reverb_predelay_var.get()),
            "limiter_enabled": self.limiter_enabled_var.get(),
            "limiter_ceiling": float(self.limiter_ceiling_var.get()),
        }

    def start_processing(self):
        if self.is_processing:
            return

        input_path = self.input_path_var.get().strip()
        if not input_path:
            messagebox.showerror("pitch-align", "Select an input audio file first.")
            self.status_var.set("Error")
            return

        output_path = make_output_path(input_path, self.output_path_var.get().strip() or None)
        self.output_path_var.set(str(output_path))

        # Get file info
        input_file = Path(input_path)
        file_size_mb = input_file.stat().st_size / (1024 * 1024)

        self.is_processing = True
        self.process_start_time = time.time()
        self.process_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.status_var.set(f"Loading audio... ({file_size_mb:.1f} MB)")
        self.root.update_idletasks()
        
        # Start elapsed time updater
        self._update_elapsed_time()
        
        self.worker_thread = threading.Thread(target=self._process_audio, args=(input_path, output_path), daemon=False)
        self.worker_thread.start()

    def _process_audio(self, input_path, output_path):
        try:
            # Load audio
            self.root.after(0, lambda: self.status_var.set("Loading audio..."))
            self.root.after(0, self.root.update_idletasks)
            
            load_start = time.time()
            audio, sr = load_audio(input_path)
            load_time = time.time() - load_start
            duration = len(audio) / sr
            self.processing_timeout_seconds = max(300.0, duration * 4.0 + 120.0)
            
            # Processing with progress callback
            def on_progress(stage):
                elapsed = time.time() - self.process_start_time
                # Dynamic timeout scales with track duration to avoid false timeouts on long files.
                if elapsed > self.processing_timeout_seconds:
                    raise TimeoutError(
                        f"Processing timeout after {elapsed:.1f}s (limit {self.processing_timeout_seconds:.1f}s)"
                    )
                status = f"Processing: {stage} ({self._format_time(elapsed)})"
                self.root.after(0, lambda: self.status_var.set(status))
                self.root.after(0, self.root.update_idletasks)
            
            self.root.after(0, lambda: self.status_var.set(f"Processing audio... ({duration:.1f}s duration)"))
            self.root.after(0, self.root.update_idletasks)
            
            process_start = time.time()
            processed = process_chain(audio, sr, self._collect_settings(), progress_callback=on_progress)
            process_time = time.time() - process_start
            
            # Saving
            self.root.after(0, lambda: self.status_var.set("Saving audio..."))
            self.root.after(0, self.root.update_idletasks)
            
            save_start = time.time()
            save_audio(output_path, processed, sr)
            save_time = time.time() - save_start
            
            total_time = time.time() - self.process_start_time
            
            self.root.after(0, lambda: self.status_var.set(f"Complete ✓ ({self._format_time(total_time)})"))
            summary = (
                f"Processing complete!\n\n"
                f"Total time: {self._format_time(total_time)}\n"
                f"  Load: {self._format_time(load_time)}\n"
                f"  Process: {self._format_time(process_time)}\n"
                f"  Save: {self._format_time(save_time)}\n\n"
                f"Output: {output_path}"
            )
            self.root.after(0, lambda: messagebox.showinfo("pitch-align", summary))
        except TimeoutError as e:
            error_text = str(e)
            self.root.after(0, lambda: self.status_var.set("Timeout ✗"))
            self.root.after(0, lambda msg=error_text: messagebox.showerror("pitch-align", f"Processing timed out:\n{msg}"))
        except Exception as exc:
            self.root.after(0, lambda: self.status_var.set("Error ✗"))
            self.root.after(0, lambda: messagebox.showerror("pitch-align", f"Processing failed:\n{str(exc)}"))
        finally:
            self.root.after(0, self._finish_processing)

    def cancel_processing(self):
        if self.is_processing and self.worker_thread:
            self.status_var.set("Cancelling...")
            self.cancel_button.config(state="disabled")
            # Note: We can't actually stop the thread, but we can update the UI
            # The thread will finish on its own, but we'll show cancelled status

    def _finish_processing(self):
        # Cancel the timer
        if self.update_timer_id:
            self.root.after_cancel(self.update_timer_id)
            self.update_timer_id = None
        
        self.is_processing = False
        self.process_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)

    def run(self):
        def on_closing():
            if self.is_processing:
                if messagebox.askyesno("pitch-align", "Processing in progress. Do you want to quit anyway?"):
                    self.root.destroy()
            else:
                self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()
