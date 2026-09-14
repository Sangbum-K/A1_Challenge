#!/usr/bin/env python3

import tkinter as tk
import configparser
import os
import rospy
import signal
import sys
import time

class ControlModeButton:
    def __init__(self):
        rospy.init_node('morai_control_button', anonymous=True)

        self.root = tk.Tk()
        self.root.title("Control Mode")
        self.root.geometry("140x110")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)

        self.main_frame = tk.Frame(self.root, bg='#EEEEEE')
        self.main_frame.pack(fill='both', expand=True)

        self.label = tk.Label(self.main_frame, text="Controller", font=('Arial', 10, 'bold'),
                              justify='center', bg='#EEEEEE')
        self.label.pack(pady=2)

        # Variable for throttling
        self.last_click_time = 0
        self.click_cooldown = 0.01  # 1 second

        # INI path
        self.ini_path = os.path.join(os.getenv('PWD'), 'config/simulation.ini')
        self.config = configparser.ConfigParser(comment_prefixes=('/', ';'), allow_no_value=True)

        # Load colors from INI or use defaults
        self.load_colors_from_ini()

        # Button definitions
        self.toggle_buttons = {
            "autonomous": {
                "var": tk.BooleanVar(),
                "section": "MoraiVehicleControl",
                "key": "udp_morai_autonomous_mode",
                "text": ("MANUAL", "AUTONOMOUS"),
                "color": ("manual", "auto"),
                "button": None
            },
            "record": {
                "var": tk.BooleanVar(),
                "section": "DatasetGenerator",
                "key": "record_signal",
                "text": ("Record OFF", "Record ON"),
                "color": ("record_off", "record_on"),
                "button": None
            }
        }

        # Create buttons
        for name in ["autonomous", "record"]:
            btn = tk.Button(
                self.main_frame,
                text="...",
                width=10,
                height=1,
                relief='raised',
                font=('Arial', 8, 'bold'),
                command=lambda n=name: self.throttled_toggle(n)
            )
            btn.pack(expand=True, pady=2)
            btn.bind("<Enter>", lambda e, n=name: self.on_hover(n, True))
            btn.bind("<Leave>", lambda e, n=name: self.on_hover(n, False))
            self.toggle_buttons[name]["button"] = btn

        # Drag behavior
        self._offsetx = 0
        self._offsety = 0
        for widget in [self.main_frame, self.label]:
            widget.bind("<Button-1>", self.click_win)
            widget.bind("<B1-Motion>", self.drag_win)

        # Update states from ini
        self.update_all_button_states()

        signal.signal(signal.SIGINT, self.signal_handler)
        self.root.after(100, self.check_ros)

    def load_colors_from_ini(self):
        """Loads button colors from the [ButtonColors] section of the INI file."""
        default_colors = {
            "manual": ("#FF4444", "#FF6666"),
            "auto": ("#44AA44", "#66CC66"),
            "record_off": ("#E37D00", "#E37D00"),
            "record_on": ("#44AAFF", "#66CCFF"),
        }
        
        self.colors = default_colors.copy()

        try:
            # --- FIX: Read with 'utf-8-sig' to handle BOM ---
            self.config.read(self.ini_path, encoding='utf-8-sig')
            if self.config.has_section('ButtonColors'):
                for key in self.colors: # 'manual', 'auto', etc.
                    base_key = f"{key}_base"
                    hover_key = f"{key}_hover"
                    
                    base_color = self.config.get('ButtonColors', base_key, fallback=self.colors[key][0])
                    hover_color = self.config.get('ButtonColors', hover_key, fallback=self.colors[key][1])
                    
                    self.colors[key] = (base_color, hover_color)
        except Exception as e:
            print(f"[WARN] Failed to read custom colors from INI, using defaults. Error: {e}")
            self.colors = default_colors

    def throttled_toggle(self, name):
        now = time.time()
        if now - self.last_click_time >= self.click_cooldown:
            self.last_click_time = now
            self.toggle_ini_value(name)

    def toggle_ini_value(self, name):
        try:
            # --- FIX: Read with 'utf-8-sig' to handle BOM ---
            with open(self.ini_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()

            key = self.toggle_buttons[name]["key"]
            found = False
            for i, line in enumerate(lines):
                # Make the search more robust
                stripped_line = line.strip()
                if stripped_line.startswith(key) and '=' in stripped_line:
                    current_value = stripped_line.split('=')[1].strip()
                    new_mode = '0' if current_value == '1' else '1'
                    lines[i] = f'{key}={new_mode}\n'
                    found = True
                    break
            
            if not found:
                print(f"Key '{key}' not found in INI.")
                return

            # --- FIX: Write with standard 'utf-8' ---
            with open(self.ini_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            self.update_button_state(name)

        except Exception as e:
            print(f"[ERROR] toggle_ini_value({name}): {e}")

    def update_all_button_states(self):
        for name in self.toggle_buttons:
            self.update_button_state(name)

    def update_button_state(self, name):
        try:
            # --- FIX: Read with 'utf-8-sig' to handle BOM ---
            self.config.read(self.ini_path, encoding='utf-8-sig')
            section = self.toggle_buttons[name]["section"]
            key = self.toggle_buttons[name]["key"]
            value = self.config.getint(section, key) == 1
            self.toggle_buttons[name]["var"].set(value)
            self.update_button_appearance(name)
        except Exception as e:
            print(f"[ERROR] update_button_state({name}): {e}")
            self.toggle_buttons[name]["var"].set(False)
            self.update_button_appearance(name)

    def update_button_appearance(self, name):
        btn_def = self.toggle_buttons[name]
        btn = btn_def["button"]
        is_on = btn_def["var"].get()
        text = btn_def["text"][1] if is_on else btn_def["text"][0]
        base_color, hover_color = self.colors[btn_def["color"][1 if is_on else 0]]
        btn.config(text=text, bg=base_color, activebackground=hover_color)

    def on_hover(self, name, enter):
        btn_def = self.toggle_buttons[name]
        is_on = btn_def["var"].get()
        color_key = btn_def["color"][1 if is_on else 0]
        base, hover = self.colors[color_key]
        btn_def["button"].config(bg=hover if enter else base)

    def click_win(self, event):
        self._offsetx = event.x
        self._offsety = event.y

    def drag_win(self, event):
        x = self.root.winfo_x() + (event.x - self._offsetx)
        y = self.root.winfo_y() + (event.y - self._offsety)
        self.root.geometry(f'+{x}+{y}')

    def check_ros(self):
        if not rospy.is_shutdown():
            self.root.after(100, self.check_ros)
        else:
            self.root.quit()

    def signal_handler(self, signum, frame):
        self.root.quit()
        sys.exit(0)

def main():
    try:
        app = ControlModeButton()
        app.root.mainloop()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    finally:
        if 'app' in locals() and hasattr(app, 'root'):
            app.root.quit()

if __name__ == "__main__":
    main()
