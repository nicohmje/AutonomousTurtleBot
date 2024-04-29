#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
import tkinter as tk
from functools import partial
import yaml  # Import the yaml library

class ParameterSetterGUI:
    def __init__(self, master):
        # ROS Node and Publisher Setup
        rospy.init_node('parameter_setter_gui', anonymous=True)
        self.alert_publisher = rospy.Publisher('/param_change_alert', Bool, queue_size=10)

        # Window setup
        self.master = master
        self.master.title("ROS Parameter Setter")
        
        # Parameters and their range values
        self.param_info = {
            'blue': [(0, 180), (0, 255), (0, 255), (0, 180), (0, 255), (0, 255)],
            'green': [(0, 180), (0, 255), (0, 255), (0, 180), (0, 255), (0, 255)],
            'yellow': [(0, 180), (0, 255), (0, 255), (0, 180), (0, 255), (0, 255)],
        }

        try:
            self.sim = rospy.get_param("use_sim_time")
        except:
            self.sim = False

        self.default_values = {
            'blue_H_l': rospy.get_param("/blue_H_l"),
            'blue_S_l': rospy.get_param("/blue_S_l"),
            'blue_V_l': rospy.get_param("/blue_V_l"),
            'blue_H_u': rospy.get_param("/blue_H_u"),
            'blue_S_u': rospy.get_param("/blue_S_u"),
            'blue_V_u': rospy.get_param("/blue_V_u"),
            'green_H_l': rospy.get_param("/green_H_l"),
            'green_S_l': rospy.get_param("/green_S_l"),
            'green_V_l': rospy.get_param("/green_V_l"),
            'green_H_u': rospy.get_param("/green_H_u"),
            'green_S_u': rospy.get_param("/green_S_u"),
            'green_V_u': rospy.get_param("/green_V_u"),
            'yellow_H_l': rospy.get_param("/yellow_H_l"),
            'yellow_S_l': rospy.get_param("/yellow_S_l"),
            'yellow_V_l': rospy.get_param("/yellow_V_l"),
            'yellow_H_u': rospy.get_param("/yellow_H_u"),
            'yellow_S_u': rospy.get_param("/yellow_S_u"),
            'yellow_V_u': rospy.get_param("/yellow_V_u"),
        }
        
        self.sliders = {}
        self.create_sliders()

    def create_sliders(self):
        row = 0
        for color, ranges in self.param_info.items():
            for i, (hsv, range) in enumerate(zip(['H_l', 'S_l', 'V_l', 'H_u', 'S_u', 'V_u'], ranges)):
                param_name = f"{color}_{hsv}"
                col = i % 3 * 2  # 2 columns per parameter
                label = tk.Label(self.master, text=param_name)
                label.grid(row=row, column=col, sticky="w")
                slider = tk.Scale(self.master, from_=range[0], to=range[1], orient=tk.HORIZONTAL,
                                  label=param_name, length=200, resolution=1)
                slider.set(self.default_values[param_name])
                slider.bind("<ButtonRelease-1>", partial(self.on_slider_change, param_name))
                slider.grid(row=row, column=col + 1, sticky="ew")
                self.sliders[param_name] = slider
                if i % 3 == 2:  # Increment the row index after every three sliders
                    row += 1

    def on_slider_change(self, param_name, event):
        value = self.sliders[param_name].get()
        rospy.set_param(param_name, value)
        self.alert_publisher.publish(True)
        self.write_to_yaml()  # Call the function to write updates to the YAML file

    def write_to_yaml(self):
        current_values = {param: slider.get() for param, slider in self.sliders.items()}
        if self.sim:
            with open('../calib/last_sim_colours.yaml', 'w') as file:
                yaml.dump(current_values, file, default_flow_style=False)
        else:
            with open('../calib/last_real_colours.yaml', 'w') as file:
                yaml.dump(current_values, file, default_flow_style=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = ParameterSetterGUI(root)
    root.mainloop()
