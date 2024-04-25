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
            'left': [(0, 180), (0, 255), (0, 255), (0, 180), (0, 255), (0, 255)],
            'right': [(0, 180), (0, 255), (0, 255), (0, 180), (0, 255), (0, 255)],
            'stepline1': [(0, 180), (0, 255), (0, 255), (0, 180), (0, 255), (0, 255)],
            'stepline2': [(0, 180), (0, 255), (0, 255), (0, 180), (0, 255), (0, 255)]
        }

        self.sim = rospy.get_param("use_sim_time")

        self.default_values = {
            'left_H_l': rospy.get_param("/left_H_l"),
            'left_S_l': rospy.get_param("/left_S_l"),
            'left_V_l': rospy.get_param("/left_V_l"),
            'left_H_u': rospy.get_param("/left_H_u"),
            'left_S_u': rospy.get_param("/left_S_u"),
            'left_V_u': rospy.get_param("/left_V_u"),
            'right_H_l': rospy.get_param("/right_H_l"),
            'right_S_l': rospy.get_param("/right_S_l"),
            'right_V_l': rospy.get_param("/right_V_l"),
            'right_H_u': rospy.get_param("/right_H_u"),
            'right_S_u': rospy.get_param("/right_S_u"),
            'right_V_u': rospy.get_param("/right_V_u"),
            'stepline1_H_l': rospy.get_param("/stepline1_H_l"),
            'stepline1_S_l': rospy.get_param("/stepline1_S_l"),
            'stepline1_V_l': rospy.get_param("/stepline1_V_l"),
            'stepline1_H_u': rospy.get_param("/stepline1_H_u"),
            'stepline1_S_u': rospy.get_param("/stepline1_S_u"),
            'stepline1_V_u': rospy.get_param("/stepline1_V_u"),
            'stepline2_H_l': rospy.get_param("/stepline2_H_l"),
            'stepline2_S_l': rospy.get_param("/stepline2_S_l"),
            'stepline2_V_l': rospy.get_param("/stepline2_V_l"),
            'stepline2_H_u': rospy.get_param("/stepline2_H_u"),
            'stepline2_S_u': rospy.get_param("/stepline2_S_u"),
            'stepline2_V_u': rospy.get_param("/stepline2_V_u"),
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
            with open('../calib/simu_colours.yaml', 'w') as file:
                yaml.dump(current_values, file, default_flow_style=False)
        else:
            with open('../calib/real_colours.yaml', 'w') as file:
                yaml.dump(current_values, file, default_flow_style=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = ParameterSetterGUI(root)
    root.mainloop()
