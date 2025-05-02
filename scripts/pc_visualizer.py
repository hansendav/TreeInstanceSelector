import argparse 
import os
from pathlib import Path
import numpy as np

import open3d as o3d
import json 

from utils import read_h5


# comment out following if not on WSL 
os.environ['XDG_SESSION_TYPE'] = 'x11'


# TODO: Fix proper closing of the annotator. 

np.random.seed(42)


class PointCloudVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()
        self.current_index = 0
        self.samples = []
        self.stop_visualization = False

        self.vis.create_window(window_name="Point Cloud Visualizer", width=800, height=600)
        self.opt = self.vis.get_render_option()
        self.opt.point_size = 2
        self.opt.background_color = [.9, .9, .9]
        self.opt.show_coordinate_frame = True
        self.point_color = [0, 0.5, 0]  # Green

        self.vis.add_geometry(self.pcd)

        # Register navigation callbacks
        self.vis.register_key_callback(ord('1'), self.next_sample_callback)
        self.vis.register_key_callback(ord('2'), self.previous_sample_callback)
        self.vis.register_key_callback(ord('q'), self.quit_callback)

    def load_samples(self, samples):
        """Load the samples to visualize."""
        self.samples = samples
        if self.samples:
            self.current_index = 0
            self.show_sample(self.current_index)

    def show_sample(self, index):
        """Display the sample at the given index."""
        if 0 <= index < len(self.samples):
            points = self.samples[index]
            self.pcd.clear()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(np.tile(self.point_color, (len(points), 1)))

            self.vis.clear_geometries()
            self.vis.add_geometry(self.pcd)
            print(f"📂 Showing sample {index + 1}/{len(self.samples)}")

    def next_sample_callback(self, vis):
        """Callback to show the next sample."""
        if self.current_index < len(self.samples) - 1:
            self.current_index += 1
            self.show_sample(self.current_index)
        else:
            print("⚠️ No more samples to show.")

    def previous_sample_callback(self, vis):
        """Callback to show the previous sample."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_sample(self.current_index)
        else:
            print("⚠️ Already at the first sample.")

    def quit_callback(self, vis):
        """Callback to quit the visualization."""
        print("❌ Exiting visualization mode.")
        self.stop_visualization = True
        vis.close()

    def run(self):
        """Run the visualizer."""
        while not self.stop_visualization:
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


def main(args): 

    instances = read_h5(args.file)
    print(f"{len(instances)} instances found in file {args.file}")

    visualizer = PointCloudVisualizer()

    visualizer.load_samples(instances)
    visualizer.run()

    visualizer.close()


if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--file', type=str) 
    args = parser.parse_args() 

    main(args)