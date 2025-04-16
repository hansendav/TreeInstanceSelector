import argparse 
import os
from pathlib import Path
import numpy as np

import open3d as o3d
import json 

from utils import list_recursive_files, read_h5


# comment out following if not on WSL 
os.environ['XDG_SESSION_TYPE'] = 'x11'

annotations = {}
LABEL_KEYS = {
    '1': 'take', 
    '2': 'leave',
    '9': 'end'
}
    

# TODO: Fix proper closing of the annotator. 


class PointCloudAnnotator: 
    def __init__(self, label_keys: dict, label_log: str): 
        self.label_keys = label_keys
        self.label_log = label_log
        self.annotations = {}
        self.load_annotations(self.label_log)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()
        self.current_file_id = None 
        self.annotation_done = False
        self.stop_annotation = False
 

        self.vis.create_window(window_name="Point Cloud Annotator", width=800, height=600)
        self.opt = self.vis.get_render_option() 
        self.opt.point_size = 4 
        self.opt.background_color = [.9, .9, .9]
        self.opt.show_coordinate_frame = True 
        self.point_color = [0, 0.5, 0] # Green

        
        self.vis.add_geometry(self.pcd)

        for key, label in self.label_keys.items():
            self.vis.register_key_callback(ord(key), self.make_callback(label))

    def set_instance(self, file_id, points): 
        self.current_file_id = file_id 
        self.pcd.clear()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(np.tile(self.point_color, (len(points), 1)))
        
        
        self.vis.clear_geometries()
        self.vis.add_geometry(self.pcd)
        self.annotation_done = False 
        print(f"\n📂 File: {file_id} — Press one of {list(self.label_keys.keys())} to label or q for quit")

    def make_callback(self, label): 
        def callback(vis):
            if label != 'end':
                self.annotations[self.current_file_id] = label
                self.annotation_done = True 
                print(f"✅ Labeled '{self.current_file_id}' as {label}")
                self.save_annotations(self.label_log)
            else: 
                print("❌ Exiting annotation mode.")
                self.save_annotations(self.label_log)
                self.stop_annotation = True  # Set the stop flag
                self.annotation_done = True  # Ensure the loop exits
                vis.close()
        return callback 
        
    def wait_for_annotation(self):
        while not self.annotation_done:
            if self.stop_annotation:  
                break
            self.vis.poll_events()
            self.vis.update_renderer()

    def save_annotations(self, label_log):
        with open(label_log, 'w') as f:
            json.dump(self.annotations, f, indent=4)
        print(f"💾 Saved to '{label_log}'")


    def close(self):
        self.vis.destroy_window()

    def load_annotations(self, label_log): 
        label_path = Path(label_log)
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Ensure the file is not empty
                        self.annotations = json.loads(content)
                        print(f"📂 Loaded {len(self.annotations)} annotations from '{label_log}'")
                    else:
                        print(f"⚠️  Annotation log '{label_log}' is empty. Starting fresh.")
                        self.annotations = {}
            except json.JSONDecodeError:
                print(f"⚠️  Annotation log '{label_log}' is corrupted. Starting fresh.")
                self.annotations = {}
        else:
            print(f"⚠️  Annotation log '{label_log}' does not exist. Starting fresh.")
            self.annotations = {}


def main(args): 
    annotation_log = Path(args.annotation_log)

    if not annotation_log.exists():
        annotation_log.touch()
    

    files_to_process = list_recursive_files(args.data_dir)

    label_keys = {
        '1': 'take', 
        '2': 'leave',
        'q': 'end'  # Ensure 'q' is mapped to 'end'
    }

    annotator = PointCloudAnnotator(label_keys=label_keys, label_log=args.annotation_log)
    

    for file in files_to_process:
        if annotator.stop_annotation:  
            break

        instances = read_h5(file)  
        print(f"{len(instances)} instances found in file {file}")
        
        file_id = Path(file).relative_to(args.data_dir)



        # Loop through each instance and process individually
        for i, instance in enumerate(instances): 
            if annotator.stop_annotation:  
                break

            instance_id = f'{file_id}_{i}'

            # Skip if the current instance is already labeled
            if instance_id in annotator.annotations:
                print(f"⏩ Skipping already-labeled instance: {instance_id}")
                continue
            
            # Otherwise, set the instance for labeling and wait for annotation
            annotator.set_instance(instance_id, instance)
            annotator.wait_for_annotation()

    # Close the annotator when done
    annotator.close()
    print("✅ All files processed. Exiting...")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_dir', type=str) 
    parser.add_argument('--annotation_log', type=str)

    args = parser.parse_args() 

    main(args)