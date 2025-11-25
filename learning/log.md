* notebook/inference.py： return self._pipeline.run， with_mesh_postprocess=True and with_texture_baking=True

* with_texture_baking=True need set sam3d_objects/model/backbone/tdfy_dit/utils/render_utils.py: renderer.rendering_options.backend = options.get("backend", "inria")

* sam3d_objects/pipeline/inference_pipeline.py: glb = postprocessing_utils.to_glb, simplify=0.6 and texture_size=512

* texture baking mode "fast" or "opt: sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py: bake_texture()

### Workflow

```bash
# run model
output = inference(image, mask, seed=42, with_mesh_postprocess=True, with_texture_baking=True, simplify=0.6, texture_size=512)
|
v
# notebook/inference.py
```

### User case
```bash
# pre-process multi-views GSV and keep the view with maximum mask area
cd /home/user/XJH/MyProjects/sam-3d-objects/notebook
python preprocessing_mask.py --mask_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/masks" --output_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/masks_max"
```

```bash
conda activate sam3d-objects
# soft link package to this version
# change /home/user/miniconda3/envs/sam3d-objects/lib/python3.11/site-packages/sam3d_objects-0.0.1.dist-info/direct_url.json
# {
#     "dir_info": {
#         "editable": true
#     },
#     "url": "file:///home/user/XJH/sam-3d-objects"
# }
# |
# v
# {
#     "dir_info": {
#         "editable": true
#     },
#     "url": "file:///home/user/XJH/MyProjects/sam-3d-objects"
# }

# edit /home/user/miniconda3/envs/sam3d-objects/lib/python3.11/site-packages/_sam3d_objects.pth

cd /home/user/XJH/MyProjects/sam-3d-objects/notebook
# 4 processes
export CUDA_VISIBLE_DEVICES=0 && python batch_demo_single_object.py --image_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/original" --mask_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/masks_max" --output_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/single_object" --simplify 0.995 --texture_size 512 --bake_texture_mode "fast" --process_index 0 --total_processes 4

export CUDA_VISIBLE_DEVICES=1 && python batch_demo_single_object.py --image_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/original" --mask_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/masks_max" --output_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/single_object" --simplify 0.995 --texture_size 512 --bake_texture_mode "fast" --process_index 1 --total_processes 4

export CUDA_VISIBLE_DEVICES=2 && python batch_demo_single_object.py --image_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/original" --mask_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/masks_max" --output_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/single_object" --simplify 0.995 --texture_size 512 --bake_texture_mode "fast" --process_index 2 --total_processes 4

export CUDA_VISIBLE_DEVICES=3 && python batch_demo_single_object.py --image_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/original" --mask_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/masks_max" --output_dir "/home/user/Synology2_FEMA_Data_labeling/FEMA_Work/airborne_lidar/manville/gsv-v1-sam3/single_object" --simplify 0.995 --texture_size 512 --bake_texture_mode "fast" --process_index 3 --total_processes 4
```